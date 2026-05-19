import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


@dataclass
class CameraRenderRequest:
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    width: int
    height: int
    timestamp: float = 0.0
    dynamics: dict = field(default_factory=dict)
    image_name: str = "camera_render"
    near_plane: float = 0.01
    far_plane: float = 500.0


@dataclass
class RenderTimingRecord:
    image_name: str
    width: int
    height: int
    timestamp: float
    camera_ms: float
    render_ms: float
    postprocess_ms: float
    total_ms: float


def as_camera_intrinsics(value, name):
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape == (3, 3):
        output = np.eye(4, dtype=np.float32)
        output[:3, :3] = matrix
        return output
    if matrix.shape == (4, 4):
        return matrix
    raise ValueError(f"{name} must be a real 3x3 or 4x4 camera intrinsic matrix")


def as_transform(value, name):
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a real 4x4 transform matrix")
    return matrix


def render_to_uint8(render_tensor):
    image = render_tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (image * 255.0 + 0.5).astype(np.uint8)


class GaussianSceneRenderer:
    def __init__(self, scene_path, collect_timing=False):
        from gaussian_renderer import GaussianModel, render
        from scene.obj_model import ObjModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for HUGSIM 3DGS rendering")

        self.scene_path = Path(scene_path)
        cfg = OmegaConf.load(self.scene_path / "cfg.yaml")

        self.gaussians = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)
        model_params, _ = torch.load(self.scene_path / "scene.pth")
        self.gaussians.restore(model_params, None)

        self.dynamic_gaussians = {}
        for ckpt_path in sorted(self.scene_path.glob("dynamic_*.pth")):
            track_id = ckpt_path.stem.split("_", 1)[1]
            dynamic_model = ObjModel(cfg.model.sh_degree, feat_mutable=False)
            dynamic_params, _ = torch.load(ckpt_path)
            dynamic_model.restore(dynamic_params, None)
            self.dynamic_gaussians[track_id] = dynamic_model

        bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._render = render
        self.previous_camera = None
        self.collect_timing = collect_timing
        self.timing_records = []

    def reset_temporal_state(self):
        self.previous_camera = None

    def enable_timing(self, clear=True):
        self.collect_timing = True
        if clear:
            self.clear_timing()

    def disable_timing(self):
        self.collect_timing = False

    def clear_timing(self):
        self.timing_records.clear()

    def timing_summary(self):
        if not self.timing_records:
            return {}
        total = np.asarray([record.total_ms for record in self.timing_records], dtype=np.float64)
        render = np.asarray([record.render_ms for record in self.timing_records], dtype=np.float64)
        return {
            "count": len(self.timing_records),
            "total_ms_avg": float(total.mean()),
            "total_ms_min": float(total.min()),
            "total_ms_max": float(total.max()),
            "render_ms_avg": float(render.mean()),
            "render_ms_min": float(render.min()),
            "render_ms_max": float(render.max()),
        }

    def print_timing_summary(self, title="Render timing"):
        summary = self.timing_summary()
        if not summary:
            print(f"{title}: no timing records")
            return
        print(
            f"{title}: count={summary['count']} "
            f"total_ms(avg/min/max)={summary['total_ms_avg']:.3f}/"
            f"{summary['total_ms_min']:.3f}/{summary['total_ms_max']:.3f} "
            f"render_ms(avg/min/max)={summary['render_ms_avg']:.3f}/"
            f"{summary['render_ms_min']:.3f}/{summary['render_ms_max']:.3f}"
        )

    def save_timing_csv(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame_index",
                    "image_name",
                    "width",
                    "height",
                    "timestamp",
                    "camera_ms",
                    "render_ms",
                    "postprocess_ms",
                    "total_ms",
                ]
            )
            for index, record in enumerate(self.timing_records):
                writer.writerow(
                    [
                        index,
                        record.image_name,
                        record.width,
                        record.height,
                        record.timestamp,
                        f"{record.camera_ms:.6f}",
                        f"{record.render_ms:.6f}",
                        f"{record.postprocess_ms:.6f}",
                        f"{record.total_ms:.6f}",
                    ]
                )
        print(f"Saved render timing to {output_path}")

    def render_image(
        self,
        intrinsics,
        camera_to_world,
        width,
        height,
        timestamp=0.0,
        dynamics=None,
        image_name="camera_render",
        near_plane=0.01,
        far_plane=500.0,
    ):
        request = CameraRenderRequest(
            intrinsics=as_camera_intrinsics(intrinsics, "camera intrinsics"),
            camera_to_world=as_transform(camera_to_world, "camera_to_world"),
            width=int(width),
            height=int(height),
            timestamp=timestamp,
            dynamics=dynamics or {},
            image_name=image_name,
            near_plane=float(near_plane),
            far_plane=float(far_plane),
        )
        return self.render_request(request)

    def render_request(self, request, postprocess=None):
        total_start = time.perf_counter()
        camera_start = time.perf_counter()
        viewpoint = self._make_camera(request)
        camera_ms = (time.perf_counter() - camera_start) * 1000.0
        previous_camera = self.previous_camera or viewpoint

        if self.collect_timing:
            render_start = torch.cuda.Event(enable_timing=True)
            render_end = torch.cuda.Event(enable_timing=True)
            render_start.record()

        with torch.no_grad():
            render_pkg = self._render(
                viewpoint,
                previous_camera,
                self.gaussians,
                self.dynamic_gaussians,
                None,
                self.background,
                False,
                near_plane=request.near_plane,
                far_plane=request.far_plane,
            )

        if self.collect_timing:
            render_end.record()
            render_end.synchronize()
            render_ms = render_start.elapsed_time(render_end)
        else:
            render_ms = 0.0

        self.previous_camera = viewpoint

        postprocess_start = time.perf_counter()
        image = render_to_uint8(render_pkg["render"])
        if postprocess is not None:
            image = postprocess(image)
        if self.collect_timing:
            torch.cuda.synchronize()
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        if self.collect_timing:
            self.timing_records.append(
                RenderTimingRecord(
                    image_name=request.image_name,
                    width=request.width,
                    height=request.height,
                    timestamp=request.timestamp,
                    camera_ms=camera_ms,
                    render_ms=render_ms,
                    postprocess_ms=postprocess_ms,
                    total_ms=total_ms,
                )
            )

        return image

    def _make_camera(self, request):
        from scene.cameras import Camera

        image = np.zeros((request.height, request.width, 3), dtype=np.float32)
        dynamics = {
            str(track_id): torch.tensor(transform, dtype=torch.float32, device="cuda")
            for track_id, transform in request.dynamics.items()
        }
        return Camera(
            width=request.width,
            height=request.height,
            image=image,
            K=request.intrinsics,
            c2w=request.camera_to_world,
            image_name=request.image_name,
            data_device="cpu",
            timestamp=request.timestamp,
            dynamics=dynamics,
        )
