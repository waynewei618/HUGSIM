import sys
import time
import csv
import math
from dataclasses import dataclass, field, replace
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
    tile_width: int = 0
    tile_height: int = 0
    tile_overlap: int = 0
    max_splat_radius: float = 0.0
    appearance_camera_to_world: np.ndarray | None = None


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
        self.default_background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.background = self.default_background.clone()
        self._render = render
        self.previous_camera = None
        self.collect_timing = collect_timing
        self.timing_records = []

    def reset_temporal_state(self):
        self.previous_camera = None

    def set_background_color(self, color):
        if color is None:
            self.background = self.default_background.clone()
            return
        background = np.asarray(color, dtype=np.float32)
        if background.shape != (3,):
            raise ValueError("background color must contain three RGB values")
        self.background = torch.tensor(background, dtype=torch.float32, device="cuda")

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
        tile_width=0,
        tile_height=0,
        tile_overlap=0,
        max_splat_radius=0.0,
        appearance_camera_to_world=None,
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
            tile_width=int(tile_width),
            tile_height=int(tile_height),
            tile_overlap=int(tile_overlap),
            max_splat_radius=float(max_splat_radius),
            appearance_camera_to_world=(
                as_transform(appearance_camera_to_world, "appearance_camera_to_world")
                if appearance_camera_to_world is not None
                else None
            ),
        )
        return self.render_request(request)

    def render_request(self, request, postprocess=None):
        if request.tile_width > 0 and request.tile_height > 0:
            return self._render_tiled_request(request, postprocess=postprocess)
        return self._render_single_request(request, postprocess=postprocess)

    def render_ray_map_request(
        self,
        request,
        map_x,
        map_y,
        tiles_x=6,
        tiles_y=4,
        tile_overlap=128,
        tile_margin=1.12,
        max_splat_radius=None,
    ):
        import cv2

        total_start = time.perf_counter()
        map_x = np.asarray(map_x, dtype=np.float32)
        map_y = np.asarray(map_y, dtype=np.float32)
        if map_x.shape != (request.height, request.width) or map_y.shape != (request.height, request.width):
            raise ValueError(
                "ray map shape must match request height and width: "
                f"expected {(request.height, request.width)}, got {map_x.shape} and {map_y.shape}"
            )

        intrinsics = np.asarray(request.intrinsics, dtype=np.float32)
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("request intrinsics must have positive fx and fy")

        ray_dirs = np.stack(
            [(map_x - cx) / fx, (map_y - cy) / fy, np.ones_like(map_x, dtype=np.float32)],
            axis=-1,
        ).astype(np.float32)
        valid = ((map_x > 0.0) | (map_y > 0.0)) & np.isfinite(map_x) & np.isfinite(map_y)
        valid &= (map_x >= 0.0) & (map_x <= request.width - 1)
        valid &= (map_y >= 0.0) & (map_y <= request.height - 1)

        accum = np.zeros((request.height, request.width, 3), dtype=np.float32)
        weight_sum = np.zeros((request.height, request.width, 1), dtype=np.float32)
        camera_ms = 0.0
        render_ms = 0.0
        tile_postprocess_ms = 0.0
        stitch_ms = 0.0

        tiles_x = max(1, int(tiles_x))
        tiles_y = max(1, int(tiles_y))
        tile_overlap = max(0, int(tile_overlap))
        tile_margin = max(1.0, float(tile_margin))
        block_width = int(math.ceil(request.width / tiles_x))
        block_height = int(math.ceil(request.height / tiles_y))
        base_camera_to_world = np.asarray(request.camera_to_world, dtype=np.float32)
        tile_max_splat_radius = (
            float(request.max_splat_radius)
            if max_splat_radius is None
            else float(max_splat_radius)
        )

        for tile_y in range(tiles_y):
            core_y0 = tile_y * block_height
            core_y1 = min(request.height, (tile_y + 1) * block_height)
            if core_y0 >= core_y1:
                continue

            for tile_x in range(tiles_x):
                core_x0 = tile_x * block_width
                core_x1 = min(request.width, (tile_x + 1) * block_width)
                if core_x0 >= core_x1:
                    continue

                crop_x0 = max(0, core_x0 - tile_overlap)
                crop_y0 = max(0, core_y0 - tile_overlap)
                crop_x1 = min(request.width, core_x1 + tile_overlap)
                crop_y1 = min(request.height, core_y1 + tile_overlap)

                crop_dirs = ray_dirs[crop_y0:crop_y1, crop_x0:crop_x1]
                crop_valid = valid[crop_y0:crop_y1, crop_x0:crop_x1]
                if not crop_valid.any():
                    continue

                center_y = min(request.height - 1, max(0, (core_y0 + core_y1) // 2))
                center_x = min(request.width - 1, max(0, (core_x0 + core_x1) // 2))
                center_ray = ray_dirs[center_y, center_x]
                if not valid[center_y, center_x]:
                    center_ray = crop_dirs[crop_valid].mean(axis=0)

                tile_rotation = self._rotation_from_camera_ray(center_ray)
                tile_dirs = crop_dirs.reshape(-1, 3) @ tile_rotation[:3, :3]
                tile_dirs = tile_dirs.reshape(crop_dirs.shape)
                tile_depth = tile_dirs[..., 2]
                usable = crop_valid & (tile_depth > 1e-4)
                if not usable.any():
                    continue

                proj_x = tile_dirs[..., 0] / np.maximum(tile_depth, 1e-4)
                proj_y = tile_dirs[..., 1] / np.maximum(tile_depth, 1e-4)
                min_x, max_x = float(proj_x[usable].min()), float(proj_x[usable].max())
                min_y, max_y = float(proj_y[usable].min()), float(proj_y[usable].max())
                min_x, max_x = self._expand_range(min_x, max_x, tile_margin)
                min_y, max_y = self._expand_range(min_y, max_y, tile_margin)

                tile_width = crop_x1 - crop_x0
                tile_height = crop_y1 - crop_y0
                tile_fx = max(8.0, (tile_width - 1) / max(max_x - min_x, 1e-6))
                tile_fy = max(8.0, (tile_height - 1) / max(max_y - min_y, 1e-6))
                tile_cx = -min_x * tile_fx
                tile_cy = -min_y * tile_fy

                tile_intrinsics = np.eye(4, dtype=np.float32)
                tile_intrinsics[0, 0] = tile_fx
                tile_intrinsics[1, 1] = tile_fy
                tile_intrinsics[0, 2] = tile_cx
                tile_intrinsics[1, 2] = tile_cy
                tile_camera_to_world = base_camera_to_world @ tile_rotation
                tile_request = replace(
                    request,
                    intrinsics=tile_intrinsics,
                    camera_to_world=tile_camera_to_world,
                    width=tile_width,
                    height=tile_height,
                    image_name=f"{request.image_name}_ray_tile_{tile_y:02d}_{tile_x:02d}",
                    tile_width=0,
                    tile_height=0,
                    tile_overlap=0,
                    max_splat_radius=tile_max_splat_radius,
                    appearance_camera_to_world=request.camera_to_world,
                )
                tile_image, tile_timing = self._render_single_request(
                    tile_request,
                    record_timing=False,
                    update_previous=False,
                    return_timing=True,
                )
                camera_ms += tile_timing.camera_ms
                render_ms += tile_timing.render_ms
                tile_postprocess_ms += tile_timing.postprocess_ms

                stitch_start = time.perf_counter()
                map_u = (proj_x * tile_fx + tile_cx).astype(np.float32)
                map_v = (proj_y * tile_fy + tile_cy).astype(np.float32)
                sampled = cv2.remap(
                    tile_image,
                    map_u,
                    map_v,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                weight = self._tile_weight(
                    crop_x0,
                    crop_y0,
                    core_x0,
                    core_y0,
                    core_x1,
                    core_y1,
                    crop_x1,
                    crop_y1,
                )
                weight *= crop_valid.astype(np.float32)
                accum[crop_y0:crop_y1, crop_x0:crop_x1] += sampled.astype(np.float32) * weight[..., None]
                weight_sum[crop_y0:crop_y1, crop_x0:crop_x1] += weight[..., None]
                stitch_ms += (time.perf_counter() - stitch_start) * 1000.0

        output = (accum / np.maximum(weight_sum, 1e-6)).clip(0.0, 255.0).astype(np.uint8)
        output[~valid] = 0
        if self.collect_timing:
            torch.cuda.synchronize()

        frame_camera_start = time.perf_counter()
        self.previous_camera = self._make_camera(request)
        camera_ms += (time.perf_counter() - frame_camera_start) * 1000.0

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
                    postprocess_ms=tile_postprocess_ms + stitch_ms,
                    total_ms=total_ms,
                )
            )
        return output

    def _render_single_request(
        self,
        request,
        postprocess=None,
        record_timing=True,
        update_previous=True,
        return_timing=False,
    ):
        total_start = time.perf_counter()
        camera_start = time.perf_counter()
        viewpoint = self._make_camera(request)
        camera_ms = (time.perf_counter() - camera_start) * 1000.0
        previous_camera = self.previous_camera or viewpoint
        if self.collect_timing:
            render_start = torch.cuda.Event(enable_timing=True)
            render_end = torch.cuda.Event(enable_timing=True)
            render_start.record()
        appearance_c2w = None
        if request.appearance_camera_to_world is not None:
            appearance_c2w = torch.tensor(
                request.appearance_camera_to_world,
                dtype=torch.float32,
                device="cuda",
            )
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
                max_radius_clip=request.max_splat_radius,
                appearance_c2w=appearance_c2w,
            )
        if self.collect_timing:
            render_end.record()
            render_end.synchronize()
            render_ms = render_start.elapsed_time(render_end)
        else:
            render_ms = 0.0
        if update_previous:
            self.previous_camera = viewpoint

        postprocess_start = time.perf_counter()
        image = render_to_uint8(render_pkg["render"])
        if postprocess is not None:
            image = postprocess(image)
        if self.collect_timing:
            torch.cuda.synchronize()
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        timing_record = RenderTimingRecord(
            image_name=request.image_name,
            width=request.width,
            height=request.height,
            timestamp=request.timestamp,
            camera_ms=camera_ms,
            render_ms=render_ms,
            postprocess_ms=postprocess_ms,
            total_ms=total_ms,
        )
        if record_timing and self.collect_timing:
            self.timing_records.append(timing_record)
        if return_timing:
            return image, timing_record
        return image

    def _render_tiled_request(self, request, postprocess=None):
        total_start = time.perf_counter()
        camera_ms = 0.0
        render_ms = 0.0
        accum = np.zeros((request.height, request.width, 3), dtype=np.float32)
        weight_sum = np.zeros((request.height, request.width, 1), dtype=np.float32)

        tile_width = min(int(request.tile_width), request.width)
        tile_height = min(int(request.tile_height), request.height)
        overlap = max(0, int(request.tile_overlap))
        tile_index = 0

        for y0 in range(0, request.height, tile_height):
            y1 = min(y0 + tile_height, request.height)
            for x0 in range(0, request.width, tile_width):
                x1 = min(x0 + tile_width, request.width)
                crop_x0 = max(0, x0 - overlap)
                crop_y0 = max(0, y0 - overlap)
                crop_x1 = min(request.width, x1 + overlap)
                crop_y1 = min(request.height, y1 + overlap)

                tile_intrinsics = np.asarray(request.intrinsics, dtype=np.float32).copy()
                tile_intrinsics[0, 2] -= crop_x0
                tile_intrinsics[1, 2] -= crop_y0
                tile_request = replace(
                    request,
                    intrinsics=tile_intrinsics,
                    width=crop_x1 - crop_x0,
                    height=crop_y1 - crop_y0,
                    image_name=f"{request.image_name}_tile_{tile_index:03d}",
                    tile_width=0,
                    tile_height=0,
                    tile_overlap=0,
                )
                tile_image, tile_timing = self._render_single_request(
                    tile_request,
                    record_timing=False,
                    update_previous=False,
                    return_timing=True,
                )
                camera_ms += tile_timing.camera_ms
                render_ms += tile_timing.render_ms

                weight = self._tile_weight(crop_x0, crop_y0, x0, y0, x1, y1, crop_x1, crop_y1)
                accum[crop_y0:crop_y1, crop_x0:crop_x1] += tile_image.astype(np.float32) * weight[..., None]
                weight_sum[crop_y0:crop_y1, crop_x0:crop_x1] += weight[..., None]
                tile_index += 1

        output = (accum / np.maximum(weight_sum, 1e-6)).clip(0.0, 255.0).astype(np.uint8)

        postprocess_start = time.perf_counter()
        if postprocess is not None:
            output = postprocess(output)
        if self.collect_timing:
            torch.cuda.synchronize()
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

        frame_camera_start = time.perf_counter()
        self.previous_camera = self._make_camera(request)
        camera_ms += (time.perf_counter() - frame_camera_start) * 1000.0

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
        return output

    def _tile_weight(self, crop_x0, crop_y0, core_x0, core_y0, core_x1, core_y1, crop_x1, crop_y1):
        wx = self._axis_tile_weight(crop_x0, core_x0, core_x1, crop_x1)
        wy = self._axis_tile_weight(crop_y0, core_y0, core_y1, crop_y1)
        return wy[:, None] * wx[None, :]

    @staticmethod
    def _axis_tile_weight(crop_start, core_start, core_end, crop_end):
        length = crop_end - crop_start
        weight = np.ones(length, dtype=np.float32)

        left = core_start - crop_start
        if left > 0:
            weight[:left] = np.linspace(0.0, 1.0, left, endpoint=False, dtype=np.float32)

        right = crop_end - core_end
        if right > 0:
            weight[-right:] = np.linspace(1.0, 0.0, right, endpoint=False, dtype=np.float32)

        return weight

    @staticmethod
    def _normalize_vector(vector):
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            raise ValueError("cannot normalize a near-zero camera ray")
        return vector / norm

    @classmethod
    def _rotation_from_camera_ray(cls, center_ray):
        z_axis = cls._normalize_vector(center_ray)
        y_reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(y_reference, z_axis))) > 0.96:
            y_reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        y_axis = y_reference - float(np.dot(y_reference, z_axis)) * z_axis
        y_axis = cls._normalize_vector(y_axis)
        x_axis = cls._normalize_vector(np.cross(y_axis, z_axis))
        y_axis = cls._normalize_vector(np.cross(z_axis, x_axis))

        rotation = np.eye(4, dtype=np.float32)
        rotation[:3, 0] = x_axis
        rotation[:3, 1] = y_axis
        rotation[:3, 2] = z_axis
        return rotation

    @staticmethod
    def _expand_range(min_value, max_value, margin):
        center = 0.5 * (min_value + max_value)
        half = max(1e-4, 0.5 * (max_value - min_value) * margin)
        return center - half, center + half

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
