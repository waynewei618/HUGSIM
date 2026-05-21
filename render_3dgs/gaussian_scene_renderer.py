import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from render_3dgs.camera_math import (
    as_camera_intrinsics,
    as_positive_float,
    as_positive_int,
    as_transform,
    render_to_uint8,
)
from render_3dgs.static_vehicle_insertion import (
    create_static_vehicle_insertion,
    ground_height,
    trajectory_pose_at_s,
    vehicle_body_to_world,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


class GaussianSceneRenderer:
    def __init__(
        self,
        scene_path,
        near_plane=0.01,
        far_plane=500.0,
        ego_trajectory=None,
        insert_vehicle_id=None,
        insert_vehicle_s=None,
        realcar_path=None,
        insert_vehicle_height=-0.3,
    ):
        from gaussian_renderer import GaussianModel, render
        from scene.obj_model import ObjModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for HUGSIM 3DGS rendering")

        self.near_plane = as_positive_float(near_plane, "near_plane")
        self.far_plane = as_positive_float(far_plane, "far_plane")
        if self.far_plane <= self.near_plane:
            raise ValueError("far_plane must be greater than near_plane")

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

        self.ego_trajectory = self._load_ego_trajectory(ego_trajectory)
        self.inserted_static_vehicle_dynamics = {}
        if insert_vehicle_id is not None:
            body_to_world = self._static_vehicle_transform(insert_vehicle_s, insert_vehicle_height)
            static_vehicle = create_static_vehicle_insertion(
                vehicle_path=self._vehicle_path(insert_vehicle_id, realcar_path),
                body_to_world=body_to_world,
                sh_degree=cfg.model.sh_degree,
            )
            self.dynamic_gaussians[static_vehicle.track_id] = static_vehicle.model
            self.inserted_static_vehicle_dynamics = static_vehicle.dynamics

        bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._render = render
        self.previous_camera = None

    def reset_temporal_state(self):
        self.previous_camera = None

    def _load_ego_trajectory(self, ego_trajectory):
        if isinstance(ego_trajectory, (str, Path)):
            with Path(ego_trajectory).open("r") as f:
                return json.load(f)
        return ego_trajectory

    def _vehicle_path(self, insert_vehicle_id, realcar_path):
        vehicle_path = Path(str(insert_vehicle_id))
        if vehicle_path.exists() or vehicle_path.suffix == ".pth":
            return vehicle_path
        realcar_root = Path(realcar_path or os.environ.get("PATH_3DRealCar", "/data/realcar3d"))
        return realcar_root / str(insert_vehicle_id)

    def _static_vehicle_transform(self, insert_vehicle_s, insert_vehicle_height):
        frames = self.ego_trajectory["frames"]
        positions = np.asarray([frame["ego_position"] for frame in frames], dtype=np.float64)
        mileages = np.asarray([frame["mileage"] for frame in frames], dtype=np.float64)
        _, position, tangent = trajectory_pose_at_s(positions, mileages, insert_vehicle_s)
        y = ground_height(self.scene_path, float(position[0]), float(position[2])) + float(insert_vehicle_height)
        return vehicle_body_to_world(position, tangent, y)

    def render_camera(
        self,
        intrinsics,
        camera_to_world,
        width,
        height,
        timestamp=0.0,
        dynamics=None,
        image_name="camera_render",
        near_plane=0.01,
        far_plane=500,
    ):
        near_plane = as_positive_float(near_plane, "near_plane")
        far_plane = as_positive_float(far_plane, "far_plane")
        if far_plane <= near_plane:
            raise ValueError("far_plane must be greater than near_plane")
        viewpoint = self._make_camera(
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            width=width,
            height=height,
            timestamp=timestamp,
            dynamics=dynamics,
            image_name=image_name,
        )
        previous_camera = self.previous_camera or viewpoint
        render_tensor = self._render_viewpoint(viewpoint, previous_camera, near_plane, far_plane)
        self.previous_camera = viewpoint
        return render_to_uint8(render_tensor)

    def _render_viewpoint(self, viewpoint, previous_camera, near_plane, far_plane):
        with torch.no_grad():
            render_pkg = self._render(
                viewpoint,
                previous_camera,
                self.gaussians,
                self.dynamic_gaussians,
                None,
                self.background,
                False,
                near_plane=near_plane,
                far_plane=far_plane,
            )
        return render_pkg["render"]

    def _make_camera(
        self,
        intrinsics,
        camera_to_world,
        width,
        height,
        timestamp=0.0,
        dynamics=None,
        image_name="camera_render",
    ):
        from scene.cameras import Camera

        intrinsics = as_camera_intrinsics(intrinsics, "camera intrinsics")
        camera_to_world = as_transform(camera_to_world, "camera_to_world")
        width = as_positive_int(width, "width")
        height = as_positive_int(height, "height")
        image = np.zeros((height, width, 3), dtype=np.float32)
        dynamics = {
            str(track_id): torch.tensor(transform, dtype=torch.float32, device="cuda")
            for track_id, transform in (dynamics or {}).items()
        }
        for track_id, transform in self.inserted_static_vehicle_dynamics.items():
            if track_id in dynamics:
                raise ValueError(f"Inserted static vehicle track id conflicts with frame dynamics: {track_id}")
            dynamics[track_id] = transform
        return Camera(
            width=width,
            height=height,
            image=image,
            K=intrinsics,
            c2w=camera_to_world,
            image_name=image_name,
            data_device="cpu",
            timestamp=float(timestamp),
            dynamics=dynamics,
        )
