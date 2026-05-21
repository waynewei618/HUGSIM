import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from render_3dgs.core.camera_math import (
    as_camera_intrinsics,
    as_positive_float,
    as_positive_int,
    as_transform,
    render_to_uint8,
)
from render_3dgs.core.static_vehicle_insertion import (
    create_static_vehicle_insertion,
    ground_height,
    trajectory_pose_at_s,
    vehicle_body_to_world,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))


class CameraViewRender:
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
        from gaussian_renderer import GaussianModel, call_rasterization, cat_bgfg, concatenate_all
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
        self.previous_camera = None
        self._call_rasterization = call_rasterization
        self._cat_bgfg = cat_bgfg
        self._concatenate_all = concatenate_all

    def reset_temporal_state(self):
        self.previous_camera = None

    def render_planes(self, near_plane=None, far_plane=None):
        near_plane = self.near_plane if near_plane is None else as_positive_float(near_plane, "near_plane")
        far_plane = self.far_plane if far_plane is None else as_positive_float(far_plane, "far_plane")
        if far_plane <= near_plane:
            raise ValueError("far_plane must be greater than near_plane")
        return near_plane, far_plane

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
        near_plane=None,
        far_plane=None,
    ):
        near_plane, far_plane = self.render_planes(near_plane, far_plane)
        viewpoint = self._make_camera(
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            width=width,
            height=height,
            timestamp=timestamp,
            dynamics=dynamics,
            image_name=image_name,
        )
        render_tensor = self.render_viewpoint(viewpoint, near_plane, far_plane)
        self.previous_camera = viewpoint
        return render_to_uint8(render_tensor)

    def render_viewpoint(self, viewpoint, near_plane=None, far_plane=None):
        """Render one Camera viewpoint and return one RGB tensor in CHW layout."""
        near_plane, far_plane = self.render_planes(near_plane, far_plane)
        return self._render_viewpoints([viewpoint], near_plane, far_plane)[0]

    def render_viewpoint_batch(self, viewpoints, near_plane=None, far_plane=None):
        """Render same-size Camera viewpoints in one batched rasterization call."""
        if len(viewpoints) == 0:
            return []

        near_plane, far_plane = self.render_planes(near_plane, far_plane)
        return self._render_viewpoints(viewpoints, near_plane, far_plane)

    def _render_viewpoints(self, viewpoints, near_plane, far_plane):
        width = viewpoints[0].width
        height = viewpoints[0].height
        if any(viewpoint.width != width or viewpoint.height != height for viewpoint in viewpoints):
            raise ValueError("batched viewpoints must have identical image dimensions")

        with torch.no_grad():
            xyz, opacities, scales, rotations, shs, _ = self._scene_tensors_for_viewpoint(viewpoints[0])
            viewmats = torch.stack([torch.linalg.inv(viewpoint.c2w) for viewpoint in viewpoints], dim=0)
            intrinsics = torch.stack([viewpoint.K[:3, :3] for viewpoint in viewpoints], dim=0)
            camera_count = len(viewpoints)

            renders, _, _ = self._call_rasterization(
                means=xyz,
                quats=rotations,
                scales=scales,
                opacities=opacities[:, 0],
                colors=shs,
                viewmats=viewmats,
                Ks=intrinsics,
                width=width,
                height=height,
                render_mode="RGB",
                sh_degree=self.gaussians.active_sh_degree,
                near_plane=near_plane,
                far_plane=far_plane,
                max_radius_clip=0.0,
                packed=False,
                backgrounds=self.background[None, :].expand(camera_count, -1),
            )

            rendered_images = renders[..., :3].permute(0, 3, 1, 2)
            rendered_images = self._apply_affine_batch(rendered_images, [viewpoint.c2w for viewpoint in viewpoints])
        return list(rendered_images)

    def _scene_tensors_for_viewpoint(self, viewpoint):
        import roma

        all_fg = []
        for track_id, body_to_world in viewpoint.dynamics.items():
            dynamic_model = self.dynamic_gaussians[track_id]
            w_dxyz = (body_to_world[:3, :3] @ dynamic_model.get_xyz.T).T + body_to_world[:3, 3]

            drot = roma.quat_wxyz_to_xyzw(dynamic_model.get_rotation)
            drot = roma.unitquat_to_rotmat(drot)
            w_drot = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(body_to_world[:3, :3] @ drot))
            all_fg.append(
                [
                    w_dxyz,
                    dynamic_model.get_opacity,
                    dynamic_model.get_scaling,
                    w_drot,
                    dynamic_model.get_features,
                    dynamic_model.get_3D_features,
                ]
            )

        all_fg = self._concatenate_all(all_fg)
        return self._cat_bgfg(self.gaussians, all_fg)

    def _apply_affine_batch(self, rendered_images, camera_to_worlds):
        if not self.gaussians.affine:
            return rendered_images

        refined_images = []
        for rendered_image, camera_to_world in zip(rendered_images, camera_to_worlds):
            cam_xyz = camera_to_world[:3, 3].cuda()
            cam_dir = camera_to_world[:3, 2].cuda()
            o_enc = self.gaussians.pos_enc(cam_xyz[None, :] / 60)
            d_enc = self.gaussians.dir_enc(cam_dir[None, :])
            appearance = self.gaussians.appearance_model(torch.cat([o_enc, d_enc], dim=1)) * 1e-1
            affine_weight, affine_bias = appearance[:, :9].view(3, 3), appearance[:, -3:]
            affine_weight = affine_weight + torch.eye(3, device=appearance.device)
            colors = rendered_image.view(3, -1).permute(1, 0)
            refined_image = (colors @ affine_weight + affine_bias).clip(0, 1).permute(1, 0).view(*rendered_image.shape)
            refined_images.append(refined_image)
        return torch.stack(refined_images, dim=0)

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
