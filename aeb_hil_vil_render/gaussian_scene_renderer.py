import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

DEFAULT_TILE_GUARD_PIXELS = 64
DEFAULT_TILE_RENDER_BATCH_SIZE = 4


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


def as_positive_int(value, name):
    output = int(value)
    if output <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return output


def as_positive_float(value, name):
    output = float(value)
    if not np.isfinite(output) or output <= 0.0:
        raise ValueError(f"{name} must be a finite value greater than 0")
    return output


def render_to_uint8(render_tensor):
    image = render_tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (image * 255.0 + 0.5).astype(np.uint8)


def tile_ranges(length, parts):
    boundaries = np.linspace(0, length, parts + 1, dtype=np.int64)
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError(f"Cannot split {length} pixels into {parts} non-empty tiles")
    return [(int(boundaries[index]), int(boundaries[index + 1])) for index in range(parts)]


def make_transform(rotation):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.astype(np.float32)
    return transform


def make_local_tile_rotation(center_ray):
    z_axis = center_ray.astype(np.float64)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("tile center ray norm is too small")
    z_axis = z_axis / z_norm

    real_x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = real_x_axis - z_axis * np.dot(real_x_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        real_y_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(real_y_axis, z_axis)
        x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("failed to build tile camera axes")
    x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)


def ray_from_pixel(k_inv, u, v):
    return k_inv @ np.asarray([u, v, 1.0], dtype=np.float32)


def tile_sample_points(x0, y0, x1, y1):
    right = max(x0, x1 - 1)
    bottom = max(y0, y1 - 1)
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    return np.asarray(
        [
            [x0, y0, 1.0],
            [right, y0, 1.0],
            [x0, bottom, 1.0],
            [right, bottom, 1.0],
            [center_x, y0, 1.0],
            [center_x, bottom, 1.0],
            [x0, center_y, 1.0],
            [right, center_y, 1.0],
        ],
        dtype=np.float32,
    )


def make_tile_intrinsics(k_real_inv, tile_to_real_rotation, x0, y0, x1, y1, tile_width, tile_height):
    sample_points = tile_sample_points(x0, y0, x1, y1)
    real_rays = (k_real_inv @ sample_points.T).T
    tile_rays = (tile_to_real_rotation.T @ real_rays.T).T
    if np.any(tile_rays[:, 2] <= 1e-6):
        raise ValueError("tile FOV includes rays behind the local tile camera")

    normalized_x = tile_rays[:, 0] / tile_rays[:, 2]
    normalized_y = tile_rays[:, 1] / tile_rays[:, 2]
    min_x, max_x = float(normalized_x.min()), float(normalized_x.max())
    min_y, max_y = float(normalized_y.min()), float(normalized_y.max())
    if max_x - min_x < 1e-8 or max_y - min_y < 1e-8:
        raise ValueError("tile FOV is too small to build intrinsics")

    fx = (tile_width - 1) / (max_x - min_x) if tile_width > 1 else 1.0
    fy = (tile_height - 1) / (max_y - min_y) if tile_height > 1 else 1.0
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = -fx * min_x
    intrinsics[1, 2] = -fy * min_y
    return intrinsics


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def feather_axis_weights(coords, core_start, core_end, guard_start, guard_end):
    weights = np.ones_like(coords, dtype=np.float32)
    if core_start > guard_start:
        left = coords < core_start
        weights[left] = smoothstep((coords[left] - guard_start) / (core_start - guard_start))
    if guard_end > core_end:
        right = coords >= core_end
        weights[right] = smoothstep((guard_end - coords[right]) / (guard_end - core_end))
    return weights


def tile_feather_weights(xs, ys, core_x0, core_y0, core_x1, core_y1, guard_x0, guard_y0, guard_x1, guard_y1):
    wx = feather_axis_weights(xs, core_x0, core_x1, guard_x0, guard_x1)
    wy = feather_axis_weights(ys, core_y0, core_y1, guard_y0, guard_y1)
    return wx * wy


class GaussianSceneRenderer:
    def __init__(self, scene_path, near_plane=0.01, far_plane=500.0):
        from gaussian_renderer import GaussianModel, call_rasterization, cat_bgfg, concatenate_all, render
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

        bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._render = render
        self._call_rasterization = call_rasterization
        self._cat_bgfg = cat_bgfg
        self._concatenate_all = concatenate_all
        self._tile_composite_cache = {}
        self.previous_camera = None

    def reset_temporal_state(self):
        self.previous_camera = None

    def render_camera(
        self,
        intrinsics,
        world_to_camera,
        width,
        height,
        timestamp=0.0,
        dynamics=None,
        image_name="camera_render",
        near_plane=None,
        far_plane=None,
        tile_rows=1,
        tile_cols=1,
    ):
        near_plane = self.near_plane if near_plane is None else as_positive_float(near_plane, "near_plane")
        far_plane = self.far_plane if far_plane is None else as_positive_float(far_plane, "far_plane")
        if far_plane <= near_plane:
            raise ValueError("far_plane must be greater than near_plane")
        tile_rows = as_positive_int(tile_rows, "tile_rows")
        tile_cols = as_positive_int(tile_cols, "tile_cols")

        if tile_rows > 1 or tile_cols > 1:
            return self._render_camera_tiled(
                intrinsics=intrinsics,
                world_to_camera=world_to_camera,
                width=width,
                height=height,
                timestamp=timestamp,
                dynamics=dynamics,
                image_name=image_name,
                near_plane=near_plane,
                far_plane=far_plane,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
            )

        viewpoint = self._make_camera(
            intrinsics=intrinsics,
            world_to_camera=world_to_camera,
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

    def _scene_tensors_for_viewpoint(self, viewpoint):
        import roma

        all_fg = []
        for track_id, b2w in viewpoint.dynamics.items():
            dynamic_model = self.dynamic_gaussians[track_id]
            w_dxyz = (b2w[:3, :3] @ dynamic_model.get_xyz.T).T + b2w[:3, 3]

            drot = roma.quat_wxyz_to_xyzw(dynamic_model.get_rotation)
            drot = roma.unitquat_to_rotmat(drot)
            w_drot = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(b2w[:3, :3] @ drot))
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

    def _apply_affine_batch(self, rendered_images, c2ws):
        if not self.gaussians.affine:
            return rendered_images

        refined_images = []
        for rendered_image, c2w in zip(rendered_images, c2ws):
            cam_xyz = c2w[:3, 3].cuda()
            cam_dir = c2w[:3, 2].cuda()
            o_enc = self.gaussians.pos_enc(cam_xyz[None, :] / 60)
            d_enc = self.gaussians.dir_enc(cam_dir[None, :])
            appearance = self.gaussians.appearance_model(torch.cat([o_enc, d_enc], dim=1)) * 1e-1
            affine_weight, affine_bias = appearance[:, :9].view(3, 3), appearance[:, -3:]
            affine_weight = affine_weight + torch.eye(3, device=appearance.device)
            colors = rendered_image.view(3, -1).permute(1, 0)
            refined_image = (colors @ affine_weight + affine_bias).clip(0, 1).permute(1, 0).view(*rendered_image.shape)
            refined_images.append(refined_image)
        return torch.stack(refined_images, dim=0)

    def _render_viewpoint_batch(self, viewpoints, near_plane, far_plane):
        width = viewpoints[0].width
        height = viewpoints[0].height
        if any(viewpoint.width != width or viewpoint.height != height for viewpoint in viewpoints):
            raise ValueError("batched tile viewpoints must have identical image dimensions")

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

    def _build_tile_render_tasks(
        self,
        intrinsics,
        world_to_camera,
        width,
        height,
        timestamp,
        dynamics,
        image_name,
        tile_rows,
        tile_cols,
    ):
        static_tasks = self._tile_static_tasks(intrinsics, width, height, tile_rows, tile_cols)
        camera_to_world = np.linalg.inv(world_to_camera).astype(np.float32)

        tasks = []
        for static_task in static_tasks:
            tile_to_real_rotation = static_task["tile_to_real_rotation"]
            tile_camera_to_world = camera_to_world @ make_transform(tile_to_real_rotation)
            tile_world_to_camera = np.linalg.inv(tile_camera_to_world).astype(np.float32)
            tile_viewpoint = self._make_camera(
                intrinsics=static_task["tile_intrinsics"],
                world_to_camera=tile_world_to_camera,
                width=static_task["tile_width"],
                height=static_task["tile_height"],
                timestamp=timestamp,
                dynamics=dynamics,
                image_name=f"{image_name}_tile_{static_task['row_index']}_{static_task['col_index']}",
            )
            tasks.append({**static_task, "viewpoint": tile_viewpoint})
        return tasks

    def _tile_cache_key(self, intrinsics, width, height, tile_rows, tile_cols):
        k_real = np.ascontiguousarray(intrinsics[:3, :3], dtype=np.float32)
        return (width, height, tile_rows, tile_cols, DEFAULT_TILE_GUARD_PIXELS, k_real.tobytes())

    def _tile_static_tasks(self, intrinsics, width, height, tile_rows, tile_cols):
        key = self._tile_cache_key(intrinsics, width, height, tile_rows, tile_cols)
        cached_tasks = self._tile_composite_cache.get(key)
        if cached_tasks is not None:
            return cached_tasks

        k_real = intrinsics[:3, :3]
        k_real_inv = np.linalg.inv(k_real).astype(np.float32)
        tasks = []
        for row_index, (tile_y0, tile_y1) in enumerate(tile_ranges(height, tile_rows)):
            for col_index, (tile_x0, tile_x1) in enumerate(tile_ranges(width, tile_cols)):
                guard_x0 = max(0, tile_x0 - DEFAULT_TILE_GUARD_PIXELS)
                guard_y0 = max(0, tile_y0 - DEFAULT_TILE_GUARD_PIXELS)
                guard_x1 = min(width, tile_x1 + DEFAULT_TILE_GUARD_PIXELS)
                guard_y1 = min(height, tile_y1 + DEFAULT_TILE_GUARD_PIXELS)
                tile_width = guard_x1 - guard_x0
                tile_height = guard_y1 - guard_y0

                center_u = (tile_x0 + tile_x1) * 0.5
                center_v = (tile_y0 + tile_y1) * 0.5
                center_ray = ray_from_pixel(k_real_inv, center_u, center_v)
                tile_to_real_rotation = make_local_tile_rotation(center_ray)
                tile_intrinsics = make_tile_intrinsics(
                    k_real_inv,
                    tile_to_real_rotation,
                    guard_x0,
                    guard_y0,
                    guard_x1,
                    guard_y1,
                    tile_width,
                    tile_height,
                )
                sample_grid, feather_weights = self._tile_composite_tensors(
                    k_real_inv,
                    tile_intrinsics,
                    tile_to_real_rotation,
                    tile_x0,
                    tile_y0,
                    tile_x1,
                    tile_y1,
                    guard_x0,
                    guard_y0,
                    guard_x1,
                    guard_y1,
                    tile_width,
                    tile_height,
                )
                tasks.append(
                    {
                        "row_index": row_index,
                        "col_index": col_index,
                        "tile_width": tile_width,
                        "tile_height": tile_height,
                        "tile_to_real_rotation": tile_to_real_rotation,
                        "tile_intrinsics": tile_intrinsics,
                        "core": (tile_x0, tile_y0, tile_x1, tile_y1),
                        "guard": (guard_x0, guard_y0, guard_x1, guard_y1),
                        "sample_grid": sample_grid,
                        "feather_weights": feather_weights,
                    }
                )

        self._tile_composite_cache[key] = tasks
        return tasks

    def _tile_composite_tensors(
        self,
        k_real_inv,
        tile_intrinsics,
        tile_to_real_rotation,
        tile_x0,
        tile_y0,
        tile_x1,
        tile_y1,
        guard_x0,
        guard_y0,
        guard_x1,
        guard_y1,
        tile_width,
        tile_height,
    ):
        xs, ys = np.meshgrid(
            np.arange(guard_x0, guard_x1, dtype=np.float32),
            np.arange(guard_y0, guard_y1, dtype=np.float32),
        )
        pixel_homogeneous = np.stack([xs, ys, np.ones_like(xs)], axis=0).reshape(3, -1)
        real_to_tile_homography = tile_intrinsics[:3, :3] @ tile_to_real_rotation.T @ k_real_inv
        tile_pixels = real_to_tile_homography @ pixel_homogeneous
        tile_pixels[:2] /= tile_pixels[2:3]

        if tile_width > 1:
            grid_x = tile_pixels[0].reshape(ys.shape) * (2.0 / (tile_width - 1)) - 1.0
        else:
            grid_x = np.zeros_like(ys, dtype=np.float32)
        if tile_height > 1:
            grid_y = tile_pixels[1].reshape(ys.shape) * (2.0 / (tile_height - 1)) - 1.0
        else:
            grid_y = np.zeros_like(ys, dtype=np.float32)
        sample_grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
        feather_weights = tile_feather_weights(
            xs,
            ys,
            tile_x0,
            tile_y0,
            tile_x1,
            tile_y1,
            guard_x0,
            guard_y0,
            guard_x1,
            guard_y1,
        ).astype(np.float32)
        return (
            torch.from_numpy(sample_grid).to(device="cuda")[None, ...],
            torch.from_numpy(feather_weights).to(device="cuda")[None, None, ...],
        )

    def _render_camera_tiled(
        self,
        intrinsics,
        world_to_camera,
        width,
        height,
        timestamp,
        dynamics,
        image_name,
        near_plane,
        far_plane,
        tile_rows,
        tile_cols,
    ):
        intrinsics = as_camera_intrinsics(intrinsics, "camera intrinsics")
        world_to_camera = as_transform(world_to_camera, "world_to_camera")
        width = as_positive_int(width, "width")
        height = as_positive_int(height, "height")
        if tile_rows > height:
            raise ValueError("tile_rows must not exceed image height")
        if tile_cols > width:
            raise ValueError("tile_cols must not exceed image width")

        reference_viewpoint = self._make_camera(
            intrinsics=intrinsics,
            world_to_camera=world_to_camera,
            width=width,
            height=height,
            timestamp=timestamp,
            dynamics=dynamics,
            image_name=image_name,
        )
        output_accum = torch.zeros((1, 3, height, width), dtype=torch.float32, device="cuda")
        weight_accum = torch.zeros((1, 1, height, width), dtype=torch.float32, device="cuda")
        tasks = self._build_tile_render_tasks(
            intrinsics,
            world_to_camera,
            width,
            height,
            timestamp,
            dynamics,
            image_name,
            tile_rows,
            tile_cols,
        )

        tasks_by_size = defaultdict(list)
        for task in tasks:
            viewpoint = task["viewpoint"]
            tasks_by_size[(viewpoint.width, viewpoint.height)].append(task)

        for batch_tasks in tasks_by_size.values():
            for batch_start in range(0, len(batch_tasks), DEFAULT_TILE_RENDER_BATCH_SIZE):
                batch = batch_tasks[batch_start : batch_start + DEFAULT_TILE_RENDER_BATCH_SIZE]
                render_tensors = self._render_viewpoint_batch(
                    [task["viewpoint"] for task in batch],
                    near_plane,
                    far_plane,
                )
                render_batch = torch.stack(render_tensors, dim=0)
                sample_grids = torch.cat([task["sample_grid"] for task in batch], dim=0)
                sampled_batch = F.grid_sample(
                    render_batch,
                    sample_grids,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )

                for task_index, task in enumerate(batch):
                    guard_x0, guard_y0, guard_x1, guard_y1 = task["guard"]
                    weights = task["feather_weights"]
                    output_accum[:, :, guard_y0:guard_y1, guard_x0:guard_x1] += (
                        sampled_batch[task_index : task_index + 1] * weights
                    )
                    weight_accum[:, :, guard_y0:guard_y1, guard_x0:guard_x1] += weights

        self.previous_camera = reference_viewpoint
        output = output_accum / torch.clamp(weight_accum, min=1e-6)
        return render_to_uint8(output[0])

    def _make_camera(
        self,
        intrinsics,
        world_to_camera,
        width,
        height,
        timestamp=0.0,
        dynamics=None,
        image_name="camera_render",
    ):
        from scene.cameras import Camera

        intrinsics = as_camera_intrinsics(intrinsics, "camera intrinsics")
        world_to_camera = as_transform(world_to_camera, "world_to_camera")
        width = as_positive_int(width, "width")
        height = as_positive_int(height, "height")
        camera_to_world = np.linalg.inv(world_to_camera).astype(np.float32)
        image = np.zeros((height, width, 3), dtype=np.float32)
        dynamics = {
            str(track_id): torch.tensor(transform, dtype=torch.float32, device="cuda")
            for track_id, transform in (dynamics or {}).items()
        }
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
