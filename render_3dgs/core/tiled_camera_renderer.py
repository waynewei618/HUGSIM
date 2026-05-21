from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from render_3dgs.core.camera_math import (
    as_camera_intrinsics,
    as_positive_int,
    as_transform,
    render_to_uint8,
)


DEFAULT_TILE_GUARD_PIXELS = 64
DEFAULT_TILE_RENDER_BATCH_SIZE = 4


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


class TiledCameraRenderer:
    def __init__(
        self,
        camera_view_render,
        guard_pixels=DEFAULT_TILE_GUARD_PIXELS,
        render_batch_size=DEFAULT_TILE_RENDER_BATCH_SIZE,
    ):
        self.camera_view_render = camera_view_render
        self.guard_pixels = as_positive_int(guard_pixels, "guard_pixels")
        self.render_batch_size = as_positive_int(render_batch_size, "render_batch_size")
        self._tile_composite_cache = {}

    @property
    def device(self):
        return self.camera_view_render.background.device

    def reset_temporal_state(self):
        self.camera_view_render.reset_temporal_state()

    def render_camera(
        self,
        *args,
        **kwargs,
    ):
        return self.render_tiled_camera(*args, **kwargs)

    def render_tiled_camera(
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
        tile_rows=1,
        tile_cols=1,
    ):
        near_plane, far_plane = self.camera_view_render.render_planes(near_plane, far_plane)
        tile_rows = as_positive_int(tile_rows, "tile_rows")
        tile_cols = as_positive_int(tile_cols, "tile_cols")
        if tile_rows == 1 and tile_cols == 1:
            raise ValueError("render_tiled_camera requires tile_rows or tile_cols greater than 1")
        return self._render_camera_tiled(
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
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

    def _build_tile_render_tasks(
        self,
        intrinsics,
        camera_to_world,
        width,
        height,
        timestamp,
        dynamics,
        image_name,
        tile_rows,
        tile_cols,
    ):
        static_tasks = self._tile_static_tasks(intrinsics, width, height, tile_rows, tile_cols)

        tasks = []
        for static_task in static_tasks:
            tile_to_real_rotation = static_task["tile_to_real_rotation"]
            tile_camera_to_world = camera_to_world @ make_transform(tile_to_real_rotation)
            tile_viewpoint = self.camera_view_render._make_camera(
                intrinsics=static_task["tile_intrinsics"],
                camera_to_world=tile_camera_to_world,
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
        return (width, height, tile_rows, tile_cols, self.guard_pixels, k_real.tobytes())

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
                guard_x0 = max(0, tile_x0 - self.guard_pixels)
                guard_y0 = max(0, tile_y0 - self.guard_pixels)
                guard_x1 = min(width, tile_x1 + self.guard_pixels)
                guard_y1 = min(height, tile_y1 + self.guard_pixels)
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
            torch.from_numpy(sample_grid).to(device=self.device)[None, ...],
            torch.from_numpy(feather_weights).to(device=self.device)[None, None, ...],
        )

    def _render_camera_tiled(
        self,
        intrinsics,
        camera_to_world,
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
        camera_to_world = as_transform(camera_to_world, "camera_to_world")
        width = as_positive_int(width, "width")
        height = as_positive_int(height, "height")
        near_plane, far_plane = self.camera_view_render.render_planes(near_plane, far_plane)
        if tile_rows > height:
            raise ValueError("tile_rows must not exceed image height")
        if tile_cols > width:
            raise ValueError("tile_cols must not exceed image width")

        reference_viewpoint = self.camera_view_render._make_camera(
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            width=width,
            height=height,
            timestamp=timestamp,
            dynamics=dynamics,
            image_name=image_name,
        )
        output_accum = torch.zeros((1, 3, height, width), dtype=torch.float32, device=self.device)
        weight_accum = torch.zeros((1, 1, height, width), dtype=torch.float32, device=self.device)
        tasks = self._build_tile_render_tasks(
            intrinsics,
            camera_to_world,
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
            for batch_start in range(0, len(batch_tasks), self.render_batch_size):
                batch = batch_tasks[batch_start : batch_start + self.render_batch_size]
                render_tensors = self.camera_view_render.render_viewpoint_batch(
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

        self.camera_view_render.previous_camera = reference_viewpoint
        output = output_accum / torch.clamp(weight_accum, min=1e-6)
        return render_to_uint8(output[0])
