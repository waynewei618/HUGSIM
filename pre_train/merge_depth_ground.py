import argparse
import json
import os
import pickle

import numpy as np
import open3d as o3d
import torch
from imageio.v2 import imread
from tqdm import tqdm

try:
    from .common import (
        FRONT_CAMERA,
        abs_data_path,
        derived_abs_path,
        frame_camera_name,
        frame_camera_params,
        frame_camtoworld,
        image_to_rgb_array,
        invert_transform,
        load_metadata,
    )
except ImportError:
    from common import (
        FRONT_CAMERA,
        abs_data_path,
        derived_abs_path,
        frame_camera_name,
        frame_camera_params,
        frame_camtoworld,
        image_to_rgb_array,
        invert_transform,
        load_metadata,
    )


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--out", dest="input", type=str, required=True)
    parser.add_argument("--total", type=int, default=1500000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_depth(path):
    try:
        depth = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        depth = torch.load(path, map_location="cpu")
    if isinstance(depth, torch.Tensor):
        return depth.detach().cpu().numpy()
    return np.asarray(depth)


def unproject_depth(depth, intrinsic):
    cx, cy, fx, fy = intrinsic[0, 2], intrinsic[1, 2], intrinsic[0, 0], intrinsic[1, 1]
    x = np.arange(depth.shape[1])
    y = np.arange(depth.shape[0])
    xx, yy = np.meshgrid(x, y)
    z = depth.reshape(-1)
    points_x = (xx.reshape(-1) - cx) * z / fx
    points_y = (yy.reshape(-1) - cy) * z / fy
    return np.stack([points_x, points_y, z], axis=1)


def read_front_camera_height(out_dir):
    front_info_path = os.path.join(out_dir, "front_info.json")
    if not os.path.exists(front_info_path):
        raise FileNotFoundError(f"{front_info_path} is required by pre_train ground merging.")
    with open(front_info_path) as f:
        return float(json.load(f)["height"])


def main():
    args = get_opts()
    out_dir = os.path.abspath(args.input)
    meta_data, camera_paras = load_metadata(out_dir)
    frames = meta_data["frames"]
    sample_per_frame = max(1, args.total // max(1, len(frames)))
    rng = np.random.default_rng(args.seed)

    points, colors, front_cam_poses = [], [], []
    for frame in tqdm(frames):
        params = frame_camera_params(camera_paras, frame)
        intrinsic = params["intrinsic"]
        c2w = frame_camtoworld(camera_paras, frame)
        if frame_camera_name(frame) == FRONT_CAMERA:
            front_cam_poses.append(c2w)

        image = image_to_rgb_array(imread(abs_data_path(out_dir, frame["rgb_path"])))
        depth_path = derived_abs_path(out_dir, frame, "depth", ".pt")
        semantics_path = derived_abs_path(out_dir, frame, "semantics", ".npy")
        for required_path in (depth_path, semantics_path):
            if not os.path.exists(required_path):
                raise FileNotFoundError(required_path)

        depth = load_depth(depth_path)
        local_points = unproject_depth(depth, intrinsic)
        local_colors = image.reshape(-1, 3).astype(np.float32) / 255.0

        semantic_mask = np.load(semantics_path).reshape(-1) <= 1
        valid_mask = semantic_mask & np.isfinite(local_points).all(axis=1)
        local_points = local_points[valid_mask]
        local_colors = local_colors[valid_mask]

        if local_points.shape[0] < sample_per_frame:
            continue
        sample_idx = rng.choice(local_points.shape[0], sample_per_frame, replace=False)
        local_points = local_points[sample_idx]
        local_colors = local_colors[sample_idx]

        world_points = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]
        points.append(world_points)
        colors.append(local_colors)

    if not points:
        raise RuntimeError("No ground points were sampled. Check depth and semantics.")
    if not front_cam_poses:
        raise RuntimeError(f"No {FRONT_CAMERA} frames were found in meta_data.json.")

    points = np.concatenate(points)
    colors = np.concatenate(colors)
    front_cam_poses = np.stack(front_cam_poses)
    front_cam_height = read_front_camera_height(out_dir)

    camera_centers = front_cam_poses[:-1, :3, 3]
    if camera_centers.shape[0] == 0:
        camera_centers = front_cam_poses[:, :3, 3]
    points_cam_dist = np.sqrt(np.sum((points[:, None, :] - camera_centers[None, :, :]) ** 2, axis=-1))
    nearest_cam_idx = np.argmin(points_cam_dist, axis=1)
    nearest_c2w = front_cam_poses[nearest_cam_idx]
    nearest_w2c = np.linalg.inv(front_cam_poses)[nearest_cam_idx]
    points_local = (
        np.einsum("nij,nj->ni", nearest_w2c[:, :3, :3], points)
        + nearest_w2c[:, :3, 3]
    )
    points_local[:, 1] = front_cam_height
    points = (
        np.einsum("nij,nj->ni", nearest_c2w[:, :3, :3], points_local)
        + nearest_c2w[:, :3, 3]
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(out_dir, "ground_points3d.ply"), pcd)

    forecast = 20
    threshold = 2.5
    high_level_commands = []
    for i, cam_pose in enumerate(front_cam_poses):
        forecast_campose = front_cam_poses[i + forecast] if i + forecast < len(front_cam_poses) else front_cam_poses[-1]
        forecast_in_curr = invert_transform(cam_pose) @ forecast_campose
        if forecast_in_curr[0, 3] > threshold:
            high_level_commands.append(0)
        elif forecast_in_curr[0, 3] < -threshold:
            high_level_commands.append(1)
        else:
            high_level_commands.append(2)

    print(high_level_commands)
    with open(os.path.join(out_dir, "ground_param.pkl"), "wb") as f:
        pickle.dump((front_cam_poses, front_cam_height, high_level_commands), f)


if __name__ == "__main__":
    main()
