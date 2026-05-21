import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class StaticVehicleInsertion:
    track_id: str
    model: object
    dynamics: dict


def create_static_vehicle_insertion(
    vehicle_path,
    body_to_world,
    sh_degree,
):
    from scene.obj_model import ObjModel

    vehicle_path = Path(vehicle_path)
    vehicle_ckpt = vehicle_path if vehicle_path.suffix == ".pth" else vehicle_path / "gs.pth"
    track_id = f"aeb_static_{vehicle_ckpt.parent.name}"

    model = ObjModel(sh_degree, feat_mutable=False)
    model_params, _ = torch.load(vehicle_ckpt)
    model.restore(model_params, None)

    return StaticVehicleInsertion(
        track_id=track_id,
        model=model,
        dynamics={track_id: torch.tensor(body_to_world, dtype=torch.float32, device="cuda")},
    )


def trajectory_pose_at_s(positions, mileages, vehicle_s):
    target_s = float(mileages[-1]) if vehicle_s is None else float(vehicle_s)
    target_s = float(np.clip(target_s, mileages[0], mileages[-1]))

    upper = int(np.searchsorted(mileages, target_s, side="right"))
    upper = min(max(upper, 1), len(mileages) - 1)
    lower = upper - 1

    segment_length = mileages[upper] - mileages[lower]
    alpha = 0.0 if segment_length <= 1e-8 else (target_s - mileages[lower]) / segment_length
    position = positions[lower] * (1.0 - alpha) + positions[upper] * alpha
    tangent = positions[upper] - positions[lower]
    return target_s, position, tangent


def vehicle_body_to_world(position, tangent, world_y):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = vehicle_rotation_from_tangent(tangent)
    transform[:3, 3] = np.asarray([position[0], world_y, position[2]], dtype=np.float32)
    return transform


def vehicle_rotation_from_tangent(tangent):
    yaw = np.arctan2(float(tangent[2]), float(tangent[0]))
    c = np.cos(-yaw)
    s = np.sin(-yaw)
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def ground_height(scene_path, u, v):
    scene_path = Path(scene_path)
    ground_param_path = scene_path / "ground_param.pkl"
    if not ground_param_path.exists():
        parent_ground_param_path = scene_path.parent / "ground_param.pkl"
        if parent_ground_param_path.exists():
            ground_param_path = parent_ground_param_path

    with ground_param_path.open("rb") as f:
        cam_poses, cam_heights, commands = pickle.load(f)
    cam_poses = dense_ground_camera_poses(np.asarray(cam_poses), commands)

    cam_dist = np.sqrt((cam_poses[:, 0, 3] - u) ** 2 + (cam_poses[:, 2, 3] - v) ** 2)
    nearest_cam_idx = int(np.argmin(cam_dist, axis=0))
    nearest_c2w = cam_poses[nearest_cam_idx]

    nearest_w2c = np.linalg.inv(nearest_c2w)
    uv_local = nearest_w2c[:3, :3] @ np.asarray([u, 0.0, v]) + nearest_w2c[:3, 3]
    uv_local[1] = 0
    uv_world = nearest_c2w[:3, :3] @ uv_local + nearest_c2w[:3, 3]
    return float(uv_world[1] + camera_height(cam_heights, nearest_cam_idx))


def camera_height(cam_heights, nearest_cam_idx):
    cam_heights = np.asarray(cam_heights, dtype=np.float64)
    if cam_heights.ndim == 0:
        return float(cam_heights)
    height_index = min(int(nearest_cam_idx), cam_heights.size - 1)
    return float(cam_heights.reshape(-1)[height_index])


def dense_ground_camera_poses(cam_poses, commands):
    from scipy.spatial.transform import Rotation as SCR

    commands = list(commands) if commands is not None else [None] * len(cam_poses)
    for _ in range(4):
        dense_poses = []
        dense_commands = []
        for index in range(cam_poses.shape[0] - 1):
            cam1 = cam_poses[index]
            cam2 = cam_poses[index + 1]
            dense_poses.append(cam1)
            dense_commands.append(commands[index] if index < len(commands) else None)
            if np.linalg.norm(cam1[:3, 3] - cam2[:3, 3]) > 0.2:
                euler1 = SCR.from_matrix(cam1[:3, :3]).as_euler("XYZ")
                euler2 = SCR.from_matrix(cam2[:3, :3]).as_euler("XYZ")
                interp_pose = np.eye(4)
                interp_pose[:3, :3] = SCR.from_euler("XYZ", (euler1 + euler2) / 2).as_matrix()
                interp_pose[:3, 3] = (cam1[:3, 3] + cam2[:3, 3]) / 2
                dense_poses.append(interp_pose)
                dense_commands.append(commands[index] if index < len(commands) else None)
        dense_poses.append(cam_poses[-1])
        dense_commands.append(commands[-1] if commands else None)
        cam_poses = np.stack(dense_poses)
        commands = dense_commands
    return cam_poses
