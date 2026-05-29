import json
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONT_CAMERA = "CAM_FRONT"
CANONICAL_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
DYNAMIC_SEMANTIC_IDS = (11, 12, 13, 14, 15, 18)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_metadata(out_dir):
    out_dir = os.path.abspath(out_dir)
    meta_data = load_json(os.path.join(out_dir, "meta_data.json"))
    camera_paras = load_json(os.path.join(out_dir, "camera_paras.json"))
    return meta_data, camera_paras


def strip_relative_prefix(path):
    path = path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    return path


def abs_data_path(out_dir, relative_path):
    return os.path.join(out_dir, strip_relative_prefix(relative_path))


def frame_camera_name(frame):
    if "camera_name" in frame:
        return frame["camera_name"]
    parts = strip_relative_prefix(frame["rgb_path"]).split("/")
    if len(parts) >= 3 and parts[0] == "images":
        return parts[1]
    raise KeyError("Frame does not contain camera_name and rgb_path cannot be parsed.")


def camera_names_from_images(out_dir):
    images_dir = os.path.join(out_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(images_dir)
    cameras = [
        name
        for name in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, name))
    ]
    order = {name: idx for idx, name in enumerate(CANONICAL_CAMERAS)}
    return sorted(cameras, key=lambda name: (order.get(name, len(order)), name))


def ensure_camera_dirs(out_dir, dirname):
    for camera in camera_names_from_images(out_dir):
        os.makedirs(os.path.join(out_dir, dirname, camera), exist_ok=True)


def derived_relative_path(frame, dirname, suffix):
    parts = strip_relative_prefix(frame["rgb_path"]).split("/")
    if len(parts) < 3 or parts[0] != "images":
        raise ValueError(f"Unsupported rgb_path: {frame['rgb_path']}")
    parts[0] = dirname
    stem, _ = os.path.splitext(parts[-1])
    parts[-1] = stem + suffix
    return os.path.join(*parts)


def derived_abs_path(out_dir, frame, dirname, suffix):
    return os.path.join(out_dir, derived_relative_path(frame, dirname, suffix))


def camera_params(camera_paras, camera_name):
    try:
        params = camera_paras["cameras"][camera_name]
    except KeyError as exc:
        raise KeyError(f"camera_paras.json has no camera entry for {camera_name}") from exc

    camera_to_ego = np.eye(4, dtype=np.float64)
    camera_to_ego[:3, :3] = np.asarray(params["rotation_matrix"], dtype=np.float64)
    camera_to_ego[:3, 3] = np.asarray(params["translation"], dtype=np.float64)

    intrinsics = params["intrinsics"]
    intrinsic = np.asarray(intrinsics["matrix"], dtype=np.float64)
    return {
        "camera_to_ego": camera_to_ego,
        "intrinsic": intrinsic,
        "width": int(intrinsics["width"]),
        "height": int(intrinsics["height"]),
        "source_camera_name": params.get("source_camera_name", camera_name),
    }


def frame_camera_params(camera_paras, frame):
    return camera_params(camera_paras, frame_camera_name(frame))


def frame_camtoworld(camera_paras, frame):
    params = frame_camera_params(camera_paras, frame)
    ego_to_world = np.asarray(frame["ego_to_world"], dtype=np.float64)
    return ego_to_world @ params["camera_to_ego"]


def frame_object_to_ego(dynamic_pose):
    if isinstance(dynamic_pose, dict) and "object_to_ego" in dynamic_pose:
        dynamic_pose = dynamic_pose["object_to_ego"]
    return np.asarray(dynamic_pose, dtype=np.float64)


def frame_object_to_world(frame, dynamic_pose):
    ego_to_world = np.asarray(frame["ego_to_world"], dtype=np.float64)
    return ego_to_world @ frame_object_to_ego(dynamic_pose)


def projection_matrix(intrinsic):
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape == (3, 3):
        projection = np.zeros((3, 4), dtype=np.float64)
        projection[:, :3] = intrinsic
        return projection
    if intrinsic.shape[0] >= 3 and intrinsic.shape[1] >= 4:
        return intrinsic[:3, :4]
    raise ValueError(f"Unsupported intrinsic shape: {intrinsic.shape}")


def invert_transform(transform):
    transform = np.asarray(transform, dtype=np.float64)
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = transform[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ transform[:3, 3]
    return inv


def image_to_rgb_array(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] > 3:
        return image[:, :, :3]
    return image
