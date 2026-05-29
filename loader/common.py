import json
import os
import shutil

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


CANONICAL_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

FRONT_CAMERA = "CAM_FRONT"

CAM_ALIGN = (
    ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"),
    ("CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"),
)

BOX_DRAW_INTERVAL = 10

BOX_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
    (0, 5),
    (1, 4),
    (2, 7),
    (3, 6),
)

BOX_FRONT_DIAGONALS = (
    (0, 3),
    (1, 2),
)

WAYMO_CAMERA_MAP = {
    1: "CAM_FRONT",
    2: "CAM_FRONT_LEFT",
    3: "CAM_FRONT_RIGHT",
    4: "CAM_BACK_LEFT",
    5: "CAM_BACK_RIGHT",
}

PANDASET_CAMERA_MAP = {
    "front_camera": "CAM_FRONT",
    "front_left_camera": "CAM_FRONT_LEFT",
    "front_right_camera": "CAM_FRONT_RIGHT",
    "back_camera": "CAM_BACK",
    "left_camera": "CAM_BACK_LEFT",
    "right_camera": "CAM_BACK_RIGHT",
}


NO_CROP = {
    "left": 0,
    "top": 0,
    "right": 0,
    "bottom": 0,
}


def normalize_crop(crop=None):
    normalized = NO_CROP.copy()
    if crop:
        normalized.update(crop)
    return {key: int(value) for key, value in normalized.items()}


def make_intrinsic_matrix(fx, fy, cx, cy):
    intrinsic = np.eye(4)
    intrinsic[0, 0] = float(fx)
    intrinsic[1, 1] = float(fy)
    intrinsic[0, 2] = float(cx)
    intrinsic[1, 2] = float(cy)
    return intrinsic


def build_final_intrinsic(raw_intrinsic, raw_width, raw_height, crop=None, downsample=1):
    crop = normalize_crop(crop)
    cropped_width = int(raw_width) - crop["left"] - crop["right"]
    cropped_height = int(raw_height) - crop["top"] - crop["bottom"]
    if cropped_width <= 0 or cropped_height <= 0:
        raise ValueError(f"Invalid crop {crop} for image size {(raw_width, raw_height)}")

    final_width = int(cropped_width // downsample)
    final_height = int(cropped_height // downsample)
    if final_width <= 0 or final_height <= 0:
        raise ValueError(f"Invalid downsample {downsample} for cropped image size {(cropped_width, cropped_height)}")

    intrinsic = np.eye(4)
    raw_intrinsic = np.asarray(raw_intrinsic, dtype=float)
    intrinsic[: raw_intrinsic.shape[0], : raw_intrinsic.shape[1]] = raw_intrinsic
    intrinsic[0, 2] -= crop["left"]
    intrinsic[1, 2] -= crop["top"]
    intrinsic[0, :] /= downsample
    intrinsic[1, :] /= downsample
    intrinsic[2, :] = [0.0, 0.0, 1.0, 0.0]
    intrinsic[3, :] = [0.0, 0.0, 0.0, 1.0]
    return intrinsic, final_width, final_height


def crop_and_downsample_image(image, crop=None, downsample=1):
    crop = normalize_crop(crop)
    h, w = image.shape[:2]
    left = crop["left"]
    top = crop["top"]
    right = w - crop["right"] if crop["right"] else w
    bottom = h - crop["bottom"] if crop["bottom"] else h
    image = image[top:bottom, left:right]
    _, final_width, final_height = build_final_intrinsic(np.eye(4), w, h, crop=crop, downsample=downsample)
    if image.shape[1] != final_width or image.shape[0] != final_height:
        image = cv2.resize(image, (final_width, final_height))
    return image


def init_output_dirs(outdir, cameras):
    os.makedirs(outdir, exist_ok=True)
    shutil.rmtree(os.path.join(outdir, "images"), ignore_errors=True)
    shutil.rmtree(os.path.join(outdir, "box"), ignore_errors=True)
    for filename in ("cam_rigid_config.json", "camera_paras.json", "geo_reference.json"):
        try:
            os.remove(os.path.join(outdir, filename))
        except FileNotFoundError:
            pass
    os.makedirs(os.path.join(outdir, "images"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "box"), exist_ok=True)
    for camera in cameras:
        os.makedirs(os.path.join(outdir, "images", camera), exist_ok=True)
        os.makedirs(os.path.join(outdir, "box", camera), exist_ok=True)


def write_front_info(outdir, height, rect_mat=None):
    front_info = {
        "height": float(height),
        "rect_mat": None if rect_mat is None else np.asarray(rect_mat).tolist(),
    }
    with open(os.path.join(outdir, "front_info.json"), "w") as f:
        json.dump(front_info, f, indent=2)


def transform_to_quat_wxyz(transform):
    quat_xyzw = R.from_matrix(transform[:3, :3]).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def matrix_to_list(matrix):
    return np.asarray(matrix, dtype=float).tolist()


def write_camera_paras(outdir, camera_paras, camera_model="OPENCV"):
    serializable = {
        "camera_model": camera_model,
        "ego_coordinate": {
            "x": "forward",
            "y": "left",
            "z": "up",
            "handedness": "right",
        },
        "camera_coordinate": {
            "x": "right",
            "y": "down",
            "z": "forward",
        },
        "cameras": {},
    }
    for camera_name, params in camera_paras.items():
        camera_to_ego = np.asarray(params["camera_to_ego"], dtype=float)
        intrinsic = np.asarray(params["intrinsic"], dtype=float)
        serializable["cameras"][camera_name] = {
            "source_camera_name": str(params["source_camera_name"]),
            "translation": [float(v) for v in camera_to_ego[:3, 3]],
            "rotation_quat_wxyz": transform_to_quat_wxyz(camera_to_ego),
            "rotation_matrix": matrix_to_list(camera_to_ego[:3, :3]),
            "intrinsics": {
                "matrix": matrix_to_list(intrinsic),
                "fx": float(intrinsic[0, 0]),
                "fy": float(intrinsic[1, 1]),
                "cx": float(intrinsic[0, 2]),
                "cy": float(intrinsic[1, 2]),
                "width": int(params["width"]),
                "height": int(params["height"]),
            },
        }

    with open(os.path.join(outdir, "camera_paras.json"), "w") as f:
        json.dump(serializable, f, indent=2)


def write_geo_reference(outdir, reference):
    payload = {
        "local_world": "first_ego_pose",
        "global_frame": "dataset_native_global",
    }
    payload.update(reference)
    with open(os.path.join(outdir, "geo_reference.json"), "w") as f:
        json.dump(payload, f, indent=2)


def invert_transform(transform):
    transform = np.asarray(transform, dtype=float)
    inv = np.eye(4)
    inv[:3, :3] = transform[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ transform[:3, 3]
    return inv


def _projection_matrix(intrinsic):
    intrinsic = np.asarray(intrinsic, dtype=float)
    if intrinsic.shape == (3, 3):
        projection = np.zeros((3, 4), dtype=float)
        projection[:, :3] = intrinsic
        return projection
    if intrinsic.shape[0] >= 3 and intrinsic.shape[1] >= 4:
        return intrinsic[:3, :4]
    raise ValueError(f"Unsupported intrinsic shape: {intrinsic.shape}")


def project_box_vertices(intrinsic, camera_to_ego, object_to_ego, vertices, min_depth=1e-3):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (8, 3):
        raise ValueError(f"Expected 8 box vertices, got shape {vertices.shape}")

    camera_from_ego = invert_transform(camera_to_ego)
    object_to_camera = camera_from_ego @ np.asarray(object_to_ego, dtype=float)
    vertices_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=float)], axis=1)
    camera_points = (object_to_camera @ vertices_h.T).T

    depth = camera_points[:, 2]
    if np.any(depth <= min_depth):
        return None

    screen_points = (_projection_matrix(intrinsic) @ camera_points.T).T
    if not np.all(np.isfinite(screen_points)):
        return None

    uv = screen_points[:, :2] / screen_points[:, 2:3]
    if not np.all(np.isfinite(uv)) or np.max(np.abs(uv)) > 1e7:
        return None

    return np.rint(uv).astype(np.int32)


def _box_overlaps_image(points_2d, image_shape):
    height, width = image_shape[:2]
    x1 = np.min(points_2d[:, 0])
    x2 = np.max(points_2d[:, 0])
    y1 = np.min(points_2d[:, 1])
    y2 = np.max(points_2d[:, 1])
    return x2 >= 0 and y2 >= 0 and x1 < width and y1 < height


def _draw_clipped_line(image, p1, p2, color, thickness):
    height, width = image.shape[:2]
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    ok, clipped_p1, clipped_p2 = cv2.clipLine((0, 0, int(width), int(height)), p1, p2)
    if ok:
        cv2.line(image, clipped_p1, clipped_p2, color, thickness=thickness)


def draw_projected_3d_box(
    image,
    intrinsic,
    camera_to_ego,
    object_to_ego,
    vertices,
    color=(255, 128, 128),
    thickness=1,
    draw_front_face=True,
):
    points_2d = project_box_vertices(intrinsic, camera_to_ego, object_to_ego, vertices)
    if points_2d is None or not _box_overlaps_image(points_2d, image.shape):
        return False

    for edge in BOX_EDGES:
        _draw_clipped_line(image, points_2d[edge[0]], points_2d[edge[1]], color, thickness)

    if draw_front_face:
        for edge in BOX_FRONT_DIAGONALS:
            _draw_clipped_line(image, points_2d[edge[0]], points_2d[edge[1]], color, thickness)

    return True


def draw_dynamic_boxes(image, intrinsic, camera_to_ego, dynamics, verts, color=(255, 128, 128), thickness=1):
    draw_count = 0
    for object_id, pose in dynamics.items():
        if object_id not in verts:
            continue
        object_to_ego = pose["object_to_ego"] if isinstance(pose, dict) and "object_to_ego" in pose else pose
        if draw_projected_3d_box(
            image,
            intrinsic,
            camera_to_ego,
            object_to_ego,
            verts[object_id],
            color=color,
            thickness=thickness,
        ):
            draw_count += 1
    return draw_count


def make_video_frame(frame_images, image_size=(400, 225)):
    rows = []
    for row_cameras in CAM_ALIGN:
        row_images = []
        for camera in row_cameras:
            if camera not in frame_images:
                continue
            image = cv2.cvtColor(frame_images[camera], cv2.COLOR_BGR2RGB)
            row_images.append(cv2.resize(image, image_size))
        if row_images:
            rows.append(cv2.hconcat(row_images))

    if not rows:
        return None

    max_width = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] == max_width:
            padded_rows.append(row)
            continue
        pad_width = max_width - row.shape[1]
        pad = np.zeros((row.shape[0], pad_width, row.shape[2]), dtype=row.dtype)
        padded_rows.append(cv2.hconcat([row, pad]))
    return cv2.vconcat(padded_rows)


def append_video_frame(video_frames, frame_images, image_size=(400, 225)):
    frame = make_video_frame(frame_images, image_size=image_size)
    if frame is not None:
        video_frames.append(frame)
