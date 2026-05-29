import argparse
import os

import cv2
import numpy as np
from imageio.v2 import imwrite

try:
    from .common import (
        DYNAMIC_SEMANTIC_IDS,
        derived_abs_path,
        ensure_camera_dirs,
        frame_camera_params,
        frame_object_to_ego,
        invert_transform,
        load_metadata,
        projection_matrix,
    )
except ImportError:
    from common import (
        DYNAMIC_SEMANTIC_IDS,
        derived_abs_path,
        ensure_camera_dirs,
        frame_camera_params,
        frame_object_to_ego,
        invert_transform,
        load_metadata,
        projection_matrix,
    )


BOX_FACES = (
    (0, 1, 4, 5, 0),
    (2, 3, 6, 7, 2),
    (0, 2, 7, 5, 0),
    (1, 3, 6, 4, 1),
    (0, 2, 3, 1, 0),
    (5, 4, 6, 7, 5),
)


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--out", "--data_path", dest="input", type=str, required=True)
    return parser.parse_args()


def project_box_to_pixels(intrinsic, camera_to_ego, object_to_ego, vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    vertices_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    object_to_camera = invert_transform(camera_to_ego) @ object_to_ego
    camera_points = (object_to_camera @ vertices_h.T).T

    projection = projection_matrix(intrinsic)
    screen = (projection @ camera_points.T).T
    depth = screen[:, 2]
    valid_depth = camera_points[:, 2] > 0
    if not np.any(valid_depth) or np.any(np.abs(depth) < 1e-8):
        return None, valid_depth

    xy = screen[:, :2] / depth[:, None]
    if not np.all(np.isfinite(xy)) or np.max(np.abs(xy)) > 1e7:
        return None, valid_depth
    return np.rint(xy).astype(np.int32), valid_depth


def main():
    args = get_opts()
    out_dir = os.path.abspath(args.input)
    ensure_camera_dirs(out_dir, "masks")
    meta_data, camera_paras = load_metadata(out_dir)
    verts = meta_data.get("verts", {})

    for frame in meta_data["frames"]:
        semantic_path = derived_abs_path(out_dir, frame, "semantics", ".npy")
        if not os.path.exists(semantic_path):
            raise FileNotFoundError(semantic_path)
        semantics = np.load(semantic_path)
        dynamic_semantics = np.isin(semantics, DYNAMIC_SEMANTIC_IDS)
        mask = np.zeros_like(dynamic_semantics, dtype=np.bool_)
        height, width = mask.shape[:2]

        params = frame_camera_params(camera_paras, frame)
        camera_to_ego = params["camera_to_ego"]
        intrinsic = params["intrinsic"]

        for object_id, pose in frame.get("dynamics", {}).items():
            if object_id not in verts:
                continue
            points_2d, valid_depth = project_box_to_pixels(
                intrinsic,
                camera_to_ego,
                frame_object_to_ego(pose),
                verts[object_id],
            )
            if points_2d is None:
                continue

            valid_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width)
            valid_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
            if not np.any(valid_x & valid_y & valid_depth):
                continue

            bbox_mask = np.zeros((height, width), dtype=np.uint8)
            for face in BOX_FACES:
                cv2.fillPoly(bbox_mask, [points_2d[list(face)]], 1)
            mask |= (bbox_mask != 0) & dynamic_semantics

        save_path = derived_abs_path(out_dir, frame, "masks", ".npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        static_mask = ~mask
        np.save(save_path, static_mask)
        imwrite(save_path.replace(".npy", ".png"), static_mask.astype(np.uint8) * 255)


if __name__ == "__main__":
    main()
