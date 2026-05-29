import argparse
import importlib
import json
import os
import sys

import cv2
import mediapy as media
import numpy as np
import open3d as o3d
from tqdm import tqdm

LOADER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(LOADER_ROOT, ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
for path in (LOADER_ROOT, DATA_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from common import (  # noqa: E402
    BOX_DRAW_INTERVAL,
    PANDASET_CAMERA_MAP,
    append_video_frame,
    build_final_intrinsic,
    crop_and_downsample_image,
    draw_dynamic_boxes,
    init_output_dirs,
    invert_transform,
    make_intrinsic_matrix,
    write_camera_paras,
    write_front_info,
    write_geo_reference,
)
from panda.utils import (  # noqa: E402
    ALLOWED_NONRIGID_CLASSES,
    ALLOWED_RIGID_CLASSES,
    _pandaset_pose_to_matrix,
    _yaw_to_rotation_matrix,
    get_vertices,
)


PANDASET_SEQ_LEN = 80
DYNAMIC_CLASSES = ALLOWED_RIGID_CLASSES + ALLOWED_NONRIGID_CLASSES
SOURCE_CAMERAS = tuple(PANDASET_CAMERA_MAP.keys())
CAMERA_CROPS = {
    "back_camera": {"bottom": 250},
}


def import_dataset_class():
    removed_paths = []
    for path in list(sys.path):
        abs_path = os.path.abspath(path or os.getcwd())
        if abs_path == PROJECT_ROOT:
            removed_paths.append(path)
            sys.path.remove(path)

    local_module = sys.modules.get("pandaset")
    if local_module is not None and not hasattr(local_module, "DataSet"):
        sys.modules.pop("pandaset", None)

    try:
        module = importlib.import_module("pandaset")
        return module.DataSet
    finally:
        for path in reversed(removed_paths):
            sys.path.insert(0, path)


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--no_video", action="store_true", default=False)
    return parser.parse_args()


def fit_ground_from_lidar(sequence, outdir):
    lidar = sequence.lidar[0]
    points_world = lidar[["x", "y", "z"]].to_numpy(dtype=np.float64)
    lidar_pose = _pandaset_pose_to_matrix(sequence.lidar.poses[0])
    world_to_lidar = np.linalg.inv(lidar_pose)
    points_lidar = (world_to_lidar[:3, :3] @ points_world.T).T + world_to_lidar[:3, 3]

    ground_mask = (np.abs(points_lidar[:, 0]) < 6) & (np.abs(points_lidar[:, 1]) < 3)
    points_lidar = points_lidar[ground_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_lidar)
    o3d.io.write_point_cloud(os.path.join(outdir, "ground_lidar.ply"), pcd)
    plane_model, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model

    front_pose_world = _pandaset_pose_to_matrix(sequence.camera["front_camera"].poses[0])
    front_pose_lidar = world_to_lidar @ front_pose_world
    front_cam_t = front_pose_lidar[:3, 3]
    ground_height = -(a * front_cam_t[0] + b * front_cam_t[1] + d) / c
    write_front_info(outdir, front_cam_t[2] - ground_height, rect_mat=None)


def collect_camera_paras(sequence, source_cameras, intrinsics, image_sizes):
    ego_to_global = _pandaset_pose_to_matrix(sequence.lidar.poses[0])
    global_to_ego = invert_transform(ego_to_global)
    camera_paras = {}
    for source_camera in source_cameras:
        canonical_camera = PANDASET_CAMERA_MAP[source_camera]
        camera_to_global = _pandaset_pose_to_matrix(sequence.camera[source_camera].poses[0])
        camera_to_ego = global_to_ego @ camera_to_global
        height, width = image_sizes[source_camera]
        camera_paras[canonical_camera] = {
            "source_camera_name": source_camera,
            "camera_to_ego": camera_to_ego,
            "intrinsic": intrinsics[source_camera],
            "width": width,
            "height": height,
        }
    return camera_paras


def load_camera_image(sequence, source_camera, frame_idx, downsample):
    curr_cam = sequence.camera[source_camera]
    image_path = curr_cam._data_structure[frame_idx]
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    raw_height, raw_width = image.shape[:2]
    intrinsic = build_raw_intrinsic(curr_cam.intrinsics)
    crop = CAMERA_CROPS.get(source_camera)
    intrinsic, final_width, final_height = build_final_intrinsic(
        intrinsic,
        raw_width,
        raw_height,
        crop=crop,
        downsample=downsample,
    )
    image = crop_and_downsample_image(image, crop=crop, downsample=downsample)
    if image.shape[:2] != (final_height, final_width):
        raise ValueError(f"PandaSet camera {source_camera} final size mismatch")

    return image, os.path.basename(image_path), final_height, final_width, intrinsic, curr_cam.timestamps[frame_idx]


def build_raw_intrinsic(intrinsics):
    return make_intrinsic_matrix(intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)


def read_geo_reference(datapath, seq):
    gps_path = os.path.join(datapath, seq, "meta", "gps.json")
    timestamps_path = os.path.join(datapath, seq, "meta", "timestamps.json")
    reference = {
        "dataset": "pandaset",
        "sequence": seq,
        "available": False,
        "source": os.path.join(seq, "meta", "gps.json"),
        "reason": "PandaSet GPS file was not found or did not contain a first-frame record.",
    }

    try:
        with open(gps_path) as f:
            gps_records = json.load(f)
    except FileNotFoundError:
        return reference

    if not gps_records:
        return reference

    gps = gps_records[0]
    timestamp = None
    try:
        with open(timestamps_path) as f:
            timestamps = json.load(f)
        if timestamps:
            timestamp = timestamps[0]
    except FileNotFoundError:
        pass

    return {
        "dataset": "pandaset",
        "sequence": seq,
        "available": True,
        "source": os.path.join(seq, "meta", "gps.json"),
        "frame_index": 0,
        "timestamp": timestamp,
        "latitude": gps.get("lat"),
        "longitude": gps.get("long"),
        "height": gps.get("height"),
        "raw": gps,
    }


def main():
    args = get_opts()
    DataSet = import_dataset_class()

    outdir = args.out
    canonical_cameras = [PANDASET_CAMERA_MAP[camera] for camera in SOURCE_CAMERAS]
    init_output_dirs(outdir, canonical_cameras)

    pandaset = DataSet(args.datapath)
    sequence = pandaset[args.seq]
    sequence.load()

    origin_ego_to_global = _pandaset_pose_to_matrix(sequence.lidar.poses[0])
    global_to_origin_ego = invert_transform(origin_ego_to_global)
    fit_ground_from_lidar(sequence, outdir)

    meta_data = {
        "camera_model": "OPENCV",
        "ego_coordinate": {"x": "forward", "y": "left", "z": "up", "handedness": "right"},
        "frames": [],
        "world_origin": "first_ego_pose",
        "origin_ego_to_global": origin_ego_to_global.tolist(),
    }
    write_geo_reference(outdir, read_geo_reference(args.datapath, args.seq))
    video_images = []
    start_timestamp = None
    first_intrinsics = {}
    first_image_sizes = {}
    image_names = {}

    for frame_idx in tqdm(range(PANDASET_SEQ_LEN)):
        frame_video_images = {}
        ego_to_global = _pandaset_pose_to_matrix(sequence.lidar.poses[frame_idx])
        ego_to_world = global_to_origin_ego @ ego_to_global
        for source_camera in SOURCE_CAMERAS:
            canonical_camera = PANDASET_CAMERA_MAP[source_camera]
            im, im_name, height, width, intrinsic, timestamp = load_camera_image(
                sequence,
                source_camera,
                frame_idx,
                downsample=args.downsample,
            )
            if start_timestamp is None:
                start_timestamp = timestamp
            if source_camera not in first_intrinsics:
                first_intrinsics[source_camera] = intrinsic
                first_image_sizes[source_camera] = (height, width)

            cv2.imwrite(os.path.join(outdir, "images", canonical_camera, im_name), im)
            image_names[(frame_idx, source_camera)] = im_name
            meta_data["frames"].append(
                {
                    "rgb_path": os.path.join("./images", canonical_camera, im_name),
                    "camera_name": canonical_camera,
                    "source_camera_name": source_camera,
                    "ego_to_world": ego_to_world.tolist(),
                    "timestamp": timestamp - start_timestamp,
                    "dynamics": {},
                }
            )
            frame_video_images[canonical_camera] = im

        if not args.no_video:
            append_video_frame(video_images, frame_video_images, image_size=(384, 216))

    def write_box_images(frame_idx, global_to_ego, frame_dynamic_boxes, frame_verts):
        if frame_idx % BOX_DRAW_INTERVAL != 0:
            return
        for source_camera in SOURCE_CAMERAS:
            canonical_camera = PANDASET_CAMERA_MAP[source_camera]
            im_name = image_names[(frame_idx, source_camera)]
            image_path = os.path.join(outdir, "images", canonical_camera, im_name)
            box_img = cv2.imread(image_path)
            if box_img is None:
                raise FileNotFoundError(image_path)
            camera_to_global = _pandaset_pose_to_matrix(sequence.camera[source_camera].poses[frame_idx])
            camera_to_ego = global_to_ego @ camera_to_global
            draw_dynamic_boxes(
                box_img,
                first_intrinsics[source_camera],
                camera_to_ego,
                frame_dynamic_boxes,
                frame_verts,
            )
            cv2.imwrite(os.path.join(outdir, "box", canonical_camera, im_name), box_img)

    cuboids = {}
    for frame_idx in tqdm(range(PANDASET_SEQ_LEN)):
        curr_cuboids = sequence.cuboids[frame_idx]
        ego_to_global = _pandaset_pose_to_matrix(sequence.lidar.poses[frame_idx])
        global_to_ego = invert_transform(ego_to_global)
        is_allowed_class = np.array([label in DYNAMIC_CLASSES for label in curr_cuboids["label"]])
        valid_mask = (~curr_cuboids["stationary"]) & is_allowed_class
        curr_cuboids = curr_cuboids[valid_mask]
        if not len(curr_cuboids):
            write_box_images(frame_idx, global_to_ego, {}, {})
            continue

        uuids = np.array(curr_cuboids["uuid"])
        labels = np.array(curr_cuboids["label"])
        yaw = curr_cuboids["yaw"].astype(np.float32)
        rot = _yaw_to_rotation_matrix(yaw)
        pos = np.vstack(
            [
                curr_cuboids["position.x"].astype(np.float32),
                curr_cuboids["position.y"].astype(np.float32),
                curr_cuboids["position.z"].astype(np.float32),
            ]
        ).T
        cuboid_poses = np.eye(4)[None].repeat(len(uuids), axis=0)
        cuboid_poses[:, :3, :3] = rot
        cuboid_poses[:, :3, 3] = pos
        cuboid_poses = global_to_ego @ cuboid_poses

        dims = np.vstack(
            [
                curr_cuboids["dimensions.x"].astype(np.float32),
                curr_cuboids["dimensions.y"].astype(np.float32),
                curr_cuboids["dimensions.z"].astype(np.float32),
            ]
        ).T

        frame_dynamic_boxes = {}
        frame_verts = {}
        for cuboid_index in range(len(uuids)):
            uuid = uuids[cuboid_index]
            vertices = get_vertices(dims[cuboid_index])
            cuboids[uuids[cuboid_index]] = {
                "label": labels[cuboid_index],
                "dims": dims[cuboid_index],
                "verts": vertices,
            }
            frame_dynamic_boxes[uuid] = {"object_to_ego": cuboid_poses[cuboid_index].tolist()}
            frame_verts[uuid] = vertices

        for cuboid_index, uuid in enumerate(uuids):
            pose = {"object_to_ego": cuboid_poses[cuboid_index].tolist()}
            for camera_index in range(len(SOURCE_CAMERAS)):
                meta_data["frames"][frame_idx * len(SOURCE_CAMERAS) + camera_index]["dynamics"][uuid] = pose

        write_box_images(frame_idx, global_to_ego, frame_dynamic_boxes, frame_verts)

    meta_data["verts"] = {uuid: cuboid["verts"].tolist() for uuid, cuboid in cuboids.items()}
    camera_paras = collect_camera_paras(sequence, SOURCE_CAMERAS, first_intrinsics, first_image_sizes)
    write_camera_paras(outdir, camera_paras)

    with open(os.path.join(outdir, "meta_data.json"), "w") as wf:
        json.dump(meta_data, wf, indent=2)

    if not args.no_video and video_images:
        media.write_video(os.path.join(outdir, "view.mp4"), video_images, fps=10)


if __name__ == "__main__":
    main()
