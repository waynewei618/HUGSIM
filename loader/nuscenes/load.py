import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import mediapy as media
import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

LOADER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(LOADER_ROOT, ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
for path in (LOADER_ROOT, DATA_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from common import (  # noqa: E402
    append_video_frame,
    build_final_intrinsic,
    crop_and_downsample_image,
    init_output_dirs,
    invert_transform,
    write_camera_paras,
    write_front_info,
    write_geo_reference,
)
from nusc.utils import (  # noqa: E402
    AVAILABLE_CAMERAS,
    WLH_TO_LWH,
    _rotation_translation_to_pose,
    find_all_sample,
    get_vertices,
)

CAMERA_CROPS = {
    "CAM_BACK": {"bottom": 80},
}


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--no_video", action="store_true", default=False)
    return parser.parse_args()


def read_first_frame_info(nusc, first_sample, dataroot, outdir):
    lidar_data = nusc.get("sample_data", first_sample["data"]["LIDAR_TOP"])
    calibrated_lidar_data = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    ego_pose_data = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    ego_pose = _rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])
    lidar_pose = _rotation_translation_to_pose(
        calibrated_lidar_data["rotation"],
        calibrated_lidar_data["translation"],
    )
    _ = ego_pose @ lidar_pose

    lidar_fn = os.path.join(dataroot, lidar_data["filename"])
    points = np.fromfile(lidar_fn, dtype=np.float32).reshape([-1, 5])[:, :3]
    ego_mask = (np.abs(points[:, 0]) < 1.5) & (np.abs(points[:, 1]) < 2.5)
    ground_mask = (np.abs(points[:, 0]) < 3) & (np.abs(points[:, 1]) < 6)
    points = points[ground_mask & (~ego_mask)]
    points = (lidar_pose[:3, :3] @ points.T).T + lidar_pose[:3, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model
    o3d.io.write_point_cloud(os.path.join(outdir, "ground_lidar.ply"), pcd)

    all_campose = {}
    for cam in AVAILABLE_CAMERAS:
        cam_data = nusc.get("sample_data", first_sample["data"][cam])
        calibrated_cam_data = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        cam_pose = _rotation_translation_to_pose(
            calibrated_cam_data["rotation"],
            calibrated_cam_data["translation"],
        )
        all_campose[cam] = cam_pose

    front_cam_t = all_campose["CAM_FRONT"][:3, 3]
    height = -(a * front_cam_t[0] + b * front_cam_t[1] + d) / c

    n = np.array([a, b, c])
    front_cam_z = all_campose["CAM_FRONT"][:3, 0]
    cos_theta = np.dot(n, front_cam_z) / (np.linalg.norm(n) * np.linalg.norm(front_cam_z))
    pitch_angle = np.arccos(cos_theta)
    rect_pitch = np.pi / 2 - pitch_angle
    rect_mat = R.from_euler("x", rect_pitch).as_matrix()
    write_front_info(outdir, front_cam_t[2] - height, rect_mat=rect_mat)
    return all_campose


def select_samples(nusc, first_sample, start, end):
    samples = find_all_sample(nusc, first_sample)
    stop = end if end and end > 0 else None
    return samples[start:stop]


def get_ego_pose(nusc, sample_data):
    ego_pose_data = nusc.get("ego_pose", sample_data["ego_pose_token"])
    return _rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])


def get_box_global(nusc, box_token):
    box = nusc.get_box(box_token)
    instance_token = nusc.get("sample_annotation", box.token)["instance_token"]
    object_to_global = np.eye(4)
    object_to_global[:3, :3] = box.orientation.rotation_matrix
    object_to_global[:3, 3] = np.array(box.center)
    object_to_global = object_to_global @ WLH_TO_LWH
    return instance_token, object_to_global, np.array(box.wlh)


def collect_camera_paras(nusc, first_sample, dataroot, downsample):
    camera_paras = {}
    for cam in AVAILABLE_CAMERAS:
        sample_data = nusc.get("sample_data", first_sample["data"][cam])
        calibrated_cam_data = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        camera_to_ego = _rotation_translation_to_pose(
            calibrated_cam_data["rotation"],
            calibrated_cam_data["translation"],
        )
        _, _, height, width, intrinsic = load_camera_image(nusc, sample_data, dataroot, downsample)
        camera_paras[cam] = {
            "source_camera_name": cam,
            "camera_to_ego": camera_to_ego,
            "intrinsic": intrinsic,
            "width": width,
            "height": height,
        }
    return camera_paras


def load_camera_image(nusc, sample_data, dataroot, downsample):
    calibrated_sensor_data = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    image_path = os.path.join(dataroot, sample_data["filename"])
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    raw_height, raw_width = image.shape[:2]
    crop = CAMERA_CROPS.get(sample_data["channel"])
    intrinsic, final_width, final_height = build_final_intrinsic(
        calibrated_sensor_data["camera_intrinsic"],
        raw_width,
        raw_height,
        crop=crop,
        downsample=downsample,
    )
    image = crop_and_downsample_image(image, crop=crop, downsample=downsample)
    if image.shape[:2] != (final_height, final_width):
        raise ValueError(f"NuScenes camera {sample_data['channel']} final size mismatch")

    return image, os.path.basename(image_path), final_height, final_width, intrinsic


def build_geo_reference(nusc, scene, version):
    log = nusc.get("log", scene["log_token"])
    return {
        "dataset": "nuscenes",
        "version": version,
        "scene": scene["name"],
        "available": False,
        "source": "ego_pose.json and log.json",
        "reason": "NuScenes public metadata exposes dataset-native metric ego poses and coarse log location, not precise latitude/longitude.",
        "location": log.get("location"),
        "logfile": log.get("logfile"),
        "vehicle": log.get("vehicle"),
        "date_captured": log.get("date_captured"),
    }


def main():
    args = get_opts()
    outdir = args.out
    init_output_dirs(outdir, AVAILABLE_CAMERAS)

    meta_data = {
        "camera_model": "OPENCV",
        "verts": {},
        "frames": [],
    }

    nusc = NuScenes(version=args.version, dataroot=args.datapath, verbose=True)
    scene = nusc.get("scene", nusc.field2token("scene", "name", args.seq)[0])
    first_sample = nusc.get("sample", scene["first_sample_token"])
    write_geo_reference(outdir, build_geo_reference(nusc, scene, args.version))

    read_first_frame_info(nusc, first_sample, args.datapath, outdir)
    samples = select_samples(nusc, first_sample, args.start, args.end)

    front_sample_data = nusc.get("sample_data", samples[0]["data"]["CAM_FRONT"])
    origin_ego_to_global = get_ego_pose(nusc, front_sample_data)
    global_to_origin_ego = invert_transform(origin_ego_to_global)
    meta_data["ego_coordinate"] = {"x": "forward", "y": "left", "z": "up", "handedness": "right"}
    meta_data["world_origin"] = "first_ego_pose"
    meta_data["origin_ego_to_global"] = origin_ego_to_global.tolist()
    camera_paras = collect_camera_paras(nusc, first_sample, args.datapath, args.downsample)
    write_camera_paras(outdir, camera_paras)

    tracks = defaultdict(list)
    for sample in samples:
        for box_token in sample["anns"]:
            instance_token, object_to_global, _ = get_box_global(nusc, box_token)
            tracks[instance_token].append(object_to_global)

    dynamic_instance = set()
    for instance_token, traj_list in tracks.items():
        dynamic = np.max(traj_list[0][:3, 3] - traj_list[-1][:3, 3]) > 2
        if dynamic:
            dynamic_instance.add(instance_token)

    video_images = []
    start_time = -1
    for frame_idx, sample in tqdm(enumerate(samples)):
        dynamics = {}
        for box_token in sample["anns"]:
            instance_token, object_to_global, lhw = get_box_global(nusc, box_token)
            if instance_token not in dynamic_instance:
                continue
            dynamics[instance_token] = object_to_global
            if instance_token not in meta_data["verts"]:
                meta_data["verts"][instance_token] = get_vertices(lhw).tolist()

        frame_video_images = {}
        for cam in AVAILABLE_CAMERAS:
            sample_data = nusc.get("sample_data", sample["data"][cam])
            im, _, _, _, _ = load_camera_image(
                nusc,
                sample_data,
                args.datapath,
                downsample=args.downsample,
            )
            im_name = f"{frame_idx:05d}.jpg"
            cv2.imwrite(os.path.join(outdir, "images", cam, im_name), im)
            ego_to_global = get_ego_pose(nusc, sample_data)
            global_to_ego = invert_transform(ego_to_global)
            ego_to_world = global_to_origin_ego @ ego_to_global
            frame_dynamics = {
                instance_token: {"object_to_ego": (global_to_ego @ object_to_global).tolist()}
                for instance_token, object_to_global in dynamics.items()
            }

            timestamp = sample["timestamp"] / 1e6
            if start_time < 0:
                start_time = timestamp
            timestamp -= start_time
            meta_data["frames"].append(
                {
                    "rgb_path": os.path.join("./images", cam, im_name),
                    "camera_name": cam,
                    "source_camera_name": cam,
                    "ego_to_world": ego_to_world.tolist(),
                    "timestamp": timestamp,
                    "dynamics": frame_dynamics,
                }
            )
            frame_video_images[cam] = im

        if not args.no_video:
            append_video_frame(video_images, frame_video_images)

    with open(os.path.join(outdir, "meta_data.json"), "w") as wf:
        json.dump(meta_data, wf, indent=2)

    if not args.no_video and video_images:
        media.write_video(os.path.join(outdir, "view.mp4"), video_images, fps=12)


if __name__ == "__main__":
    main()
