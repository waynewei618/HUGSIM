import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

ME_LOADER_ROOT = os.path.abspath(os.path.dirname(__file__))
LOADER_ROOT = os.path.abspath(os.path.join(ME_LOADER_ROOT, ".."))
if LOADER_ROOT not in sys.path:
    sys.path.insert(0, LOADER_ROOT)

from common import (  # noqa: E402
    BOX_DRAW_INTERVAL,
    append_video_frame,
    build_final_intrinsic,
    crop_and_downsample_image,
    draw_dynamic_boxes,
    init_output_dirs,
    invert_transform,
    write_camera_paras,
    write_front_info,
    write_geo_reference,
)


DEFAULT_DATAPATH = "outputs/me/resplit"
ME_LIDAR = "AT128"
ME_CAMERA_MAP = {
    "CAM_FRONT_120": "CAM_FRONT",
    "CAM_FRONT_LEFT": "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT": "CAM_FRONT_RIGHT",
    "CAM_BACK": "CAM_BACK",
    "CAM_BACK_LEFT": "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT": "CAM_BACK_RIGHT",
}
SOURCE_CAMERAS = tuple(ME_CAMERA_MAP.keys())
DYNAMIC_CATEGORIES = {"MotorVehicle", "Pedestrian", "TwoWheels", "Tricycle"}
DYNAMIC_MOVEMENT_THRESHOLD = 1.0


def camera_crop(source_camera, downsample):
    if source_camera == "CAM_FRONT_120":
        return {
            "top": 880 - 520,
            "bottom": 520,
            "left": 480 * downsample,
            "right": 480 * downsample,
        }
    if source_camera == "CAM_BACK":
        return {"bottom": 56}
    return None


@dataclass
class FrameRecord:
    annotation_path: Path
    annotation: dict
    frame_key: str
    timestamp: float
    ego_to_global: np.ndarray
    sensors: dict
    lidar_path: Path
    objects: list


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=DEFAULT_DATAPATH)
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--sub-scene", dest="sub_scene", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--downsample", type=int, default=2)
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def default_outpath(datapath, seq, sub_scene):
    input_path = Path(datapath) / seq / str(sub_scene)
    parts = list(input_path.parts)
    try:
        index = parts.index("resplit")
    except ValueError as exc:
        raise ValueError(f"Cannot derive --out because input path has no 'resplit' segment: {input_path}") from exc
    return str(Path(*parts[:index], *parts[index + 1 :]))


def annotation_files(sub_scene_dir):
    annotation_dir = sub_scene_dir / "annotations_info"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(annotation_dir)
    files = sorted(path for path in annotation_dir.glob("*.json") if path.is_file())
    if not files:
        raise RuntimeError(f"No annotation json files found in {annotation_dir}")
    return files


def timestamp_seconds(timestamp_str):
    time_text = timestamp_str.split("_")[1]
    dt = datetime.strptime(time_text, "%Y-%m-%d-%H-%M-%S-%f")
    return dt.timestamp()


def matrix4_from_meta(meta_by_frame, frame_key):
    frame_meta = meta_by_frame.get(frame_key)
    if not frame_meta:
        raise KeyError(f"Frame key not found in meta.json: {frame_key}")
    pose = frame_meta.get("pose") or []
    if not pose:
        raise KeyError(f"Frame has no ego pose in meta.json: {frame_key}")
    matrix4 = pose[0].get("matrix4") or []
    if len(matrix4) != 16:
        raise ValueError(f"Frame ego pose matrix4 must have 16 values: {frame_key}")
    return np.asarray(matrix4, dtype=np.float64).reshape(4, 4)


def sensor_to_ego(sensor):
    params = sensor.get("sensor_param") or {}
    quat_wxyz = params.get("sensor2ego_rotation")
    translation = params.get("sensor2ego_translation")
    if quat_wxyz is None or translation is None:
        raise KeyError(f"Sensor has no sensor2ego calibration: {sensor.get('sensor_id')}")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_matrix()
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def resolve_sensor_path(sub_scene_dir, sensor):
    sensor_id = sensor["sensor_id"]
    filename = Path(sensor["data_path"]).name
    if sensor_id in ME_CAMERA_MAP:
        return sub_scene_dir / "images" / sensor_id / filename
    if sensor_id == ME_LIDAR:
        return sub_scene_dir / "lidars" / ME_LIDAR / filename
    return None


def object_to_ego_from_annotation(annotation):
    x, y, z = annotation["PC_3D"][0:3]
    roll, pitch, yaw = annotation["PC_3D"][6:9]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_euler("zxy", [yaw, roll, pitch]).as_matrix()
    transform[:3, 3] = np.asarray([x, y, z], dtype=np.float64)
    return transform


def get_vertices(dim, center=None):
    if center is None:
        center = np.array([0.0, 0.0, 0.0])

    length, width, height = dim
    vertices = center[None, :].repeat(8, axis=0).astype(np.float64)
    vertices[:4, 0] += length / 2
    vertices[4:, 0] -= length / 2
    vertices[[0, 1, 4, 5], 1] += width / 2
    vertices[[2, 3, 6, 7], 1] -= width / 2
    vertices[[0, 2, 5, 7], 2] += height / 2
    vertices[[1, 3, 4, 6], 2] -= height / 2
    return vertices


def load_frames(sub_scene_dir, meta):
    meta_by_frame = meta.get("meta", {})
    frames = []
    first_timestamp = None
    for annotation_path in annotation_files(sub_scene_dir):
        annotation = load_json(annotation_path)
        frame_key = annotation_path.stem
        ego_to_global = matrix4_from_meta(meta_by_frame, frame_key)
        timestamp = timestamp_seconds(annotation["meta"]["sensor"][0]["timestamp"])
        if first_timestamp is None:
            first_timestamp = timestamp
        timestamp -= first_timestamp

        sensors = {}
        lidar_path = None
        for sensor in annotation["meta"]["sensor"]:
            sensor_id = sensor["sensor_id"]
            sensor_path = resolve_sensor_path(sub_scene_dir, sensor)
            if sensor_path is None:
                continue
            if not sensor_path.is_file():
                raise FileNotFoundError(sensor_path)
            sensor = dict(sensor)
            sensor["resolved_data_path"] = str(sensor_path)
            if sensor_id in ME_CAMERA_MAP:
                sensors[sensor_id] = sensor
            elif sensor_id == ME_LIDAR:
                lidar_path = sensor_path

        missing_cameras = [camera for camera in SOURCE_CAMERAS if camera not in sensors]
        if missing_cameras:
            raise FileNotFoundError(f"Missing camera data for {frame_key}: {missing_cameras}")
        if lidar_path is None:
            raise FileNotFoundError(f"Missing {ME_LIDAR} lidar data for {frame_key}")

        objects = []
        for item in annotation.get("annotations", []):
            if item.get("category") not in DYNAMIC_CATEGORIES:
                continue
            track_id = str(item.get("property", {}).get("track_id"))
            if track_id == "None":
                continue
            w, l, h = item["PC_3D"][3:6]
            objects.append(
                {
                    "track_id": track_id,
                    "object_to_ego": object_to_ego_from_annotation(item),
                    "center": np.asarray(item["PC_3D"][0:3], dtype=np.float64),
                    "vertices": get_vertices((l, w, h)),
                }
            )

        frames.append(
            FrameRecord(
                annotation_path=annotation_path,
                annotation=annotation,
                frame_key=frame_key,
                timestamp=timestamp,
                ego_to_global=ego_to_global,
                sensors=sensors,
                lidar_path=lidar_path,
                objects=objects,
            )
        )
    return frames


def dynamic_tracks(frames):
    tracks = {}
    for frame_idx, frame in enumerate(frames):
        for obj in frame.objects:
            center_global = frame.ego_to_global @ np.append(obj["center"], 1.0)
            track = tracks.setdefault(
                obj["track_id"],
                {"global_centers": [], "vertices": obj["vertices"], "frames": []},
            )
            track["global_centers"].append(center_global[:3])
            track["frames"].append((frame_idx, obj["object_to_ego"]))

    dynamic_ids = set()
    for track_id, track in tracks.items():
        centers = np.stack(track["global_centers"])
        movement = float(np.max(np.linalg.norm(centers[:, :2] - centers[0, :2], axis=1)))
        if movement > DYNAMIC_MOVEMENT_THRESHOLD:
            dynamic_ids.add(track_id)

    verts = {track_id: tracks[track_id]["vertices"].tolist() for track_id in dynamic_ids}
    dynamics_by_frame = {}
    for track_id in dynamic_ids:
        for frame_idx, object_to_ego in tracks[track_id]["frames"]:
            dynamics_by_frame.setdefault(frame_idx, {})[track_id] = {"object_to_ego": object_to_ego.tolist()}
    return verts, dynamics_by_frame


def process_camera_image(sensor, source_camera, downsample):
    image = cv2.imread(sensor["resolved_data_path"])
    if image is None:
        raise FileNotFoundError(sensor["resolved_data_path"])

    raw_height, raw_width = image.shape[:2]
    raw_intrinsic = np.asarray(sensor["sensor_param"]["intrinsic"], dtype=np.float64)
    crop = camera_crop(source_camera, downsample)
    intrinsic, final_width, final_height = build_final_intrinsic(
        raw_intrinsic,
        raw_width,
        raw_height,
        crop=crop,
        downsample=downsample,
    )
    image = crop_and_downsample_image(image, crop=crop, downsample=downsample)
    if image.shape[:2] != (final_height, final_width):
        raise ValueError(f"ME camera {source_camera} final size mismatch")
    return image, intrinsic, final_height, final_width


def fit_ground_from_first_lidar(first_frame, camera_to_ego, outdir):
    pcd = o3d.io.read_point_cloud(str(first_frame.lidar_path))
    lidar_points = np.asarray(pcd.points)[:, :3]
    ground_mask = (np.abs(lidar_points[:, 0]) < 6) & (np.abs(lidar_points[:, 1]) < 3)
    lidar_points = lidar_points[ground_mask]
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(lidar_points)
    o3d.io.write_point_cloud(os.path.join(outdir, "ground_lidar.ply"), ground_pcd)
    plane_model, _ = ground_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model
    front_cam_t = camera_to_ego["CAM_FRONT"][:3, 3]
    ground_height = -(a * front_cam_t[0] + b * front_cam_t[1] + d) / c
    write_front_info(outdir, front_cam_t[2] - ground_height, rect_mat=None)


def build_geo_reference(seq, sub_scene):
    return {
        "dataset": "me",
        "sequence": seq,
        "sub_scene": str(sub_scene),
        "available": False,
        "source": "meta.json pose",
        "reason": "ME metadata provides metric ego poses but no precise latitude/longitude reference.",
    }


def main():
    args = get_opts()
    datapath = Path(args.datapath)
    sub_scene_dir = datapath / args.seq / args.sub_scene
    scene_dir = datapath / args.seq
    if not sub_scene_dir.is_dir():
        raise FileNotFoundError(sub_scene_dir)

    outdir = args.out or default_outpath(args.datapath, args.seq, args.sub_scene)
    init_output_dirs(outdir, tuple(ME_CAMERA_MAP.values()))

    meta = load_json(scene_dir / "meta.json")
    frames = load_frames(sub_scene_dir, meta)
    origin_ego_to_global = frames[0].ego_to_global
    global_to_origin_ego = invert_transform(origin_ego_to_global)
    verts, dynamics_by_frame = dynamic_tracks(frames)

    meta_data = {
        "camera_model": "opencv_fisheye",
        "ego_coordinate": {"x": "forward", "y": "left", "z": "up", "handedness": "right"},
        "frames": [],
        "verts": verts,
        "world_origin": "first_ego_pose",
        "origin_ego_to_global": origin_ego_to_global.tolist(),
    }

    camera_paras = {}
    first_camera_to_ego = {}
    first_intrinsics = {}
    first_image_sizes = {}
    video_images = []

    for frame_idx, frame in tqdm(list(enumerate(frames))):
        ego_to_world = global_to_origin_ego @ frame.ego_to_global
        frame_video_images = {}
        frame_dynamics = dynamics_by_frame.get(frame_idx, {})
        for source_camera in SOURCE_CAMERAS:
            canonical_camera = ME_CAMERA_MAP[source_camera]
            sensor = frame.sensors[source_camera]
            camera_to_ego = sensor_to_ego(sensor)
            image, intrinsic, height, width = process_camera_image(sensor, source_camera, args.downsample)
            image_name = f"{frame_idx:06d}.png"
            image_path = os.path.join(outdir, "images", canonical_camera, image_name)
            cv2.imwrite(image_path, image)

            if canonical_camera not in camera_paras:
                camera_paras[canonical_camera] = {
                    "source_camera_name": source_camera,
                    "camera_to_ego": camera_to_ego,
                    "intrinsic": intrinsic,
                    "width": width,
                    "height": height,
                }
                first_camera_to_ego[canonical_camera] = camera_to_ego
                first_intrinsics[canonical_camera] = intrinsic
                first_image_sizes[canonical_camera] = (height, width)

            if frame_idx % BOX_DRAW_INTERVAL == 0:
                box_image = image.copy()
                draw_dynamic_boxes(box_image, intrinsic, camera_to_ego, frame_dynamics, verts)
                cv2.imwrite(os.path.join(outdir, "box", canonical_camera, image_name), box_image)

            meta_data["frames"].append(
                {
                    "rgb_path": os.path.join("./images", canonical_camera, image_name),
                    "camera_name": canonical_camera,
                    "source_camera_name": source_camera,
                    "ego_to_world": ego_to_world.tolist(),
                    "timestamp": frame.timestamp,
                    "dynamics": frame_dynamics,
                }
            )
            frame_video_images[canonical_camera] = image

        append_video_frame(video_images, frame_video_images, image_size=(384, 216))

    fit_ground_from_first_lidar(frames[0], first_camera_to_ego, outdir)
    write_camera_paras(outdir, camera_paras, camera_model="opencv_fisheye")
    write_geo_reference(outdir, build_geo_reference(args.seq, args.sub_scene))

    with open(os.path.join(outdir, "meta_data.json"), "w") as wf:
        json.dump(meta_data, wf, indent=2)

    if video_images:
        media.write_video(os.path.join(outdir, "view.mp4"), video_images, fps=10)


if __name__ == "__main__":
    main()
