import argparse
import json
import os
import sys

import cv2
import mediapy as media
import numpy as np
import open3d as o3d
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, utils
from tqdm import tqdm

LOADER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if LOADER_ROOT not in sys.path:
    sys.path.insert(0, LOADER_ROOT)

from common import (  # noqa: E402
    BOX_DRAW_INTERVAL,
    FRONT_CAMERA,
    WAYMO_CAMERA_MAP,
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


OPENGL_TO_WAYMO = np.array(
    [
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ]
)

TYPE_LIST = ["UNKNOWN", "VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST"]


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_path", type=str, required=True)
    parser.add_argument("-s", "--segment", type=str, required=True)
    parser.add_argument("-c", "--cameras", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("-o", "--outpath", type=str, required=True)
    parser.add_argument("--downsample", type=float, default=2)
    parser.add_argument("--no_video", action="store_true", default=False)
    return parser.parse_args()


def rotz_matrix(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def get_vertices(dim, bottom_center=np.array([0.0, 0.0, 0.0])):
    vertices = bottom_center[None, :].repeat(8, axis=0).astype(float)
    vertices[:4, 0] += dim[0] / 2
    vertices[4:, 0] -= dim[0] / 2
    vertices[[0, 1, 4, 5], 1] += dim[1] / 2
    vertices[[2, 3, 6, 7], 1] -= dim[1] / 2
    vertices[[0, 2, 5, 7], 2] += dim[2] / 2
    vertices[[1, 3, 4, 6], 2] -= dim[2] / 2
    return vertices


def validate_cameras(cameras):
    unknown = [cam for cam in cameras if cam not in WAYMO_CAMERA_MAP]
    if unknown:
        raise ValueError(f"Unsupported Waymo cameras: {unknown}")
    if 1 not in cameras:
        raise ValueError("Waymo loader requires camera 1 as CAM_FRONT")


def read_first_frame_info(datafile, cameras, outdir):
    for frame in datafile:
        lidar_points = []
        for laser_name in [
            dataset_pb2.LaserName.TOP,
            dataset_pb2.LaserName.FRONT,
            dataset_pb2.LaserName.SIDE_LEFT,
            dataset_pb2.LaserName.SIDE_RIGHT,
            dataset_pb2.LaserName.REAR,
        ]:
            laser = utils.get(frame.lasers, laser_name)
            laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)
            range_images, camera_projections, range_image_top_pose = utils.parse_range_image_and_camera_projection(laser)
            points, _ = utils.project_to_pointcloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                laser_calibration,
            )
            lidar_points.append(points[:, :3])

        lidar_points = np.concatenate(lidar_points)
        ground_mask = (np.abs(lidar_points[:, 0]) < 6) & (np.abs(lidar_points[:, 1]) < 3)
        lidar_points = lidar_points[ground_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_points)
        o3d.io.write_point_cloud(os.path.join(outdir, "ground_lidar.ply"), pcd)
        plane_model, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model

        camera_poses = {}
        for camera in frame.context.camera_calibrations:
            if camera.name not in cameras:
                continue
            c2v = np.array(camera.extrinsic.transform).reshape(4, 4)
            camera_poses[WAYMO_CAMERA_MAP[camera.name]] = c2v @ OPENGL_TO_WAYMO

        front_cam_t = camera_poses[FRONT_CAMERA][:3, 3]
        ground_height = -(a * front_cam_t[0] + b * front_cam_t[1] + d) / c
        write_front_info(outdir, front_cam_t[2] - ground_height, rect_mat=None)
        break


def main():
    args = get_opts()
    validate_cameras(args.cameras)

    seq_path = os.path.join(args.base_path, args.segment)
    datafile = WaymoDataFileReader(seq_path)
    outdir = args.outpath
    source_cameras = args.cameras
    canonical_cameras = [WAYMO_CAMERA_MAP[cam] for cam in source_cameras]

    init_output_dirs(outdir, canonical_cameras)
    read_first_frame_info(datafile, source_cameras, outdir)

    ego_poses, extr, intr, imsize, raw_imsize = {}, {}, {}, {}, {}
    vehicles, dynamics = {}, {}
    timestamps = []
    start_timestamp = None
    segment_name = None
    segment_location = None
    video_images = []

    for frame_idx, frame in tqdm(enumerate(datafile)):
        if segment_name is None:
            segment_name = frame.context.name
            segment_location = getattr(frame.context.stats, "location", "")
        if start_timestamp is None:
            start_timestamp = frame.timestamp_micros / 1e6
        timestamp = frame.timestamp_micros / 1e6 - start_timestamp
        timestamps.append(timestamp)

        frame_video_images = {}
        for img_pkg in frame.images:
            if img_pkg.name not in source_cameras:
                continue
            canonical_camera = WAYMO_CAMERA_MAP[img_pkg.name]
            img = cv2.imdecode(np.frombuffer(img_pkg.image, np.uint8), cv2.IMREAD_COLOR)
            raw_h, raw_w = img.shape[:2]
            img = crop_and_downsample_image(img, downsample=args.downsample)
            h, w = img.shape[:2]
            output_path = os.path.join(outdir, "images", canonical_camera, f"{frame_idx:06d}.png")
            cv2.imwrite(output_path, img)

            raw_imsize.setdefault(img_pkg.name, []).append((raw_h, raw_w))
            imsize.setdefault(img_pkg.name, []).append((h, w))
            ego_poses.setdefault(img_pkg.name, []).append(np.array(img_pkg.pose.transform).reshape(4, 4))
            frame_video_images[canonical_camera] = img

        if not args.no_video:
            append_video_frame(video_images, frame_video_images)

        for camera in frame.context.camera_calibrations:
            if camera.name not in source_cameras:
                continue
            intr.setdefault(camera.name, [])
            extr.setdefault(camera.name, [])

            raw_h, raw_w = raw_imsize[camera.name][-1]
            cam_intrinsic, final_w, final_h = build_final_intrinsic(
                make_intrinsic_matrix(
                    camera.intrinsic[0],
                    camera.intrinsic[1],
                    camera.intrinsic[2],
                    camera.intrinsic[3],
                ),
                raw_w,
                raw_h,
                downsample=args.downsample,
            )
            if (final_h, final_w) != imsize[camera.name][-1]:
                raise ValueError(f"Waymo camera {camera.name} final size mismatch")
            intr[camera.name].append(cam_intrinsic)

            c2v = np.array(camera.extrinsic.transform).reshape(4, 4)
            extr[camera.name].append(c2v)

        v2w = np.array(frame.pose.transform).reshape(4, 4)
        for obj in frame.laser_labels:
            type_name = TYPE_LIST[obj.type]
            height = obj.box.height
            width = obj.box.width
            length = obj.box.length
            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height / 2
            t_b2l = np.array([x, y, z, 1]).reshape((4, 1))
            t_b2w = v2w @ t_b2l
            rotation_y = -obj.box.heading - np.pi / 2
            if type_name in ["VEHICLE", "PEDESTRIAN", "CYCLIST"]:
                vehicles.setdefault(obj.id, {"rt": [], "timestamp": [], "frame": []})
                object_to_ego = np.eye(4)
                object_to_ego[:3, :3] = rotz_matrix(obj.box.heading)
                object_to_ego[:3, 3] = np.array([x, y, obj.box.center_z], dtype=float)
                vehicles[obj.id]["rt"].append(object_to_ego)
                vehicles[obj.id].setdefault("center_global", []).append(t_b2w[:3, 0])
                vehicles[obj.id].setdefault("dims", []).append(np.array([length, width, height], dtype=float))
                vehicles[obj.id]["timestamp"].append(timestamp)
                vehicles[obj.id]["frame"].append(frame_idx)

    origin_ego_to_global = ego_poses[1][0]
    global_to_origin_ego = invert_transform(origin_ego_to_global)
    camera_to_ego = {
        cam: extr[cam][0] @ OPENGL_TO_WAYMO
        for cam in source_cameras
    }

    dynamic_id = 0
    for _, infos in vehicles.items():
        infos["rt"] = np.stack(infos["rt"])
        centers_global = np.stack(infos["center_global"])
        movement = np.max(np.max(centers_global, axis=0) - np.min(centers_global, axis=0))
        if movement > 1:
            dynamics[dynamic_id] = infos
            dynamic_id += 1

    verts, rts = {}, {}
    for dynamic_id, infos in dynamics.items():
        points = get_vertices(infos["dims"][0])
        seq_visible = False
        for idx, fid in enumerate(infos["frame"]):
            object_to_ego = infos["rt"][idx]
            frame_visible = False
            for cam in source_cameras:
                camera_from_ego = invert_transform(camera_to_ego[cam])
                k = intr[cam][fid]
                h, w = imsize[cam][fid]
                object_to_camera = camera_from_ego @ object_to_ego
                points_cam = (object_to_camera[:3, :3] @ points.T).T + object_to_camera[:3, 3]
                points_screen = (k[:3, :3] @ points_cam.T).T + k[:3, 3]
                points_uv = (points_screen[:, :2] / points_screen[:, 2][:, None]).astype(int)
                valid_mask = (
                    (points_screen[:, 2] > 0)
                    & (points_uv[:, 0] >= 0)
                    & (points_uv[:, 1] >= 0)
                    & (points_uv[:, 0] < w)
                    & (points_uv[:, 1] < h)
                )
                if np.sum(valid_mask) > 0:
                    frame_visible = True
                    seq_visible = True
                    break

            if frame_visible:
                rts.setdefault(fid, {})[dynamic_id] = {"object_to_ego": object_to_ego.tolist()}

        if seq_visible:
            verts[dynamic_id] = points.tolist()

    for frame_idx in range(len(intr[1])):
        if frame_idx % BOX_DRAW_INTERVAL != 0:
            continue
        for source_camera in source_cameras:
            canonical_camera = WAYMO_CAMERA_MAP[source_camera]
            im_name = f"{frame_idx:06d}.png"
            image_path = os.path.join(outdir, "images", canonical_camera, im_name)
            box_img = cv2.imread(image_path)
            if box_img is None:
                raise FileNotFoundError(image_path)
            draw_dynamic_boxes(
                box_img,
                intr[source_camera][frame_idx],
                camera_to_ego[source_camera],
                rts.get(frame_idx, {}),
                verts,
            )
            cv2.imwrite(os.path.join(outdir, "box", canonical_camera, im_name), box_img)

    meta_data = {
        "camera_model": "OPENCV",
        "ego_coordinate": {"x": "forward", "y": "left", "z": "up", "handedness": "right"},
        "frames": [],
        "verts": verts,
        "world_origin": "first_ego_pose",
        "origin_ego_to_global": origin_ego_to_global.tolist(),
    }
    write_geo_reference(
        outdir,
        {
            "dataset": "waymo",
            "available": False,
            "source": "frame.context.stats.location and frame.pose",
            "reason": "Waymo Open Dataset exposes a dataset-native metric pose and coarse location string, not precise latitude/longitude.",
            "segment": segment_name,
            "location": segment_location,
        },
    )

    camera_paras = {}
    for source_camera in source_cameras:
        canonical_camera = WAYMO_CAMERA_MAP[source_camera]
        h, w = imsize[source_camera][0]
        camera_paras[canonical_camera] = {
            "source_camera_name": str(source_camera),
            "camera_to_ego": camera_to_ego[source_camera],
            "intrinsic": intr[source_camera][0],
            "width": w,
            "height": h,
        }
    write_camera_paras(outdir, camera_paras)

    for i in range(len(intr[1])):
        for source_camera in source_cameras:
            canonical_camera = WAYMO_CAMERA_MAP[source_camera]
            ego_to_world = global_to_origin_ego @ ego_poses[source_camera][i]
            meta_data["frames"].append(
                {
                    "rgb_path": f"./images/{canonical_camera}/{i:06d}.png",
                    "camera_name": canonical_camera,
                    "source_camera_name": str(source_camera),
                    "ego_to_world": ego_to_world.tolist(),
                    "timestamp": timestamps[i],
                    "dynamics": rts.get(i, {}),
                }
            )

    with open(os.path.join(outdir, "meta_data.json"), "w") as wf:
        json.dump(meta_data, wf, indent=2)

    if not args.no_video and video_images:
        media.write_video(os.path.join(outdir, "view.mp4"), video_images, fps=10)


if __name__ == "__main__":
    main()
