import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from aeb_hil_vil_render.gaussian_scene_renderer import (
    CameraRenderRequest,
    GaussianSceneRenderer,
    as_camera_intrinsics,
    as_transform,
)


REAL_VEHICLE_FRONT_CAMERA_ID = "front_120/cam1"
REAL_VEHICLE_FRONT_CAMERA_INTRINSICS = Path(__file__).with_name("camera_intrinsics.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render camera frames from an ego trajectory and compose a reconstruction comparison video."
    )
    parser.add_argument("scene_path", help="Exported HUGSIM scene directory.")
    parser.add_argument("original_video", help="Original collected camera video.")
    parser.add_argument("trajectory_path", help="Trajectory JSON with ego poses.")
    parser.add_argument("camera_path", help="Camera JSON with intrinsics, camera_to_ego, and resolution.")
    parser.add_argument("output_video", help="Side-by-side comparison video.")
    parser.add_argument(
        "--real-camera-intrinsics",
        default=REAL_VEHICLE_FRONT_CAMERA_INTRINSICS,
        help="Camera intrinsics JSON that contains the real vehicle AEB front camera.",
    )
    parser.add_argument(
        "--real-camera-id",
        default=REAL_VEHICLE_FRONT_CAMERA_ID,
        help="Camera ID in --real-camera-intrinsics used for the real vehicle render.",
    )
    parser.add_argument(
        "--real-camera-output",
        help="Optional output path for the real vehicle AEB camera rendered video.",
    )
    parser.add_argument(
        "--real-camera-timing-output",
        help="Optional CSV path for per-frame real vehicle camera render timing.",
    )
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def required(data, name, source):
    if name not in data:
        raise ValueError(f"{source} must contain {name}")
    return data[name]


def quaternion_values(value, order, name):
    if isinstance(value, dict):
        try:
            return np.asarray([value["w"], value["x"], value["y"], value["z"]], dtype=np.float32), "wxyz"
        except KeyError as exc:
            raise ValueError(f"{name} quaternion dict must contain w, x, y, z") from exc
    return np.asarray(value, dtype=np.float32), order


def rotation_from_quaternion(value, order, name):
    quaternion, order = quaternion_values(value, order, name)
    if quaternion.shape != (4,):
        raise ValueError(f"{name} must be a 4D quaternion")

    norm = np.linalg.norm(quaternion)
    if norm < 1e-6:
        raise ValueError(f"{name} quaternion norm is too small")
    quaternion = quaternion / norm

    if order == "wxyz":
        w, x, y, z = quaternion
    elif order == "xyzw":
        x, y, z, w = quaternion
    else:
        raise ValueError(f"{name} quaternion_order must be wxyz or xyzw")

    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rpy_to_radians(rpy, unit, name):
    angles = np.asarray(rpy, dtype=np.float32)
    if angles.shape != (3,):
        raise ValueError(f"{name} must contain roll, pitch, yaw")

    if unit in ("rad", "radian", "radians"):
        return angles
    if unit in ("deg", "degree", "degrees"):
        return np.deg2rad(angles)
    raise ValueError(f"{name} angle_unit must be rad or degree")


def rotation_from_rpy(value, unit, name):
    roll, pitch, yaw = rpy_to_radians(value, unit, name)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def angle_unit_from_frame(frame, trajectory):
    return frame.get("angle_unit", trajectory.get("angle_unit", trajectory.get("rpy_unit", "rad"))).lower()


def ego_rotation_from_frame(frame, index, trajectory):
    name = f"trajectory frame {index}"

    if "ego_quaternion_wxyz" in frame:
        return rotation_from_quaternion(frame["ego_quaternion_wxyz"], "wxyz", f"{name} ego_quaternion_wxyz")
    if "ego_quaternion_xyzw" in frame:
        return rotation_from_quaternion(frame["ego_quaternion_xyzw"], "xyzw", f"{name} ego_quaternion_xyzw")
    if "ego_quaternion" in frame:
        order = frame.get("quaternion_order", trajectory.get("quaternion_order", "wxyz")).lower()
        return rotation_from_quaternion(frame["ego_quaternion"], order, f"{name} ego_quaternion")
    if "ego_rpy" in frame:
        return rotation_from_rpy(frame["ego_rpy"], angle_unit_from_frame(frame, trajectory), f"{name} ego_rpy")
    if "ego_roll_pitch_yaw" in frame:
        return rotation_from_rpy(
            frame["ego_roll_pitch_yaw"],
            angle_unit_from_frame(frame, trajectory),
            f"{name} ego_roll_pitch_yaw",
        )
    if "ego_rpy_degrees" in frame:
        return rotation_from_rpy(frame["ego_rpy_degrees"], "degree", f"{name} ego_rpy_degrees")
    if all(key in frame for key in ("roll", "pitch", "yaw")):
        return rotation_from_rpy(
            [frame["roll"], frame["pitch"], frame["yaw"]],
            angle_unit_from_frame(frame, trajectory),
            f"{name} roll/pitch/yaw",
        )
    if all(key in frame for key in ("r", "p", "yaw")):
        return rotation_from_rpy(
            [frame["r"], frame["p"], frame["yaw"]],
            angle_unit_from_frame(frame, trajectory),
            f"{name} r/p/yaw",
        )

    raise ValueError(
        f"{name} must contain ego rotation: ego_quaternion_wxyz, "
        "ego_quaternion_xyzw, ego_quaternion, ego_rpy, or roll/pitch/yaw"
    )


def ego_pose_from_frame(frame, index, trajectory):
    if not isinstance(frame, dict):
        raise ValueError(f"trajectory frame {index} must be an object")

    position = np.asarray(frame.get("ego_position"), dtype=np.float32)
    if position.shape != (3,):
        raise ValueError(f"trajectory frame {index} ego_position must be a 3D position")

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = ego_rotation_from_frame(frame, index, trajectory)
    pose[:3, 3] = position
    return pose


def frame_timestamp(frame, index):
    if isinstance(frame, dict) and "timestamp" in frame:
        return frame["timestamp"]
    return index


def infer_fps(trajectory, camera, frames):
    if "fps" in camera:
        return float(camera["fps"])
    if "fps" in trajectory:
        return float(trajectory["fps"])

    timestamps = [float(frame.get("timestamp", index)) for index, frame in enumerate(frames)]
    if len(timestamps) > 1:
        deltas = np.diff(timestamps)
        positive = deltas[deltas > 1e-6]
        if len(positive) > 0:
            return float(1.0 / np.median(positive))
    return 10.0


def find_camera_intrinsics(camera_intrinsics, camera_id):
    for camera in camera_intrinsics.get("cameras", []):
        if camera.get("camera_id") == camera_id:
            return camera
    raise ValueError(f"camera_intrinsics.json does not contain camera_id {camera_id}")


def camera_with_real_vehicle_intrinsics(camera, camera_intrinsics_path, camera_id):
    camera_intrinsics = load_json(camera_intrinsics_path)
    real_camera = find_camera_intrinsics(camera_intrinsics, camera_id)
    image_size = required(real_camera, "image_size", camera_id)
    intrinsics = required(required(real_camera, "intrinsics", camera_id), "camera_matrix", camera_id)
    projection = real_camera.get("projection", {})

    output = dict(camera)
    output["intrinsics"] = intrinsics
    output["width"] = int(required(image_size, "width", camera_id))
    output["height"] = int(required(image_size, "height", camera_id))
    if "near" in projection:
        output["near_plane"] = float(projection["near"])
    if "far" in projection:
        output["far_plane"] = float(projection["far"])
    output["source_camera"] = camera_id
    return output


def build_camera_render_requests(trajectory, camera):
    frames = trajectory.get("frames")
    if not frames:
        raise ValueError("trajectory.json must contain a non-empty frames list")

    intrinsics = as_camera_intrinsics(required(camera, "intrinsics", "camera.json"), "camera.json intrinsics")
    camera_to_ego = as_transform(required(camera, "camera_to_ego", "camera.json"), "camera.json camera_to_ego")
    width = int(required(camera, "width", "camera.json"))
    height = int(required(camera, "height", "camera.json"))
    near_plane = float(camera.get("near_plane", camera.get("near", 0.01)))
    far_plane = float(camera.get("far_plane", camera.get("far", 500.0)))

    requests = []
    for index, frame in enumerate(frames):
        ego_to_world = ego_pose_from_frame(frame, index, trajectory)
        requests.append(
            CameraRenderRequest(
                intrinsics=intrinsics,
                camera_to_world=ego_to_world @ camera_to_ego,
                width=width,
                height=height,
                timestamp=frame_timestamp(frame, index),
                dynamics=frame.get("dynamics", {}) if isinstance(frame, dict) else {},
                image_name=f"camera_{index:06d}",
                near_plane=near_plane,
                far_plane=far_plane,
            )
        )
    return requests


def as_rgb(image):
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    return image[..., :3]


def resize_to_height(image, height):
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))


def side_by_side_frame(original_frame, rendered_frame):
    original_frame = as_rgb(original_frame)
    rendered_frame = as_rgb(rendered_frame)
    target_height = min(original_frame.shape[0], rendered_frame.shape[0])
    original_frame = resize_to_height(original_frame, target_height)
    rendered_frame = resize_to_height(rendered_frame, target_height)
    return np.concatenate([original_frame, rendered_frame], axis=1)


def write_rendered_camera_video(renderer, render_requests, fps, output_video, desc, postprocess=None):
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_video,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    frame_count = 0
    try:
        for render_request in tqdm(render_requests, desc=desc):
            writer.append_data(renderer.render_request(render_request, postprocess=postprocess))
            frame_count += 1
    finally:
        writer.close()

    print(f"Saved {frame_count} rendered frames to {output_video}")


def compose_reconstruction_compare_video(
    scene_path,
    original_video,
    trajectory_path,
    camera_path,
    output_video,
    real_camera_intrinsics_path=None,
    real_camera_id=REAL_VEHICLE_FRONT_CAMERA_ID,
    real_camera_output=None,
    real_camera_timing_output=None,
):
    trajectory = load_json(trajectory_path)
    camera = load_json(camera_path)
    frames = trajectory["frames"]
    render_requests = build_camera_render_requests(trajectory, camera)
    fps = infer_fps(trajectory, camera, frames)
    if fps <= 0:
        raise ValueError("video fps must be greater than 0")

    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    renderer = GaussianSceneRenderer(scene_path)
    original_reader = imageio.get_reader(original_video)
    writer = imageio.get_writer(
        output_video,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    frame_count = 0
    try:
        paired_frames = zip(render_requests, original_reader)
        for render_request, original_frame in tqdm(
            paired_frames,
            total=len(render_requests),
            desc="Rendering and composing compare video",
        ):
            rendered_frame = renderer.render_request(render_request)
            writer.append_data(side_by_side_frame(original_frame, rendered_frame))
            frame_count += 1
    finally:
        writer.close()
        original_reader.close()

    if frame_count != len(render_requests):
        print(f"Warning: wrote {frame_count} frames for {len(render_requests)} trajectory frames")
    print(f"Saved comparison video to {output_video}")

    if real_camera_output is not None:
        real_camera = camera_with_real_vehicle_intrinsics(
            camera,
            real_camera_intrinsics_path or REAL_VEHICLE_FRONT_CAMERA_INTRINSICS,
            real_camera_id,
        )
        real_render_requests = build_camera_render_requests(trajectory, real_camera)
        real_fps = infer_fps(trajectory, real_camera, frames)
        renderer.reset_temporal_state()
        renderer.enable_timing(clear=True)
        write_rendered_camera_video(
            renderer,
            real_render_requests,
            real_fps,
            real_camera_output,
            f"Rendering real vehicle pinhole camera {real_camera_id}",
        )
        timing_output = (
            Path(real_camera_timing_output)
            if real_camera_timing_output is not None
            else Path(real_camera_output).with_suffix(".timing.csv")
        )
        renderer.save_timing_csv(timing_output)
        renderer.print_timing_summary(f"Real vehicle camera {real_camera_id} render timing")
        renderer.disable_timing()


def main():
    args = parse_args()
    compose_reconstruction_compare_video(
        Path(args.scene_path).resolve(),
        Path(args.original_video).resolve(),
        Path(args.trajectory_path).resolve(),
        Path(args.camera_path).resolve(),
        Path(args.output_video).resolve(),
        Path(args.real_camera_intrinsics).resolve(),
        args.real_camera_id,
        Path(args.real_camera_output).resolve() if args.real_camera_output else None,
        Path(args.real_camera_timing_output).resolve() if args.real_camera_timing_output else None,
    )


if __name__ == "__main__":
    main()
