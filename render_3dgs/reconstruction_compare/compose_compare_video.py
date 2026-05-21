import argparse
import csv
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from render_3dgs.core.camera_math import as_camera_intrinsics, as_positive_int, as_transform
from render_3dgs.gaussian_scene_renderer import GaussianSceneRenderer
from render_3dgs.core.lut_distortion import LookupTableDistorter


REAL_VEHICLE_FRONT_CAMERA_ID = "front_120/cam1"
RENDER_3DGS_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REAL_VEHICLE_FRONT_CAMERA_INTRINSICS = RENDER_3DGS_DATA_DIR / "camera_intrinsics.json"
DEFAULT_FRONT_120_DISTORTION_PARAMETERS = RENDER_3DGS_DATA_DIR / "vtd_front_120" / "front_120_parameters.json"


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
    parser.add_argument(
        "--real-camera-distortion-parameters",
        help=(
            "Optional distortion parameter JSON or direct VTD lookup-table path. "
            "When omitted, front_120/cam1 uses render_3dgs/data/vtd_front_120/front_120_parameters.json."
        ),
    )
    parser.add_argument(
        "--disable-real-camera-distortion",
        action="store_true",
        help="Render the real vehicle camera as the raw pinhole image without lookup-table distortion.",
    )
    parser.add_argument(
        "--render-tile-rows",
        "--tile-rows",
        dest="render_tile_rows",
        type=int,
        default=1,
        help="Split each render_camera image into this many row tiles. Default: 1.",
    )
    parser.add_argument(
        "--render-tile-cols",
        "--tile-cols",
        dest="render_tile_cols",
        type=int,
        default=1,
        help="Split each render_camera image into this many column tiles. Default: 1.",
    )
    parser.add_argument(
        "--insert-static-vehicle-id",
        help="Optional 3DRealCar vehicle id to insert as a static non-native vehicle.",
    )
    parser.add_argument(
        "--insert-static-vehicle-s",
        type=float,
        help="Mileage s on the ego trajectory where the static vehicle is inserted. Default: trajectory max mileage.",
    )
    parser.add_argument(
        "--realcar-path",
        help="3DRealCar root directory. Default: $PATH_3DRealCar or /data/realcar3d.",
    )
    parser.add_argument(
        "--insert-static-vehicle-height",
        type=float,
        default=-0.3,
        help="Vehicle height offset relative to the estimated ground. Default: -0.3.",
    )
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def lookup_table_path_from_distortion_parameters(path):
    path = Path(path)
    if path.suffix.lower() == ".dat":
        return path

    parameters = load_json(path)
    local_files = parameters.get("local_files", {})
    lookup_table = local_files.get("lookup_table")
    if lookup_table is None:
        raise ValueError(f"{path} must contain local_files.lookup_table or be a .dat lookup table")
    return path.parent / lookup_table


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


def render_planes_from_camera(camera):
    return (
        float(camera.get("near_plane", camera.get("near", 0.01))),
        float(camera.get("far_plane", camera.get("far", 500.0))),
    )


def build_camera_inputs(trajectory, camera):
    frames = trajectory.get("frames")
    if not frames:
        raise ValueError("trajectory.json must contain a non-empty frames list")

    intrinsics = as_camera_intrinsics(required(camera, "intrinsics", "camera.json"), "camera.json intrinsics")
    camera_to_ego = as_transform(required(camera, "camera_to_ego", "camera.json"), "camera.json camera_to_ego")
    width = int(required(camera, "width", "camera.json"))
    height = int(required(camera, "height", "camera.json"))

    camera_inputs = []
    for index, frame in enumerate(frames):
        ego_to_world = ego_pose_from_frame(frame, index, trajectory)
        camera_to_world = ego_to_world @ camera_to_ego
        camera_inputs.append(
            {
                "intrinsics": intrinsics,
                "camera_to_world": camera_to_world.astype(np.float32),
                "width": width,
                "height": height,
                "timestamp": frame_timestamp(frame, index),
                "dynamics": frame.get("dynamics", {}) if isinstance(frame, dict) else {},
                "image_name": f"camera_{index:06d}",
            }
        )
    return camera_inputs


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


def save_render_timing_csv(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "image_name",
                "width",
                "height",
                "timestamp",
                "render_camera_ms",
                "postprocess_ms",
                "total_ms",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **record,
                    "render_camera_ms": f"{record['render_camera_ms']:.6f}",
                    "postprocess_ms": f"{record['postprocess_ms']:.6f}",
                    "total_ms": f"{record['total_ms']:.6f}",
                }
            )
    print(f"Saved render timing to {output_path}")


def print_render_timing_summary(records, title="Render timing"):
    if not records:
        print(f"{title}: no timing records")
        return

    total = np.asarray([record["total_ms"] for record in records], dtype=np.float64)
    render_camera = np.asarray([record["render_camera_ms"] for record in records], dtype=np.float64)
    print(
        f"{title}: count={len(records)} "
        f"total_ms(avg/min/max)={total.mean():.3f}/{total.min():.3f}/{total.max():.3f} "
        f"render_camera_ms(avg/min/max)={render_camera.mean():.3f}/"
        f"{render_camera.min():.3f}/{render_camera.max():.3f}"
    )


def write_rendered_camera_video(
    renderer,
    camera_inputs,
    fps,
    output_video,
    desc,
    postprocess=None,
    timing_records=None,
    near_plane=0.01,
    far_plane=500,
    render_tile_rows=1,
    render_tile_cols=1,
):
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
        for camera_input in tqdm(camera_inputs, desc=desc):
            render_start = time.perf_counter()
            image = renderer.render_camera(
                **camera_input,
                near_plane=near_plane,
                far_plane=far_plane,
                tile_rows=render_tile_rows,
                tile_cols=render_tile_cols,
            )
            render_camera_ms = (time.perf_counter() - render_start) * 1000.0

            postprocess_start = time.perf_counter()
            if postprocess is not None:
                image = postprocess(image)
            postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

            if timing_records is not None:
                timing_records.append(
                    {
                        "frame_index": frame_count,
                        "image_name": camera_input["image_name"],
                        "width": camera_input["width"],
                        "height": camera_input["height"],
                        "timestamp": camera_input["timestamp"],
                        "render_camera_ms": render_camera_ms,
                        "postprocess_ms": postprocess_ms,
                        "total_ms": render_camera_ms + postprocess_ms,
                    }
                )
            writer.append_data(image)
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
    render_tile_rows=1,
    render_tile_cols=1,
    real_camera_distortion=True,
    real_camera_distortion_parameters=None,
    insert_static_vehicle_id=None,
    insert_static_vehicle_s=None,
    realcar_path=None,
    insert_static_vehicle_height=-0.3,
):
    render_tile_rows = as_positive_int(render_tile_rows, "render_tile_rows")
    render_tile_cols = as_positive_int(render_tile_cols, "render_tile_cols")
    trajectory = load_json(trajectory_path)
    camera = load_json(camera_path)
    frames = trajectory["frames"]
    camera_inputs = build_camera_inputs(trajectory, camera)
    near_plane, far_plane = render_planes_from_camera(camera)
    fps = infer_fps(trajectory, camera, frames)
    if fps <= 0:
        raise ValueError("video fps must be greater than 0")

    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    renderer = GaussianSceneRenderer(
        scene_path,
        near_plane=near_plane,
        far_plane=far_plane,
        ego_trajectory=trajectory_path,
        insert_vehicle_id=insert_static_vehicle_id,
        insert_vehicle_s=insert_static_vehicle_s,
        realcar_path=realcar_path,
        insert_vehicle_height=insert_static_vehicle_height,
    )
    if insert_static_vehicle_id is not None:
        logged_s = insert_static_vehicle_s
        if logged_s is None:
            logged_s = frames[-1]["mileage"]
        print(
            "Inserted static non-native vehicle "
            f"{insert_static_vehicle_id} at s={float(logged_s):.3f}"
        )
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
        paired_frames = zip(camera_inputs, original_reader)
        for camera_input, original_frame in tqdm(
            paired_frames,
            total=len(camera_inputs),
            desc="Rendering and composing compare video",
        ):
            rendered_frame = renderer.render_camera(
                **camera_input,
                tile_rows=render_tile_rows,
                tile_cols=render_tile_cols,
            )
            writer.append_data(side_by_side_frame(original_frame, rendered_frame))
            frame_count += 1
    finally:
        writer.close()
        original_reader.close()

    if frame_count != len(camera_inputs):
        print(f"Warning: wrote {frame_count} frames for {len(camera_inputs)} trajectory frames")
    print(f"Saved comparison video to {output_video}")

    if real_camera_output is not None:
        real_camera = camera_with_real_vehicle_intrinsics(
            camera,
            real_camera_intrinsics_path or REAL_VEHICLE_FRONT_CAMERA_INTRINSICS,
            real_camera_id,
        )
        real_camera_inputs = build_camera_inputs(trajectory, real_camera)
        real_near_plane, real_far_plane = render_planes_from_camera(real_camera)
        real_fps = infer_fps(trajectory, real_camera, frames)
        renderer.reset_temporal_state()
        timing_records = []
        postprocess = None
        real_camera_desc = f"Rendering real vehicle pinhole camera {real_camera_id}"
        if real_camera_distortion:
            distortion_parameters = real_camera_distortion_parameters
            if distortion_parameters is None and real_camera_id == REAL_VEHICLE_FRONT_CAMERA_ID:
                distortion_parameters = DEFAULT_FRONT_120_DISTORTION_PARAMETERS
            if distortion_parameters is not None:
                lookup_table_path = lookup_table_path_from_distortion_parameters(distortion_parameters)
                print(f"Loading VTD lookup-table distortion from {lookup_table_path}")
                postprocess = LookupTableDistorter(lookup_table_path)
                real_camera_desc = f"Rendering real vehicle VTD-distorted camera {real_camera_id}"

        write_rendered_camera_video(
            renderer,
            real_camera_inputs,
            real_fps,
            real_camera_output,
            real_camera_desc,
            postprocess=postprocess,
            timing_records=timing_records,
            near_plane=real_near_plane,
            far_plane=real_far_plane,
            render_tile_rows=render_tile_rows,
            render_tile_cols=render_tile_cols,
        )
        timing_output = (
            Path(real_camera_timing_output)
            if real_camera_timing_output is not None
            else Path(real_camera_output).with_suffix(".timing.csv")
        )
        save_render_timing_csv(timing_records, timing_output)
        print_render_timing_summary(timing_records, f"Real vehicle camera {real_camera_id} render timing")


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
        args.render_tile_rows,
        args.render_tile_cols,
        not args.disable_real_camera_distortion,
        Path(args.real_camera_distortion_parameters).resolve()
        if args.real_camera_distortion_parameters is not None
        else None,
        args.insert_static_vehicle_id,
        args.insert_static_vehicle_s,
        Path(args.realcar_path).resolve() if args.realcar_path else None,
        args.insert_static_vehicle_height,
    )


if __name__ == "__main__":
    main()
