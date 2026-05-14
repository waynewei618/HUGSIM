import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

OUTPUT_VIDEO_NAME = "aeb_front_rendered.mp4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render an AEB HIL/VIL front-view video from ego poses and camera calibration."
    )
    parser.add_argument("scene_path", help="Exported HUGSIM scene directory.")
    parser.add_argument("trajectory_path", help="Trajectory JSON with ego positions and rotations.")
    parser.add_argument("camera_path", help="Camera JSON with real intrinsics, camera_to_ego, and resolution.")
    return parser.parse_args()


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def required(data, name, source):
    if name not in data:
        raise ValueError(f"{source} must contain {name}")
    return data[name]


def as_camera_intrinsics(value, name):
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape == (3, 3):
        output = np.eye(4, dtype=np.float32)
        output[:3, :3] = matrix
        return output
    if matrix.shape == (4, 4):
        return matrix
    raise ValueError(f"{name} must be a real 3x3 or 4x4 camera intrinsic matrix")


def as_camera_extrinsic(value, name):
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a real 4x4 camera extrinsic matrix")
    return matrix


def ego_position_from_frame(frame, index):
    if not isinstance(frame, dict):
        raise ValueError(f"trajectory frame {index} must be an object with ego_position and ego rotation")

    value = frame.get("ego_position")
    if value is None:
        raise ValueError(f"trajectory frame {index} must contain ego_position")

    position = np.asarray(value, dtype=np.float32)
    if position.shape != (3,):
        raise ValueError(f"trajectory frame {index} ego_position must be a 3D position")
    return position


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


def angle_unit_from_frame(frame, trajectory):
    return frame.get("angle_unit", trajectory.get("angle_unit", trajectory.get("rpy_unit", "rad"))).lower()


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
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = ego_rotation_from_frame(frame, index, trajectory)
    pose[:3, 3] = ego_position_from_frame(frame, index)
    return pose


def frame_timestamp(frame, index):
    if isinstance(frame, dict) and "timestamp" in frame:
        return frame["timestamp"]
    return index


def frame_dynamics(frame):
    if isinstance(frame, dict):
        return frame.get("dynamics", {})
    return {}


def infer_fps(trajectory, camera, frames):
    if "fps" in camera:
        return float(camera["fps"])
    if "fps" in trajectory:
        return float(trajectory["fps"])

    timestamps = []
    for index, frame in enumerate(frames):
        if isinstance(frame, dict) and "timestamp" in frame:
            timestamps.append(float(frame["timestamp"]))
        else:
            timestamps.append(float(index))

    if len(timestamps) > 1:
        deltas = np.diff(timestamps)
        positive = deltas[deltas > 1e-6]
        if len(positive) > 0:
            return float(1.0 / np.median(positive))
    return 10.0


def build_render_frames(trajectory, camera):
    frames = trajectory.get("frames")
    if not frames:
        raise ValueError("trajectory.json must contain a non-empty frames list")

    intrinsics = as_camera_intrinsics(required(camera, "intrinsics", "camera.json"), "camera.json intrinsics")
    camera_to_ego = as_camera_extrinsic(required(camera, "camera_to_ego", "camera.json"), "camera.json camera_to_ego")
    width = int(required(camera, "width", "camera.json"))
    height = int(required(camera, "height", "camera.json"))

    render_frames = []
    for index, frame in enumerate(frames):
        ego_to_world = ego_pose_from_frame(frame, index, trajectory)
        camera_to_world = ego_to_world @ camera_to_ego
        render_frames.append(
            {
                "name": f"frame_{index:06d}",
                "camera_to_world": camera_to_world,
                "intrinsics": intrinsics,
                "width": width,
                "height": height,
                "timestamp": frame_timestamp(frame, index),
                "dynamics": frame_dynamics(frame),
            }
        )
    return render_frames


def load_models(scene_path):
    from gaussian_renderer import GaussianModel
    from scene.obj_model import ObjModel

    cfg = OmegaConf.load(scene_path / "cfg.yaml")
    gaussians = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)
    model_params, _ = torch.load(scene_path / "scene.pth")
    gaussians.restore(model_params, None)

    dynamic_gaussians = {}
    for ckpt_path in sorted(scene_path.glob("dynamic_*.pth")):
        track_id = ckpt_path.stem.split("_", 1)[1]
        dynamic_model = ObjModel(cfg.model.sh_degree, feat_mutable=False)
        dynamic_params, _ = torch.load(ckpt_path)
        dynamic_model.restore(dynamic_params, None)
        dynamic_gaussians[track_id] = dynamic_model

    bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    return gaussians, dynamic_gaussians, background


def make_camera(frame):
    from scene.cameras import Camera

    image = np.zeros((frame["height"], frame["width"], 3), dtype=np.float32)
    dynamics = {
        str(track_id): torch.tensor(transform, dtype=torch.float32, device="cuda")
        for track_id, transform in frame["dynamics"].items()
    }
    return Camera(
        width=frame["width"],
        height=frame["height"],
        image=image,
        K=frame["intrinsics"],
        c2w=frame["camera_to_world"],
        image_name=frame["name"],
        data_device="cpu",
        timestamp=frame["timestamp"],
        dynamics=dynamics,
    )


def render_to_uint8(render_tensor):
    image = render_tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (image * 255.0 + 0.5).astype(np.uint8)


def render_front_video(scene_path, trajectory_path, camera_path):
    from gaussian_renderer import render

    trajectory = load_json(trajectory_path)
    camera = load_json(camera_path)
    frames = build_render_frames(trajectory, camera)
    fps = infer_fps(trajectory, camera, trajectory["frames"])
    if fps <= 0:
        raise ValueError("video fps must be greater than 0")

    gaussians, dynamic_gaussians, background = load_models(scene_path)
    output_path = trajectory_path.parent / OUTPUT_VIDEO_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    previous_camera = None
    try:
        for frame in tqdm(frames, desc="Rendering front video"):
            viewpoint = make_camera(frame)
            previous_camera = previous_camera or viewpoint
            with torch.no_grad():
                render_pkg = render(
                    viewpoint,
                    previous_camera,
                    gaussians,
                    dynamic_gaussians,
                    None,
                    background,
                    False,
                )
            writer.append_data(render_to_uint8(render_pkg["render"]))
            previous_camera = viewpoint
    finally:
        writer.close()

    print(f"Saved rendered video to {output_path}")


def main():
    args = parse_args()
    render_front_video(
        Path(args.scene_path).resolve(),
        Path(args.trajectory_path).resolve(),
        Path(args.camera_path).resolve(),
    )


if __name__ == "__main__":
    main()
