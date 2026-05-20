import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm


TRAJECTORY_NAME = "aeb_trajectory.json"
CAMERA_NAME = "aeb_camera.json"
ORIGINAL_VIDEO_NAME = "aeb_front_original.mp4"
TRAJECTORY_PLOT_NAME = "aeb_trajectory_plots.png"
FRONT_CAMERA_DIRS = ("cam_1", "CAM_FRONT_120", "CAM_FRONT", "front_camera")
HEIGHT_AXIS = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract minimal AEB front-render inputs from a trained HUGSIM scene."
    )
    parser.add_argument("scene_path", help="Exported HUGSIM scene directory.")
    parser.add_argument(
        "source_path",
        nargs="?",
        help="Optional original image root, images directory, or front camera image directory.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Optional directory for generated trajectory, camera, and original front video.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def save_json(path, data):
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def source_path_from_cfg(scene_path):
    cfg_path = scene_path / "cfg.yaml"
    if not cfg_path.exists():
        return None
    cfg = OmegaConf.load(cfg_path)
    source_path = cfg.get("source_path")
    if not source_path:
        return None
    return Path(source_path)


def camera_dir_from_rgb_path(rgb_path):
    parts = Path(rgb_path).parts
    if len(parts) >= 3 and parts[-3] == "images":
        return parts[-2]
    return None


def front_frames_from_meta(meta_data):
    groups = {}
    for frame in meta_data["frames"]:
        camera_dir = camera_dir_from_rgb_path(frame["rgb_path"])
        if camera_dir is not None:
            groups.setdefault(camera_dir, []).append(frame)

    if not groups:
        raise ValueError("meta_data.json does not contain image camera frames")

    camera_dir = next((name for name in FRONT_CAMERA_DIRS if name in groups), sorted(groups)[0])
    return camera_dir, groups[camera_dir]


def camera_to_world(frame):
    camtoworld = np.asarray(frame["camtoworld"], dtype=np.float32)
    if camtoworld.shape != (4, 4):
        raise ValueError("camtoworld must be a 4x4 matrix")
    return camtoworld


def quaternion_wxyz_from_rotation(rotation):
    trace = np.trace(rotation)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale

    quaternion = np.asarray([w, x, y, z], dtype=np.float32)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-6:
        raise ValueError("rotation matrix cannot be converted to a valid quaternion")
    return (quaternion / norm).tolist()


def infer_fps(frames):
    timestamps = [float(frame.get("timestamp", index)) for index, frame in enumerate(frames)]
    if len(timestamps) > 1:
        deltas = np.diff(timestamps)
        positive = deltas[deltas > 1e-6]
        if len(positive) > 0:
            return float(1.0 / np.median(positive))
    return 10.0


def cumulative_mileage(positions):
    if len(positions) == 0:
        return np.asarray([], dtype=np.float32)
    if len(positions) == 1:
        return np.asarray([0.0], dtype=np.float32)

    deltas = np.diff(np.asarray(positions, dtype=np.float32), axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    return np.concatenate([[0.0], np.cumsum(distances)]).astype(np.float32)


def make_trajectory(scene_path, frames):
    poses = [camera_to_world(frame) for frame in frames]
    positions = [pose[:3, 3] for pose in poses]
    mileages = cumulative_mileage(positions)

    output_frames = []
    for index, (frame, camtoworld) in enumerate(zip(frames, poses)):
        output_frame = {
            "index": index,
            "ego_position": camtoworld[:3, 3].tolist(),
            "ego_quaternion_wxyz": quaternion_wxyz_from_rotation(camtoworld[:3, :3]),
            "mileage": float(mileages[index]),
            "timestamp": frame.get("timestamp", index),
        }
        output_frames.append(output_frame)

    return {
        "type": "aeb_ego_trajectory",
        "schema_version": 3,
        "source_scene": str(scene_path),
        "frames": output_frames,
    }


def save_trajectory_plot(output_path, trajectory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = trajectory["frames"]
    positions = np.asarray([frame["ego_position"] for frame in frames], dtype=np.float32)
    mileages = np.asarray([frame["mileage"] for frame in frames], dtype=np.float32)
    ground_x = positions[:, 2]
    ground_y = -positions[:, 0]

    plot_path = output_path / TRAJECTORY_PLOT_NAME
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(mileages, positions[:, HEIGHT_AXIS], linewidth=1.8)
    axes[0].scatter(mileages[0], positions[0, HEIGHT_AXIS], s=24, label="start")
    axes[0].scatter(mileages[-1], positions[-1, HEIGHT_AXIS], s=24, label="end")
    axes[0].set_title("Mileage - Height")
    axes[0].set_xlabel("Mileage")
    axes[0].set_ylabel("Height (scene y)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ground_x, ground_y, linewidth=1.8)
    axes[1].scatter(ground_x[0], ground_y[0], s=24, label="start")
    axes[1].scatter(ground_x[-1], ground_y[-1], s=24, label="end")
    axes[1].set_title("Ground X-Y")
    axes[1].set_xlabel("Ground X (scene z)")
    axes[1].set_ylabel("Ground Y (-scene x)")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return plot_path


def make_camera(scene_path, camera_dir, first_frame, fps):
    return {
        "type": "aeb_front_camera",
        "schema_version": 1,
        "source_scene": str(scene_path),
        "source_camera": camera_dir,
        "intrinsics": first_frame["intrinsics"],
        "camera_to_ego": np.eye(4, dtype=np.float32).tolist(),
        "width": int(first_frame["width"]),
        "height": int(first_frame["height"]),
        "fps": fps,
    }


def relative_rgb_path(rgb_path):
    path = Path(rgb_path)
    if path.is_absolute():
        return path
    text = rgb_path[2:] if rgb_path.startswith("./") else rgb_path
    return Path(text)


def resolve_rgb_path(scene_path, source_path, rgb_path):
    path = relative_rgb_path(rgb_path)
    if path.is_absolute():
        if path.exists():
            return path
        if source_path is not None and "images" in path.parts:
            image_index = path.parts.index("images")
            image_path = Path(*path.parts[image_index:])
            candidates = [
                source_path / image_path,
                source_path / Path(*image_path.parts[1:]),
                source_path / path.name,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        return None

    candidates = [scene_path / path]
    if source_path is not None:
        candidates.append(source_path / path)
        parts = path.parts
        if len(parts) >= 3 and parts[0] == "images":
            candidates.append(source_path / Path(*parts[1:]))
            candidates.append(source_path / parts[-1])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_original_video(output_path, scene_path, source_path, frames, fps):
    image_paths = []
    for frame in frames:
        image_path = resolve_rgb_path(scene_path, source_path, frame["rgb_path"])
        if image_path is None:
            return None
        image_paths.append(image_path)

    video_path = output_path / ORIGINAL_VIDEO_NAME
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for image_path in tqdm(image_paths, desc="Writing original front video"):
            image = imageio.imread(image_path)
            writer.append_data(image[..., :3])
    finally:
        writer.close()
    return video_path


def extract_scene_inputs(scene_path, source_path=None, output_path=None):
    output_path = output_path or scene_path
    output_path.mkdir(parents=True, exist_ok=True)

    meta_path = scene_path / "meta_data.json"
    meta_data = load_json(meta_path)
    source_path = source_path or source_path_from_cfg(scene_path)
    camera_dir, frames = front_frames_from_meta(meta_data)
    fps = infer_fps(frames)

    trajectory_path = output_path / TRAJECTORY_NAME
    camera_path = output_path / CAMERA_NAME
    trajectory = make_trajectory(scene_path, frames)
    save_json(trajectory_path, trajectory)
    save_json(camera_path, make_camera(scene_path, camera_dir, frames[0], fps))
    trajectory_plot_path = save_trajectory_plot(output_path, trajectory)

    original_video_path = write_original_video(output_path, scene_path, source_path, frames, fps)

    print(f"Saved trajectory JSON to {trajectory_path}")
    print(f"Saved camera JSON to {camera_path}")
    print(f"Saved trajectory plot to {trajectory_plot_path}")
    if original_video_path is not None:
        print(f"Saved original front video to {original_video_path}")
    else:
        print("Original front images were not found; skipped original video.")


def main():
    args = parse_args()
    source_path = Path(args.source_path).resolve() if args.source_path else None
    output_path = Path(args.output_path).resolve() if args.output_path else None
    extract_scene_inputs(Path(args.scene_path).resolve(), source_path, output_path)


if __name__ == "__main__":
    main()
