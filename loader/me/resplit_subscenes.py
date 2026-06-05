#!/usr/bin/env python3
import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ME_CAMERAS = (
    "CAM_FRONT_120",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
ME_LIDAR = "AT128"

DEFAULT_DATA_ROOT = "/mnt/compute-data/e2e/me"
DEFAULT_OUTPUT_ROOT = "outputs/me/resplit"


@dataclass
class FrameRecord:
    source_sub_scene: str
    annotation_path: Path
    frame_key: str
    ego_position: np.ndarray
    image_paths: dict[str, Path]
    lidar_path: Path
    distance_from_prev: float = 0.0
    odometer: float = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Resplit ME dataset sub-scenes by ego pose distance and frame count.")
    parser.add_argument("-s", "--scene-id", required=True, help="ME scene id, for example 20250303_132635_0.")
    parser.add_argument("-d", "--data-root", default=DEFAULT_DATA_ROOT, help="ME dataset root.")
    parser.add_argument("-o", "--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output root for resplit scenes.")
    parser.add_argument("--max-frames", type=int, default=150, help="Maximum frames per new sub-scene.")
    parser.add_argument("--max-distance", type=int, default=200, help="Maximum accumulated distance in meters.")
    parser.add_argument("--overlap-frames", type=int, default=0, help="Frames carried over to the next sub-scene.")
    parser.add_argument("--min-distance", type=float, default=0.05, help="Minimum ego movement in meters between kept frames.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output scene directory before writing.")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def get_scene_path(data_root: Path, scene_id: str) -> Path:
    scene_path = data_root / scene_id
    if not scene_path.is_dir():
        raise FileNotFoundError(f"ME scene directory not found: {scene_path}")
    if not (scene_path / "meta.json").is_file():
        raise FileNotFoundError(f"ME scene meta.json not found: {scene_path / 'meta.json'}")
    return scene_path


def sorted_source_sub_scenes(scene_path: Path) -> list[Path]:
    return sorted(
        [path for path in scene_path.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def annotation_files(sub_scene_path: Path) -> list[Path]:
    annotation_dir = sub_scene_path / "annotations_info"
    if not annotation_dir.is_dir():
        return []
    return sorted(path for path in annotation_dir.glob("*.json") if path.is_file())


def pose_position(meta_by_frame: dict, frame_key: str) -> np.ndarray | None:
    frame_meta = meta_by_frame.get(frame_key)
    if not frame_meta:
        return None
    pose = frame_meta.get("pose") or []
    if not pose:
        return None
    matrix4 = pose[0].get("matrix4") or []
    if len(matrix4) != 16:
        return None
    return np.array([matrix4[3], matrix4[7], matrix4[11]], dtype=float)


def resolve_sensor_paths(annotation: dict, sub_scene_path: Path) -> tuple[dict[str, Path], Path | None]:
    image_paths = {}
    lidar_path = None
    sensors = annotation.get("meta", {}).get("sensor", [])
    for sensor in sensors:
        sensor_id = sensor.get("sensor_id")
        data_path = sensor.get("data_path") or ""
        filename = Path(data_path).name
        if not filename:
            continue
        if sensor_id in ME_CAMERAS:
            image_paths[sensor_id] = sub_scene_path / "images" / sensor_id / filename
        elif sensor_id == ME_LIDAR:
            lidar_path = sub_scene_path / "lidars" / ME_LIDAR / filename
    return image_paths, lidar_path


def validate_frame(
    sub_scene_path: Path,
    annotation_path: Path,
    meta_by_frame: dict,
    stats: dict[str, int],
) -> FrameRecord | None:
    stats["total_annotations"] += 1
    frame_key = annotation_path.stem
    ego_position = pose_position(meta_by_frame, frame_key)
    if ego_position is None:
        stats["missing_ego_pose"] += 1
        return None

    annotation = load_json(annotation_path)
    image_paths, lidar_path = resolve_sensor_paths(annotation, sub_scene_path)
    missing_cameras = [camera for camera in ME_CAMERAS if camera not in image_paths or not image_paths[camera].is_file()]
    if missing_cameras:
        stats["missing_images"] += 1
        return None
    if lidar_path is None or not lidar_path.is_file():
        stats["missing_lidar"] += 1
        return None

    stats["valid_frames"] += 1
    return FrameRecord(
        source_sub_scene=sub_scene_path.name,
        annotation_path=annotation_path,
        frame_key=frame_key,
        ego_position=ego_position,
        image_paths=image_paths,
        lidar_path=lidar_path,
    )


def collect_valid_frames(scene_path: Path, meta: dict) -> tuple[list[FrameRecord], dict[str, int]]:
    meta_by_frame = meta.get("meta", {})
    stats = {
        "total_annotations": 0,
        "missing_ego_pose": 0,
        "missing_images": 0,
        "missing_lidar": 0,
        "valid_frames": 0,
        "skipped_short_distance": 0,
    }
    frames = []
    for sub_scene_path in sorted_source_sub_scenes(scene_path):
        for annotation_path in annotation_files(sub_scene_path):
            frame = validate_frame(sub_scene_path, annotation_path, meta_by_frame, stats)
            if frame is not None:
                frames.append(frame)
    return frames, stats


def filter_by_distance(frames: list[FrameRecord], min_distance: float, stats: dict[str, int]) -> list[FrameRecord]:
    filtered = []
    prev_position = None
    odometer = 0.0
    for frame in frames:
        if prev_position is None:
            filtered.append(frame)
            prev_position = frame.ego_position
            frame.distance_from_prev = 0.0
            frame.odometer = odometer
            continue

        distance = float(np.linalg.norm(frame.ego_position - prev_position))
        if distance < min_distance:
            stats["skipped_short_distance"] += 1
            continue

        odometer += distance
        frame.distance_from_prev = distance
        frame.odometer = odometer
        filtered.append(frame)
        prev_position = frame.ego_position
    return filtered


def split_frames(
    frames: list[FrameRecord],
    max_frames: int,
    max_distance: int,
    overlap_frames: int,
) -> list[list[FrameRecord]]:
    if max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if max_distance <= 0:
        raise ValueError("--max-distance must be positive")
    if overlap_frames < 0:
        raise ValueError("--overlap-frames must be non-negative")
    if overlap_frames >= max_frames:
        raise ValueError("--overlap-frames must be smaller than --max-frames")

    sub_scenes = []
    current = []
    accumulated_distance = 0.0

    for frame in frames:
        distance = frame.distance_from_prev
        need_new_sub_scene = bool(current) and (
            len(current) >= max_frames or accumulated_distance + distance > max_distance
        )
        if need_new_sub_scene:
            sub_scenes.append(current)
            overlap = current[-overlap_frames:] if overlap_frames else []
            current = overlap.copy()
            accumulated_distance = current[-1].odometer - current[0].odometer if len(current) > 1 else 0.0

        current.append(frame)
        accumulated_distance += distance

    if current:
        sub_scenes.append(current)
    return sub_scenes


def plot_ego_trajectory(frames: list[FrameRecord], output_path: Path):
    if len(frames) < 2:
        return

    positions = np.array([frame.ego_position for frame in frames], dtype=float)
    odometer = np.array([frame.odometer for frame in frames], dtype=float)
    local_odometer = odometer - odometer[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Ego trajectory: {output_path.parent.name}", fontsize=13)

    axes[0].plot(positions[:, 0], positions[:, 1], "b-", linewidth=1.5)
    axes[0].scatter(positions[0, 0], positions[0, 1], c="green", s=45, label="Start", zorder=5)
    axes[0].scatter(positions[-1, 0], positions[-1, 1], c="red", s=45, marker="x", label="End", zorder=5)
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title("XY")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(local_odometer, positions[:, 2], "r-", linewidth=1.5)
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Z (m)")
    axes[1].set_title("Height")
    axes[1].grid(True, alpha=0.3)
    axes[1].text(
        0.02,
        0.98,
        f"Distance: {local_odometer[-1]:.2f}m, frames: {len(frames)}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def copy_frame_files(frames: list[FrameRecord], output_sub_scene_path: Path):
    annotation_dir = output_sub_scene_path / "annotations_info"
    lidar_dir = output_sub_scene_path / "lidars" / ME_LIDAR
    annotation_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)
    for camera in ME_CAMERAS:
        (output_sub_scene_path / "images" / camera).mkdir(parents=True, exist_ok=True)

    for frame in frames:
        shutil.copy2(frame.annotation_path, annotation_dir / frame.annotation_path.name)
        shutil.copy2(frame.lidar_path, lidar_dir / frame.lidar_path.name)
        for camera, image_path in frame.image_paths.items():
            shutil.copy2(image_path, output_sub_scene_path / "images" / camera / image_path.name)


def copy_resplit_scene(
    scene_path: Path,
    output_scene_path: Path,
    sub_scenes: list[list[FrameRecord]],
    overwrite: bool,
):
    if output_scene_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output scene exists, use --overwrite to replace it: {output_scene_path}")
        shutil.rmtree(output_scene_path)
    output_scene_path.mkdir(parents=True)
    shutil.copy2(scene_path / "meta.json", output_scene_path / "meta.json")

    for sub_scene_id, frames in enumerate(sub_scenes, start=1):
        output_sub_scene_path = output_scene_path / str(sub_scene_id)
        copy_frame_files(frames, output_sub_scene_path)
        plot_ego_trajectory(frames, output_sub_scene_path / "ego_trajectory.png")


def output_scene_name(scene_id: str, max_distance: int) -> str:
    return f"{scene_id}_{max_distance}m"


def print_summary(
    scene_id: str,
    scene_path: Path,
    output_scene_path: Path,
    stats: dict[str, int],
    frames: list[FrameRecord],
    sub_scenes: list[list[FrameRecord]],
    args,
):
    print(f"Scene: {scene_id}")
    print(f"Input: {scene_path}")
    print(f"Output: {output_scene_path}")
    print(
        "Parameters: "
        f"max_frames={args.max_frames}, max_distance={args.max_distance}m, "
        f"overlap_frames={args.overlap_frames}, min_distance={args.min_distance}m"
    )
    print("Validation:")
    print(f"  total annotations:       {stats['total_annotations']}")
    print(f"  missing ego pose:        {stats['missing_ego_pose']}")
    print(f"  missing images:          {stats['missing_images']}")
    print(f"  missing lidar:           {stats['missing_lidar']}")
    print(f"  valid before distance:   {stats['valid_frames']}")
    print(f"  skipped short distance:  {stats['skipped_short_distance']}")
    print(f"  valid after distance:    {len(frames)}")
    print("Resplit:")
    print(f"  new sub-scenes:          {len(sub_scenes)}")
    for index, sub_scene in enumerate(sub_scenes, start=1):
        start_frame = sub_scene[0]
        end_frame = sub_scene[-1]
        distance = sub_scene[-1].odometer - sub_scene[0].odometer
        print(
            f"  {index:03d}: frames={len(sub_scene):4d}, distance={distance:8.2f}m, "
            f"source={start_frame.source_sub_scene}->{end_frame.source_sub_scene}"
        )


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    scene_path = get_scene_path(data_root, args.scene_id)
    output_scene_path = output_root / output_scene_name(args.scene_id, args.max_distance)

    meta = load_json(scene_path / "meta.json")
    all_frames, stats = collect_valid_frames(scene_path, meta)
    frames = filter_by_distance(all_frames, args.min_distance, stats)
    if not frames:
        raise RuntimeError(f"No valid frames after filtering: {scene_path}")

    sub_scenes = split_frames(frames, args.max_frames, args.max_distance, args.overlap_frames)
    print_summary(args.scene_id, scene_path, output_scene_path, stats, frames, sub_scenes, args)
    copy_resplit_scene(scene_path, output_scene_path, sub_scenes, args.overwrite)
    print(f"Completed: {output_scene_path}")


if __name__ == "__main__":
    main()
