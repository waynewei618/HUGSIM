#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


AXES = "XYZ"


def dominant_axis(vec):
    idx = int(np.argmax(np.abs(vec)))
    sign = "+" if vec[idx] >= 0 else "-"
    return f"{AXES[idx]}{sign}"


def load_export(export_dir):
    export_dir = Path(export_dir)
    with open(export_dir / "meta_data.json") as f:
        meta_data = json.load(f)
    with open(export_dir / "camera_paras.json") as f:
        camera_paras = json.load(f)
    return export_dir, meta_data, camera_paras


def front_ego_poses(meta_data):
    poses = [
        np.asarray(frame["ego_to_world"], dtype=np.float64)
        for frame in meta_data.get("frames", [])
        if frame.get("camera_name") == "CAM_FRONT"
    ]
    if not poses:
        poses = [
            np.asarray(frame["ego_to_world"], dtype=np.float64)
            for frame in meta_data.get("frames", [])
        ]
    return poses


def check_export(export_dir, axis_threshold, require_forward_motion, min_motion):
    export_dir, meta_data, camera_paras = load_export(export_dir)
    errors = []

    cameras = camera_paras.get("cameras", {})
    front_camera = cameras.get("CAM_FRONT")
    if front_camera is None:
        errors.append("camera_paras.json missing CAM_FRONT")
        front_z = None
    else:
        rotation = np.asarray(front_camera["rotation_matrix"], dtype=np.float64)
        front_z = rotation[:, 2]
        if front_z[0] < axis_threshold or abs(front_z[0]) < max(abs(front_z[1]), abs(front_z[2])):
            errors.append(
                "CAM_FRONT camera +Z does not point to ego +X: "
                f"z_axis={np.round(front_z, 6).tolist()}, dominant={dominant_axis(front_z)}"
            )

    poses = front_ego_poses(meta_data)
    if len(poses) < 2:
        displacement = None
    else:
        displacement = poses[-1][:3, 3] - poses[0][:3, 3]
        horizontal_motion = float(np.linalg.norm(displacement[:2]))
        if require_forward_motion and horizontal_motion >= min_motion:
            if displacement[0] <= 0 or abs(displacement[0]) < abs(displacement[1]):
                errors.append(
                    "CAM_FRONT trajectory is not primarily ego +X: "
                    f"displacement={np.round(displacement, 6).tolist()}, "
                    f"dominant={dominant_axis(displacement)}"
                )

    return {
        "export_dir": export_dir,
        "errors": errors,
        "front_z": front_z,
        "displacement": displacement,
    }


def main():
    parser = argparse.ArgumentParser(description="Check HUGSIM export coordinate contract.")
    parser.add_argument("export_dirs", nargs="+", help="Export directories containing meta_data.json and camera_paras.json.")
    parser.add_argument("--axis-threshold", type=float, default=0.8)
    parser.add_argument("--require-forward-motion", action="store_true")
    parser.add_argument("--min-motion", type=float, default=5.0)
    args = parser.parse_args()

    failed = False
    for export_dir in args.export_dirs:
        result = check_export(
            export_dir,
            axis_threshold=args.axis_threshold,
            require_forward_motion=args.require_forward_motion,
            min_motion=args.min_motion,
        )
        front_z = result["front_z"]
        displacement = result["displacement"]
        print(result["export_dir"])
        if front_z is not None:
            print(f"  CAM_FRONT +Z: {np.round(front_z, 6).tolist()} dominant={dominant_axis(front_z)}")
        if displacement is not None:
            print(f"  trajectory: {np.round(displacement, 6).tolist()} dominant={dominant_axis(displacement)}")

        if result["errors"]:
            failed = True
            for error in result["errors"]:
                print(f"  ERROR: {error}")
        else:
            print("  OK")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
