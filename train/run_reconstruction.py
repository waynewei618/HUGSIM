import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_opts():
    parser = argparse.ArgumentParser(description="Run HUGSIM ground and scene reconstruction.")
    parser.add_argument("--base_cfg", type=str, default="./configs/gs_base.yaml")
    parser.add_argument("--train_cfg", type=str, default="./configs/train.yaml")
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cuda", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--resume_iteration", type=int, default=0)
    parser.add_argument("--skip_ground", action="store_true", default=False)
    parser.add_argument("--skip_scene", action="store_true", default=False)
    return parser.parse_args()


def run_stage(cmd, env):
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def main():
    args = get_opts()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda

    common_args = [
        "--base_cfg",
        args.base_cfg,
        "--train_cfg",
        args.train_cfg,
        "--source_path",
        args.source_path,
        "--model_path",
        args.model_path,
    ]

    if not args.skip_ground:
        run_stage(
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "train" / "train_ground.py"),
                *common_args,
            ],
            env,
        )

    if not args.skip_scene:
        scene_cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "train" / "train_scene.py"),
            *common_args,
        ]
        if args.resume_iteration > 0:
            scene_cmd.extend(["--resume_iteration", str(args.resume_iteration)])
        run_stage(scene_cmd, env)


if __name__ == "__main__":
    main()
