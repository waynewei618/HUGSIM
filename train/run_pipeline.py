import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_opts():
    parser = argparse.ArgumentParser(
        description="Run HUGSIM reconstruction and offline simulation preparation."
    )
    parser.add_argument("--base_cfg", type=str, default="./configs/gs_base.yaml")
    parser.add_argument("--train_cfg", type=str, default="./configs/train.yaml")
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--export_path", type=str, required=True)
    parser.add_argument("--cuda", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--resume_iteration", type=int, default=0)
    parser.add_argument("--ver0", action="store_true", default=False)
    parser.add_argument("--skip_reconstruction", action="store_true", default=False)
    parser.add_argument("--skip_offline_prepare", action="store_true", default=False)
    parser.add_argument("--skip_ground", action="store_true", default=False)
    parser.add_argument("--skip_scene", action="store_true", default=False)
    parser.add_argument("--skip_convert", action="store_true", default=False)
    return parser.parse_args()


def run_stage(cmd, env=None):
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def main():
    args = get_opts()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda

    if not args.skip_reconstruction:
        recon_cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "train" / "run_reconstruction.py"),
            "--base_cfg",
            args.base_cfg,
            "--train_cfg",
            args.train_cfg,
            "--source_path",
            args.source_path,
            "--model_path",
            args.model_path,
            "--cuda",
            args.cuda,
        ]
        if args.resume_iteration > 0:
            recon_cmd.extend(["--resume_iteration", str(args.resume_iteration)])
        if args.skip_ground:
            recon_cmd.append("--skip_ground")
        if args.skip_scene:
            recon_cmd.append("--skip_scene")
        run_stage(recon_cmd, env=env)

    if not args.skip_offline_prepare:
        offline_cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "train" / "run_offline_prepare.py"),
            "--model_path",
            args.model_path,
            "--export_path",
            args.export_path,
            "--iteration",
            str(args.iteration),
        ]
        if args.ver0:
            offline_cmd.append("--ver0")
        if args.skip_convert:
            offline_cmd.append("--skip_convert")
        run_stage(offline_cmd, env=env)


if __name__ == "__main__":
    main()
