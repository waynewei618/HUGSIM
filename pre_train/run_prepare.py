import argparse
import os
import subprocess
import sys
from pathlib import Path


PRE_TRAIN_ROOT = Path(__file__).resolve().parent


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--out", dest="input", type=str, required=True)
    parser.add_argument("--cuda", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--total", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--semantic_model_path", type=str, default=None)
    parser.add_argument("--depth_model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("INVERSEFORM_BATCH_SIZE", "2")))
    parser.add_argument("--master_port", type=int, default=int(os.environ.get("MASTER_PORT", "29500")))
    parser.add_argument("--skip_semantics", action="store_true", default=False)
    parser.add_argument("--skip_mask", action="store_true", default=False)
    parser.add_argument("--skip_depth", action="store_true", default=False)
    parser.add_argument("--skip_merge", action="store_true", default=False)
    return parser.parse_args()


def run_stage(cmd, env):
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def main():
    args = get_opts()
    out_dir = os.path.abspath(args.input)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda
    env.setdefault("HUGSIM_DISABLE_XFORMERS", "1")

    if not args.skip_semantics:
        cmd = [
            sys.executable,
            str(PRE_TRAIN_ROOT / "infer_semantics.py"),
            "--input",
            out_dir,
            "--cuda",
            args.cuda,
            "--batch_size",
            str(args.batch_size),
            "--master_port",
            str(args.master_port),
        ]
        if args.semantic_model_path:
            cmd.extend(["--model_path", args.semantic_model_path])
        run_stage(cmd, env)

    if not args.skip_mask:
        run_stage(
            [
                sys.executable,
                str(PRE_TRAIN_ROOT / "create_dynamic_mask.py"),
                "--input",
                out_dir,
            ],
            env,
        )

    if not args.skip_depth:
        cmd = [
            sys.executable,
            str(PRE_TRAIN_ROOT / "estimate_depth.py"),
            "--input",
            out_dir,
        ]
        if args.depth_model_path:
            cmd.extend(["--model_path", args.depth_model_path])
        run_stage(cmd, env)

    if not args.skip_merge:
        run_stage(
            [
                sys.executable,
                str(PRE_TRAIN_ROOT / "merge_depth_wo_ground.py"),
                "--input",
                out_dir,
                "--total",
                str(args.total),
                "--seed",
                str(args.seed),
            ],
            env,
        )
        run_stage(
            [
                sys.executable,
                str(PRE_TRAIN_ROOT / "merge_depth_ground.py"),
                "--input",
                out_dir,
                "--total",
                str(args.total),
                "--seed",
                str(args.seed),
            ],
            env,
        )


if __name__ == "__main__":
    main()
