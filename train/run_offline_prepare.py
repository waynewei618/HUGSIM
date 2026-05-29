import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_opts():
    parser = argparse.ArgumentParser(description="Export and convert a reconstructed HUGSIM scene.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--export_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--ver0", action="store_true", default=False)
    parser.add_argument("--skip_convert", action="store_true", default=False)
    return parser.parse_args()


def run_stage(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    args = get_opts()
    export_cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "eval_render" / "export_scene.py"),
        "--model_path",
        args.model_path,
        "--output_path",
        args.export_path,
        "--iteration",
        str(args.iteration),
    ]
    if args.ver0:
        export_cmd.append("--ver0")
    run_stage(export_cmd)

    if not args.skip_convert:
        run_stage(
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "eval_render" / "convert_scene.py"),
                "--model_path",
                args.export_path,
            ]
        )


if __name__ == "__main__":
    main()
