import argparse
import os
import subprocess
import sys

try:
    from .common import PROJECT_ROOT, camera_names_from_images, ensure_camera_dirs
except ImportError:
    from common import PROJECT_ROOT, camera_names_from_images, ensure_camera_dirs


INVERSEFORM_DIR = PROJECT_ROOT / "pre_train" / "InverseForm"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "hrnet48_OCR_HMS_IF_checkpoint.pth"


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--out", dest="input", type=str, required=True)
    parser.add_argument("--cuda", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("INVERSEFORM_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
    )
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("INVERSEFORM_BATCH_SIZE", "2")))
    parser.add_argument("--master_port", type=int, default=int(os.environ.get("MASTER_PORT", "29500")))
    parser.add_argument("--cameras", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = get_opts()
    out_dir = os.path.abspath(args.input)
    model_path = os.path.abspath(args.model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"InverseForm model checkpoint not found: {model_path}")

    cameras = args.cameras or camera_names_from_images(out_dir)
    ensure_camera_dirs(out_dir, "semantics")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda

    for idx, camera in enumerate(cameras):
        input_dir = os.path.join(out_dir, "images", camera)
        output_dir = os.path.join(out_dir, "semantics", camera)
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(input_dir)
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--master_port",
            str(args.master_port + idx),
            "validation.py",
            "--input_dir",
            input_dir,
            "--output_dir",
            output_dir,
            "--model_path",
            model_path,
            "--arch",
            "ocrnet.HRNet_Mscale",
            "--hrnet_base",
            "48",
            "--has_edge",
            "True",
            "--batch_size",
            str(args.batch_size),
        ]
        print(f"[semantics] {camera}")
        subprocess.run(cmd, cwd=str(INVERSEFORM_DIR), env=env, check=True)


if __name__ == "__main__":
    main()
