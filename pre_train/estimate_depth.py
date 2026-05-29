import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from .common import PROJECT_ROOT, abs_data_path, derived_abs_path, ensure_camera_dirs, frame_camera_params, load_metadata
except ImportError:
    from common import PROJECT_ROOT, abs_data_path, derived_abs_path, ensure_camera_dirs, frame_camera_params, load_metadata


sys.path.insert(0, str(PROJECT_ROOT))
try:
    from .model_cache import checkpoint_path, configure_model_cache  # noqa: E402
except ImportError:
    from model_cache import checkpoint_path, configure_model_cache  # noqa: E402


configure_model_cache()

from unidepth.models import UniDepthV2  # noqa: E402


if os.environ.get("HUGSIM_DISABLE_XFORMERS") == "1":
    try:
        from unidepth.models.backbones.metadinov2 import attention as metadinov2_attention

        metadinov2_attention.XFORMERS_AVAILABLE = False
    except Exception:
        pass


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--out", dest="input", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("UNIDEPTH_MODEL_PATH", checkpoint_path("unidepth-v2-vitl14")),
    )
    return parser.parse_args()


def main():
    args = get_opts()
    out_dir = os.path.abspath(args.input)
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"UniDepth checkpoint directory not found: {args.model_path}")

    ensure_camera_dirs(out_dir, "depth")
    meta_data, camera_paras = load_metadata(out_dir)

    print("loading depth model...")
    model = UniDepthV2.from_pretrained(args.model_path, local_files_only=True)
    model = model.to("cuda")
    model.eval()
    print("Depth model loaded")

    for frame in tqdm(meta_data["frames"]):
        image_path = abs_data_path(out_dir, frame["rgb_path"])
        intrinsic = frame_camera_params(camera_paras, frame)["intrinsic"]
        k_tensor = torch.from_numpy(np.asarray(intrinsic[:3, :3])).float().cuda()
        image = torch.from_numpy(np.array(Image.open(image_path).convert("RGB"))).permute(2, 0, 1)
        prediction = model.infer(image, k_tensor)
        depth = prediction["depth"][0][0].detach().cpu()

        depth_path = derived_abs_path(out_dir, frame, "depth", ".pt")
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        torch.save(depth, depth_path)


if __name__ == "__main__":
    main()
