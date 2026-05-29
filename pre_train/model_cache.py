import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
TORCH_HOME = CHECKPOINTS_DIR / "torch"
TORCH_HUB_DIR = TORCH_HOME / "hub"
HF_HOME = CHECKPOINTS_DIR / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"


def configure_model_cache():
    """Keep third-party model downloads inside the project checkpoints folder."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
    HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_CACHE))

    try:
        import torch

        torch.hub.set_dir(str(TORCH_HUB_DIR))
    except Exception:
        pass


def checkpoint_path(*parts):
    return str(CHECKPOINTS_DIR.joinpath(*parts))
