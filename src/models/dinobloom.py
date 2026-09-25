"""
DinoBloom hematology foundation model weights (Koch et al., MICCAI 2024).

Loads DINOv2 ViT architecture + DinoBloom checkpoints from Hugging Face or Zenodo.
https://github.com/marrlab/DinoBloom
"""

from __future__ import annotations

import urllib.request
import warnings
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from config.settings import REPO_ROOT

# backbone_name -> (torch.hub dinov2 fn, embed_dim, short tag)
DINOBLOOM_VARIANTS: dict[str, Tuple[str, int, str]] = {
    "dinobloom_s": ("dinov2_vits14", 384, "s"),
    "dinobloom_b": ("dinov2_vitb14", 768, "b"),
    "dinobloom_l": ("dinov2_vitl14", 1024, "l"),
    "dinobloom_g": ("dinov2_vitg14", 1536, "g"),
}

HF_REPO = "MarrLab/DinoBloom"
ZENODO_BASE = "https://zenodo.org/records/10908163/files"
ZENODO_FILES = {
    "s": "DinoBloom-S.pth",
    "b": "DinoBloom-B.pth",
    "l": "DinoBloom-L.pth",
    "g": "DinoBloom-G.pth",
}
HF_FILES = {k: f"pytorch_model_{k}.bin" for k in "sblg"}


def default_weights_dir() -> Path:
    return REPO_ROOT / "weights" / "dinobloom"


def _zenodo_path(short: str, root: Path) -> Path:
    return root / ZENODO_FILES[short]


def _hf_path(short: str, root: Path) -> Path:
    return root / HF_FILES[short]


def _download_zenodo(short: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ZENODO_BASE}/{ZENODO_FILES[short]}?download=1"
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _download_hf(short: str, dest: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Install huggingface_hub to download DinoBloom: pip install huggingface_hub"
        ) from e
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=HF_REPO, filename=HF_FILES[short])
    import shutil
    shutil.copy2(cached, dest)
    return dest


def ensure_dinobloom_weights(
    variant: str,
    weights_dir: Path | None = None,
    prefer_hf: bool = True,
) -> Path:
    """Return local checkpoint path, downloading if missing."""
    if variant not in DINOBLOOM_VARIANTS:
        raise ValueError(f"Unknown DinoBloom variant: {variant}")
    _, _, short = DINOBLOOM_VARIANTS[variant]
    root = weights_dir or default_weights_dir()

    hf = _hf_path(short, root)
    zen = _zenodo_path(short, root)
    if hf.exists():
        return hf
    if zen.exists():
        return zen

    if prefer_hf:
        try:
            return _download_hf(short, hf)
        except Exception as exc:
            print(f"HF download failed ({exc}); trying Zenodo...")
    return _download_zenodo(short, zen)


def _parse_checkpoint(ckpt_path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "teacher" in raw:
        return {
            k.replace("backbone.", ""): v
            for k, v in raw["teacher"].items()
            if "dino_head" not in k and "ibot_head" not in k
        }
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    if isinstance(raw, dict):
        # Hugging Face .bin: direct state dict
        return raw
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


def load_dinobloom_vit(
    variant: str,
    img_size: int = 224,
    weights_dir: Path | None = None,
    weights_path: Path | None = None,
) -> nn.Module:
    """Build DINOv2 ViT and load DinoBloom hematology weights."""
    if variant not in DINOBLOOM_VARIANTS:
        raise ValueError(f"Unknown DinoBloom variant: {variant}")
    hub_name, embed_dim, short = DINOBLOOM_VARIANTS[variant]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="xFormers is not available",
            category=UserWarning,
        )
        model = torch.hub.load(
            "facebookresearch/dinov2",
            hub_name,
            pretrained=False,
        )
    num_tokens = 1 + (img_size // 14) ** 2
    model.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

    ckpt = weights_path or ensure_dinobloom_weights(variant, weights_dir)
    state = _parse_checkpoint(Path(ckpt))
    model.load_state_dict(state, strict=True)
    return model


def is_dinobloom(name: str) -> bool:
    return name in DINOBLOOM_VARIANTS
