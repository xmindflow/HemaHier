"""
Frozen foundation-model backbones for HemaHier head-only training.

Backbones:
  - dinobloom_s / dinobloom_b / dinobloom_l: hematology foundation (used in the paper)
  - dinov2_vitb14 / dinov2_vitl14: generic DINOv2 fallback
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.lora import LoRALinear, _inject_lora_into_dino
from models.dinobloom import DINOBLOOM_VARIANTS, is_dinobloom, load_dinobloom_vit


_DINOV2_HUB = {
    "dinov2_vitb14": "dinov2_vitb14",
    "dinov2_vitl14": "dinov2_vitl14",
}

VIT_BACKBONES = frozenset({*_DINOV2_HUB.keys(), *DINOBLOOM_VARIANTS.keys()})

# Earlier ViT block for morphology-preserving features (0-indexed).
INTERMEDIATE_BLOCK_INDEX = -4


def _apply_lora_and_freeze(model: nn.Module, lora_rank: int, lora_alpha: float) -> None:
    _inject_lora_into_dino(model, rank=lora_rank, alpha=lora_alpha)
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_a.parameters():
                p.requires_grad = True
            for p in m.lora_b.parameters():
                p.requires_grad = True


class DINOv2CLSBackbone(nn.Module):
    """DINOv2 / DinoBloom ViT with CLS-token pooling and optional multiscale."""

    def __init__(
        self,
        variant: str = "dinobloom_s",
        pretrained: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 128.0,
        img_size: int = 224,
    ):
        super().__init__()
        self.variant = variant
        if is_dinobloom(variant):
            if not pretrained:
                raise ValueError("DinoBloom backbones require pretrained weights.")
            self.model = load_dinobloom_vit(variant, img_size=img_size)
        elif variant in _DINOV2_HUB:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                _DINOV2_HUB[variant],
                pretrained=pretrained,
            )
        else:
            raise ValueError(f"Unknown foundation backbone: {variant}")

        self.feat_dim = self.model.embed_dim
        self.use_lora = lora_rank > 0
        if self.use_lora:
            _apply_lora_and_freeze(self.model, lora_rank, lora_alpha)
        elif pretrained:
            for p in self.model.parameters():
                p.requires_grad = False

    def _intermediate_block_index(self) -> int:
        n = len(self.model.blocks)
        idx = INTERMEDIATE_BLOCK_INDEX
        return idx if idx >= 0 else n + idx

    def forward_multiscale_tokens(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return late CLS h_L (B,D) and mid patch tokens (B,N,D) before pooling."""
        tokens = self.model.prepare_tokens_with_masks(x)
        mid_idx = self._intermediate_block_index()
        mid_patches: Optional[Tensor] = None
        for i, blk in enumerate(self.model.blocks):
            tokens = blk(tokens)
            if i == mid_idx:
                mid_patches = tokens[:, 1:, :]
        tokens = self.model.norm(tokens)
        h_l = tokens[:, 0]
        if mid_patches is None:
            mid_patches = h_l.unsqueeze(1)
        return h_l, mid_patches

    def forward_tokens(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return late CLS feature h_L and pooled intermediate patch feature h_M."""
        h_l, mid_patches = self.forward_multiscale_tokens(x)
        h_m = mid_patches.mean(dim=1) if mid_patches.ndim == 3 else mid_patches
        return h_l, h_m

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_tokens(x)[0]

    def forward_linear_probe_features(self, x: Tensor, layers: int = 4) -> Tensor:
        """
        DINOv2 / DinoBloom linear-classifier features (see DinoBloom dinov2/hub/classifiers.py).

        layers=1: [CLS, mean(patch)]  → 2 * D
        layers=4: CLS from 4 blocks + mean patch from deepest → 5 * D
        """
        if layers not in (1, 4):
            raise ValueError(f"linear probe layers must be 1 or 4, got {layers}")
        if layers == 1:
            feat = self.model.forward_features(x)
            cls_token = feat["x_norm_clstoken"]
            patch_tokens = feat["x_norm_patchtokens"]
            return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        layer_out = self.model.get_intermediate_layers(x, n=4, return_class_token=True)
        return torch.cat(
            [
                layer_out[0][1],
                layer_out[1][1],
                layer_out[2][1],
                layer_out[3][1],
                layer_out[3][0].mean(dim=1),
            ],
            dim=1,
        )

    def forward_multilayer_features(
        self,
        x: Tensor,
        layer_indices: Sequence[int] = (-1, -2, -4, -6),
    ) -> Tensor:
        """
        Concatenate CLS + mean-patch features from selected ViT blocks.

        Returns (B, n_layers * 2 * D).
        """
        tokens = self.model.prepare_tokens_with_masks(x)
        n_blocks = len(self.model.blocks)
        abs_indices = sorted({i if i >= 0 else n_blocks + i for i in layer_indices})
        layer_feats: dict[int, Tuple[Tensor, Tensor]] = {}

        for i, blk in enumerate(self.model.blocks):
            tokens = blk(tokens)
            if i in abs_indices:
                t = tokens
                if i == n_blocks - 1:
                    t = self.model.norm(t)
                cls = t[:, 0]
                patch = t[:, 1:, :].mean(dim=1) if t.shape[1] > 1 else cls
                layer_feats[i] = (cls, patch)

        if n_blocks - 1 not in layer_feats:
            t = self.model.norm(tokens)
            layer_feats[n_blocks - 1] = (t[:, 0], t[:, 1:, :].mean(dim=1))

        parts: List[Tensor] = []
        for idx in layer_indices:
            abs_i = idx if idx >= 0 else n_blocks + idx
            cls, patch = layer_feats[abs_i]
            parts.extend([cls, patch])
        return torch.cat(parts, dim=-1)


def build_foundation_backbone(
    name: str,
    pretrained: bool = True,
    lora_rank: int = 0,
    lora_alpha: float = 128.0,
    img_size: int = 224,
) -> nn.Module:
    if name in VIT_BACKBONES:
        return DINOv2CLSBackbone(
            variant=name,
            pretrained=pretrained,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            img_size=img_size,
        )
    raise ValueError(f"Unknown foundation backbone: {name}")
