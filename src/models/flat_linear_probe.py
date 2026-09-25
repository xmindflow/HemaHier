"""
Frozen DinoBloom / DINOv2 linear probe (DinoBloom dinov2/hub/classifiers.py).

Backbone features go directly into a single linear head, with no re-embedder MLP.
Default: 4 intermediate CLS tokens + mean patch from the deepest block.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from data.ontology import CANONICAL_CLASSES, MAPPING_MATRIX
from models.foundation_backbone import VIT_BACKBONES, build_foundation_backbone


class FlatLinearProbe(nn.Module):
    """Frozen ViT backbone + linear head on concatenated intermediate features."""

    def __init__(
        self,
        backbone: str = "dinobloom_s",
        pretrained: bool = True,
        linear_layers: int = 4,
        lora_rank: int = 0,
        lora_alpha: float = 128.0,
        num_fine_classes: Optional[int] = None,
        img_size: int = 224,
    ):
        super().__init__()
        if backbone not in VIT_BACKBONES:
            raise ValueError(
                f"Flat linear probe requires a ViT/DinoBloom backbone, got {backbone!r}"
            )
        if linear_layers not in (1, 4):
            raise ValueError(f"linear_layers must be 1 or 4, got {linear_layers}")

        self.flat_only = True
        self.hemahier_mode = False
        self.flat_matched = False
        self.linear_probe = True
        self.linear_layers = linear_layers

        self.encoder = build_foundation_backbone(
            backbone,
            pretrained=pretrained,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            img_size=img_size,
        )
        d = self.encoder.feat_dim
        self.probe_dim = (1 + linear_layers) * d
        self.K_f = num_fine_classes or len(CANONICAL_CLASSES)
        self.linear_head = nn.Linear(self.probe_dim, self.K_f)
        self.register_buffer("M", MAPPING_MATRIX.clone())

    def encode_probe(self, images: Tensor) -> Tensor:
        return self.encoder.forward_linear_probe_features(images, layers=self.linear_layers)

    def forward(
        self,
        images: Tensor,
        return_features: bool = True,
        coarse_labels=None,
        global_step=None,
    ) -> Dict[str, Tensor]:
        feats = images if images.ndim == 2 else self.encode_probe(images)
        logits_f = self.linear_head(feats)
        return {
            "logits_fine": logits_f,
            "logits_fine_raw": logits_f,
            "logits_coarse": None,
            "fine_feat": feats if return_features else None,
            "coarse_feat": None,
            "mat_pred": None,
            "lineage_mat_preds": {},
            "logits_fine_bayes": logits_f,
            "attn_weights": None,
            "pooled_feat": feats,
        }


__all__ = ["FlatLinearProbe"]
