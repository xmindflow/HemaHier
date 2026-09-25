"""Small building blocks shared by the HemaHier cascade."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ReEmbedder(nn.Module):
    """Two-layer MLP projector with LN + GELU."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(out_dim, in_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
