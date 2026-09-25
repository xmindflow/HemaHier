"""LoRA on frozen DINO attention layers (Hu et al., 2021)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 64, alpha: float = 128.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * self.lora_b(self.lora_a(x))


def _inject_lora_into_dino(model: nn.Module, rank: int, alpha: float) -> None:
    for block in model.blocks:
        attn = block.attn
        attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha)
        attn.proj = LoRALinear(attn.proj, rank=rank, alpha=alpha)
