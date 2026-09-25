"""Learning-rate schedule and dataloader helpers."""

from __future__ import annotations

import math
from typing import Iterator


def cosine_lr(epoch: int, total: int, warmup: int) -> float:
    if epoch < warmup:
        return epoch / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def finalshot_decay_lr_multiplier(
    step: int,
    max_iters: int,
    decay_start: int = 500,
) -> float:
    """Peak LR at training start, cosine decay after ``decay_start`` steps.

  Decouples LR from auxiliary-loss ramps: classifier fits at full LR while
  lineage ramps, then LR decays as maturity turns on.
    """
    if max_iters <= 0:
        return 1.0
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / max(max_iters - decay_start, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def infinite_loader(loader) -> Iterator:
    """Yield batches forever, restarting the loader each pass."""
    while True:
        for batch in loader:
            yield batch
