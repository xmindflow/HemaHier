"""Global RNG seeding for reproducible training and split generation."""

from __future__ import annotations

import os
import random
from typing import Callable, Optional

import numpy as np
import torch


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = True,
) -> None:
    """Seed Python, NumPy, PyTorch CPU/CUDA and configure cuDNN."""
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Per-worker seeding for DataLoader subprocesses."""

    def _init(worker_id: int) -> None:
        worker_seed = int(base_seed) + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _init


def dataloader_generators(
    seed: int,
) -> tuple[torch.Generator, torch.Generator, Callable[[int], None]]:
    """Sampler generator, shuffle generator, and worker_init_fn for a given seed."""
    seed = int(seed)
    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(seed)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(seed + 10_000)
    return sampler_gen, loader_gen, worker_init_fn(seed)
