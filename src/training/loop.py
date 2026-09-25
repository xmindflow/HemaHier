"""One training epoch with mixed-precision and gradient clipping."""

from __future__ import annotations

import time

import torch

try:
    from torch.amp import autocast
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast
    _AMP_DEVICE = None


def _autocast_ctx():
    if _AMP_DEVICE is not None:
        return autocast(_AMP_DEVICE)
    return autocast()


def _clip_model_gradients(model, max_norm: float = 1.0) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def train_one_epoch(
    model,
    data_iter,
    n_steps,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    writer=None,
    global_step_start=0,
    log_every=50,
    scheduler=None,
):
    model.train()
    running = {
        "L_total": 0.0, "L_ce_fine": 0.0, "L_ce_coarse": 0.0,
        "L_hba": 0.0, "L_ham": 0.0, "L_supcon": 0.0,
        "L_ordinal": 0.0, "L_rank": 0.0, "beta": 0.0,
        "lambda_c": 0.0, "lambda_o": 0.0,
    }
    n = 0
    t0 = time.time()
    for batch_idx in range(n_steps):
        images, fine, coarse, mat_pos, mat_mask, _ = next(data_iter)
        images = images.to(device, non_blocking=True)
        fine = fine.long().to(device, non_blocking=True)
        coarse = coarse.long().to(device, non_blocking=True)
        mat_pos = mat_pos.float().to(device, non_blocking=True)
        mat_mask = mat_mask.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        step = global_step_start + batch_idx
        with _autocast_ctx():
            if getattr(model, "hemahier_mode", False):
                out = model(
                    images,
                    return_features=True,
                    coarse_labels=coarse,
                    fine_labels=fine,
                    global_step=step,
                )
            else:
                out = model(images, return_features=True)
            loss, comps = criterion(
                out, fine, coarse, mat_pos, mat_mask, epoch, global_step=step,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        _clip_model_gradients(model, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        for k in running:
            running[k] += comps.get(k, 0.0)
        for k in ("gamma_joint", "L_ce_joint", "L_ce_raw", "lambda_raw_fine_aux"):
            if k in comps:
                running.setdefault(k, 0.0)
                running[k] += comps[k]
        n += 1
    elapsed = time.time() - t0
    if n:
        for k in running:
            running[k] /= n
    return running, elapsed
