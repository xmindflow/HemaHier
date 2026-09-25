"""
Fine-class decoding modes for hierarchical models.

Modes
-----
argmax      : argmax over fine logits (ignores coarse head at inference).
bayes       : argmax over P_fine(f) * P_coarse(parent(f))  (current H5 default).
strict      : c_hat = argmax P_coarse; f_hat = argmax_{f in children(c_hat)} P_fine(f).
conditional : same as strict but scores = log P_coarse(c_hat) + log P_fine(f).

Note on Bayes consistency
-------------------------
Product decoding (bayes) does *not* guarantee that the parent lineage of the
selected fine class equals argmax P_coarse.  Only ``strict`` and ``conditional``
enforce coarse/fine consistency by construction.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
from torch import Tensor

DecodeMode = Literal[
    "argmax", "bayes", "strict", "conditional", "tree_risk",
    "raw", "joint", "joint_maturity", "bio_posterior",
]


def decode_fine_predictions(
    logits_fine: Tensor,
    logits_coarse: Optional[Tensor],
    mapping_matrix: Tensor,
    mode: DecodeMode = "argmax",
    logits_fine_bayes: Optional[Tensor] = None,
    logits_fine_raw: Optional[Tensor] = None,
    logits_fine_joint: Optional[Tensor] = None,
    logits_fine_ordinal: Optional[Tensor] = None,
) -> Tensor:
    """Return (B,) fine-class indices for the requested decoding mode."""
    if mode == "raw":
        return (logits_fine_raw if logits_fine_raw is not None else logits_fine).argmax(dim=-1)

    if mode == "joint":
        return (logits_fine_joint if logits_fine_joint is not None else logits_fine).argmax(dim=-1)

    if mode == "joint_maturity":
        lt = logits_fine_ordinal if logits_fine_ordinal is not None else logits_fine_joint
        if lt is None:
            lt = logits_fine
        return lt.argmax(dim=-1)

    if mode == "bio_posterior":
        lt = logits_fine_joint if logits_fine_joint is not None else logits_fine
        return lt.argmax(dim=-1)

    if mode == "argmax":
        return logits_fine.argmax(dim=-1)

    if mode == "bayes":
        if logits_fine_bayes is not None:
            return logits_fine_bayes.argmax(dim=-1)
        return _bayes_decode(logits_fine, logits_coarse, mapping_matrix)

    if mode == "tree_risk":
        return _tree_risk_decode(logits_fine)

    if logits_coarse is None:
        return logits_fine.argmax(dim=-1)

    if mode == "strict":
        return _strict_decode(logits_fine, logits_coarse, mapping_matrix, use_coarse_log=False)
    if mode == "conditional":
        return _strict_decode(logits_fine, logits_coarse, mapping_matrix, use_coarse_log=True)

    raise ValueError(f"Unknown decode mode: {mode}")


def _tree_risk_decode(logits_fine: Tensor) -> Tensor:
    """Bayes decision rule under the fixed hematopoietic tree-distance cost."""
    from data.ontology import FINE_TREE_DISTANCE

    posterior = logits_fine.softmax(dim=-1)
    distance = torch.as_tensor(
        FINE_TREE_DISTANCE,
        device=logits_fine.device,
        dtype=posterior.dtype,
    )
    # risk[b, j] = E_{f~posterior[b]} d(f, j)
    risk = posterior @ distance
    return risk.argmin(dim=-1)


def _bayes_decode(
    logits_fine: Tensor,
    logits_coarse: Tensor,
    mapping_matrix: Tensor,
) -> Tensor:
    p_f = logits_fine.softmax(dim=-1)
    p_c = logits_coarse.softmax(dim=-1)
    coarse_of_f = mapping_matrix.argmax(dim=-1)
    p_c_for_f = p_c[:, coarse_of_f]
    p_bayes = p_f * p_c_for_f
    p_bayes = p_bayes / (p_bayes.sum(dim=-1, keepdim=True) + 1e-12)
    return p_bayes.argmax(dim=-1)


def _strict_decode(
    logits_fine: Tensor,
    logits_coarse: Tensor,
    mapping_matrix: Tensor,
    use_coarse_log: bool,
) -> Tensor:
    """Per-sample masked argmax within the predicted coarse lineage."""
    c_hat = logits_coarse.argmax(dim=-1)
    B, K_f = logits_fine.shape
    M = mapping_matrix.to(logits_fine.device)
    preds = []
    log_p_f = logits_fine.log_softmax(dim=-1)
    for i in range(B):
        c = int(c_hat[i].item())
        mask = M[:, c] > 0
        scores = log_p_f[i].clone()
        scores[~mask] = float("-inf")
        if use_coarse_log:
            log_p_c = logits_coarse[i].log_softmax(dim=-1)[c]
            scores = scores + log_p_c
        preds.append(int(scores.argmax().item()))
    return torch.tensor(preds, device=logits_fine.device, dtype=torch.long)


def coarse_predictions(
    logits_coarse: Optional[Tensor],
    fine_preds: Tensor,
    mapping_matrix: Tensor,
) -> Tensor:
    if logits_coarse is not None:
        return logits_coarse.argmax(dim=-1)
    parent = mapping_matrix.argmax(dim=-1).to(fine_preds.device)
    return parent[fine_preds]


def verify_bayes_consistency(
    logits_fine: Tensor,
    logits_coarse: Tensor,
    mapping_matrix: Tensor,
) -> Dict[str, float]:
    """Check whether Bayes fine preds agree with coarse-head argmax."""
    fine_bayes = _bayes_decode(logits_fine, logits_coarse, mapping_matrix)
    c_from_head = logits_coarse.argmax(dim=-1)
    parent = mapping_matrix.argmax(dim=-1).to(fine_bayes.device)
    c_from_fine = parent[fine_bayes]
    agree = (c_from_head == c_from_fine).float()
    return {
        "bayes_coarse_consistency_rate": float(agree.mean().item()),
        "n_samples": int(fine_bayes.shape[0]),
    }
