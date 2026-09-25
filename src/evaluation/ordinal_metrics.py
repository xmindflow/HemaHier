"""
Ordinal maturity metrics for HemaHier.

Evaluates whether lineage-specific maturity heads capture biological ordering,
independent of fine macro-F1.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

from data.ontology import (
    CANONICAL_CLASSES,
    FINE_TREE_DISTANCE,
    LINEAGES,
    LINEAGES_WITH_MATURATION,
    MATURATION_CHAINS,
    MATURATION_TABLE,
    _FINE_CHAIN_NAME,
)


def _chain_name_for_fine(fine_idx: int) -> Optional[str]:
    return _FINE_CHAIN_NAME[fine_idx]


def maturity_from_fine_labels(fine_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Class-derived normalized maturity and validity mask."""
    positions = np.array([MATURATION_TABLE[int(f)][1] for f in fine_idx], dtype=np.float32)
    mask = np.array([MATURATION_TABLE[int(f)][1] >= 0.0 for f in fine_idx], dtype=bool)
    return positions, mask


def compute_class_derived_maturity_metrics(
    fine_pred: np.ndarray,
    mat_true: np.ndarray,
    mat_mask: np.ndarray,
) -> Dict[str, float]:
    """Maturity metrics when maturity is read from predicted fine class."""
    pred_pos, pred_mask = maturity_from_fine_labels(fine_pred)
    mask = mat_mask.astype(bool) & pred_mask
    if mask.sum() == 0:
        return {
            "maturity_mae_class_derived": float("nan"),
            "maturity_spearman_class_derived": float("nan"),
        }
    true_m = mat_true[mask]
    pred_m = pred_pos[mask]
    mae = float(np.mean(np.abs(pred_m - true_m)))
    if len(np.unique(true_m)) > 1:
        rho, _ = spearmanr(true_m, pred_m)
        spearman = float(rho) if not np.isnan(rho) else 0.0
    else:
        spearman = float("nan")
    return {
        "maturity_mae_class_derived": mae,
        "maturity_spearman_class_derived": spearman,
    }


@torch.no_grad()
def collect_maturity_predictions(
    model,
    loader,
    device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns mat_pred, mat_true, mat_mask, coarse_true, fine_true."""
    preds, targets, masks, coarse, fine = [], [], [], [], []
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        fine_lab = batch[1].long().to(device)
        coarse_lab = batch[2].long().to(device)
        mat_pos = batch[3].float().to(device)
        mat_m = batch[4].bool().to(device)

        model_kwargs = {"coarse_labels": coarse_lab}
        if getattr(model, "requires_fine_for_maturity", False):
            model_kwargs["fine_labels"] = fine_lab
        out = model(images, return_features=False, **model_kwargs)
        mp = out.get("mat_pred")
        if mp is None:
            continue
        preds.append(mp.cpu().numpy())
        targets.append(mat_pos.cpu().numpy())
        masks.append(mat_m.cpu().numpy())
        coarse.append(coarse_lab.cpu().numpy())
        fine.append(fine_lab.cpu().numpy())

    if not preds:
        return (
            np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]),
        )
    return (
        np.concatenate(preds),
        np.concatenate(targets),
        np.concatenate(masks),
        np.concatenate(coarse),
        np.concatenate(fine),
    )


def compute_ordinal_metrics(
    mat_pred: np.ndarray,
    mat_true: np.ndarray,
    mat_mask: np.ndarray,
    coarse_true: np.ndarray,
    fine_true: Optional[np.ndarray] = None,
    embeddings: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Maturity MAE, Spearman (global + per-lineage), clinical error rates."""
    result: Dict[str, float] = {}
    mask = mat_mask.astype(bool)
    if mask.sum() == 0:
        return {"maturity_mae": float("nan"), "maturity_spearman": float("nan")}

    pred_m = mat_pred[mask]
    true_m = mat_true[mask]
    result["maturity_mae"] = float(np.mean(np.abs(pred_m - true_m)))

    if len(np.unique(true_m)) > 1:
        rho, _ = spearmanr(true_m, pred_m)
        result["maturity_spearman"] = float(rho) if not np.isnan(rho) else 0.0
    else:
        result["maturity_spearman"] = float("nan")

    per_lin: List[float] = []
    for li in LINEAGES_WITH_MATURATION:
        lin_mask = mask & (coarse_true == li)
        if lin_mask.sum() < 3:
            continue
        t, p = mat_true[lin_mask], mat_pred[lin_mask]
        if len(np.unique(t)) > 1:
            rho, _ = spearmanr(t, p)
            if not np.isnan(rho):
                per_lin.append(float(rho))
                result[f"maturity_spearman_{LINEAGES[li]}"] = float(rho)
    result["maturity_spearman_mean_lineage"] = (
        float(np.mean(per_lin)) if per_lin else float("nan")
    )

    if fine_true is not None and embeddings is not None and len(embeddings) > 10:
        result.update(_embedding_tree_correlation(embeddings, fine_true))
    if fine_true is not None:
        result.update(pairwise_maturity_ordering_accuracy(
            pred_maturity=mat_pred,
            fine_true=fine_true,
            mat_mask=mat_mask,
        ))

    return result


def pairwise_maturity_ordering_accuracy(
    pred_maturity: np.ndarray,
    fine_true: np.ndarray,
    mat_mask: np.ndarray,
) -> Dict[str, float]:
    """Fraction of cross-stage sample pairs ordered correctly within chains."""
    correct = 0.0
    total = 0
    result: Dict[str, float] = {}
    valid = mat_mask.astype(bool)
    for chain_name, members in MATURATION_CHAINS.items():
        chain_correct = 0.0
        chain_total = 0
        for earlier_pos, earlier_name in enumerate(members):
            earlier_idx = CANONICAL_CLASSES.index(earlier_name)
            earlier_scores = pred_maturity[valid & (fine_true == earlier_idx)]
            for later_name in members[earlier_pos + 1:]:
                later_idx = CANONICAL_CLASSES.index(later_name)
                later_scores = pred_maturity[valid & (fine_true == later_idx)]
                if not len(earlier_scores) or not len(later_scores):
                    continue
                differences = later_scores[:, None] - earlier_scores[None, :]
                chain_correct += float((differences > 0).sum())
                chain_correct += 0.5 * float((differences == 0).sum())
                chain_total += int(differences.size)
        if chain_total:
            result[f"maturity_pairwise_accuracy_{chain_name}"] = chain_correct / chain_total
            correct += chain_correct
            total += chain_total
    result["maturity_pairwise_accuracy"] = correct / total if total else float("nan")
    return result


def compute_chain_head_ordinal_metrics(
    chain_preds: Dict[str, np.ndarray],
    fine_true: np.ndarray,
) -> Dict[str, float]:
    """Evaluate every branch with its own head and normalized chain targets."""
    maes: List[float] = []
    rhos: List[float] = []
    total_correct = 0.0
    total_pairs = 0
    result: Dict[str, float] = {}
    for chain_name, members in MATURATION_CHAINS.items():
        if chain_name not in chain_preds:
            continue
        pred_all = np.asarray(chain_preds[chain_name])
        class_indices = [CANONICAL_CLASSES.index(name) for name in members]
        positions = np.linspace(0.0, 1.0, len(members), dtype=np.float32)
        mask = np.isin(fine_true, class_indices)
        if not mask.any():
            continue
        targets = np.zeros(int(mask.sum()), dtype=np.float32)
        selected_labels = fine_true[mask]
        for class_idx, position in zip(class_indices, positions):
            targets[selected_labels == class_idx] = position
        predictions = pred_all[mask]
        mae = float(np.mean(np.abs(predictions - targets)))
        maes.append(mae)
        result[f"maturity_mae_chain_{chain_name}"] = mae
        if len(np.unique(targets)) > 1:
            rho, _ = spearmanr(targets, predictions)
            if not np.isnan(rho):
                rhos.append(float(rho))
                result[f"maturity_spearman_chain_{chain_name}"] = float(rho)

        chain_correct = 0.0
        chain_pairs = 0
        for earlier_idx in range(len(class_indices) - 1):
            earlier = pred_all[fine_true == class_indices[earlier_idx]]
            for later_idx in range(earlier_idx + 1, len(class_indices)):
                later = pred_all[fine_true == class_indices[later_idx]]
                if not len(earlier) or not len(later):
                    continue
                differences = later[:, None] - earlier[None, :]
                chain_correct += float((differences > 0).sum())
                chain_correct += 0.5 * float((differences == 0).sum())
                chain_pairs += int(differences.size)
        if chain_pairs:
            result[f"maturity_pairwise_accuracy_chain_{chain_name}"] = (
                chain_correct / chain_pairs
            )
            total_correct += chain_correct
            total_pairs += chain_pairs

    result["maturity_mae_mean_chain"] = float(np.mean(maes)) if maes else float("nan")
    result["maturity_spearman_mean_chain"] = (
        float(np.mean(rhos)) if rhos else float("nan")
    )
    result["maturity_pairwise_accuracy_chain"] = (
        total_correct / total_pairs if total_pairs else float("nan")
    )
    return result


def compute_maturity_sanity_metrics(
    mat_pred_head: np.ndarray,
    mat_true: np.ndarray,
    mat_mask: np.ndarray,
    fine_pred: np.ndarray,
) -> Dict[str, float]:
    """Head-based vs class-derived maturity (sanity check for HemaHier)."""
    head = compute_ordinal_metrics(mat_pred_head, mat_true, mat_mask, coarse_true=np.zeros_like(mat_true))
    derived = compute_class_derived_maturity_metrics(fine_pred, mat_true, mat_mask)
    out = {
        "maturity_mae_head": head.get("maturity_mae", float("nan")),
        "maturity_spearman_head": head.get("maturity_spearman", float("nan")),
    }
    out.update(derived)
    return out


def _embedding_tree_correlation(
    embeddings: np.ndarray,
    fine_true: np.ndarray,
    n_pairs: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(fine_true)
    if n < 4:
        return {}
    idx_i = rng.integers(0, n, size=n_pairs)
    idx_j = rng.integers(0, n, size=n_pairs)
    emb_d, tree_d = [], []
    for i, j in zip(idx_i, idx_j):
        if i == j:
            continue
        emb_d.append(float(np.linalg.norm(embeddings[i] - embeddings[j])))
        tree_d.append(float(FINE_TREE_DISTANCE[fine_true[i], fine_true[j]]))
    if len(emb_d) < 5:
        return {}
    rho, _ = spearmanr(emb_d, tree_d)
    return {"embedding_tree_spearman": float(rho) if not np.isnan(rho) else 0.0}


def evaluate_ordinal_metrics(
    model,
    loader,
    device,
    collect_embeddings: bool = False,
) -> Dict[str, float]:
    mat_pred, mat_true, mat_mask, coarse_true, fine_true = collect_maturity_predictions(
        model, loader, device,
    )
    embs = None
    if collect_embeddings and hasattr(model, "encode"):
        embs_list, fine_list = [], []
        for batch in loader:
            images = batch[0].to(device)
            fine_list.append(batch[1].numpy())
            with torch.no_grad():
                embs_list.append(model.encode(images).cpu().numpy())
        if embs_list:
            embs = np.concatenate(embs_list)
            fine_true = np.concatenate(fine_list)

    return compute_ordinal_metrics(
        mat_pred, mat_true, mat_mask, coarse_true, fine_true, embs,
    )


def mean_tree_distance_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset: Optional[str] = None,
) -> float:
    if len(y_true) == 0:
        return float("nan")
    from data.ontology import fine_index_distance
    dists = [fine_index_distance(int(t), int(p), dataset=dataset) for t, p in zip(y_true, y_pred)]
    return float(np.mean(dists))
