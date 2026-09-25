"""
Pathology-focused metrics for healthy vs abnormal cell recognition.

Stratifies fine-class predictions into:
  - off-chain / dysplastic / atypical identities (ontology)
  - pathologic classes from labels.json (biologic_status)
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from config.settings import REPO_ROOT
from data.datasets import resolve_canonical
from data.ontology import (
    CANONICAL_CLASSES,
    FINE_IDX_TO_COARSE_IDX,
    MATURATION_TABLE,
    MATURITY_VARIANT_OF,
)
from data.ontology import OFF_CHAIN_IDENTITIES, OFF_CHAIN_LINEAGE, is_off_chain
from data.ontology import fine_index_distance

_FINE_TO_COARSE = np.asarray(FINE_IDX_TO_COARSE_IDX)
_LABELS_PATH = REPO_ROOT / "configs" / "labels_v2.json"


@lru_cache(maxsize=1)
def pathologic_canonical_names() -> frozenset[str]:
    """Canonical fine classes with dysplastic/pathologic biologic status."""
    try:
        from data.ontology import is_v2_labels, metadata_by_canonical, is_off_chain_status
        if is_v2_labels():
            return frozenset(
                cid for cid, row in metadata_by_canonical().items()
                if is_off_chain_status(row)
                and str(row.get("biologic_status")) == "pathologic"
                and cid in CANONICAL_CLASSES
            )
    except Exception:
        pass
    names: Set[str] = set()
    from data.datasets import _labels_path
    for row in json.loads(_labels_path().read_text()):
        if row.get("biologic_status") not in (
            "non_healthy_or_pathologic", "pathologic",
        ):
            continue
        canon = resolve_canonical(row.get("canonical_id"))
        if canon and canon in CANONICAL_CLASSES:
            names.add(canon)
    return frozenset(names)


@lru_cache(maxsize=1)
def pathologic_fine_indices() -> frozenset[int]:
    names = pathologic_canonical_names()
    return frozenset(
        CANONICAL_CLASSES.index(n) for n in names if n in CANONICAL_CLASSES
    )


@lru_cache(maxsize=1)
def offchain_fine_indices() -> frozenset[int]:
    return frozenset(
        CANONICAL_CLASSES.index(n) for n in OFF_CHAIN_IDENTITIES if n in CANONICAL_CLASSES
    )


@lru_cache(maxsize=1)
def healthy_onchain_fine_indices() -> frozenset[int]:
    """On maturation chain, not pathologic, not off-chain."""
    pathologic = pathologic_fine_indices()
    out: Set[int] = set()
    for idx, name in enumerate(CANONICAL_CLASSES):
        if idx in pathologic or is_off_chain(name):
            continue
        chain_id, _ = MATURATION_TABLE[idx]
        if chain_id >= 0:
            out.add(idx)
    return frozenset(out)


def _gt_lineage_idx(fine_idx: int) -> int:
    name = CANONICAL_CLASSES[fine_idx]
    if name in OFF_CHAIN_LINEAGE:
        from data.ontology import LINEAGE_TO_IDX
        return LINEAGE_TO_IDX[OFF_CHAIN_LINEAGE[name]]
    return int(_FINE_TO_COARSE[fine_idx])


def _is_on_chain_healthy_pred(fine_idx: int) -> bool:
    name = CANONICAL_CLASSES[fine_idx]
    if is_off_chain(name):
        return False
    chain_id, _ = MATURATION_TABLE[fine_idx]
    return chain_id >= 0


def _is_false_healthy_normalization(gt: int, pred: int) -> bool:
    if gt == pred:
        return False
    gt_name = CANONICAL_CLASSES[gt]
    if gt_name not in pathologic_canonical_names() and not is_off_chain(gt_name):
        return False
    if not _is_on_chain_healthy_pred(pred):
        return False
    ref = MATURITY_VARIANT_OF.get(gt_name)
    if ref is not None and CANONICAL_CLASSES[pred] == ref:
        return True
    return _gt_lineage_idx(gt) == int(_FINE_TO_COARSE[pred])


def _macro_on_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_indices: List[int],
) -> Dict[str, float]:
    if not label_indices:
        return {"macro_f1": float("nan"), "macro_recall": float("nan"), "n_samples": 0}
    mask = np.isin(y_true, label_indices)
    n = int(mask.sum())
    if n == 0:
        return {"macro_f1": float("nan"), "macro_recall": float("nan"), "n_samples": 0}
    present = sorted(set(y_true[mask].tolist()))
    f1 = f1_score(
        y_true[mask], y_pred[mask], labels=present, average="macro", zero_division=0,
    )
    _, rec, _, _ = precision_recall_fscore_support(
        y_true[mask], y_pred[mask], labels=present, average="macro", zero_division=0,
    )
    return {"macro_f1": float(f1), "macro_recall": float(rec), "n_samples": n}


def pathology_metrics_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset: Optional[str] = None,
) -> Dict[str, float]:
    """Compute pathology stratification metrics on fine-class indices."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    if n == 0:
        return {}

    # Source-label datasets (e.g. MLL v1 folder codes) use local class indices.
    if int(np.max(y_true)) >= len(CANONICAL_CLASSES):
        return {}

    pathologic_idx = sorted(pathologic_fine_indices())
    offchain_idx = sorted(offchain_fine_indices())
    healthy_onchain_idx = sorted(healthy_onchain_fine_indices())

    offchain_present = sorted(set(y_true.tolist()) & set(offchain_idx))
    pathologic_present = sorted(set(y_true.tolist()) & set(pathologic_idx))
    healthy_present = sorted(set(y_true.tolist()) & set(healthy_onchain_idx))

    offchain = _macro_on_labels(y_true, y_pred, offchain_present)
    pathologic = _macro_on_labels(y_true, y_pred, pathologic_present)
    healthy_onchain = _macro_on_labels(y_true, y_pred, healthy_present)

    gt_pathologic = np.isin(y_true, pathologic_idx)
    pred_pathologic = np.isin(y_pred, pathologic_idx)
    detection_f1 = (
        float(f1_score(
            gt_pathologic, pred_pathologic, labels=[False, True], zero_division=0,
        ))
        if gt_pathologic.any() or pred_pathologic.any() else float("nan")
    )
    detection_recall = (
        float((pred_pathologic & gt_pathologic).sum() / gt_pathologic.sum())
        if gt_pathologic.any() else float("nan")
    )

    abnormal_gt = np.isin(y_true, list(pathologic_fine_indices() | offchain_fine_indices()))
    false_healthy = sum(
        _is_false_healthy_normalization(int(t), int(p))
        for t, p in zip(y_true.tolist(), y_pred.tolist())
    )
    false_healthy_rate = (
        false_healthy / int(abnormal_gt.sum()) if abnormal_gt.any() else float("nan")
    )

    pathologic_mask = gt_pathologic
    if pathologic_mask.any():
        distances = [
            fine_index_distance(int(t), int(p), dataset=dataset)
            for t, p in zip(y_true[pathologic_mask], y_pred[pathologic_mask])
        ]
        pathologic_tree_distance = float(np.mean(distances))
    else:
        pathologic_tree_distance = float("nan")

    return {
        "offchain_macro_f1": offchain["macro_f1"],
        "offchain_macro_recall": offchain["macro_recall"],
        "offchain_n_samples": float(offchain["n_samples"]),
        "offchain_classes_present": len(offchain_present),
        "pathologic_macro_f1": pathologic["macro_f1"],
        "pathologic_recall": pathologic["macro_recall"],
        "pathologic_n_samples": float(pathologic["n_samples"]),
        "pathologic_classes_present": len(pathologic_present),
        "pathologic_detection_f1": detection_f1,
        "pathologic_detection_recall": detection_recall,
        "false_healthy_rate": false_healthy_rate,
        "false_healthy_n": float(false_healthy),
        "abnormal_gt_n": float(abnormal_gt.sum()),
        "pathologic_mean_tree_distance": pathologic_tree_distance,
        "healthy_onchain_macro_f1": healthy_onchain["macro_f1"],
        "healthy_onchain_n_samples": float(healthy_onchain["n_samples"]),
    }
