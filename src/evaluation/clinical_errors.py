"""
Clinically grouped error rates for fine-class predictions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from data.ontology import CANONICAL_CLASSES, FINE_IDX_TO_COARSE_IDX, chains_for_fine

_FINE_TO_COARSE = np.asarray(FINE_IDX_TO_COARSE_IDX)

# fine_idx -> (chain_name or None, position_in_chain or -1)
_CHAIN_LOOKUP = [chains_for_fine(name) for name in CANONICAL_CLASSES]

_PROGENITOR_IDX = CANONICAL_CLASSES.index("blast_unspecified")
_BLAST_LIKE = {
    CANONICAL_CLASSES.index(n)
    for n in (
        "blast_unspecified", "myeloblast",
        "monoblast", "megakaryoblast", "proerythroblast",
    )
    if n in CANONICAL_CLASSES
}


def clinical_error_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Fraction of samples in each clinically motivated error bucket."""
    n = len(y_true)
    if n == 0:
        return {}

    exact = 0
    same_lin_adj = 0
    same_lin_nonadj = 0
    same_chain = 0
    same_lineage_other = 0
    cross_lin = 0
    blast_conf = 0

    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t == p:
            exact += 1
            continue

        ct, cp = _FINE_TO_COARSE[t], _FINE_TO_COARSE[p]
        if ct != cp:
            cross_lin += 1
            if (t in _BLAST_LIKE) ^ (p in _BLAST_LIKE):
                blast_conf += 1
            elif t == _PROGENITOR_IDX or p == _PROGENITOR_IDX:
                blast_conf += 1
            continue

        shared_gaps = [
            abs(pos_t - pos_p)
            for chain_t, pos_t in _CHAIN_LOOKUP[t]
            for chain_p, pos_p in _CHAIN_LOOKUP[p]
            if chain_t == chain_p
        ]
        if shared_gaps:
            same_chain += 1
            if min(shared_gaps) == 1:
                same_lin_adj += 1
            else:
                same_lin_nonadj += 1
        else:
            same_lin_nonadj += 1
            same_lineage_other += 1

    return {
        "clinical_exact_correct": exact / n,
        "clinical_same_lineage_adjacent": same_lin_adj / n,
        "clinical_same_lineage_nonadjacent": same_lin_nonadj / n,
        "clinical_same_chain": same_chain / n,
        "clinical_same_lineage_other": same_lineage_other / n,
        "clinical_cross_lineage": cross_lin / n,
        "clinical_blast_progenitor_confusion": blast_conf / n,
    }
