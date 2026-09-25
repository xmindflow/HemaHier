"""
Bone-marrow ontology: labels_v2.json, taxonomy, training hierarchy, evaluation distances.

Single module for class lists, maturation chains, label mappings, and D_eval.
"""

from __future__ import annotations

# --- labels_v2.json ---------------------------------------------------------


import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_V2_PATH = REPO_ROOT / "configs" / "labels_v2.json"

# Dataset key in configs/labels_v2.json for each dataset the code can load.
# The "ontology" rows carry the taxonomy itself and belong to no dataset.
_LABELS_DATASET_KEYS = {"mllv1": "mll_v1"}

LINEAGE_V2_TO_COARSE: Dict[str, str] = {
    "multipotential_blast": "progenitor",
    "granulocytopoiesis": "granulocytic",
    "myeloid": "granulocytic",
    "erythropoiesis": "erythroid",
    "monocytopoiesis": "monocytic",
    "lymphocytopoiesis": "lymphoid",
    "megakaryocytopoiesis": "megakaryocytic",
    "mast_cell_lineage": "mast",
    "artifact_background": "other",
    "non_hematopoietic_stromal": "other",
}

# Branch → maturation chain head name (HemaHier z_d conditioning).
BRANCH_TO_CHAIN: Dict[str, str] = {
    "neutrophilic": "neutrophil",
    "eosinophilic": "eosinophil",
    "basophilic": "basophil",
    "erythroid": "erythroid",
    "monocytic": "monocytic",
    "lymphoid": "lymphoid",
    "megakaryocytic": "megakaryocytic",
    "granulocytic": "granulocytic",
    "plasma_cell": "lymphoid",
    "macrophage": "monocytic",
    "mast_cell": "mast",
    "b_cell": "lymphoid",
    "root": "root",
}

VIRTUAL_ROOTS = frozenset({"progenitor", "blast_unspecified"})


def labels_v2_path() -> Path:
    try:
        from config.settings import cfg_paths
        rel = cfg_paths().get("labels_file", "labels_v2.json")
        p = REPO_ROOT / str(rel)
        if p.is_file():
            return p
    except Exception:
        pass
    return DEFAULT_LABELS_V2_PATH


def is_v2_labels(path: Optional[Path] = None) -> bool:
    p = path or labels_v2_path()
    return p.name == "labels_v2.json" and p.is_file()


@lru_cache(maxsize=1)
def load_rows() -> Tuple[dict, ...]:
    path = labels_v2_path()
    return tuple(json.loads(path.read_text()))


def clear_cache() -> None:
    load_rows.cache_clear()
    build_taxonomy.cache_clear()
    metadata_by_canonical.cache_clear()
    parent_map_global.cache_clear()


def is_kept_row(row: Mapping[str, object]) -> bool:
    if row.get("is_artifact"):
        return False
    bs = str(row.get("biologic_status") or "")
    hc = str(row.get("health_category") or "")
    if hc == "artifact" or bs in ("artifact", "ignore", "process", "non_hematopoietic"):
        return False
    if hc == "healthy" or bs in ("healthy_normal", "healthy_or_non_neoplastic"):
        return True
    if bs in ("dysplastic", "pathologic", "non_healthy_or_pathologic"):
        return True
    if hc == "non_healthy":
        return True
    return False


def is_included_row(row: Mapping[str, object], *, include_artifacts: bool = False) -> bool:
    """Whether a labels row participates in source→canonical mapping."""
    if is_kept_row(row):
        return True
    if include_artifacts and row.get("is_artifact") and row.get("canonical_id"):
        return True
    return False


def is_off_chain_status(row: Mapping[str, object]) -> bool:
    return str(row.get("biologic_status") or "") in (
        "dysplastic", "pathologic", "non_healthy_or_pathologic",
    )


def is_healthy_chain_row(row: Mapping[str, object]) -> bool:
    return str(row.get("biologic_status") or "") in (
        "healthy_normal", "healthy_or_non_neoplastic",
    )


def rows_for_dataset(dataset_name: str) -> List[dict]:
    key = _LABELS_DATASET_KEYS.get(dataset_name, dataset_name)
    return [dict(r) for r in load_rows() if r.get("dataset") == key]


def _normalize_parent(parent: Optional[str]) -> Optional[str]:
    if parent in (None, ""):
        return None
    if parent == "progenitor":
        return "blast_unspecified"
    return str(parent)


def _order_branch_chain(members: Set[str], parent_of: Dict[str, Optional[str]]) -> List[str]:
    if not members:
        return []
    children: Dict[str, List[str]] = {m: [] for m in members}
    roots: List[str] = []
    for node in members:
        parent = parent_of.get(node)
        if parent in members:
            children[parent].append(node)
        else:
            roots.append(node)
    roots.sort()
    ordered: List[str] = []
    seen: Set[str] = set()

    def walk(n: str) -> None:
        if n in seen:
            return
        seen.add(n)
        ordered.append(n)
        for kid in sorted(children.get(n, [])):
            walk(kid)

    for root in roots:
        walk(root)
    for node in sorted(members):
        if node not in seen:
            ordered.append(node)
    return ordered


@lru_cache(maxsize=1)
def metadata_by_canonical(
    datasets: Tuple[str, ...] = ("ontology",),
) -> Dict[str, dict]:
    """Merged metadata per canonical_id, over the taxonomy rows."""
    out: Dict[str, dict] = {}
    for ds in datasets:
        for row in rows_for_dataset(ds):
            if not is_kept_row(row):
                continue
            cid = row.get("canonical_id")
            if cid and str(cid) not in out:
                out[str(cid)] = dict(row)
    return out


def _resolve_parent_conflicts(edges: Dict[str, Set[str]]) -> Dict[str, Optional[str]]:
    """Pick one parent per node when datasets disagree (prefer deepest parent)."""
    resolved: Dict[str, Optional[str]] = {}
    for cid, pars in edges.items():
        pars = {p for p in pars if p}
        if not pars:
            resolved[cid] = None
            continue
        if "myeloblast" in pars:
            pars.discard("blast_unspecified")
        if len(pars) > 1 and "blast_unspecified" in pars:
            deeper = {p for p in pars if p != "blast_unspecified"}
            if deeper:
                pars = deeper
        if len(pars) == 1:
            resolved[cid] = next(iter(pars))
        else:
            resolved[cid] = sorted(pars)[0]
    if "myeloblast" in edges or "myeloblast" in resolved:
        if resolved.get("myeloblast") in (None, ""):
            resolved["myeloblast"] = "blast_unspecified"
    return resolved


@lru_cache(maxsize=1)
def parent_map_global(
    datasets: Tuple[str, ...] = ("ontology",),
) -> Dict[str, Optional[str]]:
    edges: Dict[str, Set[str]] = {}
    for ds in datasets:
        for row in rows_for_dataset(ds):
            if not is_kept_row(row):
                continue
            cid = str(row["canonical_id"])
            par = _normalize_parent(row.get("parent"))
            if par:
                edges.setdefault(cid, set()).add(par)
            else:
                edges.setdefault(cid, set())
    return _resolve_parent_conflicts(edges)


@lru_cache(maxsize=1)
def build_taxonomy(
    datasets: Tuple[str, ...] = ("ontology",),
) -> dict:
    """Build CANONICAL_CLASSES, LINEAGE_HIERARCHY, MATURATION_CHAINS from labels_v2."""
    kept_ids: List[str] = []
    seen: Set[str] = set()
    for ds in datasets:
        for row in rows_for_dataset(ds):
            if not is_kept_row(row):
                continue
            cid = str(row["canonical_id"])
            if cid not in seen:
                seen.add(cid)
                kept_ids.append(cid)

    meta = metadata_by_canonical()
    lineage_hierarchy: Dict[str, List[str]] = {
        "granulocytic": [],
        "erythroid": [],
        "monocytic": [],
        "lymphoid": [],
        "megakaryocytic": [],
        "progenitor": [],
        "mast": [],
        "other": [],
    }

    for cid in kept_ids:
        row = meta.get(cid, {})
        coarse = LINEAGE_V2_TO_COARSE.get(str(row.get("lineage") or ""), "other")
        if cid not in lineage_hierarchy[coarse]:
            lineage_hierarchy[coarse].append(cid)

    parent_of = parent_map_global()
    chains: Dict[str, List[str]] = {}
    branch_members: Dict[str, Set[str]] = {}

    for cid in kept_ids:
        row = meta.get(cid, {})
        if not is_healthy_chain_row(row):
            continue
        branch = str(row.get("branch") or "")
        chain_name = BRANCH_TO_CHAIN.get(branch)
        if not chain_name or chain_name in ("root", "mast", "granulocytic"):
            continue
        branch_members.setdefault(chain_name, set()).add(cid)

    def _include_granulocytic_ancestors(members: Set[str]) -> Set[str]:
        out = set(members)
        for node in list(members):
            cur = parent_of.get(node)
            while cur and cur not in VIRTUAL_ROOTS:
                if cur in kept_set and cur not in out:
                    prow = meta.get(cur, {})
                    if is_healthy_chain_row(prow):
                        out.add(cur)
                cur = parent_of.get(cur)
        return out

    kept_set = set(kept_ids)
    branch_members = {
        name: _include_granulocytic_ancestors(members)
        for name, members in branch_members.items()
    }

    for chain_name, members in branch_members.items():
        chains[chain_name] = _order_branch_chain(
            {m for m in members if m not in VIRTUAL_ROOTS},
            parent_of,
        )

    chain_parent_lineage = {
        "neutrophil": "granulocytic",
        "eosinophil": "granulocytic",
        "basophil": "granulocytic",
        "granulocytic": "granulocytic",
        "erythroid": "erythroid",
        "monocytic": "monocytic",
        "lymphoid": "lymphoid",
        "megakaryocytic": "megakaryocytic",
    }

    off_chain = sorted(
        cid for cid in kept_ids
        if is_off_chain_status(meta.get(cid, {}))
    )
    off_chain_lineage = {
        cid: LINEAGE_V2_TO_COARSE.get(str(meta[cid].get("lineage") or ""), "granulocytic")
        for cid in off_chain
        if cid in meta
    }

    return {
        "canonical_classes": kept_ids,
        "lineage_hierarchy": lineage_hierarchy,
        "maturation_chains": chains,
        "chain_parent_lineage": chain_parent_lineage,
        "off_chain_identities": off_chain,
        "off_chain_lineage": off_chain_lineage,
        "parent_map": parent_of,
        "metadata": meta,
    }


def build_mapping(
    dataset_name: str,
    *,
    include_artifacts: bool = False,
) -> Dict[str, Optional[str]]:
    """source_label → canonical_id (None if excluded)."""
    rows = rows_for_dataset(dataset_name)
    kept = {
        str(r["canonical_id"])
        for r in rows
        if is_included_row(r, include_artifacts=include_artifacts)
    }
    mapping: Dict[str, Optional[str]] = {}
    for row in rows:
        src = str(row["source_label"])
        if not is_included_row(row, include_artifacts=include_artifacts):
            mapping[src] = None
            continue
        cid = str(row["canonical_id"])
        mapping[src] = cid if cid in kept else None
    return mapping


def row_for_source(dataset_name: str, source_label: str) -> Optional[dict]:
    for row in rows_for_dataset(dataset_name):
        if row.get("source_label") == source_label:
            return row
    return None


def _ancestors(node: str, parent_of: Dict[str, Optional[str]]) -> List[str]:
    chain = [node]
    cur = parent_of.get(node)
    seen = {node}
    while cur and cur not in seen:
        chain.append(cur)
        seen.add(cur)
        cur = parent_of.get(cur)
    return chain


def _tree_hops(a: str, b: str, parent_of: Dict[str, Optional[str]]) -> int:
    if a == b:
        return 0
    anc_a = set(_ancestors(a, parent_of))
    for i, node in enumerate(_ancestors(b, parent_of)):
        if node in anc_a:
            lca = node
            break
    else:
        return 10_000
    ia = _ancestors(a, parent_of).index(lca)
    ib = _ancestors(b, parent_of).index(lca)
    return ia + ib


def v2_pairwise_distance(
    name_a: str,
    name_b: str,
    *,
    eval_params: Optional[Mapping[str, float]] = None,
) -> float:
    """Unified D_eval from labels_v2 parent tree + off-chain rules."""
    if name_a == name_b:
        return 0.0

    p = dict(eval_params or {})
    d_cross = float(p.get("cross_lineage", 1.0))
    d_off_parent = float(p.get("off_chain_to_healthy", 0.25))
    d_off_intra = float(p.get("off_chain_intra", 0.10))
    d_max_chain = float(p.get("max_same_chain", 0.85))
    d_branch = float(p.get("same_lineage_branch", 0.75))

    meta = metadata_by_canonical()
    parent_of = parent_map_global()
    tax = build_taxonomy()
    chains = tax["maturation_chains"]
    off_chain = set(tax["off_chain_identities"])

    def anchor(node: str) -> str:
        if node in off_chain:
            par = parent_of.get(node)
            return par if par else node
        return node

    a, b = anchor(name_a), anchor(name_b)
    row_a, row_b = meta.get(name_a, {}), meta.get(name_b, {})

    la = LINEAGE_V2_TO_COARSE.get(str(row_a.get("lineage") or ""), "other")
    lb = LINEAGE_V2_TO_COARSE.get(str(row_b.get("lineage") or ""), "other")
    if la != lb:
        return d_cross

    # Off-chain rules take precedence over the chain-gap: an off-chain
    # (pathological/atypical) identity anchors to its healthy parent, so if the
    # other class *is* that parent the chain-gap would be 0.  Confusing a
    # pathological cell with its healthy counterpart must still carry the
    # off-chain penalty rather than zero cost.
    if name_a in off_chain and name_b in off_chain:
        return d_off_intra if la == lb else d_cross

    if name_a in off_chain or name_b in off_chain:
        healthy, off = (a, b) if name_b in off_chain else (b, a)
        hops = _tree_hops(healthy, off, parent_of)
        if hops < 10_000:
            return d_off_parent
        return d_branch

    for chain in chains.values():
        if a in chain and b in chain:
            denom = max(len(chain) - 1, 1)
            gap = abs(chain.index(a) - chain.index(b)) / denom
            return min(gap * d_max_chain, d_max_chain)

    hops = _tree_hops(a, b, parent_of)
    if hops >= 10_000:
        return d_cross
    return min(d_branch, d_max_chain)

# --- hierarchy (training tensors) ----------------------------------------


from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Fine class list (must match the HemaAug canonical list for label
# compatibility with the existing labels.json).
# ---------------------------------------------------------------------------

CANONICAL_CLASSES: List[str] = [
    # Progenitor roots (undifferentiated and granulocytic blasts are separate)
    "blast_unspecified",
    "myeloblast",
    # Granulocytic
    "promyelocyte",
    "promyelocyte_atypical",
    "myelocyte_neutrophil",
    "myelocyte_eosinophil",
    "metamyelocyte",
    "neutrophil_band",
    "segmented_neutrophil",
    "eosinophil",
    "eosinophil_immature",
    "eosinophil_atypical",
    "basophil",
    "basophil_immature",
    "basophil_atypical",
    "pseudo_pelger_pattern",
    # Erythroid
    "proerythroblast",
    "basophilic_erythroblast",
    "polychromatic_erythroblast",
    "orthochromatic_erythroblast",
    "erythroblast_dysplastic",
    # Monocytic
    "monoblast",
    "monocyte",
    "macrophage",
    # Lymphoid
    "lymphocyte",
    "lymphocyte_reactive",
    "lymphocyte_neoplastic",
    "lymphocyte_large_granular",
    "plasma_cell",
    # Megakaryocytic
    "megakaryoblast",
    "megakaryocyte",
    "megakaryocyte_dysplastic",
    "micromegakaryocyte",
    # Other (kept for inference; excluded from training in dataset.py)
    "mast_cell",
    "hairy_cell",
    "other_cell",
]


# ---------------------------------------------------------------------------
# Coarse level: LINEAGE groups
# ---------------------------------------------------------------------------

LINEAGE_HIERARCHY: Dict[str, List[str]] = {
    # Mature granulocyte lineages share staining and morphology more with each
    # other than with anything else, so we group them under a single coarse
    # node and let the fine head resolve the sub-lineage.
    "granulocytic": [
        "myeloblast",
        "promyelocyte",
        "promyelocyte_atypical",
        "myelocyte_neutrophil",
        "myelocyte_eosinophil",
        "metamyelocyte",
        "neutrophil_band",
        "segmented_neutrophil",
        "pseudo_pelger_pattern",
        "eosinophil",
        "eosinophil_immature",
        "eosinophil_atypical",
        "basophil",
        "basophil_immature",
        "basophil_atypical",
    ],
    "erythroid": [
        "proerythroblast",
        "basophilic_erythroblast",
        "polychromatic_erythroblast",
        "orthochromatic_erythroblast",
        "erythroblast_dysplastic",
    ],
    "monocytic": [
        "monoblast",
        "monocyte",
        "macrophage",
    ],
    "lymphoid": [
        "lymphocyte",
        "lymphocyte_reactive",
        "lymphocyte_neoplastic",
        "lymphocyte_large_granular",
        "plasma_cell",
    ],
    "megakaryocytic": [
        "megakaryoblast",
        "megakaryocyte",
        "megakaryocyte_dysplastic",
        "micromegakaryocyte",
    ],
    # Multipotential progenitor root (name says "unspecified", not "unknown").
    "progenitor": [
        "blast_unspecified",
    ],
    "mast": [
        "mast_cell",
        "hairy_cell",
    ],
    "other": [
        "other_cell",
    ],
}

LINEAGES: List[str] = list(LINEAGE_HIERARCHY.keys())


# ---------------------------------------------------------------------------
# Maturation chains: ORDINAL positions within a lineage
# (0 = least mature, max = most mature). Used by an auxiliary ordinal
# regression head that gives the model extra biological prior.
# ---------------------------------------------------------------------------

# Parent compartment (coarse lineage) for each healthy maturation branch.
CHAIN_PARENT_LINEAGE: Dict[str, str] = {
    "neutrophil": "granulocytic",
    "eosinophil": "granulocytic",
    "basophil": "granulocytic",
    "erythroid": "erythroid",
    "monocytic": "monocytic",
    "lymphoid": "lymphoid",
    "megakaryocytic": "megakaryocytic",
}

MATURATION_CHAINS: Dict[str, List[str]] = {
    "neutrophil": [
        "myeloblast",
        "promyelocyte",
        "myelocyte_neutrophil",
        "metamyelocyte",
        "neutrophil_band",
        "segmented_neutrophil",
    ],
    "eosinophil": [
        "promyelocyte",
        "eosinophil_immature",
        "eosinophil",
    ],
    "basophil": [
        "promyelocyte",
        "basophil_immature",
        "basophil",
    ],
    "erythroid": [
        "proerythroblast",
        "basophilic_erythroblast",
        "polychromatic_erythroblast",
        "orthochromatic_erythroblast",
    ],
    "monocytic": [
        "monoblast",
        "monocyte",
        "macrophage",
    ],
    "lymphoid": [
        "lymphocyte",
        "plasma_cell",
    ],
    "megakaryocytic": [
        "megakaryoblast",
        "megakaryocyte",
    ],
}

# Explicit same-stage variants.  These stay in their biological lineage but
# never become normal maturation steps.  The mapping is used only by the
# evaluation distance (variant ↔ reference stage = 0.25).
MATURITY_VARIANT_OF: Dict[str, str] = {
    "promyelocyte_atypical": "promyelocyte",
}


# ---------------------------------------------------------------------------
# Derived structures
# ---------------------------------------------------------------------------

FINE_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
LINEAGE_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(LINEAGES)}

# fine_name -> coarse_name
FINE_TO_COARSE: Dict[str, str] = {}
for lineage, members in LINEAGE_HIERARCHY.items():
    for cls in members:
        FINE_TO_COARSE[cls] = lineage

# fine_idx -> coarse_idx
FINE_IDX_TO_COARSE_IDX: List[int] = [
    LINEAGE_TO_IDX[FINE_TO_COARSE[c]] for c in CANONICAL_CLASSES
]


def fine_to_maturation_position(fine: str) -> Tuple[int, float]:
    """
    Returns (chain_id, normalized_position). Valid positions are in [0, 1]
    (0 = least mature). chain_id = -1 and position = -1 mean the cell has no
    normal maturation ordering and must be ignored by the ordinal head.
    """
    for chain_id, (_, members) in enumerate(MATURATION_CHAINS.items()):
        if fine in members:
            pos = members.index(fine)
            denom = max(len(members) - 1, 1)
            return chain_id, pos / denom
    return -1, -1.0


# Pre-compute (chain_id, normalized_position) for every fine class.
MATURATION_TABLE: List[Tuple[int, float]] = [
    fine_to_maturation_position(c) for c in CANONICAL_CLASSES
]


# ---------------------------------------------------------------------------
# Mapping matrix M ∈ R^{K_f × K_c}
# ---------------------------------------------------------------------------

def build_mapping_matrix() -> torch.Tensor:
    """Binary matrix M with M[f, c] = 1 if fine class f belongs to coarse c."""
    K_f, K_c = len(CANONICAL_CLASSES), len(LINEAGES)
    M = torch.zeros(K_f, K_c)
    for f, fine in enumerate(CANONICAL_CLASSES):
        M[f, LINEAGE_TO_IDX[FINE_TO_COARSE[fine]]] = 1.0
    return M


MAPPING_MATRIX: torch.Tensor = build_mapping_matrix()


def fine_logits_to_coarse_probs(
    fine_logits: torch.Tensor,
    mapping_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert a (B, K_f) tensor of fine logits into a (B, K_c) tensor of
    coarse probabilities by summing the softmax mass that falls inside
    each coarse class.  This is the HBA equation (4) from Jin et al. 2025.
    """
    M = mapping_matrix if mapping_matrix is not None else MAPPING_MATRIX
    M = M.to(fine_logits.device)
    p_fine = fine_logits.softmax(dim=-1)
    return p_fine @ M


# ---------------------------------------------------------------------------
# Public helpers used during training
# ---------------------------------------------------------------------------

def fine_label_to_coarse_label(fine_labels: torch.Tensor) -> torch.Tensor:
    """Vectorised fine_idx → coarse_idx."""
    table = torch.tensor(FINE_IDX_TO_COARSE_IDX, device=fine_labels.device)
    return table[fine_labels]


def fine_label_to_maturation_target(
    fine_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (positions, valid_mask):
        positions     - (B,) float, normalized chain position
        valid_mask    - (B,) bool, True if the cell has an ordinal position.
    """
    positions = torch.zeros_like(fine_labels, dtype=torch.float32)
    mask = torch.zeros_like(fine_labels, dtype=torch.bool)
    table = MATURATION_TABLE
    for i, fl in enumerate(fine_labels.tolist()):
        chain_id, pos = table[fl]
        if chain_id >= 0:
            positions[i] = pos
            mask[i] = True
    return positions, mask


# ---------------------------------------------------------------------------
# Rare class list (used for rare-F1 metric, mirrors HemaAug list).
# ---------------------------------------------------------------------------

RARE_CLASSES: List[str] = [
    "blast_unspecified",
    "promyelocyte", "promyelocyte_atypical",
    "myelocyte_neutrophil", "myelocyte_eosinophil",
    "metamyelocyte",
    "basophil", "basophil_immature",
    "eosinophil_immature",
    "proerythroblast",
    "monoblast", "macrophage",
    "megakaryoblast", "megakaryocyte",
    "mast_cell",
    "plasma_cell", "hairy_cell",
    "lymphocyte_neoplastic",
]


# ---------------------------------------------------------------------------
# Lineage-specific ordinal (HemaHier-v2)
# ---------------------------------------------------------------------------

_FINE_CHAIN_NAME: List[Optional[str]] = []
_FINE_CHAIN_POS: List[int] = []
for _fine_name in CANONICAL_CLASSES:
    _found_name, _found_pos = None, -1
    for _cn, _members in MATURATION_CHAINS.items():
        if _fine_name in _members:
            _found_name, _found_pos = _cn, _members.index(_fine_name)
            break
    _FINE_CHAIN_NAME.append(_found_name)
    _FINE_CHAIN_POS.append(_found_pos)


def lineages_with_maturation_heads() -> List[int]:
    """Coarse lineage indices that contain at least one chain-defined fine class."""
    out: List[int] = []
    for li, lineage in enumerate(LINEAGES):
        for fine in LINEAGE_HIERARCHY[lineage]:
            chain_id, _ = fine_to_maturation_position(fine)
            if chain_id >= 0:
                out.append(li)
                break
    return out


LINEAGES_WITH_MATURATION: List[int] = lineages_with_maturation_heads()


def assemble_lineage_mat_pred(
    lineage_preds: Dict[int, torch.Tensor],
    coarse_labels: torch.Tensor,
) -> torch.Tensor:
    """Gather per-lineage head outputs using true coarse labels (for loss/metrics)."""
    device = coarse_labels.device
    out = torch.zeros(coarse_labels.shape[0], device=device, dtype=next(iter(lineage_preds.values())).dtype)
    for li, pred in lineage_preds.items():
        mask = coarse_labels == li
        if mask.any():
            out[mask] = pred[mask]
    return out


def chains_for_fine(fine: str) -> List[Tuple[str, int]]:
    """All explicit normal-chain memberships for a fine identity."""
    out: List[Tuple[str, int]] = []
    for chain_name, members in MATURATION_CHAINS.items():
        if fine in members:
            out.append((chain_name, members.index(fine)))
    return out


def primary_chain_for_fine(
    fine_idx: int,
    coarse_idx: Optional[int] = None,
) -> Optional[str]:
    """Pick the branch chain used for scalar maturity readout."""
    fine_name = CANONICAL_CLASSES[fine_idx]
    chains = chains_for_fine(fine_name)
    if not chains:
        return None
    if coarse_idx is not None:
        lineage = LINEAGES[coarse_idx]
        for chain_name, _ in chains:
            if CHAIN_PARENT_LINEAGE.get(chain_name) == lineage:
                return chain_name
    return chains[0][0]


def chain_target_position(chain_name: str, fine_idx: int) -> Optional[float]:
    """Normalized maturity target pi in [0, 1] for a fine class within a branch."""
    fine_name = CANONICAL_CLASSES[fine_idx]
    members = MATURATION_CHAINS.get(chain_name)
    if members is None or fine_name not in members:
        return None
    pos = members.index(fine_name)
    denom = max(len(members) - 1, 1)
    return pos / denom


def assemble_chain_mat_pred(
    chain_preds: Dict[str, torch.Tensor],
    fine_labels: torch.Tensor,
    coarse_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather branch-head outputs using true fine (and optional coarse) labels."""
    device = fine_labels.device
    sample = next(iter(chain_preds.values()))
    out = torch.zeros(fine_labels.shape[0], device=device, dtype=sample.dtype)
    for i, fine_idx in enumerate(fine_labels.tolist()):
        coarse_idx = (
            int(coarse_labels[i].item()) if coarse_labels is not None else None
        )
        chain_name = primary_chain_for_fine(int(fine_idx), coarse_idx)
        if chain_name is not None and chain_name in chain_preds:
            out[i] = chain_preds[chain_name][i]
    return out


def _pairwise_tree_distance(fine_a: int, fine_b: int) -> float:
    fine_index_distance
    return fine_index_distance(fine_a, fine_b)


@lru_cache(maxsize=1)
def fine_tree_distance_matrix() -> "np.ndarray":
    build_distance_matrix
    return build_distance_matrix()


def __getattr__(name: str):
    if name == "FINE_TREE_DISTANCE":
        return fine_tree_distance_matrix()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _rebuild_derived_taxonomy() -> None:
    """Recompute index maps and tensors after CANONICAL_CLASSES / chains change."""
    global FINE_TO_IDX, LINEAGE_TO_IDX, FINE_TO_COARSE, FINE_IDX_TO_COARSE_IDX
    global MATURATION_TABLE, MAPPING_MATRIX, LINEAGES_WITH_MATURATION
    global _FINE_CHAIN_NAME, _FINE_CHAIN_POS

    FINE_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
    LINEAGE_TO_IDX = {c: i for i, c in enumerate(LINEAGES)}
    FINE_TO_COARSE = {}
    for lineage, members in LINEAGE_HIERARCHY.items():
        for cls in members:
            FINE_TO_COARSE[cls] = lineage
    FINE_IDX_TO_COARSE_IDX = [
        LINEAGE_TO_IDX[FINE_TO_COARSE[c]] for c in CANONICAL_CLASSES
    ]
    MATURATION_TABLE = [fine_to_maturation_position(c) for c in CANONICAL_CLASSES]
    MAPPING_MATRIX = build_mapping_matrix()

    _FINE_CHAIN_NAME = []
    _FINE_CHAIN_POS = []
    for _fine_name in CANONICAL_CLASSES:
        _found_name, _found_pos = None, -1
        for _cn, _members in MATURATION_CHAINS.items():
            if _fine_name in _members:
                _found_name, _found_pos = _cn, _members.index(_fine_name)
                break
        _FINE_CHAIN_NAME.append(_found_name)
        _FINE_CHAIN_POS.append(_found_pos)
    LINEAGES_WITH_MATURATION = lineages_with_maturation_heads()


def apply_labels_v2_taxonomy() -> bool:
    """Load taxonomy from labels_v2.json when configured. Returns True if applied."""
    try:
        if not is_v2_labels():
            return False
        tax = build_taxonomy()
    except Exception:
        return False

    global CANONICAL_CLASSES, LINEAGE_HIERARCHY, MATURATION_CHAINS, CHAIN_PARENT_LINEAGE
    global LINEAGES, MATURITY_VARIANT_OF, RARE_CLASSES

    CANONICAL_CLASSES = list(tax["canonical_classes"])
    LINEAGE_HIERARCHY = dict(tax["lineage_hierarchy"])
    MATURATION_CHAINS = dict(tax["maturation_chains"])
    CHAIN_PARENT_LINEAGE = dict(tax["chain_parent_lineage"])
    LINEAGES = list(LINEAGE_HIERARCHY.keys())

    meta = tax["metadata"]
    parent_of = tax["parent_map"]
    MATURITY_VARIANT_OF = {}
    for cid in tax["off_chain_identities"]:
        par = parent_of.get(cid)
        if par and par in meta and meta[par].get("biologic_status") in (
            "healthy_normal", "healthy_or_non_neoplastic",
        ):
            MATURITY_VARIANT_OF[cid] = par

    RARE_CLASSES = [
        c for c in CANONICAL_CLASSES
        if c in tax["off_chain_identities"]
        or c.endswith("_atypical")
        or "dysplastic" in c
        or c in ("blast_unspecified", "myeloblast", "megakaryoblast", "monoblast")
    ]
    _rebuild_derived_taxonomy()
    return True


apply_labels_v2_taxonomy()


def _maybe_split_granulocytic() -> None:
    """Env-gated (``HEMAHIER_SPLIT_GRAN=1``): split the dominant granulocytic
    coarse node into neutrophil / eosinophil / basophil sub-lineages plus a
    shared-precursor node, so the coarse head has real discriminative work on
    granulocyte-heavy corpora. Default off; leaves every other
    dataset's behaviour unchanged."""
    import os
    if os.environ.get("HEMAHIER_SPLIT_GRAN", "0") != "1":
        return
    global LINEAGE_HIERARCHY, LINEAGES
    gran = LINEAGE_HIERARCHY.get("granulocytic")
    if not gran:
        return
    sub = {"gran_precursor": [], "neutrophil_lin": [], "eosinophil_lin": [], "basophil_lin": []}
    for cls in gran:
        chset = {c for c, _ in chains_for_fine(cls)}
        if len(chset) != 1:          # off-chain or shared precursor (myeloblast/promyelocyte)
            sub["gran_precursor"].append(cls)
        elif "neutrophil" in chset:
            sub["neutrophil_lin"].append(cls)
        elif "eosinophil" in chset:
            sub["eosinophil_lin"].append(cls)
        elif "basophil" in chset:
            sub["basophil_lin"].append(cls)
        else:
            sub["gran_precursor"].append(cls)
    new_h = {k: v for k, v in LINEAGE_HIERARCHY.items() if k != "granulocytic"}
    for k, v in sub.items():
        if v:
            new_h[k] = v
    LINEAGE_HIERARCHY = new_h
    LINEAGES = list(LINEAGE_HIERARCHY.keys())
    _rebuild_derived_taxonomy()
    # Repoint any off-chain lineage entries that still name the old merged node.
    for _name, _lin in list(OFF_CHAIN_LINEAGE.items()):
        if _lin == "granulocytic":
            OFF_CHAIN_LINEAGE[_name] = FINE_TO_COARSE.get(_name, "gran_precursor")


def _maybe_restrict_to_dataset_classes() -> None:
    """Env-gated (``HEMAHIER_DATASET_CLASSES=<path to splits/<ds>/summary.json>``):
    restrict the taxonomy to the classes one dataset actually contains.

    The datasets assign *compacted* label indices over the classes they hold,
    while every tensor derived here is indexed against the full
    ``CANONICAL_CLASSES``.  Nothing re-indexes one onto the other, so for a
    dataset that is missing any canonical class, fine index *i* means
    ``dataset_class_list[i]`` in the data but ``CANONICAL_CLASSES[i]`` in the
    hierarchy -- and the lineage head, the coupled posterior and the maturity
    lookup are all supervised through that shifted map.

    Restricting the taxonomy to the dataset's own class list removes the gaps
    and makes index *i* mean the same class on both sides.  The dataset order
    must already be the canonical order with absent classes dropped; we assert
    that rather than reordering, so a mismatch fails loudly instead of silently
    producing a different misalignment.

    Default off, so every existing run reproduces unchanged.
    """
    import os
    path = os.environ.get("HEMAHIER_DATASET_CLASSES", "")
    if not path:
        return

    global CANONICAL_CLASSES, LINEAGE_HIERARCHY, MATURATION_CHAINS
    global RARE_CLASSES, MATURITY_VARIANT_OF

    with open(path) as fh:
        class_list = list(json.load(fh)["class_list"])
    keep = set(class_list)

    filtered = [c for c in CANONICAL_CLASSES if c in keep]
    if filtered != class_list:
        raise ValueError(
            "HEMAHIER_DATASET_CLASSES: dataset class order is not the canonical "
            f"order with absent classes dropped ({path}). Refusing to guess."
        )

    CANONICAL_CLASSES = filtered
    LINEAGE_HIERARCHY = {
        lin: [c for c in members if c in keep]
        for lin, members in LINEAGE_HIERARCHY.items()
    }
    # LINEAGES (and therefore K_coarse) is deliberately left alone, so an empty
    # lineage stays a never-predicted column rather than changing the head size.
    MATURATION_CHAINS = {
        q: [c for c in members if c in keep]
        for q, members in MATURATION_CHAINS.items()
    }
    MATURATION_CHAINS = {q: m for q, m in MATURATION_CHAINS.items() if len(m) >= 2}
    RARE_CLASSES = [c for c in RARE_CLASSES if c in keep]
    MATURITY_VARIANT_OF = {
        k: v for k, v in MATURITY_VARIANT_OF.items() if k in keep and v in keep
    }
    _rebuild_derived_taxonomy()


# --- config helpers (configs/data.yaml ontology section) -------------------


from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Set

from config.settings import cfg_ontology


@lru_cache(maxsize=1)
def load_ontology_config() -> dict:
    return cfg_ontology()


def canonical_aliases(config: Optional[dict] = None) -> Dict[str, str]:
    return dict((config or load_ontology_config()).get("canonical_aliases", {}))


def training_excluded_canonical(config: Optional[dict] = None) -> FrozenSet[str]:
    return frozenset((config or load_ontology_config()).get("training_excluded_canonical", []))


def artifact_canonical(config: Optional[dict] = None) -> FrozenSet[str]:
    return frozenset((config or load_ontology_config()).get("artifact_canonical", []))


def unhealthy_dysplastic(config: Optional[dict] = None) -> Dict[str, dict]:
    """Dysplastic / atypical cells with lineage anchor (off-tree but linked)."""
    unhealthy = (config or load_ontology_config()).get("unhealthy") or {}
    return dict(unhealthy.get("dysplastic") or {})


def unhealthy_pathologic(config: Optional[dict] = None) -> Dict[str, dict]:
    """Broad pathologic cells without a fixed tree position (distance-only)."""
    unhealthy = (config or load_ontology_config()).get("unhealthy") or {}
    return dict(unhealthy.get("pathologic") or {})


def off_chain_identities(config: Optional[dict] = None) -> Set[str]:
    try:
        # inlined
        if is_v2_labels():
            return set(build_taxonomy()["off_chain_identities"])
    except Exception:
        pass
    return set((config or load_ontology_config()).get("off_chain_identities", []))


def off_chain_lineage(config: Optional[dict] = None) -> Dict[str, str]:
    try:
        # inlined
        if is_v2_labels():
            return dict(build_taxonomy()["off_chain_lineage"])
    except Exception:
        pass
    return dict((config or load_ontology_config()).get("off_chain_lineage", {}))


def eval_distance_params(config: Optional[dict] = None) -> dict:
    return dict((config or load_ontology_config()).get("eval_distance", {}))


def progenitor_config(config: Optional[dict] = None) -> dict:
    """Shared blast root (blast_unspecified) and granulocytic entry (myeloblast)."""
    return dict((config or load_ontology_config()).get("progenitor") or {})


def eval_distance_value(key: str, default: float, config: Optional[dict] = None) -> float:
    return float(eval_distance_params(config).get(key, default))


def branch_chains(config: Optional[dict] = None) -> Dict[str, List[str]]:
    """Branch name → ordered healthy maturation chain (from configs/data.yaml branches)."""
    branches = (config or load_ontology_config()).get("branches") or {}
    return {
        name: list(meta.get("chain") or [])
        for name, meta in branches.items()
        if isinstance(meta, dict)
    }

_cfg = load_ontology_config()
OFF_CHAIN_IDENTITIES: Set[str] = off_chain_identities()
OFF_CHAIN_LINEAGE: Dict[str, str] = off_chain_lineage()

_maybe_split_granulocytic()  # env-gated finer granulocytic taxonomy (after OFF_CHAIN_LINEAGE)
_maybe_restrict_to_dataset_classes()  # env-gated per-dataset index alignment (must follow the above)

D_CROSS_LINEAGE = eval_distance_value("cross_lineage", 1.0)
D_OFF_CHAIN_TO_HEALTHY = eval_distance_value("off_chain_to_healthy", 0.25)
D_OFF_CHAIN_INTRA = eval_distance_value("off_chain_intra", 0.10)
D_MAX_SAME_CHAIN = eval_distance_value("max_same_chain", 0.85)
D_SAME_LINEAGE_BRANCH = eval_distance_value("same_lineage_branch", 0.75)

# --- evaluation distances --------------------------------------------------


from functools import lru_cache
from typing import Dict, List, Optional, Set

import numpy as np



def is_off_chain(name: str) -> bool:
    return name in OFF_CHAIN_IDENTITIES


def _same_chain_gap(name_a: str, name_b: str) -> Optional[float]:
    best: Optional[float] = None
    for chain_a, pos_a in chains_for_fine(name_a):
        for chain_b, pos_b in chains_for_fine(name_b):
            if chain_a != chain_b:
                continue
            denom = max(len(MATURATION_CHAINS[chain_a]) - 1, 1)
            d = abs(pos_a - pos_b) / denom
            if best is None or d < best:
                best = d
    return best


def _off_chain_lineage(name: str) -> Optional[str]:
    if name in OFF_CHAIN_LINEAGE:
        return OFF_CHAIN_LINEAGE[name]
    return FINE_TO_COARSE.get(name)


def pairwise_distance(name_a: str, name_b: str) -> float:
    """Maturity-aware tree distance in [0, 1] between two canonical classes.

    A same-chain confusion costs the normalized maturity gap between the two
    stages, a same-lineage off-chain confusion an intermediate constant, and a
    cross-lineage confusion the maximum. Used to measure error severity only,
    never as a training loss.
    """
    return v2_pairwise_distance(name_a, name_b, eval_params=eval_distance_params())


def build_distance_matrix(classes: Optional[List[str]] = None) -> np.ndarray:
    names = classes or CANONICAL_CLASSES
    n = len(names)
    d = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            d[i, j] = pairwise_distance(a, b)
    return d


def fine_index_distance(fine_a: int, fine_b: int, dataset: Optional[str] = None) -> float:
    """`dataset` is accepted for call-site compatibility; the ontology defines
    one tree, so the distance does not depend on it."""
    return pairwise_distance(CANONICAL_CLASSES[fine_a], CANONICAL_CLASSES[fine_b])


@lru_cache(maxsize=1)
def maturity_chain_mask() -> np.ndarray:
    """Boolean mask: True where both indices share a healthy maturation chain (D_maturity)."""
    n = len(CANONICAL_CLASSES)
    m = np.zeros((n, n), dtype=bool)
    for i, a in enumerate(CANONICAL_CLASSES):
        for j, b in enumerate(CANONICAL_CLASSES):
            if i == j:
                m[i, j] = True
                continue
            for ca, _ in chains_for_fine(a):
                for cb, _ in chains_for_fine(b):
                    if ca == cb:
                        m[i, j] = True
                        break
    return m
