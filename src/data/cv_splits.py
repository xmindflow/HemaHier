"""
5-fold stratified cross-validation splits, one dataset at a time.

Files written under  splits/<dataset>/
  config.json      : snapshot of the dataset config used to build splits
  manifest.json    : all kept samples (before fold assignment)
  fold_0.json … fold_4.json
  summary.json     : per-fold train/val sizes and class lists
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.datasets import SampleRecord, collect_dataset_samples, drop_unreadable_samples, output_class_order
from config.settings import DEFAULT_CONFIG_PATH, REPO_ROOT, cfg_paths, dataset_config as settings_dataset_config
from tools.reproducibility import set_global_seed

_paths = cfg_paths()
DEFAULT_SPLITS_DIR = REPO_ROOT / _paths.get("splits_dir", "splits")


def load_dataset_config(
    dataset_name: str,
    config_path: Optional[Path] = None,
) -> dict:
    if config_path is not None:
        raise ValueError(
            "Per-dataset JSON configs were removed; edit configs/config.yaml "
            f"datasets.{dataset_name} instead."
        )
    unified = settings_dataset_config(dataset_name)
    if unified and unified.get("dataset") == dataset_name:
        return dict(unified)
    fallback = DEFAULT_SPLITS_DIR / dataset_name / "config.json"
    if fallback.exists():
        cfg = json.loads(fallback.read_text())
        if cfg.get("dataset") == dataset_name:
            return cfg
    raise FileNotFoundError(
        f"No config for dataset {dataset_name!r} in configs/config.yaml "
        f"and no snapshot at {fallback}"
    )


def splits_dir_for(dataset_name: str, splits_root: Optional[Path] = None) -> Path:
    return (splits_root or DEFAULT_SPLITS_DIR) / dataset_name


def fold_path(
    dataset_name: str,
    fold: int,
    splits_root: Optional[Path] = None,
) -> Path:
    return splits_dir_for(dataset_name, splits_root) / f"fold_{fold}.json"


def load_fold_split(
    dataset_name: str,
    fold: int,
    splits_root: Optional[Path] = None,
) -> dict:
    path = fold_path(dataset_name, fold, splits_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/make_cv_splits.py {dataset_name}"
        )
    return json.loads(path.read_text())


def _class_counts(records: List[SampleRecord]) -> Dict[str, int]:
    return dict(sorted(Counter(r.canonical for r in records).items()))


def _records_to_items(records: List[SampleRecord]) -> List[dict]:
    return [r.to_dict() for r in records]


def _split_cfg(cfg: dict) -> dict:
    """Flatten datasets.<name>.cv fields into the split-generation config."""
    out = dict(cfg)
    cv = out.pop("cv", None) or {}
    for key, val in cv.items():
        out.setdefault(key, val)
    return out


def _per_class_kfold_indices(
    samples: List[SampleRecord],
    n_folds: int,
    seed: int,
) -> List[Tuple[List[int], List[int]]]:
    """
    Disjoint K-fold CV with per-class ~80/20 train/val in each fold.

    Each sample appears in validation exactly once across all folds.  Within
    each fold, every class contributes floor(n_c*(n_folds-1)/n_folds) train and
    ceil(n_c/n_folds) val samples (80/20 when n_folds=5).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, rec in enumerate(samples):
        by_class[int(rec.fine_idx)].append(i)

    fold_val: List[List[int]] = [[] for _ in range(n_folds)]
    rng = np.random.RandomState(seed)
    for cls_idxs in by_class.values():
        arr = np.asarray(cls_idxs, dtype=int)
        rng.shuffle(arr)
        chunks = np.array_split(arr, n_folds)
        for fold_idx, chunk in enumerate(chunks):
            fold_val[fold_idx].extend(int(i) for i in chunk.tolist())

    seen_val: set[int] = set()
    folds: List[Tuple[List[int], List[int]]] = []
    n = len(samples)
    for fold_idx in range(n_folds):
        val_idx = sorted(set(fold_val[fold_idx]))
        overlap = seen_val.intersection(val_idx)
        if overlap:
            raise RuntimeError(
                f"Fold {fold_idx} reuses validation indices: {sorted(overlap)[:5]}..."
            )
        seen_val.update(val_idx)
        val_set = set(val_idx)
        train_idx = [i for i in range(n) if i not in val_set]
        folds.append((train_idx, val_idx))

    if len(seen_val) != n:
        missing = set(range(n)) - seen_val
        raise RuntimeError(
            f"Not all samples assigned to a validation fold ({len(missing)} missing)"
        )
    return folds


def _per_class_split_ratios(
    samples: List[SampleRecord],
    train_idx: List[int],
    val_idx: List[int],
) -> Dict[str, Dict[str, float]]:
    """Per-class train/val counts and val fraction for one fold."""
    train_counts = Counter(samples[i].canonical for i in train_idx)
    val_counts = Counter(samples[i].canonical for i in val_idx)
    ratios: Dict[str, Dict[str, float]] = {}
    for cls in sorted(set(train_counts) | set(val_counts)):
        tr = train_counts.get(cls, 0)
        va = val_counts.get(cls, 0)
        total = tr + va
        ratios[cls] = {
            "train": tr,
            "val": va,
            "val_fraction": round(va / total, 4) if total else 0.0,
        }
    return ratios


def generate_cv_splits(
    dataset_name: str,
    root: Optional[str] = None,
    config: Optional[dict] = None,
    config_path: Optional[Path] = None,
    splits_root: Optional[Path] = None,
    overwrite: bool = False,
) -> dict:
    """Build stratified K-fold splits and write JSON files."""
    cfg = _split_cfg(config or load_dataset_config(dataset_name, config_path))
    n_folds = int(cfg.get("n_folds", 5))
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    min_n = int(cfg.get("min_samples_per_class", 5))
    bio = cfg.get("biologic_filter", "hematopoietic")
    use_source = bool(cfg.get("use_source_labels", False))

    out_dir = splits_dir_for(dataset_name, splits_root)
    if out_dir.exists() and not overwrite:
        existing = list(out_dir.glob("fold_*.json"))
        if existing:
            raise FileExistsError(
                f"Splits already exist in {out_dir} "
                f"({len(existing)} folds). Pass overwrite=True to rebuild."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, manifest_summary = collect_dataset_samples(
        dataset_name,
        root=root or cfg.get("root"),
        min_samples_per_class=min_n,
        biologic_filter=bio,
        exclude_artifacts=cfg.get("exclude_artifacts", True),
        use_source_labels=use_source,
    )
    if len(samples) < n_folds:
        raise ValueError(f"Only {len(samples)} samples, cannot make {n_folds} folds.")
    if manifest_summary["n_classes"] < 2:
        raise ValueError("Need at least 2 classes after filtering.")

    if cfg.get("validate_images", False):
        samples, unreadable = drop_unreadable_samples(samples)
        if unreadable:
            manifest_summary = {
                **manifest_summary,
                "n_kept": len(samples),
                "n_classes": len({r.canonical for r in samples}),
                "dropped_unreadable": unreadable,
            }

    from data.ontology import CANONICAL_CLASSES

    if use_source:
        class_list = sorted({r.canonical for r in samples})
        class_order = "source_labels"
        n_output = len(class_list)
    else:
        kept = {r.canonical for r in samples}
        include_artifacts = not cfg.get("exclude_artifacts", True)
        class_list = output_class_order(kept, include_artifacts=include_artifacts)
        has_artifacts = bool(set(class_list) - set(CANONICAL_CLASSES))
        class_order = "CANONICAL_CLASSES+artifacts" if has_artifacts else "CANONICAL_CLASSES"
        n_output = len(class_list)
        if len(class_list) != manifest_summary["n_classes"]:
            raise ValueError(
                f"Class list size mismatch "
                f"({len(class_list)} vs {manifest_summary['n_classes']})"
            )

    counts = Counter(r.canonical for r in samples)
    too_few = {c: n for c, n in counts.items() if n < n_folds}
    if too_few:
        raise ValueError(
            f"These classes have < {n_folds} samples after min_samples filter: {too_few}"
        )

    fold_assignments = _per_class_kfold_indices(samples, n_folds, seed)

    # Save config snapshot + manifest.
    cfg_snapshot = {**cfg, "generated_at": datetime.now(timezone.utc).isoformat()}
    (out_dir / "config.json").write_text(json.dumps(cfg_snapshot, indent=2))

    manifest = {
        "summary": manifest_summary,
        "class_list": class_list,
        "class_order": class_order,
        "n_output_classes": n_output,
        "samples": _records_to_items(samples),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    fold_summaries = []
    for fold_idx, (train_idx, val_idx) in enumerate(fold_assignments):
        train_recs = [samples[i] for i in train_idx]
        val_recs = [samples[i] for i in val_idx]
        class_ratios = _per_class_split_ratios(samples, train_idx, val_idx)

        fold_doc = {
            "dataset": dataset_name,
            "fold": fold_idx,
            "n_folds": n_folds,
            "seed": seed,
            "config_path": str(config_path or DEFAULT_DATA_CONFIG_PATH),
            "root": manifest_summary["root"],
            "classes": class_list,
            "class_order": class_order,
            "n_output_classes": n_output,
            "train": {
                "n": len(train_recs),
                "class_counts": _class_counts(train_recs),
                "items": _records_to_items(train_recs),
            },
            "val": {
                "n": len(val_recs),
                "class_counts": _class_counts(val_recs),
                "items": _records_to_items(val_recs),
            },
            "per_class_split": class_ratios,
        }
        (out_dir / f"fold_{fold_idx}.json").write_text(json.dumps(fold_doc, indent=2))
        fold_summaries.append({
            "fold": fold_idx,
            "train_n": len(train_recs),
            "val_n": len(val_recs),
            "train_classes": len(fold_doc["train"]["class_counts"]),
            "val_classes": len(fold_doc["val"]["class_counts"]),
            "mean_val_fraction": round(
                float(np.mean([v["val_fraction"] for v in class_ratios.values()])), 4,
            ),
        })

    summary = {
        "dataset": dataset_name,
        "n_folds": n_folds,
        "seed": seed,
        "n_samples": len(samples),
        "n_classes": manifest_summary["n_classes"],
        "class_list": class_list,
        "class_order": class_order,
        "n_output_classes": n_output,
        "dropped_classes": manifest_summary["dropped_classes"],
        "folds": fold_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
