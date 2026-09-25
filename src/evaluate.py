#!/usr/bin/env python3
"""Evaluate a finished run directory: no training, no re-fitting.

A run directory is the leaf that ``train.py`` writes:
``<output_dir>/<exp_id>/{config.json, best_checkpoint.pt, history.json}``.
This script rebuilds the model from that saved ``config.json``, loads the
checkpoint, rebuilds the validation loader for the fold the run was trained on,
and recomputes every metric the paper reports.

It is the reproduction entry point: the recomputed numbers must match the
``val`` block stored inside the checkpoint (the epoch selected by validation
macro-F1), which is what the paper tables aggregate.

Usage:
  PYTHONPATH=src python src/evaluate.py runs/mllv1_hemahier/fold0/HemaHier \\
      --dataset_root "$MLLV1_ROOT"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from config.settings import EXP_CONFIGS
from config.model_factory import build_model_and_criterion
from data.datasets import DATASETS, build_cv_dataloaders
from evaluation.metrics import evaluate as evaluate_model
from tools.reproducibility import set_global_seed

# Metrics quoted in the paper's main table, in table order.
PAPER_METRICS = (
    "fine_accuracy",
    "fine_balanced_accuracy",
    "fine_macro_f1",
    "fine_weighted_f1",
    "fine_rare_macro_f1",
    "clinical_cross_lineage",
    "tree_distance_error",
    "maturity_spearman",
)


def find_run_dirs(root: Path) -> list[Path]:
    """Every leaf run directory (holding a checkpoint) at or below ``root``."""
    if (root / "best_checkpoint.pt").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("best_checkpoint.pt"))


def resolve_decode_mode(cfg: dict, saved: dict) -> str:
    """Reproduce train.py's decode-mode resolution."""
    mode = saved.get("decode_mode")
    if mode is None:
        mode = cfg.get("default_decode_mode") or None
    if mode is None:
        if cfg.get("hemahier_version") == "oc":
            mode = "joint"
        else:
            mode = "bayes" if cfg.get("use_bayes_eval") else "argmax"
    return mode


def evaluate_run(
    run_dir: Path,
    dataset_root: str | None = None,
    splits_dir: str | None = None,
    num_workers: int = 8,
    batch_size: int | None = None,
    decode_mode: str | None = None,
    device: torch.device | None = None,
) -> dict:
    """Recompute validation metrics for one finished run directory."""
    saved = json.loads((run_dir / "config.json").read_text())
    exp_id = saved["exp_id"]
    # train.py passes the *registry* config (not the arg-merged one) to the model
    # factory, so collisions such as `pooling` resolve the same way here.
    cfg = EXP_CONFIGS[exp_id]
    args = SimpleNamespace(**saved)
    if batch_size is not None:
        args.batch_size = batch_size
    args.num_workers = num_workers

    ds_name = saved.get("single_dataset")
    if ds_name is None:
        raise SystemExit(
            f"{run_dir}: only single-dataset CV runs can be re-evaluated "
            "(config.json has no `single_dataset`)."
        )
    fold = saved.get("fold")
    if fold is None:
        raise SystemExit(f"{run_dir}: config.json has no `fold`.")

    root = (
        dataset_root
        or saved.get(f"{ds_name}_root")
        or os.environ.get(DATASETS[ds_name].path_env_var)
    )
    if not root:
        raise SystemExit(
            f"{run_dir}: no root for {ds_name}; pass --dataset_root or set "
            f"{DATASETS[ds_name].path_env_var}."
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(int(saved.get("seed", 42)))

    _, val_loader, train_ds, _ = build_cv_dataloaders(
        ds_name,
        fold=int(fold),
        dataset_root=root,
        splits_dir=splits_dir or saved.get("splits_dir"),
        batch_size=args.batch_size,
        img_size=int(saved.get("img_size", 224)),
        num_workers=num_workers,
        preload_images=False,
        seed=int(saved.get("seed", 42)),
    )
    args.num_fine_classes = train_ds.num_fine_classes

    model, _ = build_model_and_criterion(args, cfg, device, n_epochs=1)
    ckpt = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mode = decode_mode or resolve_decode_mode(cfg, saved)
    with torch.no_grad():
        metrics = evaluate_model(
            model, val_loader, device,
            decode_mode=mode,
            return_extended=True,
            dataset_name=ds_name,
        )

    return {
        "run_dir": str(run_dir),
        "exp_id": exp_id,
        "dataset": ds_name,
        "fold": int(fold),
        "seed": int(saved.get("seed", 42)),
        "lora_rank": int(saved.get("lora_rank", 0)),
        "decode_mode": mode,
        "checkpoint_epoch": ckpt.get("epoch"),
        "recomputed": metrics,
        "checkpoint_val": ckpt.get("val", {}),
    }


def report(result: dict, tolerance: float) -> bool:
    """Print a recomputed-vs-stored comparison. Returns True if all match."""
    rec, stored = result["recomputed"], result["checkpoint_val"]
    print(f"\n{result['run_dir']}  (fold {result['fold']}, "
          f"epoch {result['checkpoint_epoch']}, decode={result['decode_mode']})")
    ok = True
    for key in PAPER_METRICS:
        r = rec.get(key)
        s = stored.get(key)
        if r is None:
            continue
        if s is None:
            print(f"  {key:26s} {r:8.4f}   (not stored)")
            continue
        delta = abs(r - s)
        flag = "" if delta <= tolerance else "  <-- MISMATCH"
        ok = ok and delta <= tolerance
        print(f"  {key:26s} {r:8.4f}  stored {s:8.4f}  Δ{delta:.5f}{flag}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path,
                   help="Run directory (leaf with best_checkpoint.pt), or a root with --recurse.")
    p.add_argument("--recurse", action="store_true",
                   help="Evaluate every run directory found beneath run_dir.")
    p.add_argument("--dataset_root", default=None,
                   help="Image root; defaults to the run's config or the dataset env var.")
    p.add_argument("--splits_dir", default=None, help="Override splits/ root.")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--decode_mode", default=None,
                   help="Override the run's decoder (argmax, bayes, joint, ...).")
    p.add_argument("--json", type=Path, default=None, help="Write results here.")
    p.add_argument("--tolerance", type=float, default=1e-4,
                   help="Max |recomputed - stored| before a metric counts as a mismatch.")
    args = p.parse_args()

    run_dirs = find_run_dirs(args.run_dir) if args.recurse else [args.run_dir]
    if not run_dirs:
        raise SystemExit(f"No run directories under {args.run_dir}")

    results, all_ok = [], True
    for run_dir in run_dirs:
        result = evaluate_run(
            run_dir,
            dataset_root=args.dataset_root,
            splits_dir=args.splits_dir,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            decode_mode=args.decode_mode,
        )
        results.append(result)
        all_ok &= report(result, args.tolerance)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json}")

    print("\nAll metrics match the stored checkpoint values."
          if all_ok else "\nSome metrics differ from the stored values (see MISMATCH above).")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
