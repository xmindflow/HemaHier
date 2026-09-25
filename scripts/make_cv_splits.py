#!/usr/bin/env python3
"""Build the 5-fold CV splits and check them against the paper's digests.

The folds are a deterministic function of the images on disk and the split seed,
so they are rebuilt locally rather than shipped. After writing them this script
hashes each fold's file list and compares it with `splits/<dataset>/CHECKSUMS.json`;
a mismatch means your copy of the dataset differs from the one used in the paper.

Writes under splits/<dataset>/: config.json, manifest.json, summary.json and
fold_0.json .. fold_4.json.

Example:
  export MLLV1_ROOT=/path/to/Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer_v1
  PYTHONPATH=src python scripts/make_cv_splits.py mllv1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.settings import DEFAULT_DATA_CONFIG_PATH
from data.cv_splits import DEFAULT_SPLITS_DIR, generate_cv_splits, load_dataset_config
from data.datasets import DATASETS


def fold_digests(splits_dir: Path) -> dict:
    """sha256 over each fold's ordered list of dataset-relative paths."""
    out: dict = {}
    for fold_path in sorted(splits_dir.glob("fold_*.json")):
        data = json.loads(fold_path.read_text())
        entry = {}
        for split in ("train", "val"):
            rel = [item["rel_path"] for item in data[split]["items"]]
            entry[split] = {
                "n": len(rel),
                "sha256": hashlib.sha256("\n".join(rel).encode()).hexdigest(),
            }
        out[fold_path.stem] = entry
    return out


def check_against_reference(splits_dir: Path) -> bool:
    """Compare the folds just written with the reference digests. True if equal."""
    reference_path = splits_dir / "CHECKSUMS.json"
    if not reference_path.exists():
        print(f"[warn] no {reference_path.name}; skipping the check")
        return True

    reference = json.loads(reference_path.read_text())["folds"]
    digests = fold_digests(splits_dir)
    ok = True
    for fold, expected in sorted(reference.items()):
        for split in ("train", "val"):
            got = digests.get(fold, {}).get(split)
            if got == expected[split]:
                print(f"  OK  {fold}/{split}  n={got['n']}")
            else:
                ok = False
                print(f"  MISMATCH {fold}/{split}: expected {expected[split]}, got {got}")
    return ok


def resolve_root(cfg: dict) -> str | None:
    env = cfg.get("root_env")
    return os.environ.get(env) if env and os.environ.get(env) else cfg.get("root")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=sorted(DATASETS.keys()), help="Dataset name (mllv1).")
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    p.add_argument("--overwrite", action="store_true", help="Rebuild existing splits.")
    p.add_argument("--write-checksums", action="store_true",
                   help="Record the digests as the new reference instead of checking them.")
    args = p.parse_args()

    cfg = load_dataset_config(args.dataset)
    root = resolve_root(cfg)
    if not root or not Path(root).is_dir():
        raise SystemExit(
            f"{args.dataset}: set {cfg.get('root_env')} to the image root (got {root!r})."
        )

    print(f"=== {args.dataset}  root={root} ===")
    splits_dir = args.splits_dir / args.dataset
    folds_exist = any(splits_dir.glob("fold_*.json"))

    if folds_exist and not args.overwrite and not args.write_checksums:
        # Already built: just verify against the paper digests (README path).
        print(f"Folds already in {splits_dir}/; checking digests (pass --overwrite to rebuild).")
        print("\nChecking against the folds used in the paper:")
        if check_against_reference(splits_dir):
            print("\nSplits match the reference.")
            return
        raise SystemExit(
            "\nSplits differ from the reference: your copy of the dataset is not the "
            "one used in the paper, and the numbers will not be comparable."
        )

    summary = generate_cv_splits(
        args.dataset,
        root=root,
        config=cfg,
        config_path=DEFAULT_DATA_CONFIG_PATH,
        splits_root=args.splits_dir,
        overwrite=args.overwrite,
    )
    print(f"Wrote {summary['n_folds']} folds over {summary['n_samples']} cells "
          f"to {splits_dir}/")

    if args.write_checksums:
        payload = {
            "dataset": args.dataset,
            "seed": summary.get("seed"),
            "n_samples": summary.get("n_samples"),
            "n_classes": summary.get("n_classes"),
            "class_list": summary.get("class_list"),
            "folds": fold_digests(splits_dir),
        }
        (splits_dir / "CHECKSUMS.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {splits_dir / 'CHECKSUMS.json'}")
        return

    print("\nChecking against the folds used in the paper:")
    if check_against_reference(splits_dir):
        print("\nSplits match the reference.")
    else:
        raise SystemExit(
            "\nSplits differ from the reference: your copy of the dataset is not the "
            "one used in the paper, and the numbers will not be comparable."
        )


if __name__ == "__main__":
    main()
