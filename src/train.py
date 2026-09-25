"""Train HemaHier or the flat linear probe on one MLLv1 fold.

  PYTHONPATH=src python src/train.py --exp_id HemaHier --mllv1_root $MLLV1_ROOT \\
      --fold 0 --output_dir runs/mllv1_hemahier/fold0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.optim as optim

try:
    from torch.amp import GradScaler
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler
    _AMP_DEVICE = None


def _make_grad_scaler():
    if _AMP_DEVICE is not None:
        return GradScaler(_AMP_DEVICE)
    return GradScaler()

REPO_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_SRC))

from config.model_factory import build_model_and_criterion
from config.settings import EXP_CONFIGS, cfg_model, cfg_training
from data.datasets import DATASETS, build_cv_dataloaders
from data.ontology import CANONICAL_CLASSES, LINEAGES
from evaluation.metrics import evaluate
from tools.reproducibility import set_global_seed
from tools.run_metadata import build_run_metadata, write_run_metadata
from training.loop import train_one_epoch
from training.schedule import cosine_lr

TRAIN_IMPLEMENTATION_VERSION = 5

_TR = cfg_training()
_MD = cfg_model()


def parse_args():
    p = argparse.ArgumentParser(description="Train HemaHier / Flat on one MLLv1 fold.")
    p.add_argument("--exp_id", default="HemaHier", choices=list(EXP_CONFIGS.keys()))
    p.add_argument("--mllv1_root", default=None)
    p.add_argument("--output_dir", default="./runs")
    p.add_argument("--fold", type=int, required=True, help="CV fold 0..4")
    p.add_argument("--splits_dir", default=None)
    p.add_argument("--epochs", type=int, default=_TR.get("epochs", 10))
    p.add_argument("--warmup_epochs", type=int, default=_TR.get("warmup_epochs", 1))
    p.add_argument("--batch_size", type=int, default=_TR.get("batch_size", 64))
    p.add_argument("--lr", type=float, default=_TR.get("lr", 3e-4))
    p.add_argument("--weight_decay", type=float, default=_TR.get("weight_decay", 1e-4))
    p.add_argument("--img_size", type=int, default=_TR.get("img_size", 224))
    p.add_argument("--num_workers", type=int, default=_TR.get("num_workers", 8))
    p.add_argument("--seed", type=int, default=_TR.get("seed", 2026))
    p.add_argument("--backbone", default=_MD.get("backbone", "dinobloom_s"))
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--lora_rank", type=int, default=_MD.get("lora_rank", 0))
    p.add_argument("--lora_alpha", type=float, default=_MD.get("lora_alpha", 128.0))
    p.add_argument("--fine_dim", type=int, default=_MD.get("fine_dim", 256))
    p.add_argument("--coarse_dim", type=int, default=_MD.get("coarse_dim", 512))
    p.add_argument("--decode_mode", default=None,
                   choices=["argmax", "bayes", "joint", "raw", "joint_maturity"])
    p.add_argument("--loss_schedule", default=_TR.get("loss_schedule", "finalshot"))
    p.add_argument("--lr_schedule", default=_TR.get("lr_schedule", "cosine_epoch"))
    p.add_argument("--maturity_target_mode", default=_TR.get("maturity_target_mode", "hard"))
    p.add_argument("--maturity_soft_sigma_stages", type=float,
                   default=_TR.get("maturity_soft_sigma_stages", 1.0))
    p.add_argument("--max_iters", type=int, default=None)
    p.add_argument("--iters_per_epoch", type=int, default=_TR.get("iters_per_epoch", 250))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preload_images", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.single_dataset = "mllv1"
    args.label_smoothing = float(_MD.get("label_smoothing", 0.05))
    cfg = EXP_CONFIGS[args.exp_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_config = {
        **cfg,
        **vars(args),
        "implementation_version": TRAIN_IMPLEMENTATION_VERSION,
        "ontology_revision": "committed_blast_v2",
        "metric_revision": "present_class_f1_rare_quartile_v2",
    }

    print(f"\n{'=' * 70}")
    print(f"  {args.exp_id}: {cfg.get('description', '')}")
    print(f"  Device: {device}   fold={args.fold}")
    print(f"{'=' * 70}\n")

    set_global_seed(args.seed)
    out_dir = Path(args.output_dir) / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    write_run_metadata(
        out_dir / "run_metadata.json",
        build_run_metadata(args.exp_id, cfg, args),
    )

    root = args.mllv1_root or os.environ.get(DATASETS["mllv1"].path_env_var)
    if not root:
        raise SystemExit("Pass --mllv1_root or set MLLV1_ROOT")

    train_loader, val_loader, train_ds, val_ds = build_cv_dataloaders(
        "mllv1",
        fold=args.fold,
        dataset_root=root,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        preload_images=args.preload_images,
        seed=args.seed,
    )

    if args.max_iters is not None:
        iters_per_epoch = max(1, args.iters_per_epoch)
        n_epochs = max(1, math.ceil(args.max_iters / iters_per_epoch))
        warmup_epochs = args.warmup_epochs
        args.iters_per_epoch = iters_per_epoch
    else:
        args.iters_per_epoch = len(train_loader)
        n_epochs = args.epochs
        warmup_epochs = args.warmup_epochs

    if hasattr(train_ds, "num_fine_classes"):
        args.num_fine_classes = train_ds.num_fine_classes
        print(f"Fine classes: {args.num_fine_classes}")

    model, criterion = build_model_and_criterion(args, cfg, device, n_epochs)
    if hasattr(criterion, "set_class_prior") and hasattr(train_ds, "samples"):
        n_fine = getattr(args, "num_fine_classes", None) or getattr(
            train_ds, "num_fine_classes", None)
        if n_fine:
            counts = torch.zeros(int(n_fine))
            for s in train_ds.samples:
                fi = int(s[2])
                if 0 <= fi < counts.numel():
                    counts[fi] += 1
            criterion.set_class_prior(counts)

    if args.freeze_backbone or args.lora_rank > 0:
        enc = model.encoder
        inner = getattr(enc, "inner", enc)
        for name, p in inner.named_parameters():
            if args.lora_rank > 0 and ("lora_a" in name or "lora_b" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False
        mode = "LoRA adapters" if args.lora_rank > 0 else "fully frozen"
        print(f"Backbone ({args.backbone}): {mode}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"params: {n_params:.1f}M ({n_trainable:.1f}M trainable)")
    k_fine = getattr(model, "K_f", getattr(args, "num_fine_classes", len(CANONICAL_CLASSES)))
    print(f"K_fine = {k_fine}   K_coarse = {len(LINEAGES)}")

    params = [p for p in model.parameters() if p.requires_grad]
    params += list(criterion.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: cosine_lr(e, n_epochs, warmup_epochs),
    )
    scaler = _make_grad_scaler()

    decode_mode = args.decode_mode or cfg.get("default_decode_mode") or "argmax"

    best_score, history = -1.0, []
    start_epoch = 1
    last_checkpoint_path = out_dir / "last_checkpoint.pt"
    if args.resume and last_checkpoint_path.exists():
        payload = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        criterion.load_state_dict(payload.get("loss_state", {}))
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state"):
            scaler.load_state_dict(payload["scaler_state"])
        history = payload.get("history", [])
        best_score = float(payload.get("best_score", -1.0))
        start_epoch = int(payload["epoch"]) + 1
        print(f"Resumed at epoch {start_epoch - 1}/{n_epochs} (best={best_score:.4f})")

    steps_per_epoch = args.iters_per_epoch if args.max_iters is None else args.iters_per_epoch
    if args.max_iters is None:
        steps_per_epoch = len(train_loader)

    for epoch in range(start_epoch, n_epochs + 1):
        train_stats, elapsed = train_one_epoch(
            model, iter(train_loader), steps_per_epoch, criterion, optimizer, scaler,
            device, epoch, writer=None, global_step_start=(epoch - 1) * steps_per_epoch,
        )
        scheduler.step()
        val = evaluate(
            model, val_loader, device,
            decode_mode=decode_mode,
            return_extended=True,
            dataset_name="mllv1",
        )
        score = val["fine_macro_f1"]
        print(
            f"Epoch {epoch:3d}/{n_epochs}  "
            f"L={train_stats['L_total']:.3f}  "
            f"fineF1={val['fine_macro_f1']:.3f}  wF1={val['fine_weighted_f1']:.3f}  "
            f"rareF1={val['fine_rare_macro_f1']:.3f}  "
            f"({elapsed:.0f}s)"
        )
        history.append({"epoch": epoch, "train": train_stats, "val": val, "score": score})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if score > best_score:
            best_score = score
            torch.save({
                "implementation_version": TRAIN_IMPLEMENTATION_VERSION,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "loss_state": criterion.state_dict(),
                "config": run_config,
                "val": val,
            }, out_dir / "best_checkpoint.pt")
            print(f"  ★ New best fine_macro_f1 = {best_score:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "loss_state": criterion.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "best_score": best_score,
            "config": run_config,
        }, out_dir / "last_checkpoint.pt")

    (out_dir / "completed.json").write_text(json.dumps({
        "epochs": n_epochs,
        "implementation_version": TRAIN_IMPLEMENTATION_VERSION,
        "ontology_revision": run_config["ontology_revision"],
        "metric_revision": run_config["metric_revision"],
    }, indent=2))
    print(f"\nDone. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
