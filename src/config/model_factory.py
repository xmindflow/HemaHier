"""Build the two public methods: Flat linear probe and HemaHier cascade."""

from __future__ import annotations

from loss import FlatCELoss, HemaHierLoss
from models.flat_linear_probe import FlatLinearProbe
from models.hemahier_cascade import HemaHierCascade


def build_model_and_criterion(args, cfg: dict, device, n_epochs: int):
    n_fine = getattr(args, "num_fine_classes", None)
    ls = float(cfg.get("label_smoothing", getattr(args, "label_smoothing", 0.05)))

    if cfg.get("flat_only"):
        model = FlatLinearProbe(
            backbone=args.backbone,
            pretrained=True,
            linear_layers=int(cfg.get("linear_probe_layers", 4)),
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            num_fine_classes=n_fine,
            img_size=getattr(args, "img_size", 224),
        ).to(device)
        return model, FlatCELoss(label_smoothing=ls).to(device)

    if not cfg.get("use_hemahier"):
        raise SystemExit(f"Unknown experiment {getattr(args, 'exp_id', '?')!r}")

    use_mat = cfg.get("use_maturity", True)
    model = HemaHierCascade(
        pretrained=True,
        backbone=args.backbone,
        context_dim=args.fine_dim,
        feature_mode=cfg.get("feature_mode", "probe"),
        feature_layers=int(cfg.get("feature_layers", 4)),
        img_size=getattr(args, "img_size", 224),
        detach_aux_backbone=cfg.get("detach_aux_backbone", False),
        fine_from_maturity=cfg.get("fine_from_maturity", True),
        use_maturity=use_mat,
        joint_posterior=cfg.get("joint_posterior", True),
        chain_cond_dim=int(cfg.get("chain_cond_dim", 32)),
        maturity_hidden_dim=int(cfg.get("maturity_hidden_dim", 64)),
        maturity_activation=cfg.get("maturity_activation", "identity"),
        maturity_eps=float(cfg.get("maturity_eps", 0.05)),
        beta_maturity_decode=float(cfg.get("beta_maturity_decode", 0.0)),
        maturity_decode_sigma=float(cfg.get("maturity_decode_sigma", 0.25)),
        joint_coarse_temperature=float(cfg.get("joint_coarse_temperature", 1.0)),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    ).to(device)
    max_iters = args.max_iters or (n_epochs * args.iters_per_epoch)
    criterion = HemaHierLoss(
        lambda_lineage=float(cfg.get("lambda_lineage", 0.05)),
        lambda_maturity=float(cfg.get("lambda_maturity", 0.005)),
        use_maturity=use_mat,
        use_ranking=cfg.get("use_ranking", True),
        lambda_rank=float(cfg.get("lambda_rank", 0.003)),
        lambda_raw_fine_aux=float(cfg.get("lambda_raw_fine_aux", 0.2)),
        label_smoothing=ls,
        max_iters=max_iters,
        loss_schedule=getattr(args, "loss_schedule", "finalshot"),
        maturity_target_mode=getattr(args, "maturity_target_mode", "hard"),
        maturity_soft_sigma_stages=getattr(args, "maturity_soft_sigma_stages", 1.0),
        rank_margin=float(cfg.get("rank_margin", 0.05)),
        rank_temperature=float(cfg.get("rank_temperature", 1.0)),
        use_joint_fine=cfg.get("joint_posterior", True),
        joint_loss_warmup=bool(cfg.get("joint_loss_warmup", False)),
        joint_loss_ramp_start=int(cfg.get("joint_loss_ramp_start", 0)),
        joint_loss_ramp_end=int(cfg.get("joint_loss_ramp_end", 1000)),
    ).to(device)
    return model, criterion
