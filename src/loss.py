"""Training losses: HemaHierLoss (paper method) and FlatCELoss (linear probe)."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.ontology import (
    CANONICAL_CLASSES,
    FINE_IDX_TO_COARSE_IDX,
    FINE_TO_COARSE,
    FINE_TO_IDX,
    LINEAGE_HIERARCHY,
    LINEAGES_WITH_MATURATION,
    MAPPING_MATRIX,
    MATURATION_CHAINS,
    chains_for_fine,
)

# --- chain maturity (HemaHier branch heads) -------------------------------

MaturityTargetMode = Literal["hard", "soft"]


def _chain_spec(chain_name: str, device: torch.device) -> Tuple[Tensor, Tensor]:
    members = MATURATION_CHAINS[chain_name]
    indices = torch.tensor([FINE_TO_IDX[name] for name in members], device=device)
    positions = torch.linspace(0.0, 1.0, len(members), device=device)
    return indices, positions


def _targets_from_membership(
    membership: Tensor,
    positions: Tensor,
    *,
    target_mode: MaturityTargetMode = "hard",
    soft_sigma_stages: float = 1.0,
) -> Tensor:
    """Hard point target or Gaussian-soft target along chain stage indices."""
    mem = membership.float()
    if target_mode == "hard":
        return (mem * positions[None, :]).sum(dim=1)

    stage_idx = mem.argmax(dim=1)
    k = torch.arange(positions.numel(), device=positions.device, dtype=torch.float32)
    w = torch.exp(-0.5 * ((k[None, :] - stage_idx[:, None].float()) / soft_sigma_stages) ** 2)
    denom = w.sum(dim=1).clamp_min(1e-6)
    return (w * positions[None, :]).sum(dim=1) / denom


def chain_head_losses(
    chain_preds: Dict[str, Tensor],
    fine_labels: Tensor,
    eps: float = 1e-7,
    target_mode: MaturityTargetMode = "hard",
    soft_sigma_stages: float = 1.0,
    rank_margin: float = 0.0,
    rank_temperature: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Smooth-L1 maturity loss and within-chain pairwise ranking.

    For each chain q, only samples whose fine label belongs to q supervise
    s_q(x). A class may belong to multiple chains; in that case it contributes
    to each corresponding branch head, which is intentional for shared branch
    origins such as promyelocyte.

    Ranking is computed over all ordered pairs (i, j) with target_i < target_j.
    The previous implementation used an upper-triangular mask and therefore
    ignored pairs whenever the less mature sample appeared later in the batch.
    """
    regression, ranking = [], []
    for chain_name, pred in chain_preds.items():
        if chain_name not in MATURATION_CHAINS:
            continue

        indices, positions = _chain_spec(chain_name, fine_labels.device)
        membership = fine_labels[:, None] == indices[None, :]
        mask = membership.any(dim=1)
        if not mask.any():
            continue

        targets_all = _targets_from_membership(
            membership, positions,
            target_mode=target_mode,
            soft_sigma_stages=soft_sigma_stages,
        )
        targets = targets_all[mask].float()
        scores = pred[mask].float()

        regression.append(F.smooth_l1_loss(scores, targets))

        if scores.numel() > 1:
            target_diff = targets[:, None] - targets[None, :]
            score_diff = scores[:, None] - scores[None, :]
            valid = target_diff < -eps  # target_i < target_j
            if valid.any():
                if rank_margin > 0.0:
                    gap = -target_diff[valid]
                    desired = rank_margin * gap
                    ranking.append(
                        F.softplus((score_diff[valid] + desired) / max(rank_temperature, 1e-6)).mean()
                    )
                else:
                    ranking.append(
                        F.softplus(score_diff[valid] / max(rank_temperature, 1e-6)).mean()
                    )

    zero = torch.zeros((), device=fine_labels.device)
    return (
        torch.stack(regression).mean() if regression else zero,
        torch.stack(ranking).mean() if ranking else zero,
    )



# --- HemaHier --------------------------------------------------------------


def _share_maturation_chain(fine_a: int, fine_b: int) -> bool:
    chains_a = {name for name, _ in chains_for_fine(CANONICAL_CLASSES[fine_a])}
    chains_b = {name for name, _ in chains_for_fine(CANONICAL_CLASSES[fine_b])}
    return bool(chains_a & chains_b)


def ramp_lambda(step: int, target: float, start: int, end: int) -> float:
    if target <= 0.0 or end <= start:
        return 0.0 if step < end else target
    if step < start:
        return 0.0
    if step >= end:
        return target
    return target * (step - start) / (end - start)


def scheduled_lineage_weight(step: int, target: float) -> float:
    """Final-shot schedule: lineage ramps during warmup (steps 0–500)."""
    if target <= 0.0:
        return 0.0
    return ramp_lambda(step, target, 0, 500)


def scheduled_maturity_weight(
    step: int,
    target: float,
    loss_schedule: str = "finalshot",
) -> float:
    """Maturity λ schedule (see ``loss_schedule``)."""
    if target <= 0.0:
        return 0.0
    if loss_schedule == "finalshot_decay":
        return ramp_lambda(step, target, 1500, 3000)
    return ramp_lambda(step, target, 500, 1500)


def scheduled_rank_weight(
    step: int,
    target: float,
    loss_schedule: str = "finalshot",
) -> float:
    """Independent delayed ranking schedule (not multiplied by lambda_maturity)."""
    if target <= 0.0:
        return 0.0
    if loss_schedule in ("finalshot", "finalshot_decay"):
        return ramp_lambda(step, target, 1000, 2000)
    return ramp_lambda(step, target, 500, 1500)


def lineage_smooth_l1_loss(
    lineage_preds: Dict[int, Tensor],
    mat_target: Tensor,
    mat_mask: Tensor,
    coarse_labels: Tensor,
) -> Tensor:
    losses = []
    mask_all = mat_mask.bool()
    for li in LINEAGES_WITH_MATURATION:
        if li not in lineage_preds:
            continue
        mask = mask_all & (coarse_labels == li)
        if mask.sum() == 0:
            continue
        losses.append(F.smooth_l1_loss(lineage_preds[li][mask], mat_target[mask]))
    if not losses:
        return torch.zeros((), device=mat_target.device)
    return torch.stack(losses).mean()


def lineage_pairwise_ranking_loss(
    lineage_preds: Dict[int, Tensor],
    mat_target: Tensor,
    mat_mask: Tensor,
    coarse_labels: Tensor,
    fine_labels: Tensor,
    margin: float = 0.05,
    max_pairs: int = 256,
) -> Tensor:
    device = mat_target.device
    total, n_pairs = torch.zeros((), device=device), 0
    mask_all = mat_mask.bool()

    for li in LINEAGES_WITH_MATURATION:
        if li not in lineage_preds:
            continue
        idx = (mask_all & (coarse_labels == li)).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            continue
        scores = lineage_preds[li][idx]
        targets = mat_target[idx]
        pairs_done = 0
        for a in range(idx.numel()):
            for b in range(a + 1, idx.numel()):
                if not _share_maturation_chain(
                    int(fine_labels[idx[a]].item()), int(fine_labels[idx[b]].item()),
                ):
                    continue
                if targets[a] < targets[b]:
                    total = total + F.relu(scores[a] - scores[b] + margin)
                    n_pairs += 1
                elif targets[b] < targets[a]:
                    total = total + F.relu(scores[b] - scores[a] + margin)
                    n_pairs += 1
                pairs_done += 1
                if pairs_done >= max_pairs:
                    break
            if pairs_done >= max_pairs:
                break

    if n_pairs == 0:
        return torch.zeros((), device=device)
    return total / n_pairs


def global_pairwise_ranking_loss(
    pred: Tensor,
    fine_labels: Tensor,
    mat_target: Tensor,
    mat_mask: Tensor,
    margin: float = 0.05,
    max_pairs: int = 256,
) -> Tensor:
    """Pairwise ranking for the deliberately weaker global-maturity ablation."""
    idx = mat_mask.bool().nonzero(as_tuple=False).squeeze(-1)
    total = torch.zeros((), device=pred.device)
    n_pairs = 0
    for a in range(idx.numel()):
        for b in range(a + 1, idx.numel()):
            ia, ib = idx[a], idx[b]
            if not _share_maturation_chain(
                int(fine_labels[ia].item()), int(fine_labels[ib].item()),
            ):
                continue
            if mat_target[ia] < mat_target[ib]:
                total = total + F.relu(pred[ia] - pred[ib] + margin)
                n_pairs += 1
            elif mat_target[ib] < mat_target[ia]:
                total = total + F.relu(pred[ib] - pred[ia] + margin)
                n_pairs += 1
            if n_pairs >= max_pairs:
                break
        if n_pairs >= max_pairs:
            break
    return total / n_pairs if n_pairs else torch.zeros((), device=pred.device)


class HemaHierLoss(nn.Module):
    """
  L = L_joint + λ_raw L_raw + λ_c(t)L_c + λ_m(t)L_m + λ_r(t)L_r
    """

    def __init__(
        self,
        lambda_lineage: float = 0.2,
        lambda_maturity: float = 0.05,
        use_maturity: bool = True,
        use_ranking: bool = True,
        lambda_rank: float = 0.1,
        lambda_raw_fine_aux: float = 0.2,
        label_smoothing: float = 0.0,
        max_iters: int = 5000,
        loss_schedule: str = "finalshot",
        maturity_target_mode: str = "hard",
        maturity_soft_sigma_stages: float = 1.0,
        rank_margin: float = 0.05,
        rank_temperature: float = 1.0,
        use_joint_fine: bool = True,
        joint_loss_warmup: bool = False,
        joint_loss_ramp_start: int = 0,
        joint_loss_ramp_end: int = 1000,
        logit_adjust_tau: float = 0.0,
    ):
        super().__init__()
        self.loss_schedule = loss_schedule
        self.lambda_lineage_target = lambda_lineage
        self.lambda_maturity_target = lambda_maturity
        self.use_maturity = use_maturity
        self.use_ranking = use_ranking
        self.lambda_rank_target = lambda_rank
        self.lambda_raw_fine_aux = lambda_raw_fine_aux
        self.ls = label_smoothing
        self.max_iters = max_iters
        self.maturity_target_mode = maturity_target_mode
        self.maturity_soft_sigma_stages = maturity_soft_sigma_stages
        self.rank_margin = rank_margin
        self.rank_temperature = rank_temperature
        self.use_joint_fine = use_joint_fine
        self.joint_loss_warmup = joint_loss_warmup
        self.joint_loss_ramp_start = int(joint_loss_ramp_start)
        self.joint_loss_ramp_end = int(joint_loss_ramp_end)
        self.logit_adjust_tau = float(logit_adjust_tau)
        self.register_buffer("mapping", MAPPING_MATRIX.clone())
        # Long-tail logit adjustment (Menon et al., 2021): additive log-prior on
        # the direct fine cross-entropy, disabled until class frequencies are set.
        self.register_buffer("fine_log_prior", torch.zeros(1))

    def set_class_prior(self, counts) -> None:
        """Store the fine-class log-prior for logit adjustment.

        `counts` is a 1-D tensor of per-fine-class training frequencies.
        """
        counts = torch.as_tensor(counts, dtype=torch.float32)
        freq = counts.clamp_min(1.0) / counts.clamp_min(1.0).sum()
        self.fine_log_prior = torch.log(freq)

    def lambdas_at(self, step: int) -> Tuple[float, float, float]:
        ll = scheduled_lineage_weight(step, self.lambda_lineage_target)
        lm = (
            scheduled_maturity_weight(
                step, self.lambda_maturity_target, self.loss_schedule,
            )
            if self.use_maturity else 0.0
        )
        lr = (
            scheduled_rank_weight(step, self.lambda_rank_target, self.loss_schedule)
            if self.use_ranking else 0.0
        )
        return ll, lm, lr

    def forward(
        self,
        out: Dict[str, Tensor],
        fine_labels: Tensor,
        coarse_labels: Tensor,
        mat_target: Tensor,
        mat_mask: Tensor,
        epoch: int,
        global_step: int = 0,
    ) -> Tuple[Tensor, Dict[str, float]]:
        ll, lm, lr = self.lambdas_at(global_step)

        logits_f_raw = out.get("logits_fine_raw")
        if logits_f_raw is None:
            logits_f_raw = out.get("logits_fine")
        if logits_f_raw is None:
            raise KeyError("model output must include logits_fine_raw or logits_fine")
        logits_f_joint = out.get("logits_fine_joint")

        if self.use_joint_fine and logits_f_joint is not None:
            L_ce_joint = F.nll_loss(logits_f_joint, fine_labels)
        else:
            L_ce_joint = F.cross_entropy(
                logits_f_raw, fine_labels, label_smoothing=self.ls,
            )

        logits_f_raw_adj = logits_f_raw
        if self.logit_adjust_tau > 0.0 and self.fine_log_prior.numel() > 1:
            la = (self.logit_adjust_tau * self.fine_log_prior).to(logits_f_raw.device)
            logits_f_raw_adj = logits_f_raw + la.unsqueeze(0)
        L_ce_raw = F.cross_entropy(
            logits_f_raw_adj, fine_labels, label_smoothing=self.ls,
        )
        if self.joint_loss_warmup and self.use_joint_fine and logits_f_joint is not None:
            gamma = ramp_lambda(
                global_step, 1.0,
                self.joint_loss_ramp_start, self.joint_loss_ramp_end,
            )
            L_fine = (
                gamma * L_ce_joint
                + (1.0 - gamma) * L_ce_raw
                + self.lambda_raw_fine_aux * L_ce_raw
            )
        else:
            gamma = 1.0 if (self.use_joint_fine and logits_f_joint is not None) else 0.0
            L_fine = L_ce_joint + self.lambda_raw_fine_aux * L_ce_raw

        L_ce_lineage = F.cross_entropy(
            out["logits_coarse"], coarse_labels, label_smoothing=self.ls,
        )

        chain_preds = out.get("chain_mat_raw") or out.get("chain_mat_preds") or {}
        lineage_preds = out.get("lineage_mat_preds") or {}
        mat_scores = out.get("mat_raw_gt")
        if mat_scores is None:
            mat_scores = out.get("mat_pred")
        if self.use_maturity and chain_preds:
            L_mat, L_rank = chain_head_losses(
                chain_preds,
                fine_labels,
                target_mode=self.maturity_target_mode,  # type: ignore[arg-type]
                soft_sigma_stages=self.maturity_soft_sigma_stages,
                rank_margin=self.rank_margin,
                rank_temperature=self.rank_temperature,
            )
            if not self.use_ranking:
                L_rank = torch.zeros((), device=fine_labels.device)
        elif self.use_maturity and lineage_preds:
            L_mat = lineage_smooth_l1_loss(
                lineage_preds, mat_target, mat_mask, coarse_labels,
            )
            L_rank = (
                lineage_pairwise_ranking_loss(
                    lineage_preds, mat_target, mat_mask, coarse_labels,
                    fine_labels,
                )
                if self.use_ranking else torch.zeros((), device=fine_labels.device)
            )
        elif self.use_maturity and mat_scores is not None:
            valid = mat_mask.bool()
            L_mat = (
                F.smooth_l1_loss(mat_scores[valid], mat_target[valid])
                if valid.any() else torch.zeros((), device=fine_labels.device)
            )
            L_rank = (
                global_pairwise_ranking_loss(
                    mat_scores, fine_labels, mat_target, mat_mask,
                )
                if self.use_ranking else torch.zeros((), device=fine_labels.device)
            )
        else:
            L_mat = torch.zeros((), device=fine_labels.device)
            L_rank = torch.zeros((), device=fine_labels.device)

        total = (
            L_fine
            + ll * L_ce_lineage
            + lm * L_mat
            + lr * L_rank
        )

        components = {
            "L_total": float(total.item()),
            "L_ce_fine": float(L_fine.item()),
            "L_ce_joint": float(L_ce_joint.item()),
            "L_ce_raw": float(L_ce_raw.item()),
            "L_ce_coarse": float(L_ce_lineage.item()),
            "L_hba": 0.0,
            "L_ham": 0.0,
            "L_supcon": 0.0,
            "L_ordinal": float(L_mat.item()),
            "L_rank": float(L_rank.item()),
            "L_lineage_weighted": float((ll * L_ce_lineage).item()),
            "L_maturity_weighted": float((lm * L_mat).item()),
            "L_rank_weighted": float((lr * L_rank).item()),
            "L_raw_fine_weighted": float((self.lambda_raw_fine_aux * L_ce_raw).item()),
            "gamma_joint": float(gamma),
            "lambda_raw_fine_aux": float(self.lambda_raw_fine_aux),
            "beta": 0.0,
            "lambda_lineage": ll,
            "lambda_maturity": lm,
            "lambda_rank": lr,
            "lambda_rank_target": self.lambda_rank_target,
            "lambda_c": ll,
            "lambda_o": lm,
        }
        return total, components

    def beta(self, epoch: int) -> float:
        return 0.0


class FlatCELoss(nn.Module):
    """Fine cross-entropy only (flat linear probe)."""

    def __init__(self, label_smoothing: float = 0.05):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, fine_labels, *_args, epoch=0, global_step=0):
        loss = self.ce(out["logits_fine"], fine_labels)
        val = float(loss.item())
        comps = {
            "L_total": val,
            "L_ce_fine": val,
            "L_ce_coarse": 0.0,
            "L_hba": 0.0,
            "L_ham": 0.0,
            "L_supcon": 0.0,
            "L_ordinal": 0.0,
            "L_rank": 0.0,
            "L_consistency": 0.0,
            "beta": 0.0,
            "lambda_c": 0.0,
            "lambda_o": 0.0,
        }
        return loss, comps

    def beta(self, epoch: int) -> float:
        return 0.0
