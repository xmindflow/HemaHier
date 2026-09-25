"""
HemaHier v4 (Cascade): staged 256-d contexts with multiscale lineage → fine → maturation.

  h = [h_L; h_M]
  c_coarse = φ_c(h)
  c_fine   = φ_f([c_coarse; h])
  c_mat    = φ_m([c_coarse; c_fine])

Heads consume context vectors (d=256), not raw backbone width.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.ontology import (
    CANONICAL_CLASSES,
    LINEAGES,
    MAPPING_MATRIX,
    MATURATION_CHAINS,
    MATURATION_TABLE,
    assemble_chain_mat_pred,
)
from models.foundation_backbone import VIT_BACKBONES, build_foundation_backbone
from models.layers import ReEmbedder

_CHAIN_NAMES = tuple(MATURATION_CHAINS.keys())


class HemaHierCascade(nn.Module):
    """Cascade-context HemaHier with 256-d coarse / fine / maturation contexts."""

    def __init__(
        self,
        backbone: str = "dinobloom_b",
        pretrained: bool = True,
        context_dim: int = 256,
        dropout: float = 0.3,
        use_maturity: bool = True,
        joint_posterior: bool = True,
        chain_cond_dim: int = 32,
        maturity_hidden_dim: int = 64,
        maturity_activation: str = "identity",
        maturity_eps: float = 0.05,
        beta_maturity_decode: float = 0.0,
        maturity_decode_sigma: float = 0.25,
        joint_coarse_temperature: float = 1.0,
        feature_mode: str = "multiscale",
        feature_layers: int = 4,
        img_size: int = 224,
        detach_aux_backbone: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 128.0,
        fine_from_maturity: bool = True,
    ):
        super().__init__()
        self.flat_only = False
        self.hemahier_mode = True
        self.hemahier_version = "cascade"
        self.feature_mode = feature_mode
        self.feature_layers = int(feature_layers)
        # When False, the fine head reads only [c_coarse; c_fine] and NOT the
        # maturity context c_mat.  The maturity axis is shaped by an ordering
        # objective that smears adjacent early stages (e.g. myeloblast into
        # promyelocyte); feeding it into the fine head corrupts fine
        # discrimination of blast/immature classes.  Decoupling keeps maturity
        # as a read-out head without polluting fine classification.
        self.fine_from_maturity = bool(fine_from_maturity)
        # When True, lineage/maturity heads read detached contexts so their
        # gradients do not perturb the (LoRA-adapted) backbone; the encoder then
        # adapts under the fine objective alone, like a flat probe, and the
        # hierarchy is learned on top -> a floor at flat under adaptation.
        self.detach_aux_backbone = bool(detach_aux_backbone)
        self.joint_posterior = bool(joint_posterior)
        self.use_maturity = use_maturity
        self.context_dim = int(context_dim)
        self.chain_cond_dim = int(chain_cond_dim)
        self.beta_maturity_decode = float(beta_maturity_decode)
        self.maturity_decode_sigma = float(maturity_decode_sigma)
        self.joint_coarse_temperature = float(joint_coarse_temperature)
        self.requires_fine_for_maturity = use_maturity
        self.maturity_activation = maturity_activation
        self.maturity_eps = float(maturity_eps)
        self.maturity_hidden_dim = int(maturity_hidden_dim)

        if backbone not in VIT_BACKBONES:
            raise ValueError(f"Unsupported backbone {backbone!r}")
        self.encoder = build_foundation_backbone(
            backbone, pretrained=pretrained,
            lora_rank=lora_rank, lora_alpha=lora_alpha, img_size=img_size,
        )

        D = int(self.encoder.feat_dim)
        D_m = int(getattr(self.encoder, "mid_feat_dim", D))
        self.backbone_dim = D
        self.mid_dim = D_m
        # "probe": all-level CLS tokens + mean patch (same rich features as the
        # flat linear probe), feeding the coarse -> fine -> maturity cascade.
        # "multiscale": [h_L; h_M] (paper default).
        if feature_mode == "probe":
            self.multiscale_dim = (1 + self.feature_layers) * D
        else:
            self.multiscale_dim = D + D_m

        d = self.context_dim
        self.phi_c = ReEmbedder(self.multiscale_dim, d, dropout=dropout)
        self.phi_f = ReEmbedder(d + self.multiscale_dim, d, dropout=dropout)
        self.phi_m = ReEmbedder(2 * d, d, dropout=dropout)

        self.K_f = len(CANONICAL_CLASSES)
        self.K_c = len(LINEAGES)
        self.coarse_head = nn.Linear(d, self.K_c)
        self.fine_head = nn.Linear((3 if self.fine_from_maturity else 2) * d, self.K_f)

        if use_maturity:
            self.chain_embed = nn.Embedding(len(_CHAIN_NAMES), self.chain_cond_dim)
            mat_in = d + self.chain_cond_dim
            self.shared_maturity_head = self._make_maturity_head(mat_in, dropout=dropout)
        else:
            self.chain_embed = None
            self.shared_maturity_head = None

        self.register_buffer("M", MAPPING_MATRIX.clone())
        self.register_buffer(
            "fine_maturity_target",
            torch.tensor([pos for _, pos in MATURATION_TABLE], dtype=torch.float32),
        )
        self.register_buffer(
            "fine_maturity_mask",
            torch.tensor([chain >= 0 for chain, _ in MATURATION_TABLE], dtype=torch.bool),
        )
        self._init_heads()

    @property
    def v2_mode(self) -> bool:
        return True

    def _make_maturity_head(self, in_dim: int, dropout: float) -> nn.Module:
        if self.maturity_hidden_dim <= 0:
            return nn.Linear(in_dim, 1)
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.maturity_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.maturity_hidden_dim, 1),
        )

    def _final_linear(self, head: nn.Module) -> nn.Linear:
        if isinstance(head, nn.Linear):
            return head
        if isinstance(head, nn.Sequential) and isinstance(head[-1], nn.Linear):
            return head[-1]
        raise TypeError(f"Cannot find final Linear in maturity head: {head}")

    def _init_heads(self) -> None:
        for head in (self.coarse_head, self.fine_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
        if self.shared_maturity_head is not None:
            final = self._final_linear(self.shared_maturity_head)
            bias = 0.5 if self.maturity_activation == "identity" else 0.0
            nn.init.xavier_uniform_(final.weight, gain=0.5)
            nn.init.constant_(final.bias, bias)
        if self.chain_embed is not None:
            nn.init.normal_(self.chain_embed.weight, std=0.02)

    @staticmethod
    def _pool_feature(x: Tensor) -> Tensor:
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            return x.mean(dim=1)
        if x.ndim == 4:
            return x.mean(dim=(-2, -1))
        raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")

    def encode_multiscale(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        if hasattr(self.encoder, "forward_multiscale_tokens"):
            h_l, mid_tokens = self.encoder.forward_multiscale_tokens(images)
            h_m = mid_tokens.mean(dim=1) if mid_tokens.ndim == 3 else mid_tokens
            return self._pool_feature(h_l), self._pool_feature(h_m)
        if hasattr(self.encoder, "forward_tokens"):
            h_l, h_m = self.encoder.forward_tokens(images)
            return self._pool_feature(h_l), self._pool_feature(h_m)
        h = self._pool_feature(self.encoder(images))
        return h, h

    def encode(self, images: Tensor) -> Tensor:
        h_l, h_m = self.encode_multiscale(images)
        return torch.cat([h_l, h_m], dim=-1)

    def _multiscale_from_input(self, images: Tensor) -> Tensor:
        if images.ndim == 2:
            h = images
            if h.shape[-1] == self.multiscale_dim:
                return h
            if h.shape[-1] == self.backbone_dim and self.mid_dim == self.backbone_dim:
                return torch.cat([h, h], dim=-1)
            raise ValueError(
                f"Cached features must be {self.multiscale_dim}-d, got {h.shape[-1]}"
            )
        if self.feature_mode == "probe" and hasattr(self.encoder, "forward_linear_probe_features"):
            return self.encoder.forward_linear_probe_features(images, layers=self.feature_layers)
        h_l, h_m = self.encode_multiscale(images)
        return torch.cat([h_l, h_m], dim=-1)

    def _contexts(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        h = self._multiscale_from_input(images)
        c_coarse = self.phi_c(h)
        c_fine = self.phi_f(torch.cat([c_coarse, h], dim=-1))
        c_mat = self.phi_m(torch.cat([c_coarse, c_fine], dim=-1))
        return c_coarse, c_fine, c_mat, h

    def _apply_maturity_activation(self, raw: Tensor) -> Tensor:
        if self.maturity_activation == "identity":
            return raw
        if self.maturity_activation == "sigmoid":
            return torch.sigmoid(raw)
        return self.maturity_eps + (1.0 - 2.0 * self.maturity_eps) * torch.sigmoid(raw)

    def _shared_chain_outputs(self, c_mat: Tensor) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        chain_raw: Dict[str, Tensor] = {}
        chain_preds: Dict[str, Tensor] = {}
        b = c_mat.shape[0]
        for chain_idx, chain_name in enumerate(_CHAIN_NAMES):
            e_q = self.chain_embed.weight[chain_idx].expand(b, -1)
            raw = self.shared_maturity_head(torch.cat([c_mat, e_q], dim=-1)).squeeze(-1)
            chain_raw[chain_name] = raw
            chain_preds[chain_name] = self._apply_maturity_activation(raw)
        return chain_raw, chain_preds

    def _hierarchical_joint_logits(self, logits_f_raw: Tensor, logits_c: Tensor) -> Tensor:
        out_dtype = logits_f_raw.dtype
        logits_f_raw = logits_f_raw.float()
        logits_c = logits_c.float()
        valid_lineages = self.M.sum(dim=0) > 0
        logits_c_masked = logits_c.masked_fill(~valid_lineages.unsqueeze(0), float("-inf"))
        t = max(float(self.joint_coarse_temperature), 1e-6)
        log_p_c = (logits_c_masked / t).log_softmax(dim=-1)
        parent = self.M.argmax(dim=-1).to(logits_f_raw.device)

        log_p_f_given_c = torch.zeros_like(logits_f_raw)
        for c in range(self.K_c):
            mask = parent == c
            if not mask.any():
                continue
            group_logits = logits_f_raw[:, mask]
            log_p_f_given_c[:, mask] = group_logits - torch.logsumexp(
                group_logits, dim=-1, keepdim=True,
            )

        logits_f_joint = log_p_c[:, parent] + log_p_f_given_c
        return (
            logits_f_joint - torch.logsumexp(logits_f_joint, dim=-1, keepdim=True)
        ).to(dtype=out_dtype)

    def _bayes_logits(self, logits_fine: Tensor, logits_coarse: Tensor) -> Tensor:
        p_f = logits_fine.softmax(dim=-1)
        p_c = logits_coarse.softmax(dim=-1)
        parent = self.M.argmax(dim=-1).to(logits_fine.device)
        p_c_for_f = p_c[:, parent]
        p_bayes = p_f * p_c_for_f
        p_bayes = p_bayes / (p_bayes.sum(dim=-1, keepdim=True) + 1e-12)
        return (p_bayes + 1e-12).log()

    def _maturity_compat_scores(self, chain_preds: Dict[str, Tensor]) -> Tensor:
        from data.ontology import chain_target_position, chains_for_fine

        device = next(iter(chain_preds.values())).device
        b = next(iter(chain_preds.values())).shape[0]
        compat = torch.zeros(b, self.K_f, device=device)
        sigma = max(self.maturity_decode_sigma, 1e-6)
        for f_idx, fine_name in enumerate(CANONICAL_CLASSES):
            chain_list = chains_for_fine(fine_name)
            if not chain_list:
                continue
            best: Tensor | None = None
            for chain_name, _ in chain_list:
                if chain_name not in chain_preds:
                    continue
                pos = chain_target_position(chain_name, f_idx)
                if pos is None:
                    continue
                s_q = chain_preds[chain_name]
                c = -0.5 * ((s_q - pos) / sigma) ** 2
                best = c if best is None else torch.maximum(best, c)
            if best is not None:
                compat[:, f_idx] = best
        return compat

    def forward(
        self,
        images: Tensor,
        return_features: bool = True,
        coarse_labels: Optional[Tensor] = None,
        fine_labels: Optional[Tensor] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, Tensor | Dict[str, Tensor] | None]:
        del global_step
        c_coarse, c_fine, c_mat, h = self._contexts(images)
        if self.fine_from_maturity:
            zf = torch.cat([c_coarse, c_fine, c_mat], dim=-1)
        else:
            zf = torch.cat([c_coarse, c_fine], dim=-1)

        logits_f_raw = self.fine_head(zf)
        c_coarse_aux = c_coarse.detach() if self.detach_aux_backbone else c_coarse
        logits_c = self.coarse_head(c_coarse_aux)

        if self.joint_posterior:
            logits_f_joint = self._hierarchical_joint_logits(logits_f_raw, logits_c)
        else:
            logits_f_joint = logits_f_raw

        logits_f_bayes = self._bayes_logits(logits_f_raw, logits_c)

        chain_preds: Dict[str, Tensor] = {}
        chain_raw: Dict[str, Tensor] = {}
        mat_pred_gt = None
        mat_raw_gt = None
        if self.use_maturity and self.shared_maturity_head is not None:
            c_mat_aux = c_mat.detach() if self.detach_aux_backbone else c_mat
            chain_raw, chain_preds = self._shared_chain_outputs(c_mat_aux)
            if fine_labels is not None:
                mat_raw_gt = assemble_chain_mat_pred(chain_raw, fine_labels, coarse_labels)
                mat_pred_gt = assemble_chain_mat_pred(chain_preds, fine_labels, coarse_labels)

        logits_fine_ordinal = logits_f_joint
        if chain_preds and self.beta_maturity_decode > 0.0:
            logits_fine_ordinal = (
                logits_f_joint + self.beta_maturity_decode * self._maturity_compat_scores(chain_preds)
            )

        return {
            "logits_fine": logits_f_joint,
            "logits_fine_raw": logits_f_raw,
            "logits_fine_joint": logits_f_joint,
            "logits_fine_ordinal": logits_fine_ordinal,
            "logits_coarse": logits_c,
            "logits_fine_bayes": logits_f_bayes,
            "fine_feat": F.normalize(zf, dim=-1) if return_features else None,
            "coarse_feat": F.normalize(c_coarse, dim=-1) if return_features else None,
            "detail_feat": F.normalize(c_fine, dim=-1) if return_features else None,
            "maturation_feat": F.normalize(c_mat, dim=-1) if return_features else None,
            "mat_pred": mat_pred_gt,
            "mat_pred_gt": mat_pred_gt,
            "mat_pred_gt_clamped": None if mat_pred_gt is None else mat_pred_gt.clamp(0.0, 1.0),
            "mat_pred_clamped": None if mat_pred_gt is None else mat_pred_gt.clamp(0.0, 1.0),
            "mat_raw": mat_raw_gt,
            "mat_raw_gt": mat_raw_gt,
            "chain_mat_preds": chain_preds,
            "chain_mat_preds_clamped": {k: v.clamp(0.0, 1.0) for k, v in chain_preds.items()},
            "chain_mat_raw": chain_raw,
            "lineage_mat_preds": {},
            "attn_weights": None,
            "pooled_feat": h[:, : self.backbone_dim],
        }
