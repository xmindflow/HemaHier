"""
Evaluation metrics: flat AND hierarchical.

We report:
    fine_accuracy, fine_macro_f1, fine_rare_macro_f1, mcc
    coarse_accuracy, coarse_macro_f1
    hierarchical_consistency     - fraction of preds where coarse_of(fine) = coarse_pred
    lineage_correct_rate         - fraction where predicted coarse matches true coarse
    per_class_f1                 - fine, dict form
    confusion_matrix             - optional

The lineage-correct rate treats a prediction as useful when the coarse lineage
is right even if the fine class is wrong.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_fscore_support, recall_score,
)

from data.ontology import (
    CANONICAL_CLASSES, FINE_IDX_TO_COARSE_IDX, LINEAGES,
)
from evaluation.clinical_errors import clinical_error_breakdown
from evaluation.decoding import DecodeMode, coarse_predictions, decode_fine_predictions
from evaluation.ordinal_metrics import mean_tree_distance_error
from evaluation.pathology_metrics import pathology_metrics_breakdown


_FINE_TO_COARSE = np.asarray(FINE_IDX_TO_COARSE_IDX)


def bottom_support_quartile(y_true: np.ndarray) -> List[int]:
    """Class indices in the bottom support quartile (deterministic ties)."""
    present = sorted(set(np.asarray(y_true).tolist()))
    if not present:
        return []
    support = {cls: int((y_true == cls).sum()) for cls in present}
    n_rare = max(1, int(np.ceil(len(present) * 0.25)))
    return sorted(present, key=lambda cls: (support[cls], cls))[:n_rare]


def _balanced_accuracy_present(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
) -> float:
    """Macro recall over classes present in *y_true* (avoids sklearn warnings).

    CV val folds omit rare classes; the model may still predict them.
    ``balanced_accuracy_score`` without an explicit label set triggers
    ``y_pred contains classes not in y_true``.
    """
    if not labels:
        return 0.0
    return float(
        recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    use_bayes: bool = True,
    decode_mode: DecodeMode | None = None,
    return_confusion: bool = False,
    return_extended: bool = False,
    dataset_name: Optional[str] = None,
) -> Dict:
    model.eval()
    if decode_mode is None:
        decode_mode = "bayes" if use_bayes else "argmax"

    fine_preds, coarse_preds, fine_true, coarse_true = [], [], [], []
    coarse_head_preds = []
    maturity_preds, maturity_true, maturity_masks = [], [], []
    chain_maturity_preds: Dict[str, List[np.ndarray]] = {}
    mapping = getattr(model, "M", None)

    for batch in loader:
        # Dataset returns (img, fine, coarse, mat_pos, mat_mask, ds_id)
        images = batch[0].to(device, non_blocking=True)
        f_lab  = batch[1]
        c_lab  = batch[2]

        if getattr(model, "hemahier_mode", False):
            model_kwargs = {
                "coarse_labels": c_lab.to(device, non_blocking=True),
            }
            if getattr(model, "requires_fine_for_maturity", False):
                model_kwargs["fine_labels"] = f_lab.to(device, non_blocking=True)
            out = model(images, return_features=False, **model_kwargs)
        else:
            out = model(images, return_features=False)

        if mapping is None:
            mapping = torch.tensor(
                [[1 if FINE_IDX_TO_COARSE_IDX[f] == c else 0
                  for c in range(len(LINEAGES))]
                 for f in range(len(CANONICAL_CLASSES))],
                device=device,
            )

        f_pred = decode_fine_predictions(
            out["logits_fine"],
            out.get("logits_coarse"),
            mapping,
            mode=decode_mode,
            logits_fine_bayes=out.get("logits_fine_bayes"),
            logits_fine_raw=out.get("logits_fine_raw"),
            logits_fine_joint=out.get("logits_fine_joint"),
            logits_fine_ordinal=out.get("logits_fine_ordinal"),
        ).cpu().numpy()

        c_pred = coarse_predictions(
            out.get("logits_coarse"), torch.as_tensor(f_pred, device=device), mapping,
        ).cpu().numpy()

        fine_preds.append(f_pred)
        coarse_preds.append(c_pred)
        fine_true.append(f_lab.numpy())
        coarse_true.append(c_lab.numpy())
        if out.get("logits_coarse") is not None:
            coarse_head_preds.append(out["logits_coarse"].argmax(dim=-1).cpu().numpy())
        use_clamped = getattr(model, "maturity_activation", "") == "identity"
        mat_out = out.get("mat_pred_gt_clamped" if use_clamped else "mat_pred_gt")
        if mat_out is None:
            mat_out = out.get("mat_pred_clamped" if use_clamped else "mat_pred")
        if mat_out is not None:
            maturity_preds.append(mat_out.detach().cpu().numpy())
            maturity_true.append(batch[3].numpy())
            maturity_masks.append(batch[4].numpy())
        chain_out = out.get("chain_mat_preds_clamped" if use_clamped else "chain_mat_preds") or {}
        for chain_name, chain_pred in chain_out.items():
            chain_maturity_preds.setdefault(chain_name, []).append(
                chain_pred.detach().cpu().numpy()
            )

    y_pred_f = np.concatenate(fine_preds)
    y_pred_c = np.concatenate(coarse_preds)
    y_true_f = np.concatenate(fine_true)
    y_true_c = np.concatenate(coarse_true)

    has_explicit_coarse_head = getattr(model, "coarse_head", None) is not None

    present_f = sorted(set(y_true_f.tolist()))
    present_c = sorted(set(y_true_c.tolist()))

    # ---- Fine ---------------------------------------------------------
    fine_acc = accuracy_score(y_true_f, y_pred_f)
    fine_bal = _balanced_accuracy_present(y_true_f, y_pred_f, present_f)
    fine_f1 = f1_score(
        y_true_f, y_pred_f, labels=present_f, average="macro", zero_division=0,
    )
    fine_wf1 = f1_score(
        y_true_f, y_pred_f, labels=present_f, average="weighted", zero_division=0,
    )
    fine_mcc = matthews_corrcoef(y_true_f, y_pred_f) if len(present_f) > 1 else 0.0

    # Dataset-relative rare classes: bottom support quartile among classes
    # present in this evaluation split. CV is stratified, so this is stable
    # across folds and matches the metric definition in the manuscript.
    rare_present = bottom_support_quartile(y_true_f)
    rare_f1 = (
        f1_score(y_true_f, y_pred_f, labels=rare_present, average="macro", zero_division=0)
        if rare_present else 0.0
    )

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_f, y_pred_f, labels=present_f, average=None, zero_division=0,
    )
    per_class_f1 = {CANONICAL_CLASSES[c]: float(f1[i]) for i, c in enumerate(present_f)}
    per_class_support = {CANONICAL_CLASSES[c]: int(support[i]) for i, c in enumerate(present_f)}

    # ---- Coarse -------------------------------------------------------
    coarse_acc = accuracy_score(y_true_c, y_pred_c)
    coarse_f1 = f1_score(
        y_true_c, y_pred_c, labels=present_c, average="macro", zero_division=0,
    )
    coarse_wf1 = f1_score(
        y_true_c, y_pred_c, labels=present_c, average="weighted", zero_division=0,
    )
    coarse_bal = _balanced_accuracy_present(y_true_c, y_pred_c, present_c)

    per_lin_prec, per_lin_rec, per_lin_f1, _ = precision_recall_fscore_support(
        y_true_c, y_pred_c, labels=present_c, average=None, zero_division=0,
    )
    per_lineage_f1 = {LINEAGES[c]: float(per_lin_f1[i]) for i, c in enumerate(present_c)}

    # ---- Hierarchical / lineage metrics --------------------------------
    coarse_of_fine_pred = _FINE_TO_COARSE[y_pred_f]
    coarse_of_fine_true = _FINE_TO_COARSE[y_true_f]

    # Lineage via predicted fine class (meaningful for all models).
    lineage_correct_via_fine = float((coarse_of_fine_pred == y_true_c).mean())

    if has_explicit_coarse_head:
        y_coarse_head = np.concatenate(coarse_head_preds)

        coarse_head_acc = accuracy_score(y_true_c, y_coarse_head)
        coarse_head_f1 = f1_score(
            y_true_c, y_coarse_head, labels=present_c, average="macro", zero_division=0,
        )
        coarse_fine_consistency = float((y_coarse_head == coarse_of_fine_pred).mean())

        lineage_correct = float((y_coarse_head == y_true_c).mean())
        consistency = coarse_fine_consistency
    else:
        coarse_head_acc = None
        coarse_head_f1 = None
        coarse_fine_consistency = None
        lineage_correct = lineage_correct_via_fine
        consistency = None

    result = {
        "decode_mode":               decode_mode,
        "has_explicit_coarse_head":  has_explicit_coarse_head,
        "fine_accuracy":             float(fine_acc),
        "fine_balanced_accuracy":    float(fine_bal),
        "fine_macro_f1":             float(fine_f1),
        "fine_weighted_f1":          float(fine_wf1),
        "fine_rare_macro_f1":        float(rare_f1),
        "fine_rare_classes":         [CANONICAL_CLASSES[c] for c in rare_present],
        "fine_mcc":                  float(fine_mcc),
        "coarse_accuracy":           float(coarse_acc),
        "coarse_balanced_accuracy":  float(coarse_bal),
        "coarse_macro_f1":           float(coarse_f1),
        "coarse_weighted_f1":        float(coarse_wf1),
        "lineage_correct_via_fine":  lineage_correct_via_fine,
        "lineage_correct_rate":      lineage_correct,
        "per_class_f1":              per_class_f1,
        "per_class_support":         per_class_support,
        "per_lineage_f1":            per_lineage_f1,
        "n_samples":                 int(len(y_true_f)),
    }
    if has_explicit_coarse_head:
        result["coarse_head_accuracy"] = float(coarse_head_acc)
        result["coarse_head_macro_f1"] = float(coarse_head_f1)
        result["coarse_fine_consistency"] = coarse_fine_consistency
        result["hierarchical_consistency"] = consistency
    else:
        result["coarse_head_accuracy"] = None
        result["coarse_head_macro_f1"] = None
        result["coarse_fine_consistency"] = None
        result["hierarchical_consistency"] = None
    result.update(clinical_error_breakdown(y_true_f, y_pred_f))
    result["tree_distance_error"] = mean_tree_distance_error(
        y_true_f, y_pred_f, dataset=dataset_name,
    )
    n_model_fine = int(getattr(model, "num_fine_classes", len(CANONICAL_CLASSES)))
    if n_model_fine == len(CANONICAL_CLASSES):
        result.update(pathology_metrics_breakdown(y_true_f, y_pred_f, dataset=dataset_name))

    if return_extended and getattr(model, "v2_mode", False) and getattr(model, "use_maturity", False):
        from evaluation.ordinal_metrics import (
            compute_chain_head_ordinal_metrics,
            compute_ordinal_metrics,
        )
        if maturity_preds:
            result.update(compute_ordinal_metrics(
                np.concatenate(maturity_preds),
                np.concatenate(maturity_true),
                np.concatenate(maturity_masks),
                y_true_c,
                y_true_f,
            ))
        if chain_maturity_preds:
            result.update(compute_chain_head_ordinal_metrics(
                {name: np.concatenate(values) for name, values in chain_maturity_preds.items()},
                y_true_f,
            ))

    if return_confusion or return_extended:
        n_fine = int(getattr(model, "num_fine_classes", 0)) or len(CANONICAL_CLASSES)
        fine_cm_labels = list(range(n_fine))
        n_coarse = len(LINEAGES)
        coarse_cm_labels = list(range(n_coarse))
        result["fine_confusion"] = confusion_matrix(
            y_true_f, y_pred_f, labels=fine_cm_labels,
        ).tolist()
        if fold_doc_labels := getattr(model, "fine_class_names", None):
            result["fine_confusion_labels"] = list(fold_doc_labels)
        else:
            result["fine_confusion_labels"] = [
                CANONICAL_CLASSES[c] if c < len(CANONICAL_CLASSES) else str(c)
                for c in fine_cm_labels
            ]
        result["coarse_confusion"] = confusion_matrix(
            y_true_c, y_pred_c, labels=coarse_cm_labels,
        ).tolist()
        result["coarse_confusion_labels"] = [LINEAGES[c] for c in coarse_cm_labels]
    return result


# ---------------------------------------------------------------------------
# LaTeX table helpers
# ---------------------------------------------------------------------------

def format_main_table(results_by_method: Dict[str, Dict]) -> str:
    header = (
        r"\begin{tabular}{lcccccc}" + "\n"
        r"\toprule" + "\n"
        r"Method & FineAcc & Fine-F1 & Rare-F1 & MCC & "
        r"CoarseAcc & CoarseF1 \\" + "\n"
        r"\midrule" + "\n"
    )
    rows = []
    for name, m in results_by_method.items():
        rows.append(
            f"{name} & "
            f"{m['fine_accuracy']:.3f} & "
            f"{m['fine_macro_f1']:.3f} & "
            f"{m['fine_rare_macro_f1']:.3f} & "
            f"{m['fine_mcc']:.3f} & "
            f"{m['coarse_accuracy']:.3f} & "
            f"{m['coarse_macro_f1']:.3f} \\\\"
        )
    return header + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}"
