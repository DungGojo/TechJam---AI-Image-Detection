"""Binary detection metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

FIXED_FPRS: tuple[float, ...] = (0.01, 0.05)


def tpr_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float
) -> tuple[float, float]:
    """Return the best TPR and threshold at or below the FPR limit."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    usable = fpr <= target_fpr
    if not usable.any():
        return 0.0, float("inf")
    idx = int(np.flatnonzero(usable)[-1])
    return float(tpr[idx]), float(thresholds[idx])


def compute(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
    fixed_fprs: tuple[float, ...] = FIXED_FPRS,
) -> dict[str, float]:
    """Compute binary metrics, returning NaN for degenerate inputs."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")

    out: dict[str, float] = {
        "n": float(len(y_true)),
        "n_real": float((y_true == 0).sum()),
        "n_fake": float((y_true == 1).sum()),
        "mean_score_real": float(y_score[y_true == 0].mean()) if (y_true == 0).any() else float("nan"),
        "mean_score_fake": float(y_score[y_true == 1].mean()) if (y_true == 1).any() else float("nan"),
    }

    single_class = len(np.unique(y_true)) < 2
    out["accuracy"] = (
        float(accuracy_score(y_true, (y_score >= threshold).astype(int)))
        if len(y_true)
        else float("nan")
    )
    if single_class:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
        for f in fixed_fprs:
            out[f"tpr_at_fpr{f:g}"] = float("nan")
            out[f"threshold_at_fpr{f:g}"] = float("nan")
        return out

    out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    out["average_precision"] = float(average_precision_score(y_true, y_score))
    for f in fixed_fprs:
        tpr, thr = tpr_at_fpr(y_true, y_score, f)
        out[f"tpr_at_fpr{f:g}"] = tpr
        out[f"threshold_at_fpr{f:g}"] = thr
    return out


METRIC_ORDER: tuple[str, ...] = (
    "roc_auc",
    "accuracy",
    "average_precision",
    "tpr_at_fpr0.01",
    "tpr_at_fpr0.05",
    "n",
    "n_real",
    "n_fake",
)
