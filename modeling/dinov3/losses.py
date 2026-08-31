"""Losses for robust image detection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

EPS = 1e-6


def _flatten_tensor(tensor: Tensor) -> Tensor:
    """Flatten tensor to 1D for unified binary loss calculation."""
    return tensor.reshape(-1)


class FocalLoss(nn.Module):
    """Binary focal loss on logits with class-balance weighting.

    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    When gamma=0, reduces exactly to standard binary cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        logits, target = _flatten_tensor(logits), _flatten_tensor(target).float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        prob_target = prob * target + (1.0 - prob) * (1.0 - target)
        loss = bce_loss * (1.0 - prob_target).clamp(min=EPS).pow(self.gamma)
        if self.alpha >= 0:
            alpha_target = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
            loss = alpha_target * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class PredictionKL(nn.Module):
    """Measure Bernoulli KL divergence from detached clean predictions to degraded predictions.

    D_KL(Bernoulli(p) || Bernoulli(q)) = p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
    """

    def forward(self, logits_clean: Tensor, logits_degraded: Tensor) -> Tensor:
        prob_clean = (
            torch.sigmoid(_flatten_tensor(logits_clean)).detach().clamp(EPS, 1 - EPS)
        )
        prob_degraded = torch.sigmoid(_flatten_tensor(logits_degraded)).clamp(
            EPS, 1 - EPS
        )
        kl_divergence = (
            prob_clean * (prob_clean / prob_degraded).log()
            + (1 - prob_clean) * ((1 - prob_clean) / (1 - prob_degraded)).log()
        )
        return kl_divergence.mean()


class FeatureMSE(nn.Module):
    """Cosine/MSE representation loss comparing detached clean features with corrected degraded features."""

    def __init__(self, normalise: bool = True):
        super().__init__()
        self.normalise = normalise

    def forward(self, feat_clean: Tensor, feat_degraded: Tensor) -> Tensor:
        clean_normalized = feat_clean.detach()
        degraded_normalized = feat_degraded
        if self.normalise:
            clean_normalized = F.normalize(clean_normalized, dim=-1)
            degraded_normalized = F.normalize(degraded_normalized, dim=-1)
        return F.mse_loss(degraded_normalized, clean_normalized)


@dataclass
class LossWeights:
    """Weights for the 4-term robust objective."""

    focal_gamma: float = 2.0
    focal_alpha: float = 0.5
    cls_degraded: float = 1.0
    kl: float = 0.5
    feat_mse: float = 0.25


class RobustLoss(nn.Module):
    """Combine classification, prediction-consistency, and feature-consistency losses."""

    def __init__(self, weights: LossWeights | None = None):
        super().__init__()
        self.weights = weights or LossWeights()
        self.focal = FocalLoss(self.weights.focal_gamma, self.weights.focal_alpha)
        self.kl = PredictionKL()
        self.fmse = FeatureMSE()

    def forward(
        self,
        logits_clean: Tensor,
        target: Tensor,
        logits_degraded: Tensor | None = None,
        feat_clean: Tensor | None = None,
        feat_degraded: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the total weighted loss and return per-component scalars for logging."""
        loss_components: dict[str, float] = {}
        total_loss = self.focal(logits_clean, target)
        loss_components["cls_clean"] = float(total_loss.detach())

        if logits_degraded is not None:
            if self.weights.cls_degraded > 0:
                loss_degraded_cls = self.focal(logits_degraded, target)
                total_loss = total_loss + self.weights.cls_degraded * loss_degraded_cls
                loss_components["cls_degraded"] = float(loss_degraded_cls.detach())
            if self.weights.kl > 0:
                loss_kl_consistency = self.kl(logits_clean, logits_degraded)
                total_loss = total_loss + self.weights.kl * loss_kl_consistency
                loss_components["kl"] = float(loss_kl_consistency.detach())

        if (
            self.weights.feat_mse > 0
            and feat_clean is not None
            and feat_degraded is not None
        ):
            loss_feature_mse = self.fmse(feat_clean, feat_degraded)
            total_loss = total_loss + self.weights.feat_mse * loss_feature_mse
            loss_components["feat_mse"] = float(loss_feature_mse.detach())

        loss_components["total"] = float(total_loss.detach())
        return total_loss, loss_components
