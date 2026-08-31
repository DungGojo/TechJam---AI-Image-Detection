"""Losses for robust image detection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

EPS = 1e-6


def _flat(x: Tensor) -> Tensor:
    return x.reshape(-1)


class FocalLoss(nn.Module):
    """Binary focal loss on logits. gamma=0 reduces exactly to weighted BCE."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        logits, target = _flat(logits), _flat(target).float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        loss = bce * (1.0 - p_t).clamp(min=EPS).pow(self.gamma)
        if self.alpha >= 0:
            a_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
            loss = a_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class PredictionKL(nn.Module):
    """Measure Bernoulli KL from detached clean to degraded predictions."""

    def forward(self, logits_clean: Tensor, logits_degraded: Tensor) -> Tensor:
        p = torch.sigmoid(_flat(logits_clean)).detach().clamp(EPS, 1 - EPS)
        q = torch.sigmoid(_flat(logits_degraded)).clamp(EPS, 1 - EPS)
        kl = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
        return kl.mean()


class FeatureMSE(nn.Module):
    """Compare detached clean features with corrected degraded features."""

    def __init__(self, normalise: bool = True):
        super().__init__()
        self.normalise = normalise

    def forward(self, feat_clean: Tensor, feat_degraded: Tensor) -> Tensor:
        a = feat_clean.detach()
        b = feat_degraded
        if self.normalise:
            a = F.normalize(a, dim=-1)
            b = F.normalize(b, dim=-1)
        return F.mse_loss(b, a)


@dataclass
class LossWeights:
    """Defaults reproduce the NTIRE 2026 third-place configuration."""
    focal_gamma: float = 2.0
    focal_alpha: float = 0.5
    cls_degraded: float = 1.0
    kl: float = 0.5
    feat_mse: float = 0.25


class RobustLoss(nn.Module):
    """Combine classification, prediction-consistency, and feature losses."""

    def __init__(self, w: LossWeights | None = None):
        super().__init__()
        self.w = w or LossWeights()
        self.focal = FocalLoss(self.w.focal_gamma, self.w.focal_alpha)
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
        parts: dict[str, float] = {}
        total = self.focal(logits_clean, target)
        parts["cls_clean"] = float(total.detach())

        if logits_degraded is not None:
            if self.w.cls_degraded > 0:
                l_deg = self.focal(logits_degraded, target)
                total = total + self.w.cls_degraded * l_deg
                parts["cls_degraded"] = float(l_deg.detach())
            if self.w.kl > 0:
                l_kl = self.kl(logits_clean, logits_degraded)
                total = total + self.w.kl * l_kl
                parts["kl"] = float(l_kl.detach())

        if self.w.feat_mse > 0 and feat_clean is not None and feat_degraded is not None:
            l_f = self.fmse(feat_clean, feat_degraded)
            total = total + self.w.feat_mse * l_f
            parts["feat_mse"] = float(l_f.detach())

        parts["total"] = float(total.detach())
        return total, parts
