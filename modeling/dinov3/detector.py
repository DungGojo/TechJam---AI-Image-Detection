"""DINOv3 detection expert."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from modeling.dinov3.backbone import Backbone, BackboneConfig


@dataclass
class ExpertConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    head_hidden: int = 512
    head_dropout: float = 0.0
    feature_correction: bool = False
    input_size: int = 224
    crop_mode: str = "random_crop"


class FeatureCorrection(nn.Module):
    """Map degraded features toward clean features with a residual FFN: (B, D) -> (B, D)."""

    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class Expert(nn.Module):
    """Classification expert: (B, 3, H, W) -> (B,) logit and (B, D) feature embeddings."""

    def __init__(self, cfg: ExpertConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg.backbone)
        embed_dim = self.backbone.dim
        self.correction = (
            FeatureCorrection(embed_dim) if cfg.feature_correction else None
        )
        layers: list[nn.Module] = [nn.LayerNorm(embed_dim)]
        if cfg.head_hidden:
            layers += [nn.Linear(embed_dim, cfg.head_hidden), nn.GELU()]
            if cfg.head_dropout > 0:
                layers.append(nn.Dropout(cfg.head_dropout))
            layers.append(nn.Linear(cfg.head_hidden, 1))
        else:
            layers.append(nn.Linear(embed_dim, 1))
        self.head = nn.Sequential(*layers)

    def features(self, x: Tensor, degraded: bool = False) -> Tensor:
        """Extract pooled backbone features, with optional residual correction if degraded."""
        feature_embedding = self.backbone(x)
        if degraded and self.correction is not None:
            feature_embedding = self.correction(feature_embedding)
        return feature_embedding

    def forward(self, x: Tensor, degraded: bool = False) -> tuple[Tensor, Tensor]:
        """Forward pass returning (logits, features).

        Args:
            x: Input image tensor of shape (B, 3, H, W).
            degraded: Whether inputs are degraded (applies FeatureCorrection if enabled).

        Returns:
            Tuple of (logits of shape (B,), feature_embedding of shape (B, D)).
        """
        feature_embedding = self.features(x, degraded=degraded)
        return self.head(feature_embedding).squeeze(-1), feature_embedding

    @torch.no_grad()
    def predict_crops(self, crops: Tensor, reduce: str = "mean") -> Tensor:
        """Aggregate multi-crop logits into one probability per image.

        Args:
            crops: Tensor of shape (B, K, 3, H, W) where K is number of crops per image.
            reduce: Aggregation strategy ('mean', 'max', 'logit_mean').

        Returns:
            Tensor of shape (B,) containing aggregated probabilities in [0, 1].
        """
        batch_size, num_crops = crops.shape[:2]
        logits, _ = self(crops.flatten(0, 1))
        crop_probabilities = torch.sigmoid(logits).view(batch_size, num_crops)
        if reduce == "max":
            return crop_probabilities.max(dim=1).values
        if reduce == "mean":
            return crop_probabilities.mean(dim=1)
        if reduce == "logit_mean":
            return torch.sigmoid(logits.view(batch_size, num_crops).mean(dim=1))
        raise ValueError(f"unknown reduce: {reduce}")

    def param_counts(self) -> tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        return total, trainable

    def describe(self) -> str:
        total, trainable = self.param_counts()
        return (
            f"{self.cfg.backbone.name} @ {self.cfg.input_size}px "
            f"| pool={self.cfg.backbone.pooling} crop={self.cfg.crop_mode} "
            f"| {total / 1e6:.1f}M params, {trainable / 1e6:.2f}M trainable"
        )
