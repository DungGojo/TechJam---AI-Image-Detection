"""DINOv3 detection expert."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from Modeling.dinov3.backbone import Backbone, BackboneConfig


@dataclass
class ExpertConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    head_hidden: int = 512
    head_dropout: float = 0.0
    feature_correction: bool = False
    input_size: int = 224
    crop_mode: str = "random_crop"


class FeatureCorrection(nn.Module):
    """Map degraded features toward clean features with a residual FFN."""

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
    """Returns a single logit. P(AI-generated) = sigmoid(logit)."""

    def __init__(self, cfg: ExpertConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg.backbone)
        d = self.backbone.dim
        self.correction = FeatureCorrection(d) if cfg.feature_correction else None
        layers: list[nn.Module] = [nn.LayerNorm(d)]
        if cfg.head_hidden:
            layers += [nn.Linear(d, cfg.head_hidden), nn.GELU()]
            if cfg.head_dropout > 0:
                layers.append(nn.Dropout(cfg.head_dropout))
            layers.append(nn.Linear(cfg.head_hidden, 1))
        else:
            layers.append(nn.Linear(d, 1))
        self.head = nn.Sequential(*layers)


    def features(self, x: Tensor, degraded: bool = False) -> Tensor:
        z = self.backbone(x)
        if degraded and self.correction is not None:
            z = self.correction(z)
        return z

    def forward(self, x: Tensor, degraded: bool = False) -> tuple[Tensor, Tensor]:
        z = self.features(x, degraded=degraded)
        return self.head(z).squeeze(-1), z

    @torch.no_grad()
    def predict_crops(self, crops: Tensor, reduce: str = "mean") -> Tensor:
        """Aggregate crop logits into one probability per image."""
        b, k = crops.shape[:2]
        logits, _ = self(crops.flatten(0, 1))
        p = torch.sigmoid(logits).view(b, k)
        if reduce == "max":
            return p.max(dim=1).values
        if reduce == "mean":
            return p.mean(dim=1)
        if reduce == "logit_mean":
            return torch.sigmoid(logits.view(b, k).mean(dim=1))
        raise ValueError(f"unknown reduce: {reduce}")

    def param_counts(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def describe(self) -> str:
        total, trainable = self.param_counts()
        return (
            f"{self.cfg.backbone.name} @ {self.cfg.input_size}px "
            f"| pool={self.cfg.backbone.pooling} crop={self.cfg.crop_mode} "
            f"| {total/1e6:.1f}M params, {trainable/1e6:.2f}M trainable"
        )
