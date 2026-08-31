"""Inference adapter for a fitted stacking head."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from Modeling.common.device import get_device
from Modeling.common.interfaces import Scorer
from Modeling.dinov3.backbone_cache import FrozenBackbone
from Modeling.registry import get, register
from Modeling.stacking.head import EPS, load


class StackingScorer(Scorer):
    name = "stacking"

    def __init__(
        self,
        checkpoint: str = "weights/stacking/default/best_robust.pt",
        device: str | None = None,
        n_crops: int | None = None,
        seed: int = 1337,
        batch_size: int = 64,
    ):
        self.device = get_device(device)
        self.seed = seed
        self.batch_size = batch_size

        self.model, state = load(Path(checkpoint), map_location=self.device)
        self.model.to(self.device).eval()
        self.n_crops = int(n_crops or state.get("crops", 5))

        # Never pair a fitted head with a different backbone.
        self.backbone = FrozenBackbone(
            state["backbone"],
            crop_size=int(state.get("crop_size", 224)),
            device=str(self.device),
            allow_fallback=False,
        )

        self.experts = [get(name) for name in self.model.cfg.expert_names]
        if len(self.experts) != self.model.cfg.extra_dim:
            raise RuntimeError(
                f"checkpoint expects {self.model.cfg.extra_dim} expert inputs but names "
                f"{len(self.experts)} of them; the checkpoint is inconsistent"
            )

    def n_params(self) -> int:
        return (
            self.backbone.n_params()
            + self.model.n_params()
            + sum(expert.n_params() for expert in self.experts)
        )

    def describe(self) -> str:
        experts = ", ".join(self.model.cfg.expert_names) or "none"
        return (
            f"{self.backbone.name} (frozen) + {self.model.describe()}\n"
            f"  crops={self.n_crops} experts={experts} total={self.n_params():,} params"
        )

    @torch.inference_mode()
    def score(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty(0, dtype=np.float64)

        expert_logits = None
        if self.experts:
            probabilities = np.stack(
                [expert.score_checked(images) for expert in self.experts], axis=1
            ).astype(np.float32)
            clipped = np.clip(probabilities, EPS, 1 - EPS)
            expert_logits = torch.from_numpy(
                np.log(clipped) - np.log1p(-clipped)
            ).to(self.device)

        crops: list[Image.Image] = []
        for index, image in enumerate(images):
            rng = np.random.default_rng(self.seed + index)
            crops += self.backbone.crops(image, self.n_crops, rng)

        embeddings = np.empty((len(crops), self.backbone.dim), dtype=np.float16)
        for start in range(0, len(crops), self.batch_size):
            chunk = crops[start : start + self.batch_size]
            embeddings[start : start + len(chunk)] = self.backbone.embed(chunk)

        features = torch.from_numpy(embeddings.astype(np.float32)).to(self.device)
        extra = None
        if expert_logits is not None:
            extra = expert_logits.repeat_interleave(self.n_crops, dim=0)

        logits, _ = self.model(features, extra, degraded=False)
        per_crop = torch.sigmoid(logits.float()).view(len(images), self.n_crops)
        return per_crop.mean(dim=1).cpu().numpy().astype(np.float64)


register("stacking")(lambda **kwargs: StackingScorer(**kwargs))
