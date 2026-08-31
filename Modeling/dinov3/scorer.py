"""Inference adapter for a trained DINOv3 expert checkpoint."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from Data_Pipeline.datasets import take_crop, to_tensor
from Modeling.common.device import get_device
from Modeling.common.interfaces import Scorer
from Modeling.dinov3.backbone import BackboneConfig
from Modeling.dinov3.detector import Expert, ExpertConfig
from Modeling.registry import register


class DINOv3Scorer(Scorer):
    name = "dinov3"

    def __init__(
        self,
        checkpoint: str = "weights/dinov3/default/best_robust.pt",
        device: str | None = None,
        n_crops: int = 8,
        seed: int = 1337,
    ):
        self.device = get_device(device)
        self.n_crops = n_crops
        self.seed = seed
        state = torch.load(
            Path(checkpoint), map_location=self.device, weights_only=False
        )
        raw = dict(state.get("model_config") or {})
        if not raw:
            raise RuntimeError(
                "checkpoint has no model_config; retrain or use a new-format checkpoint"
            )
        backbone = BackboneConfig(**raw.pop("backbone"))
        allowed = {field.name for field in fields(ExpertConfig)} - {"backbone"}
        config = ExpertConfig(
            backbone=backbone, **{k: v for k, v in raw.items() if k in allowed}
        )
        self.model = Expert(config).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @torch.inference_mode()
    def score(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty(0, dtype=np.float64)
        batches = []
        for index, image in enumerate(images):
            rng = np.random.default_rng((self.seed, index))
            crops = [
                to_tensor(
                    take_crop(
                        image,
                        self.model.cfg.input_size,
                        self.model.cfg.crop_mode,
                        rng,
                    )
                )
                for _ in range(self.n_crops)
            ]
            batches.append(torch.stack(crops))
        probabilities = self.model.predict_crops(torch.stack(batches).to(self.device))
        return probabilities.float().cpu().numpy().astype(np.float64)


register("dinov3")(lambda **kwargs: DINOv3Scorer(**kwargs))
