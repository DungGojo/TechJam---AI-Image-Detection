"""Inference adapter for a trained DINOv3 expert checkpoint."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_pipeline.datasets import take_crop, to_tensor
from modeling.common.device import get_device
from modeling.common.interfaces import Scorer
from modeling.dinov3.backbone import BackboneConfig
from modeling.dinov3.detector import Expert, ExpertConfig
from modeling.registry import register


class DINOv3Scorer(Scorer):
    """Inference scorer evaluating images with multi-crop Test-Time Augmentation (TTA) on fine-tuned DINOv3."""

    name = "dinov3"

    def __init__(
        self,
        checkpoint: str = "checkpoints/dinov3/default/best_robust.pt",
        device: str | None = None,
        n_crops: int = 8,
        seed: int = 1337,
    ):
        self.device = get_device(device)
        self.n_crops = n_crops
        self.seed = seed
        checkpoint_state = torch.load(
            Path(checkpoint), map_location=self.device, weights_only=False
        )
        raw_model_config = dict(checkpoint_state.get("model_config") or {})
        if not raw_model_config:
            raise RuntimeError(
                "checkpoint has no model_config; retrain or use a new-format checkpoint"
            )
        backbone = BackboneConfig(**raw_model_config.pop("backbone"))
        allowed_expert_fields = {field.name for field in fields(ExpertConfig)} - {
            "backbone"
        }
        config = ExpertConfig(
            backbone=backbone,
            **{k: v for k, v in raw_model_config.items() if k in allowed_expert_fields},
        )
        self.model = Expert(config).to(self.device)
        self.model.load_state_dict(checkpoint_state["model"])
        self.model.eval()

    def n_params(self) -> int:
        """Return total parameter count."""
        return sum(parameter.numel() for parameter in self.model.parameters())

    @torch.inference_mode()
    def score(self, images: list[Image.Image]) -> np.ndarray:
        """Score images using n_crops test-time crops with reproducible per-image seeds."""
        if not images:
            return np.empty(0, dtype=np.float64)
        batch_crop_tensors = []
        for index, image in enumerate(images):
            rng = np.random.default_rng((self.seed, index))
            image_crop_tensors = [
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
            batch_crop_tensors.append(torch.stack(image_crop_tensors))
        probabilities = self.model.predict_crops(
            torch.stack(batch_crop_tensors).to(self.device)
        )
        return probabilities.float().cpu().numpy().astype(np.float64)


register("dinov3")(lambda **kwargs: DINOv3Scorer(**kwargs))
