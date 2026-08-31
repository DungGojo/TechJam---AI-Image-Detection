"""Released Community Forensics checkpoint behind the shared scorer API."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from modeling.common.device import get_device
from modeling.common.interfaces import Scorer
from modeling.registry import register

REPOSITORY = "buildborderless/CommunityForensics-DeepfakeDet-ViT"


class CommunityForensicsScorer(Scorer):
    """ViT-S/16 baseline using its checkpoint-defined preprocessing."""

    name = "community_forensics"

    def __init__(
        self, device: str | None = None, batch_dtype: torch.dtype = torch.float32
    ):
        """Load pretrained Community Forensics ViT-S/16 model and processor."""
        from transformers import AutoImageProcessor, ViTForImageClassification

        self.device = get_device(device)
        self.dtype = batch_dtype
        self.processor = AutoImageProcessor.from_pretrained(REPOSITORY)
        self.model = ViTForImageClassification.from_pretrained(REPOSITORY)
        self.model.eval().to(self.device)
        if int(getattr(self.model.config, "num_labels", 1)) != 1:
            raise RuntimeError(
                "Community Forensics checkpoint must expose one output logit"
            )

    def n_params(self) -> int:
        """Return total parameter count for the ViT model."""
        return sum(param.numel() for param in self.model.parameters())

    @torch.inference_mode()
    def score(self, images: list[Image.Image]) -> np.ndarray:
        """Score a list of RGB images directly without multi-crop TTA."""
        if not images:
            return np.empty(0, dtype=np.float64)
        batch = self.processor(
            images=[image.convert("RGB") for image in images], return_tensors="pt"
        )
        pixels = batch["pixel_values"].to(self.device, dtype=self.dtype)
        logits = self.model(pixel_values=pixels).logits
        return (
            torch.sigmoid(logits.float().squeeze(-1)).cpu().numpy().astype(np.float64)
        )


register("community_forensics")(lambda **kwargs: CommunityForensicsScorer(**kwargs))
