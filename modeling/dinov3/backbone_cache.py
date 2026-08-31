"""Frozen backbone feature extraction."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from modeling.common.device import get_device

PRIMARY = "vit_large_patch16_dinov3.lvd1689m"
FALLBACK = "vit_large_patch14_dinov2.lvd142m"


class FrozenBackbone:
    """timm backbone in eval mode, multi-crop at native resolution."""

    def __init__(
        self,
        name: str = PRIMARY,
        *,
        crop_size: int = 224,
        device: str | None = None,
        allow_fallback: bool = True,
    ):
        import timm

        self.device = get_device(device)
        self.crop_size = crop_size
        self.requested = name

        try:
            self.model = timm.create_model(
                name, pretrained=True, num_classes=0, dynamic_img_size=True
            )
            self.name = name
            self.used_fallback = False
        except Exception as exc:  # noqa: BLE001 - the DINOv3 gate is expected
            if not allow_fallback:
                raise
            print(
                f"[backbone] {name} unavailable ({exc.__class__.__name__}); "
                f"falling back to {FALLBACK}"
            )
            self.model = timm.create_model(
                FALLBACK, pretrained=True, num_classes=0, dynamic_img_size=True
            )
            self.name = FALLBACK
            self.used_fallback = True

        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
        self.dim = int(self.model.num_features)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def crops(
        self, img: Image.Image, n: int, rng: np.random.Generator
    ) -> list[Image.Image]:
        """Create reproducible native-resolution crops without upscaling."""
        img = img.convert("RGB")
        w, h = img.size
        size = self.crop_size

        if w < size or h < size:
            arr = np.asarray(img)
            pad_h, pad_w = max(0, size - h), max(0, size - w)
            arr = np.pad(
                arr,
                (
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                    (0, 0),
                ),
                mode="reflect",
            )
            img = Image.fromarray(arr, "RGB")
            w, h = img.size

        out = []
        for _ in range(n):
            left = int(rng.integers(0, w - size + 1))
            top = int(rng.integers(0, h - size + 1))
            out.append(img.crop((left, top, left + size, top + size)))
        return out

    @torch.inference_mode()
    def embed(self, crops: list[Image.Image]) -> np.ndarray:
        """(len(crops), dim) float16 embeddings."""
        if not crops:
            return np.empty((0, self.dim), dtype=np.float16)
        feats = self.model(self._tensor(crops).to(self.device))
        return feats.float().cpu().numpy().astype(np.float16)

    def _tensor(self, crops: list[Image.Image]) -> torch.Tensor:
        batch = (
            torch.from_numpy(np.stack([np.asarray(c, dtype=np.uint8) for c in crops]))
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
        )
        return (batch - self.mean) / self.std

    @torch.inference_mode()
    def token_layers(self, crops: list[Image.Image], layers: list[int]) -> np.ndarray:
        """Collect normalized tokens from selected transformer blocks in one pass."""
        if not crops:
            return np.empty((0, len(layers), 0, self.dim), dtype=np.float16)
        if not layers:
            raise ValueError("at least one intermediate layer is required")
        depth = len(getattr(self.model, "blocks", ()))
        if depth and (min(layers) < 0 or max(layers) >= depth):
            raise ValueError(f"layers must be in [0, {depth - 1}], got {layers}")
        forward = getattr(self.model, "forward_intermediates", None)
        if forward is None:
            raise TypeError(f"{self.name!r} does not expose forward_intermediates")

        outputs = forward(
            self._tensor(crops).to(self.device),
            indices=layers,
            return_prefix_tokens=True,
            norm=True,
            output_fmt="NLC",
            intermediates_only=True,
        )
        joined = []
        for item in outputs:
            if isinstance(item, tuple):
                patches, prefix = item
                item = torch.cat([prefix, patches], dim=1)
            joined.append(item)
        tokens = torch.stack(joined, dim=1)
        return tokens.float().cpu().numpy().astype(np.float16)
