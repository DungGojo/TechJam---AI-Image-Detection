"""Manifest-backed clean and degraded image datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from Data_Pipeline.image_io import load_image, resolve_path
from Modeling.common.degradations import apply_chain, pick_tier, sample_chain, tier_weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MEAN_TENSOR = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD_TENSOR = torch.tensor(IMAGENET_STD).view(3, 1, 1)


load_rgb = load_image


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return (t - _MEAN_TENSOR) / _STD_TENSOR


def take_crop(img: Image.Image, size: int, mode: str, rng: np.random.Generator) -> Image.Image:
    """Return a square crop using the configured crop policy."""
    w, h = img.size
    if mode == "resize":
        return img.resize((size, size), Image.BICUBIC)
    if mode == "crop_then_resize":
        side = int(rng.integers(int(min(w, h) * 0.5), min(w, h) + 1)) if min(w, h) > 8 else min(w, h)
        x = int(rng.integers(0, max(1, w - side + 1)))
        y = int(rng.integers(0, max(1, h - side + 1)))
        return img.crop((x, y, x + side, y + side)).resize((size, size), Image.BICUBIC)
    if mode != "random_crop":
        raise ValueError(f"unknown crop mode: {mode}")
    if w < size or h < size:
        pad = Image.new("RGB", (max(w, size), max(h, size)))
        for ox in range(0, pad.size[0], w):
            for oy in range(0, pad.size[1], h):
                pad.paste(img, (ox, oy))
        img, (w, h) = pad, pad.size
    x = int(rng.integers(0, w - size + 1))
    y = int(rng.integers(0, h - size + 1))
    return img.crop((x, y, x + size, y + size))


@dataclass
class DataConfig:
    manifest: str
    input_size: int = 224
    crop_mode: str = "random_crop"
    pairwise: bool = True
    schedule: str = "escalate"
    total_epochs: int = 10
    pool: list[str] | None = None
    hflip: bool = True
    seed: int = 0


class PairedManifestDataset(Dataset):
    """Reads a manifest CSV with at least `path` and `label` columns."""

    def __init__(self, cfg: DataConfig, split: str | None = None):
        self.cfg = cfg
        df = pd.read_csv(resolve_path(cfg.manifest))
        missing = {"path", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"manifest has no rows for split={split!r}")
        self.df = df
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call from the training loop -- this is what makes the curriculum escalate."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        # Keep each epoch reproducible while varying augmentations across epochs.
        rng = np.random.default_rng((self.cfg.seed, self.epoch, i))
        img = load_rgb(row["path"])
        crop = take_crop(img, self.cfg.input_size, self.cfg.crop_mode, rng)
        if self.cfg.hflip and rng.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

        label = torch.tensor(float(row["label"]))
        clean = to_tensor(crop)
        if not self.cfg.pairwise:
            weights = tier_weights(self.epoch, self.cfg.total_epochs, self.cfg.schedule)
            chain = sample_chain(rng, pick_tier(rng, weights), self.cfg.pool)
            return to_tensor(apply_chain(crop, chain, rng=rng)), label

        weights = tier_weights(self.epoch, self.cfg.total_epochs, self.cfg.schedule)
        chain = sample_chain(rng, pick_tier(rng, weights), self.cfg.pool)
        degraded = to_tensor(_fit(apply_chain(crop, chain, rng=rng), self.cfg.input_size))
        return clean, degraded, label


def _fit(img: Image.Image, size: int) -> Image.Image:
    """Restore the target size by padding without resampling pixels."""
    if img.size == (size, size):
        return img
    w, h = img.size
    if w >= size and h >= size:
        left, top = (w - size) // 2, (h - size) // 2
        return img.crop((left, top, left + size, top + size))
    out = Image.new("RGB", (size, size))
    for ox in range(0, size, max(1, w)):
        for oy in range(0, size, max(1, h)):
            out.paste(img, (ox, oy))
    return out


class EvalCropDataset(Dataset):
    """Multi-crop evaluation: returns (K, 3, H, W) per image for test-time aggregation."""

    def __init__(self, paths: list[str], input_size: int, n_crops: int = 8, seed: int = 0,
                 crop_mode: str = "random_crop"):
        self.paths = paths
        self.input_size = input_size
        self.n_crops = n_crops
        self.seed = seed
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, i))
        img = load_rgb(self.paths[i])
        crops = [
            to_tensor(take_crop(img, self.input_size, self.crop_mode, rng))
            for _ in range(self.n_crops)
        ]
        return torch.stack(crops), self.paths[i]
