"""Manifest-backed clean and degraded image datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_pipeline.image_io import load_image, resolve_path
from modeling.common.degradations import apply_chain, pick_tier, sample_chain, tier_weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MEAN_TENSOR = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD_TENSOR = torch.tensor(IMAGENET_STD).view(3, 1, 1)

load_rgb = load_image


def to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image in [0, 255] to ImageNet-normalized (3, H, W) float32 tensor."""
    normalized_arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor_img = torch.from_numpy(normalized_arr).permute(2, 0, 1)
    return (tensor_img - _MEAN_TENSOR) / _STD_TENSOR


def take_crop(img: Image.Image, size: int, mode: str, rng: np.random.Generator) -> Image.Image:
    """Return a square crop of `size` x `size` using the configured crop policy."""
    width, height = img.size
    if mode == "resize":
        return img.resize((size, size), Image.BICUBIC)
    if mode == "crop_then_resize":
        side = int(rng.integers(int(min(width, height) * 0.5), min(width, height) + 1)) if min(width, height) > 8 else min(width, height)
        x = int(rng.integers(0, max(1, width - side + 1)))
        y = int(rng.integers(0, max(1, height - side + 1)))
        return img.crop((x, y, x + side, y + side)).resize((size, size), Image.BICUBIC)
    if mode != "random_crop":
        raise ValueError(f"unknown crop mode: {mode}")
    if width < size or height < size:
        pad = Image.new("RGB", (max(width, size), max(height, size)))
        for offset_x in range(0, pad.size[0], width):
            for offset_y in range(0, pad.size[1], height):
                pad.paste(img, (offset_x, offset_y))
        img, (width, height) = pad, pad.size
    x = int(rng.integers(0, width - size + 1))
    y = int(rng.integers(0, height - size + 1))
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
    """Dataset yielding clean and degraded views originating from the same native crop.

    Note: DataLoader workers must use persistent_workers=False so workers re-read self.epoch
    when the curriculum escalates each epoch.
    """

    def __init__(self, cfg: DataConfig, split: str | None = None):
        self.cfg = cfg
        manifest_df = pd.read_csv(resolve_path(cfg.manifest))
        missing_cols = {"path", "label"} - set(manifest_df.columns)
        if missing_cols:
            raise ValueError(f"manifest is missing columns: {sorted(missing_cols)}")
        if split is not None and "split" in manifest_df.columns:
            manifest_df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        if len(manifest_df) == 0:
            raise ValueError(f"manifest has no rows for split={split!r}")
        self.df = manifest_df
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current training epoch to advance the curriculum tier schedule."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        # Keep each epoch reproducible while varying augmentations across epochs.
        rng = np.random.default_rng((self.cfg.seed, self.epoch, index))
        img = load_rgb(row["path"])
        crop = take_crop(img, self.cfg.input_size, self.cfg.crop_mode, rng)
        if self.cfg.hflip and rng.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

        label = torch.tensor(float(row["label"]))
        clean = to_tensor(crop)

        curriculum_weights = tier_weights(self.epoch, self.cfg.total_epochs, self.cfg.schedule)
        degradation_chain = sample_chain(rng, pick_tier(rng, curriculum_weights), self.cfg.pool)
        degraded_image = apply_chain(crop, degradation_chain, rng=rng)

        if not self.cfg.pairwise:
            return to_tensor(degraded_image), label

        degraded = to_tensor(_fit(degraded_image, self.cfg.input_size))
        return clean, degraded, label


def _fit(img: Image.Image, size: int) -> Image.Image:
    """Restore the target size by padding without resampling pixels."""
    if img.size == (size, size):
        return img
    width, height = img.size
    if width >= size and height >= size:
        left, top = (width - size) // 2, (height - size) // 2
        return img.crop((left, top, left + size, top + size))
    out = Image.new("RGB", (size, size))
    for offset_x in range(0, size, max(1, width)):
        for offset_y in range(0, size, max(1, height)):
            out.paste(img, (offset_x, offset_y))
    return out


class EvalCropDataset(Dataset):
    """Multi-crop evaluation dataset: returns (K, 3, H, W) per image for test-time aggregation."""

    def __init__(self, paths: list[str], input_size: int, n_crops: int = 8, seed: int = 0,
                 crop_mode: str = "random_crop"):
        self.paths = paths
        self.input_size = input_size
        self.n_crops = n_crops
        self.seed = seed
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng((self.seed, idx))
        img = load_rgb(self.paths[idx])
        crops = [
            to_tensor(take_crop(img, self.input_size, self.crop_mode, rng))
            for _ in range(self.n_crops)
        ]
        return torch.stack(crops), self.paths[idx]

