"""Robust validation shared by training and standalone evaluation."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from data_pipeline.datasets import load_rgb, take_crop, to_tensor
from evaluation.metrics import compute
from modeling.common.degradations import Cell, apply_chain, discrete_grid


@torch.no_grad()
def evaluate_robust(
    model: nn.Module,
    manifest_df,
    input_size: int,
    crop_mode: str,
    device: torch.device,
    cells: list[Cell] | None = None,
    max_images: int = 2000,
    batch_size: int = 32,
    seed: int = 0,
    show_progress: bool = True,
    progress_interval_seconds: float = 300.0,
) -> dict[str, dict[str, float]]:
    """Evaluate a PyTorch model on clean validation images and across every cell in the degradation grid.

    Args:
        model: Detection network in eval mode.
        manifest_df: Validation DataFrame with 'path' and 'label' columns.
        input_size: Target square crop size in pixels.
        crop_mode: Policy for cropping images ('random_crop', 'resize', 'crop_then_resize').
        device: Hardware device (cuda/mps/cpu).
        cells: Optional explicit list of degradation cells (defaults to discrete_grid()).
        max_images: Subsampling ceiling for fast validation.
        batch_size: evaluation batch size.
        seed: Random seed for subsampling and crop generation.
        show_progress: Whether to display a progress bar.
        progress_interval_seconds: Minimum seconds between progress updates.

    Returns:
        Dict mapping cell labels ('clean', 'jpeg=50', etc.) to metric dicts (auc, tpr@1fpr, acc).
    """
    model.eval()
    sampled_df = manifest_df if len(manifest_df) <= max_images else manifest_df.sample(
        max_images, random_state=seed
    )
    paths = sampled_df["path"].tolist()
    labels = np.asarray(sampled_df["label"].tolist(), dtype=int)
    results: dict[str, dict[str, float]] = {}
    evaluation_cells = [None, *(cells if cells is not None else discrete_grid())]
    batches_per_cell = math.ceil(len(paths) / batch_size)
    progress = tqdm(
        total=len(evaluation_cells) * batches_per_cell,
        desc="Validation",
        unit="batch",
        mininterval=progress_interval_seconds,
        maxinterval=progress_interval_seconds,
        miniters=0,
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    for cell in evaluation_cells:
        label = "clean" if cell is None else cell.label
        progress.set_description(f"Validation ({label})", refresh=False)
        scores = np.zeros(len(paths), dtype=np.float64)
        for start in range(0, len(paths), batch_size):
            batch_tensors = []
            batch_paths = paths[start : start + batch_size]
            for offset, path in enumerate(batch_paths):
                rng = np.random.default_rng((seed, start + offset))
                image = load_rgb(path)
                if cell is not None:
                    image = apply_chain(image, [cell], rng=rng)
                batch_tensors.append(to_tensor(take_crop(image, input_size, crop_mode, rng)))
            logits = model(torch.stack(batch_tensors).to(device))
            logits = logits[0] if isinstance(logits, tuple) else logits
            scores[start : start + len(batch_paths)] = torch.sigmoid(
                logits.detach().float().reshape(-1)
            ).cpu().numpy()
            progress.update(1)
            if progress.n >= progress.total:
                progress.refresh()
        raw_metrics = compute(labels, scores)
        key = "clean" if cell is None else cell.label
        results[key] = {
            "auc": raw_metrics["roc_auc"],
            "tpr@1fpr": raw_metrics["tpr_at_fpr0.01"],
            "tpr@5fpr": raw_metrics["tpr_at_fpr0.05"],
            "acc": raw_metrics["accuracy"],
        }
    progress.close()
    model.train()
    return results


def summarise(grid: dict[str, dict[str, float]]) -> dict[str, float]:
    """Aggregate per-cell validation metrics into clean AUC, robust mean AUC, worst cell, and degradation gap."""
    clean = grid.get("clean", {}).get("auc", float("nan"))
    degraded = [
        value["auc"] for key, value in grid.items()
        if key != "clean" and not math.isnan(value["auc"])
    ]
    robust = float(np.mean(degraded)) if degraded else float("nan")
    return {
        "clean_auc": clean,
        "robust_auc": robust,
        "worst_cell_auc": float(np.min(degraded)) if degraded else float("nan"),
        "gap": clean - robust if degraded else float("nan"),
    }

