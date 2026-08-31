"""Robust validation shared by training and standalone evaluation."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from Data_Pipeline.datasets import load_rgb, take_crop, to_tensor
from Evaluation.metrics import compute
from Modeling.common.degradations import Cell, apply_chain, discrete_grid


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
    model.eval()
    df = manifest_df if len(manifest_df) <= max_images else manifest_df.sample(
        max_images, random_state=seed
    )
    paths = df["path"].tolist()
    labels = np.asarray(df["label"].tolist(), dtype=int)
    results: dict[str, dict[str, float]] = {}
    grid_cells = [None, *(cells if cells is not None else discrete_grid())]
    batches_per_cell = math.ceil(len(paths) / batch_size)
    progress = tqdm(
        total=len(grid_cells) * batches_per_cell,
        desc="Validation",
        unit="batch",
        mininterval=progress_interval_seconds,
        maxinterval=progress_interval_seconds,
        miniters=0,
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    for cell in grid_cells:
        label = "clean" if cell is None else cell.label
        progress.set_description(f"Validation ({label})", refresh=False)
        scores = np.zeros(len(paths), dtype=np.float64)
        for start in range(0, len(paths), batch_size):
            batch = []
            chunk = paths[start : start + batch_size]
            for offset, path in enumerate(chunk):
                rng = np.random.default_rng((seed, start + offset))
                image = load_rgb(path)
                if cell is not None:
                    image = apply_chain(image, [cell], rng=rng)
                batch.append(to_tensor(take_crop(image, input_size, crop_mode, rng)))
            logits = model(torch.stack(batch).to(device))
            logits = logits[0] if isinstance(logits, tuple) else logits
            scores[start : start + len(chunk)] = torch.sigmoid(
                logits.detach().float().reshape(-1)
            ).cpu().numpy()
            progress.update(1)
            if progress.n >= progress.total:
                progress.refresh()
        raw = compute(labels, scores)
        key = "clean" if cell is None else cell.label
        results[key] = {
            "auc": raw["roc_auc"],
            "tpr@1fpr": raw["tpr_at_fpr0.01"],
            "tpr@5fpr": raw["tpr_at_fpr0.05"],
            "acc": raw["accuracy"],
        }
    progress.close()
    model.train()
    return results


def summarise(grid: dict[str, dict[str, float]]) -> dict[str, float]:
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
