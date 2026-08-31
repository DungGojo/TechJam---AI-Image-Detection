"""Trainable head for cached frozen features."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn

from Modeling.common.device import get_device
from Modeling.dinov3.detector import FeatureCorrection
from Modeling.dinov3.losses import LossWeights, RobustLoss
from Modeling.stacking.features import FeatureSet

EPS = 1e-6


@dataclass
class StackingConfig:
    dim: int = 768
    extra_dim: int = 0
    head_hidden: int = 256
    head_dropout: float = 0.1
    feature_correction: bool = True
    normalise: bool = True
    expert_names: list[str] = field(default_factory=list)


class StackingHead(nn.Module):
    """(features[, expert logits]) -> one logit. P(AI-generated) = sigmoid(logit)."""

    def __init__(self, cfg: StackingConfig):
        super().__init__()
        self.cfg = cfg
        self.correction = FeatureCorrection(cfg.dim) if cfg.feature_correction else None
        width = cfg.dim + cfg.extra_dim
        layers: list[nn.Module] = [nn.LayerNorm(width)]
        if cfg.head_hidden:
            layers += [nn.Linear(width, cfg.head_hidden), nn.GELU()]
            if cfg.head_dropout > 0:
                layers.append(nn.Dropout(cfg.head_dropout))
            layers.append(nn.Linear(cfg.head_hidden, 1))
        else:
            layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

    def features(self, feat: Tensor, degraded: bool = False) -> Tensor:
        z = feat
        if self.cfg.normalise:
            z = torch.nn.functional.normalize(z, dim=-1)
        if degraded and self.correction is not None:
            z = self.correction(z)
        return z

    def forward(
        self, feat: Tensor, extra: Tensor | None = None, degraded: bool = False
    ) -> tuple[Tensor, Tensor]:
        z = self.features(feat, degraded=degraded)
        joined = z if extra is None else torch.cat([z, extra], dim=-1)
        return self.head(joined).squeeze(-1), z

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def describe(self) -> str:
        return (
            f"stacking head over {self.cfg.dim}-d features"
            f"{f' + {self.cfg.extra_dim} expert logits' if self.cfg.extra_dim else ''} "
            f"| hidden={self.cfg.head_hidden} correction={self.cfg.feature_correction} "
            f"| {self.n_params():,} trainable params"
        )


@dataclass
class FitConfig:
    epochs: int = 30
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.02
    seed: int = 1337
    device: str | None = None
    val_every: int = 1
    loss: LossWeights = field(default_factory=LossWeights)


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float32), EPS, 1 - EPS)
    return np.log(q) - np.log1p(-q)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


@torch.no_grad()
def predict_variant(
    model: StackingHead,
    feature_set: FeatureSet,
    variant: str,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Per-row P(AI-generated), averaged over the row's cached crops."""
    model.eval()
    features = feature_set.features(variant)                 # (n, crops, dim)
    n, crops, _ = features.shape
    flat = torch.from_numpy(features.reshape(n * crops, -1))
    scores = feature_set.extra_for(variant)
    extra = None if scores is None else torch.from_numpy(np.repeat(_logit(scores), crops, axis=0))

    degraded = variant != "clean"
    out = np.empty(n * crops, dtype=np.float64)
    for start in range(0, n * crops, batch_size):
        chunk = flat[start : start + batch_size].to(device)
        chunk_extra = None if extra is None else extra[start : start + batch_size].to(device)
        logits, _ = model(chunk, chunk_extra, degraded=degraded)
        out[start : start + batch_size] = torch.sigmoid(logits.float()).cpu().numpy()
    return out.reshape(n, crops).mean(axis=1)


def evaluate(
    model: StackingHead, feature_set: FeatureSet, *, device: torch.device
) -> dict[str, float]:
    """Compute clean, mean robust, worst-cell AUC, and their gap."""
    labels = feature_set.labels
    summary: dict[str, float] = {}
    summary["clean_auc"] = _auc(labels, predict_variant(model, feature_set, "clean", device=device))

    cells = {}
    for name in feature_set.degraded_names():
        cells[name] = _auc(labels, predict_variant(model, feature_set, name, device=device))
    for name, value in cells.items():
        summary[f"auc_{name.replace('=', '')}"] = value

    values = [v for v in cells.values() if not np.isnan(v)]
    summary["robust_auc"] = float(np.mean(values)) if values else summary["clean_auc"]
    summary["worst_cell_auc"] = float(np.min(values)) if values else summary["clean_auc"]
    summary["gap"] = summary["clean_auc"] - summary["robust_auc"]
    return summary


def fit(
    train_set: FeatureSet,
    val_set: FeatureSet | None,
    cfg: StackingConfig,
    fit_cfg: FitConfig | None = None,
    *,
    progress=None,
) -> dict:
    """Fit the stacking head on paired cached clean and degraded features."""
    fit_cfg = fit_cfg or FitConfig()
    device = get_device(fit_cfg.device)
    torch.manual_seed(fit_cfg.seed)
    rng = np.random.default_rng(fit_cfg.seed)

    model = StackingHead(cfg).to(device)
    criterion = RobustLoss(fit_cfg.loss).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=fit_cfg.lr, weight_decay=fit_cfg.weight_decay
    )

    clean = torch.from_numpy(train_set.features("clean"))          # (n, crops, dim)
    degraded_names = train_set.degraded_names()
    degraded = {name: torch.from_numpy(train_set.features(name)) for name in degraded_names}
    labels = torch.from_numpy(train_set.labels).float()
    extra_clean = train_set.extra_for("clean")
    extra_clean = None if extra_clean is None else torch.from_numpy(_logit(extra_clean))
    extra_degraded = {
        name: torch.from_numpy(_logit(train_set.extra_for(name))) for name in degraded_names
    } if train_set.extra_dim else {}

    n_rows, n_crops, _ = clean.shape
    pairs = np.array([(r, c) for r in range(n_rows) for c in range(n_crops)], dtype=np.int64)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, fit_cfg.epochs)
    )

    history: list[dict] = []
    best_state, best_robust = None, -np.inf

    for epoch in range(fit_cfg.epochs):
        model.train()
        order = rng.permutation(len(pairs))
        totals: dict[str, float] = {}
        n_batches = 0

        for start in range(0, len(order), fit_cfg.batch_size):
            picks = pairs[order[start : start + fit_cfg.batch_size]]
            rows = torch.from_numpy(picks[:, 0])
            crops = torch.from_numpy(picks[:, 1])

            feat_clean = clean[rows, crops].to(device)
            target = labels[rows].to(device)
            batch_extra = None if extra_clean is None else extra_clean[rows].to(device)

            logit_clean, z_clean = model(feat_clean, batch_extra, degraded=False)

            logit_degraded = z_degraded = None
            paired = False
            if degraded_names:
                name = degraded_names[int(rng.integers(len(degraded_names)))]
                paired = train_set.variants[name].aligned
                feat_degraded = degraded[name][rows, crops].to(device)
                degraded_extra = (
                    extra_degraded[name][rows].to(device) if extra_degraded else None
                )
                logit_degraded, z_degraded = model(
                    feat_degraded, degraded_extra, degraded=True
                )

            total, parts = criterion(
                logit_clean,
                target,
                logit_degraded,
                z_clean if paired else None,
                z_degraded if paired else None,
            )
            if logit_degraded is not None and not paired:
                # KL requires paired views; degraded classification does not.
                total = total - fit_cfg.loss.kl * criterion.kl(logit_clean, logit_degraded)
                parts.pop("kl", None)

            optimiser.zero_grad(set_to_none=True)
            total.backward()
            optimiser.step()

            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1

        scheduler.step()
        train_row = {"epoch": epoch}
        for key, value in totals.items():
            train_row[f"train_{key}"] = value / max(1, n_batches)

        summary: dict[str, float] = {}
        if val_set is not None and (epoch + 1) % fit_cfg.val_every == 0:
            summary = evaluate(model, val_set, device=device)
            if summary["robust_auc"] > best_robust:
                best_robust = summary["robust_auc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append({"train": train_row, "summary": summary})
        if progress is not None:
            progress(epoch, train_row, summary)

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_robust_auc": None if best_state is None else float(best_robust),
        "device": str(device),
    }




def save(
    path: str | Path,
    model: StackingHead,
    *,
    backbone: str,
    crops: int,
    crop_size: int,
    history: list[dict] | None = None,
    best_robust_auc: float | None = None,
    variants: list[str] | None = None,
) -> Path:
    """Write an inference checkpoint. Everything needed to rebuild the scorer."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "stacking_config": asdict(model.cfg),
            "backbone": backbone,
            "crops": crops,
            "crop_size": crop_size,
            "history": history or [],
            "best_robust_auc": best_robust_auc,
            "variants": variants or [],
        },
        target,
    )
    return target


def load(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[StackingHead, dict]:
    state = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "stacking_config" not in state:
        raise RuntimeError(
            f"{path} is not a stacking checkpoint (no stacking_config). A DINOv3 "
            "expert checkpoint loads with Modeling.dinov3.scorer instead."
        )
    cfg = StackingConfig(**state["stacking_config"])
    model = StackingHead(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def history_frame(history: list[dict]):
    """Flatten to the same column layout the DINOv3 trainer's history produces."""
    import pandas as pd

    rows = []
    for entry in history:
        row: dict = {}
        row.update(entry.get("train", {}))
        row.update(entry.get("summary", {}))
        rows.append(row)
    return pd.DataFrame(rows)
