"""Post-fusion temperature calibration fitted on held-out validation data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TemperatureCalibrator:
    temperature: float = 1.0
    eps: float = 1e-6

    def fit(self, probabilities, labels, *, max_iter: int = 100) -> "TemperatureCalibrator":
        probs = torch.as_tensor(np.asarray(probabilities), dtype=torch.float64).clamp(
            self.eps, 1 - self.eps
        )
        targets = torch.as_tensor(np.asarray(labels), dtype=torch.float64)
        if probs.shape != targets.shape or probs.numel() == 0:
            raise ValueError("probabilities and labels must be non-empty with matching shapes")
        log_temperature = torch.tensor(
            np.log(max(self.temperature, self.eps)), dtype=torch.float64, requires_grad=True
        )
        logits = torch.logit(probs)
        optimizer = torch.optim.LBFGS(
            [log_temperature], max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits / log_temperature.exp(), targets
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_temperature.detach().exp().clamp_min(self.eps))
        return self

    def transform(self, probabilities) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=np.float64)
        clipped = np.clip(probs, self.eps, 1 - self.eps)
        logits = np.log(clipped) - np.log1p(-clipped)
        scaled = np.clip(logits / self.temperature, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-scaled))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"temperature": self.temperature}, indent=2) + "\n")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureCalibrator":
        return cls(**json.loads(Path(path).read_text()))
