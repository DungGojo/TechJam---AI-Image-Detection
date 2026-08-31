"""Stable inference contract implemented by every detector."""

from __future__ import annotations

import abc

import numpy as np
from PIL import Image


class Scorer(abc.ABC):
    """Anything that returns one P(AI-generated) value per input image."""

    name: str = "unnamed"

    @abc.abstractmethod
    def score(self, images: list[Image.Image]) -> np.ndarray:
        """Return probabilities in input order."""

    def n_params(self) -> int:
        return 0

    def score_checked(self, images: list[Image.Image]) -> np.ndarray:
        out = np.asarray(self.score(images), dtype=np.float64).ravel()
        if out.shape[0] != len(images):
            raise ValueError(
                f"{self.name}.score returned {out.shape[0]} scores for {len(images)} images"
            )
        if not np.all(np.isfinite(out)):
            raise ValueError(f"{self.name}.score returned non-finite values")
        if len(out) and (out.min() < 0.0 or out.max() > 1.0):
            raise ValueError(
                f"{self.name}.score returned values outside [0, 1]: "
                f"[{out.min():.4f}, {out.max():.4f}]"
            )
        return out
