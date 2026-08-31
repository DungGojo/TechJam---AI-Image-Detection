"""Stable inference contract implemented by every detector."""

from __future__ import annotations

import abc

import numpy as np
from PIL import Image


class Scorer(abc.ABC):
    """Abstract base class for anything that returns one P(AI-generated) probability per input image."""

    name: str = "unnamed"

    @abc.abstractmethod
    def score(self, images: list[Image.Image]) -> np.ndarray:
        """Return AI-generated probabilities in input order.

        Args:
            images: List of PIL Image objects in RGB format.

        Returns:
            np.ndarray of shape (len(images),) containing float values representing P(AI-generated).
        """

    def n_params(self) -> int:
        """Return the total number of parameters across all internal networks/backbones."""
        return 0

    def score_checked(self, images: list[Image.Image]) -> np.ndarray:
        """Score a list of images with strict output shape, finiteness, and range [0, 1] validation.

        Args:
            images: List of PIL Image objects.

        Returns:
            1D np.ndarray of shape (len(images),) with dtype float64.

        Raises:
            ValueError: If the output length mismatches, contains non-finite numbers, or has values outside [0, 1].
        """
        scores_array = np.asarray(self.score(images), dtype=np.float64).ravel()
        if scores_array.shape[0] != len(images):
            raise ValueError(
                f"{self.name}.score returned {scores_array.shape[0]} scores for {len(images)} images"
            )
        if not np.all(np.isfinite(scores_array)):
            raise ValueError(f"{self.name}.score returned non-finite values")
        if len(scores_array) and (scores_array.min() < 0.0 or scores_array.max() > 1.0):
            raise ValueError(
                f"{self.name}.score returned values outside [0, 1]: "
                f"[{scores_array.min():.4f}, {scores_array.max():.4f}]"
            )
        return scores_array

