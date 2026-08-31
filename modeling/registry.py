"""Detector scorer registry."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

from modeling.common.interfaces import Scorer

MAX_PARAMS = 2_000_000_000

_REGISTRY: dict[str, Callable[..., Scorer]] = {}


def register(name: str) -> Callable[[Callable[..., Scorer]], Callable[..., Scorer]]:
    """Register a scorer factory under the specified unique identifier."""
    def decorator(factory: Callable[..., Scorer]) -> Callable[..., Scorer]:
        if name in _REGISTRY:
            raise ValueError(f"scorer {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def available() -> list[str]:
    """Return the sorted list of registered scorer names."""
    return sorted(_REGISTRY)


def get(name: str, **kwargs) -> Scorer:
    """Instantiate and return a registered Scorer by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown scorer {name!r}; available: {available()}")
    scorer = _REGISTRY[name](**kwargs)
    scorer.name = name
    return scorer


def check_param_budget(scorer: Scorer) -> int:
    """Validate that the scorer's total parameter count is within the competition cap (2B)."""
    total_param_count = int(scorer.n_params())
    if total_param_count > MAX_PARAMS:
        raise RuntimeError(
            f"{scorer.name} has {total_param_count:,} parameters, over the {MAX_PARAMS:,} limit"
        )
    return total_param_count


class RandomScorer(Scorer):
    """Uniform-random baseline scorer used to validate the evaluation harness."""

    name = "random"

    def __init__(self, seed: int = 1337):
        self.rng = np.random.default_rng(seed)

    def score(self, images: list[Image.Image]) -> np.ndarray:
        return self.rng.uniform(0.0, 1.0, size=len(images))


class ConstantScorer(Scorer):
    """Constant-probability baseline scorer for degenerate-case and fallback testing."""

    name = "constant"

    def __init__(self, value: float = 0.5):
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be a probability in [0, 1]")
        self.value = float(value)

    def score(self, images: list[Image.Image]) -> np.ndarray:
        return np.full(len(images), self.value, dtype=np.float64)


class MeanIntensityScorer(Scorer):
    """Heuristic baseline that scores mean grayscale pixel intensity."""

    name = "mean_intensity"

    def score(self, images: list[Image.Image]) -> np.ndarray:
        return np.array(
            [float(np.asarray(image.convert("L"), dtype=np.float64).mean()) / 255.0 for image in images]
        )


register("random")(lambda **kwargs: RandomScorer(**kwargs))
register("constant")(lambda **kwargs: ConstantScorer(**kwargs))
register("mean_intensity")(lambda **kwargs: MeanIntensityScorer(**kwargs))


def _load_baselines() -> None:
    """Register optional baseline scorers without making them mandatory if extra deps are missing."""
    import importlib
    import sys

    for module in (
        "modeling.community_forensics.model",
        "modeling.dinov3.scorer",
        "modeling.stacking.scorer",
        "modeling.ensemble.scorer",
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:
            print(
                f"[registry] {module} unavailable: {exc.__class__.__name__}: {exc}",
                file=sys.stderr,
            )


_load_baselines()

