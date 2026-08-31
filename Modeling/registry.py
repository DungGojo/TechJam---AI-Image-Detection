"""Detector scorer registry."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

from Modeling.common.interfaces import Scorer

MAX_PARAMS = 2_000_000_000


_REGISTRY: dict[str, Callable[..., Scorer]] = {}


def register(name: str) -> Callable[[Callable[..., Scorer]], Callable[..., Scorer]]:
    def decorator(factory: Callable[..., Scorer]) -> Callable[..., Scorer]:
        if name in _REGISTRY:
            raise ValueError(f"scorer {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str, **kwargs) -> Scorer:
    if name not in _REGISTRY:
        raise KeyError(f"unknown scorer {name!r}; available: {available()}")
    scorer = _REGISTRY[name](**kwargs)
    scorer.name = name
    return scorer


def check_param_budget(scorer: Scorer) -> int:
    """Hard rule 1. Raises rather than warns."""
    n = int(scorer.n_params())
    if n > MAX_PARAMS:
        raise RuntimeError(
            f"{scorer.name} has {n:,} parameters, over the {MAX_PARAMS:,} limit"
        )
    return n




class RandomScorer(Scorer):
    """Uniform-random scorer used to validate the evaluation harness."""

    name = "random"

    def __init__(self, seed: int = 1337):
        self.rng = np.random.default_rng(seed)

    def score(self, images: list[Image.Image]) -> np.ndarray:
        return self.rng.uniform(0.0, 1.0, size=len(images))


class ConstantScorer(Scorer):
    """Constant-probability scorer for degenerate-case testing."""

    name = "constant"

    def __init__(self, value: float = 0.5):
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be a probability")
        self.value = float(value)

    def score(self, images: list[Image.Image]) -> np.ndarray:
        return np.full(len(images), self.value, dtype=np.float64)


class MeanIntensityScorer(Scorer):
    """Weak baseline that scores mean pixel intensity."""

    name = "mean_intensity"

    def score(self, images: list[Image.Image]) -> np.ndarray:
        return np.array(
            [float(np.asarray(im.convert("L"), dtype=np.float64).mean()) / 255.0 for im in images]
        )


register("random")(lambda **kw: RandomScorer(**kw))
register("constant")(lambda **kw: ConstantScorer(**kw))
register("mean_intensity")(lambda **kw: MeanIntensityScorer(**kw))


def _load_baselines() -> None:
    """Register optional baseline scorers without making them mandatory."""
    import importlib
    import sys

    for module in (
        "Modeling.community_forensics.model",
        "Modeling.dinov3.scorer",
        "Modeling.stacking.scorer",
        "Modeling.ensemble.scorer",
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:
            print(
                f"[registry] {module} unavailable: {exc.__class__.__name__}: {exc}",
                file=sys.stderr,
            )


_load_baselines()
