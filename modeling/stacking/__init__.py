"""Frozen-feature stacking detector."""

from modeling.stacking.features import FeatureSet, Variant, crops_aligned
from modeling.stacking.head import FitConfig, StackingConfig, StackingHead, fit, load, save

__all__ = [
    "FeatureSet",
    "FitConfig",
    "StackingConfig",
    "StackingHead",
    "Variant",
    "crops_aligned",
    "fit",
    "load",
    "save",
]
