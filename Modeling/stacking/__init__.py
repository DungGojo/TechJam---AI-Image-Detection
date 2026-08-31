"""Frozen-feature stacking detector."""

from Modeling.stacking.features import FeatureSet, Variant, crops_aligned
from Modeling.stacking.head import FitConfig, StackingConfig, StackingHead, fit, load, save

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
