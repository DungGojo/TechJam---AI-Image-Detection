"""DINOv3 detector, training loop, and checkpoint inference."""

from .backbone import BackboneConfig
from .detector import Expert, ExpertConfig

__all__ = ["BackboneConfig", "Expert", "ExpertConfig"]
