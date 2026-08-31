"""Late fusion and calibration across detector families."""

from .calibration import TemperatureCalibrator
from .fusion import FusionConfig, fuse

__all__ = ["FusionConfig", "TemperatureCalibrator", "fuse"]
