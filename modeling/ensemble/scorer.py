"""Config-driven fusion of any registered detector scorers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from modeling.common.interfaces import Scorer
from modeling.ensemble.calibration import TemperatureCalibrator
from modeling.ensemble.fusion import FusionConfig, GateConfig, fuse
from modeling.registry import get, register


class EnsembleScorer(Scorer):
    name = "ensemble"

    def __init__(self, config: str = "configs/ensemble.yaml"):
        spec = yaml.safe_load(Path(config).read_text())
        experts = spec.get("experts", {})
        if not experts:
            raise ValueError("ensemble config must define at least one expert")
        self.experts = {}
        for name, expert_spec in experts.items():
            model_name = expert_spec.get("model", name)
            if model_name == "ensemble":
                raise ValueError("an ensemble cannot contain itself")
            kwargs = {key: value for key, value in expert_spec.items() if key != "model"}
            self.experts[name] = get(model_name, **kwargs)

        fusion_spec = dict(spec.get("fusion", {}))
        if isinstance(fusion_spec.get("gate"), dict):
            fusion_spec["gate"] = GateConfig(**fusion_spec["gate"])
        self.fusion = FusionConfig(**fusion_spec)
        calibration = spec.get("calibration")
        self.calibrator = None
        if isinstance(calibration, str) and Path(calibration).exists():
            self.calibrator = TemperatureCalibrator.load(calibration)
        elif isinstance(calibration, dict) and "temperature" in calibration:
            self.calibrator = TemperatureCalibrator(float(calibration["temperature"]))

    def n_params(self) -> int:
        return sum(expert.n_params() for expert in self.experts.values())

    def score(self, images: list[Image.Image]) -> np.ndarray:
        probabilities = {
            name: torch.from_numpy(expert.score_checked(images)).float()
            for name, expert in self.experts.items()
        }
        combined = fuse(probabilities, cfg=self.fusion).cpu().numpy().astype(np.float64)
        return self.calibrator.transform(combined) if self.calibrator else combined


register("ensemble")(lambda **kwargs: EnsembleScorer(**kwargs))
