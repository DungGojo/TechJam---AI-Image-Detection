"""Fit temperature calibration from held-out labels and fused probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from modeling.ensemble.calibration import TemperatureCalibrator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--score-column", default="pred")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/fusion/temperature.json")
    )
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.predictions)
    missing = {args.score_column, args.label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"predictions CSV is missing columns: {sorted(missing)}")
    calibrator = TemperatureCalibrator().fit(
        frame[args.score_column].to_numpy(), frame[args.label_column].to_numpy()
    )
    calibrator.save(args.output)
    print(f"temperature={calibrator.temperature:.6f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
