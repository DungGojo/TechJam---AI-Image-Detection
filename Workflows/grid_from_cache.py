"""Evaluate robustness from cached features."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from Evaluation.metrics import compute  # noqa: E402
from Evaluation.robustness import CLEAN, CleanCell, is_held_out, save_report, to_markdown  # noqa: E402
from Modeling.common.degradations import Cell  # noqa: E402
from Modeling.common.device import get_device  # noqa: E402
from Modeling.ensemble.calibration import TemperatureCalibrator  # noqa: E402
from Modeling.stacking.features import DEFAULT_ROOT, FeatureSet, cache_paths  # noqa: E402
from Modeling.stacking.head import load, predict_variant  # noqa: E402
from Workflows.cache_scores import DEFAULT_ROOT as SCORE_ROOT  # noqa: E402
from Workflows.fit_stacker import expert_table  # noqa: E402

# Cached names omit "="; the alphabetic/numeric boundary restores it.
_TAG = re.compile(r"^([a-z_]+)([\d.]+)$")


def parse_tag(tag: str):
    """Cache filename tag -> a Cell (or CleanCell)."""
    if tag == CLEAN:
        return CleanCell()
    match = _TAG.match(tag)
    if not match:
        raise ValueError(f"cannot parse variant tag {tag!r}")
    return Cell(match.group(1), float(match.group(2)))


def discover_variants(
    manifest: Path, *, backbone: str, crops: int, crop_size: int, root: Path
) -> list[str]:
    """Every variant cached for this manifest, clean first then by severity."""
    directory = Path(root) / backbone.replace("/", "_")
    prefix = f"{manifest.stem}_{crops}x{crop_size}_"
    found = []
    for path in sorted(directory.glob(f"{prefix}*.npy")):
        if path.name.endswith(".done.npy"):
            continue
        tag = path.name[len(prefix) : -len(".npy")]
        if tag == CLEAN:
            found.append(CLEAN)
            continue
        # Preserve textual precision: blur1.0 and blur1 name different caches.
        match = _TAG.match(tag)
        if not match:
            raise ValueError(f"cannot parse variant tag {tag!r} in {path.name}")
        found.append(f"{match.group(1)}={match.group(2)}")
    return sorted(found, key=lambda v: (v != CLEAN, v))


def grid_rows(model_name: str, labels: np.ndarray, per_variant: dict[str, np.ndarray]):
    rows = []
    for variant, scores in per_variant.items():
        cell = parse_tag(variant.replace("=", ""))
        rows.append({
            "model": model_name,
            "cell": cell.label,
            "transform": cell.family,
            "severity": float(cell.value),
            "held_out": is_held_out(cell),
            "n_failed": 0,          # unreadable rows were excluded when caching
            **compute(labels, scores),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--experts", nargs="*", default=None,
                   help="defaults to the experts the checkpoint was fitted with")
    p.add_argument("--variants", nargs="*", default=None,
                   help="defaults to every variant cached for this manifest")
    p.add_argument("--feature-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--score-root", type=Path, default=SCORE_ROOT)
    p.add_argument("--calibration", type=Path, default=None,
                   help="temperature JSON from Workflows.calibrate")
    p.add_argument("--branch-a-weight", type=float, default=0.70)
    p.add_argument("--branch-b-weight", type=float, default=0.30)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/robustness"))
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = get_device(args.device)
    model, state = load(args.checkpoint, map_location=device)
    model.to(device).eval()

    backbone = state["backbone"]
    crops, crop_size = int(state["crops"]), int(state["crop_size"])
    experts = args.experts if args.experts is not None else list(model.cfg.expert_names)

    variants = args.variants or discover_variants(
        args.manifest, backbone=backbone, crops=crops,
        crop_size=crop_size, root=args.feature_root,
    )
    if CLEAN not in variants:
        print(f"no clean cache for {args.manifest}; the grid needs a clean reference",
              file=sys.stderr)
        return 1

    print(f"checkpoint {args.checkpoint}")
    print(f"backbone   {backbone}  ({crops}x{crop_size})")
    print(f"manifest   {args.manifest}")
    print(f"variants   {len(variants)}: {', '.join(variants)}\n")

    feature_set = FeatureSet(
        args.manifest, backbone=backbone, crops=crops, crop_size=crop_size,
        variants=variants, root=args.feature_root,
        extra=expert_table(args.manifest, experts, variants, args.score_root),
    )
    labels = feature_set.labels
    print(f"{len(feature_set):,} usable rows "
          f"({int((labels == 0).sum()):,} authentic, {int((labels == 1).sum()):,} generated)\n")

    calibrator = (
        TemperatureCalibrator.load(args.calibration)
        if args.calibration and args.calibration.exists() else None
    )

    branch_a, branch_b, fused = {}, {}, {}
    for variant in variants:
        a = predict_variant(model, feature_set, variant, device=device)
        branch_a[variant] = a
        if experts:
            scores = feature_set.extra_for(variant)
            b = scores[:, 0].astype(np.float64)
            branch_b[variant] = b
            combined = (args.branch_a_weight * a + args.branch_b_weight * b) / (
                args.branch_a_weight + args.branch_b_weight
            )
            fused[variant] = calibrator.transform(combined) if calibrator else combined

    grids = {"stacking": grid_rows("stacking", labels, branch_a)}
    if experts:
        grids[experts[0]] = grid_rows(experts[0], labels, branch_b)
        grids["fused"] = grid_rows("fused", labels, fused)

    written = []
    for name, rows in grids.items():
        results = pd.DataFrame(rows)
        print(to_markdown(results))
        print()
        paths = save_report(results, name, out_dir=args.out_dir)
        written += list(paths.values())

    print("wrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
