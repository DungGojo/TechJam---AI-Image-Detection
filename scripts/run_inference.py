#!/usr/bin/env python3
"""Unified inference and evaluation script for AI-Generated Image Detection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set before any Hugging Face import so downloads land in the project cache
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "cache/hf"))

import numpy as np

from modeling.common.device import get_device
from workflows.classify import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DINOV3_CHECKPOINT,
    DEFAULT_FUSION_ARTIFACT,
    DEFAULT_SEED,
    DEFAULT_STACKING_CHECKPOINT,
    classify_images,
    collect_images,
    display_path,
    print_metrics,
    print_table,
    prompt_for_inputs,
)
from workflows.predict import FALLBACK_PRED, _as_given, predict_dir


def label_from_path(path: Path) -> int | None:
    """Read ground truth from parent folders containing 'real' or 'fake'."""
    for part in reversed(path.parent.parts):
        part_lower = part.lower()
        if part_lower in ("real", "fake"):
            return 0 if part_lower == "real" else 1
        if "(real)" in part_lower or "[real]" in part_lower or "_real" in part_lower:
            return 0
        if "(fake)" in part_lower or "[fake]" in part_lower or "_fake" in part_lower:
            return 1
    return None



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified inference and evaluation tool for Real vs AI-Generated Image Detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Inputs: supports positional inputs or explicit flag
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[],
        help="Input image file(s) or folder(s) (searched recursively). Optional second positional arg can be output path.",
    )
    parser.add_argument(
        "--input",
        "-i",
        "--input_dir",
        dest="flag_input",
        nargs="+",
        default=None,
        help="Path(s) to image file(s) or directory containing images.",
    )
    # Outputs: supports --output, --json, --csv
    parser.add_argument(
        "--output",
        "-o",
        dest="flag_output",
        type=Path,
        default=None,
        help="Path to output file (.json or .csv). Format is inferred from extension.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Export prediction results to this JSON file.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Export prediction results and branch scores to this CSV file.",
    )
    # evaluation
    parser.add_argument(
        "--evaluate",
        "-e",
        action="store_true",
        help="Evaluate ground-truth labels inferred from parent directory names (real/ vs fake/) and report full metrics.",
    )
    # Model Selection
    parser.add_argument(
        "--model",
        "-m",
        default="pipeline",
        choices=["pipeline", "dinov3", "stacking", "community_forensics", "ensemble"],
        help="Model to execute: 'pipeline' (full 3-detector SVM fusion, recommended) or a specific baseline.",
    )
    # Hyperparameters & Paths
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Decision threshold on P(fake); default is loaded from the fusion artifact (0.70).",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Force compute hardware: 'cuda', 'mps', 'cpu' (default: auto-detected).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=None,
        help="Batch size for model inference (default: tuned per detector branch).",
    )
    parser.add_argument(
        "--dinov3-checkpoint",
        type=Path,
        default=DEFAULT_DINOV3_CHECKPOINT,
        help="Path to fine-tuned DINOv3-L checkpoint.",
    )
    parser.add_argument(
        "--stacking-checkpoint",
        type=Path,
        default=DEFAULT_STACKING_CHECKPOINT,
        help="Path to trained DINOv3-B Stacking Head checkpoint.",
    )
    parser.add_argument(
        "--fusion",
        type=Path,
        default=DEFAULT_FUSION_ARTIFACT,
        help="Path to trained fusion artifact (.joblib).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for caching intermediate detector scores.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass score cache and re-evaluate all images fresh.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for crop sampling in multi-crop TTA.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress bars and setup banners.",
    )

    args = parser.parse_args(argv)

    # 1. Resolve raw input paths
    raw_inputs: list[str] = []
    if args.flag_input:
        raw_inputs.extend(args.flag_input)
    elif args.inputs:
        # Check if last positional argument looks like an output path (.json or .csv) when 2+ args are given
        if len(args.inputs) >= 2 and args.inputs[-1].endswith((".json", ".csv")) and not Path(args.inputs[-1]).is_dir():
            raw_inputs.extend(args.inputs[:-1])
            if args.flag_output is None and args.json is None and args.csv is None:
                args.flag_output = Path(args.inputs[-1])
        else:
            raw_inputs.extend(args.inputs)

    # If no inputs provided and interactive terminal, prompt user
    if not raw_inputs:
        if sys.stdin.isatty():
            raw_inputs = prompt_for_inputs()
        if not raw_inputs:
            parser.error("No input files or directories provided.")

    args.raw_inputs = raw_inputs

    # 2. Resolve output files
    output_target = args.flag_output or args.json or args.csv
    args.output_file = output_target.resolve() if output_target is not None else None
    args.output_json = args.json.resolve() if args.json is not None else (
        args.output_file if (args.output_file and args.output_file.suffix.lower() == ".json") else None
    )
    args.output_csv = args.csv.resolve() if args.csv is not None else (
        args.output_file if (args.output_file and args.output_file.suffix.lower() == ".csv") else None
    )

    return args


def write_json_output(
    image_paths: list[Path],
    probabilities: np.ndarray,
    output_path: Path,
    base_dir: Path | None = None,
) -> None:
    """Export standard JSON prediction array [ { 'image_path': ..., 'pred': ... } ]."""
    records = []
    for path, prob in zip(image_paths, probabilities):
        pred_val = float(FALLBACK_PRED) if (prob is None or np.isnan(prob)) else float(np.clip(prob, 0.0, 1.0))
        rel_path = _as_given(base_dir, path) if base_dir else display_path(path)
        records.append({
            "image_path": rel_path,
            "pred": round(pred_val, 6),
        })

    records.sort(key=lambda r: str(r["image_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2) + "\n")


def write_csv_output(
    image_paths: list[Path],
    scores: dict[str, np.ndarray],
    final_probabilities: np.ndarray,
    verdicts: np.ndarray,
    output_path: Path,
    base_dir: Path | None = None,
) -> None:
    """Export detailed CSV table with per-branch detector breakdown and final verdict."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["image_path", "pred", "verdict"] + list(scores.keys())
        writer.writerow(header)
        for i, path in enumerate(image_paths):
            rel_path = _as_given(base_dir, path) if base_dir else display_path(path)
            prob = final_probabilities[i]
            verdict_str = "FAKE" if verdicts[i] == 1 else ("REAL" if verdicts[i] == 0 else "UNREADABLE")
            row = [
                rel_path,
                f"{prob:.6f}" if np.isfinite(prob) else "NaN",
                verdict_str,
            ] + [f"{scores[col][i]:.6f}" if np.isfinite(scores[col][i]) else "NaN" for col in scores]
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 1. Collect and deduplicate image paths
    image_paths, errors = collect_images(args.raw_inputs)
    if errors and not args.quiet:
        for err in errors:
            print(f"WARNING: {err}", file=sys.stderr)

    if not image_paths:
        print("ERROR: No valid images found from the supplied inputs.", file=sys.stderr)
        return 1

    base_dir = Path(args.raw_inputs[0]).resolve() if (len(args.raw_inputs) == 1 and Path(args.raw_inputs[0]).is_dir()) else None
    labels = [label_from_path(p) for p in image_paths] if args.evaluate else [None] * len(image_paths)

    # 2. Run inference
    if args.model == "pipeline" and args.fusion.exists():
        # Full 3-detector fusion model
        scores, final_probabilities, verdicts, meta = classify_images(image_paths, args)
    else:
        # Single baseline model
        model_name = "dinov3" if args.model == "pipeline" else args.model
        checkpoint = (
            str(args.dinov3_checkpoint) if model_name == "dinov3"
            else (str(args.stacking_checkpoint) if model_name == "stacking" else None)
        )
        resolved_device = get_device(args.device)
        records = predict_dir(
            base_dir or PROJECT_ROOT,
            model=model_name,
            batch_size=args.batch_size or 32,
            device=str(resolved_device),
            checkpoint=checkpoint,
            quiet=args.quiet,
        )
        scores = {model_name: np.array([r["pred"] for r in records], dtype=np.float64)}
        final_probabilities = scores[model_name]
        threshold = args.threshold or 0.50
        verdicts = (final_probabilities >= threshold).astype(int)
        meta = {
            "model_name": model_name,
            "threshold": threshold,
            "feature_order": [model_name],
            "device": str(resolved_device),
        }

    # 3. Export to JSON if requested
    if args.output_json:
        write_json_output(image_paths, final_probabilities, args.output_json, base_dir=base_dir)
        if not args.quiet:
            print(f"Exported JSON predictions to: {args.output_json} ({len(image_paths)} images)", file=sys.stderr)

    # 4. Export to CSV if requested
    if args.output_csv:
        write_csv_output(image_paths, scores, final_probabilities, verdicts, args.output_csv, base_dir=base_dir)
        if not args.quiet:
            print(f"Exported CSV report to: {args.output_csv}", file=sys.stderr)

    # 5. Display terminal table & evaluation metrics if requested or no file export specified
    if args.evaluate or (not args.output_json and not args.output_csv):
        print_table(
            image_paths,
            scores,
            final_probabilities,
            verdicts,
            labels,
            meta,
            colour=True,
        )

    if args.evaluate:
        print_metrics(labels, final_probabilities, verdicts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
