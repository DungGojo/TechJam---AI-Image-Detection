"""Run directory inference with a registered scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Data_Pipeline.image_io import iter_image_paths, load_image_safe  # noqa: E402
from Modeling.common.device import get_device  # noqa: E402
from Modeling.registry import MAX_PARAMS, available, check_param_budget, get  # noqa: E402

# Unreadable inputs remain maximally uncertain.
FALLBACK_PRED = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score images for the probability they are AI-generated.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_dir", required=True, type=Path, help="directory to recurse")
    p.add_argument("--output", required=True, type=Path, help="JSON file to write")
    p.add_argument("--model", default="constant", help=f"scorer name; one of {available()}")
    p.add_argument("--checkpoint", default=None, help="checkpoint path for trained models")
    p.add_argument("--ensemble-config", default=None, help="YAML config for --model ensemble")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="force cpu/mps/cuda (default: auto)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def predict_dir(
    input_dir: Path,
    model: str = "constant",
    *,
    batch_size: int = 32,
    device: str | None = None,
    checkpoint: str | None = None,
    ensemble_config: str | None = None,
    quiet: bool = False,
) -> list[dict[str, object]]:
    """Score every image under `input_dir`. Returns the JSON-ready records."""
    resolved_device = get_device(device)
    scorer_kwargs = (
        {"device": str(resolved_device)}
        if model in {"community_forensics", "dinov3", "stacking"}
        else {}
    )
    if checkpoint is not None:
        scorer_kwargs["checkpoint"] = checkpoint
    if ensemble_config is not None:
        scorer_kwargs["config"] = ensemble_config
    scorer = get(model, **scorer_kwargs)

    n_params = check_param_budget(scorer)

    if not quiet:
        print(f"model            {model}", file=sys.stderr)
        print(f"device           {resolved_device}", file=sys.stderr)
        print(f"total parameters {n_params:,}  (limit {MAX_PARAMS:,})", file=sys.stderr)

    paths = iter_image_paths(input_dir)
    if not quiet:
        print(f"images found     {len(paths)}", file=sys.stderr)

    records: list[dict[str, object]] = []
    n_failed = 0

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        images, kept = [], []

        for path in chunk:
            img, error = load_image_safe(path)
            if img is None:
                n_failed += 1
                print(f"WARNING: unreadable, scoring {FALLBACK_PRED}: {error}", file=sys.stderr)
                records.append({"image_path": _as_given(input_dir, path), "pred": FALLBACK_PRED})
                continue
            images.append(img)
            kept.append(path)

        if not images:
            continue
        scores = scorer.score_checked(images)
        for path, score in zip(kept, scores):
            records.append({"image_path": _as_given(input_dir, path), "pred": float(score)})

    # Failures are appended early, so restore deterministic path order.
    records.sort(key=lambda r: str(r["image_path"]))

    if not quiet and n_failed:
        print(f"unreadable files {n_failed} (scored {FALLBACK_PRED})", file=sys.stderr)
    return records


def _as_given(input_dir: Path, path: Path) -> str:
    """Preserve caller-relative paths in prediction output."""
    try:
        return str(Path(input_dir) / path.relative_to(Path(input_dir).resolve()))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input_dir.exists():
        print(f"ERROR: no such directory: {args.input_dir}", file=sys.stderr)
        return 2

    records = predict_dir(
        args.input_dir,
        args.model,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint=args.checkpoint,
        ensemble_config=args.ensemble_config,
        quiet=args.quiet,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")

    if not args.quiet:
        print(f"wrote            {args.output}  ({len(records)} records)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
