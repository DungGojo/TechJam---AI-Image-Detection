"""Execute resumable feature and score caching plans."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from workflows import cache_features, cache_scores


def balanced_subsample(frame: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Equal rows per label, shuffled, deterministic in `seed`."""
    pieces = [
        group.sample(min(per_class, len(group)), random_state=seed + int(label))
        for label, group in frame.groupby("label")
    ]
    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def resolve_manifest(step: dict, manifest_dir: Path) -> tuple[Path, int]:
    """The manifest this step caches against, building a subsample if asked."""
    source = Path(step["manifest"])
    per_class = step.get("per_class")
    if not per_class:
        return source, len(pd.read_csv(source))

    manifest_dir.mkdir(parents=True, exist_ok=True)
    target = manifest_dir / f"{step['name']}.csv"
    seed = int(step.get("seed", 1337))
    if not target.exists():
        sample = balanced_subsample(pd.read_csv(source), int(per_class), seed)
        sample.to_csv(target, index=False)
        print(
            f"  wrote subsample {target} ({len(sample):,} rows, per_class={per_class}, seed={seed})"
        )
    return target, len(pd.read_csv(target))


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--dry-run", action="store_true", help="print the plan and estimates only"
    )
    p.add_argument("--only", default=None, help="run one step by name")
    p.add_argument("--skip-scores", action="store_true", help="features only")
    args = p.parse_args(argv)

    spec = yaml.safe_load(args.config.read_text())
    backbone = spec["backbone"]
    crops = int(spec.get("crops", 3))
    crop_size = int(spec.get("crop_size", 224))
    workers = int(spec.get("workers", 4))
    batch_size = int(spec.get("batch_size", 8))
    manifest_dir = Path(spec.get("manifest_dir", "data/plan"))
    crops_per_second = float(spec.get("throughput_crops_per_second", 30))
    images_per_second = float(spec.get("throughput_expert_images_per_second", 30))

    steps = [s for s in spec["steps"] if args.only in (None, s["name"])]
    if not steps:
        print(f"no step named {args.only!r}", file=sys.stderr)
        return 1

    print(f"backbone   {backbone}")
    print(f"crops      {crops} x {crop_size}px, {workers} decode workers\n")

    plan_rows, total_seconds = [], 0.0
    for step in steps:
        manifest, rows = resolve_manifest(step, manifest_dir)
        variants = step.get("variants", ["clean"])
        experts = [] if args.skip_scores else step.get("experts", [])
        feature_seconds = rows * crops * len(variants) / crops_per_second
        score_seconds = rows * len(variants) * len(experts) / images_per_second
        total_seconds += feature_seconds + score_seconds
        step["_manifest"] = manifest
        step["_rows"] = rows
        step["_variants"] = variants
        step["_experts"] = experts
        plan_rows.append(
            {
                "step": step["name"],
                "manifest": manifest.name,
                "rows": f"{rows:,}",
                "variants": len(variants),
                "crops": f"{rows * crops * len(variants):,}",
                "experts": len(experts),
                "estimate": format_duration(feature_seconds + score_seconds),
            }
        )

    print(pd.DataFrame(plan_rows).to_string(index=False))
    print(f"\nestimated total: {format_duration(total_seconds)}")
    print("(estimates use the configured throughput; resumed steps cost nothing)\n")

    if args.dry_run:
        return 0

    started = time.time()
    for step in steps:
        manifest, variants, experts = (
            step["_manifest"],
            step["_variants"],
            step["_experts"],
        )
        for variant in variants:
            print(f"\n=== {step['name']} | features | {variant} ===", flush=True)
            code = cache_features.main(
                [
                    "--manifest",
                    str(manifest),
                    "--backbone",
                    backbone,
                    "--crops",
                    str(crops),
                    "--crop-size",
                    str(crop_size),
                    "--variant",
                    variant,
                    "--batch-size",
                    str(batch_size),
                    "--workers",
                    str(workers),
                ]
            )
            if code != 0:
                print(f"step {step['name']} failed on {variant}", file=sys.stderr)
                return code
        for expert in experts:
            for variant in variants:
                print(f"\n=== {step['name']} | {expert} | {variant} ===", flush=True)
                code = cache_scores.main(
                    [
                        "--manifest",
                        str(manifest),
                        "--model",
                        expert,
                        "--variant",
                        variant,
                        "--batch-size",
                        str(max(batch_size, 16)),
                    ]
                )
                if code != 0:
                    print(
                        f"step {step['name']} failed scoring {variant}", file=sys.stderr
                    )
                    return code
        print(
            f"\n--- {step['name']} done, {format_duration(time.time() - started)} elapsed ---"
        )

    print(f"\nplan complete in {format_duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
