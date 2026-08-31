"""Fit the frozen-feature stacking head."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from modeling.dinov3.losses import LossWeights
from modeling.stacking.features import DEFAULT_ROOT, FeatureSet
from modeling.stacking.head import (
    FitConfig,
    StackingConfig,
    evaluate,
    fit,
    history_frame,
    save,
)
from workflows.cache_scores import DEFAULT_ROOT as SCORE_ROOT
from workflows.cache_scores import load_scores


def expert_table(
    manifest: Path, experts: list[str], variants: list[str], root: Path
) -> dict[str, np.ndarray] | None:
    """Load row-aligned cached expert probabilities by variant."""
    if not experts:
        return None
    table: dict[str, np.ndarray] = {}
    for variant in variants:
        columns = []
        for model in experts:
            try:
                scores, done = load_scores(
                    manifest, model=model, variant=variant, root=root
                )
            except FileNotFoundError:
                if variant == "clean":
                    raise
                columns = []
                break
            if not done.all():
                print(
                    f"WARNING: {model} scores for {manifest.name} [{variant}] are "
                    f"{int(done.sum()):,}/{len(done):,} complete; missing rows sit at "
                    "the 0.5 fallback. Finish cache_scores for a clean run.",
                    file=sys.stderr,
                )
            columns.append(scores)
        if len(columns) == len(experts):
            table[variant] = np.stack(columns, axis=1)
    return table


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--train-manifest", required=True, type=Path)
    p.add_argument("--val-manifest", required=True, type=Path)
    p.add_argument("--backbone", default="vit_base_patch16_dinov3.lvd1689m")
    p.add_argument("--crops", type=int, default=3)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument(
        "--variants",
        nargs="*",
        default=[],
        help="degradation cells cached alongside clean, e.g. jpeg=50 blur=1.0",
    )
    p.add_argument(
        "--val-variants",
        nargs="*",
        default=None,
        help="variants to select on; defaults to --variants",
    )
    p.add_argument(
        "--experts", nargs="*", default=[], help="cached scorers to feed the head"
    )
    p.add_argument("--feature-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--score-root", type=Path, default=SCORE_ROOT)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument(
        "--head-hidden", type=int, default=256, help="0 for a pure linear probe"
    )
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--no-correction", action="store_true", help="drop the FeatureCorrection block"
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--out", type=Path, default=Path("checkpoints/stacking/default/best_robust.pt")
    )
    args = p.parse_args(argv)

    train_variants = ["clean", *args.variants]
    val_variants = [
        "clean",
        *(args.val_variants if args.val_variants is not None else args.variants),
    ]

    print(f"backbone   {args.backbone}")
    print(f"train      {args.train_manifest}  variants={train_variants}")
    print(f"validation {args.val_manifest}  variants={val_variants}")

    train_set = FeatureSet(
        args.train_manifest,
        backbone=args.backbone,
        crops=args.crops,
        crop_size=args.crop_size,
        variants=train_variants,
        root=args.feature_root,
        extra=expert_table(
            args.train_manifest, args.experts, train_variants, args.score_root
        ),
    )
    val_set = FeatureSet(
        args.val_manifest,
        backbone=args.backbone,
        crops=args.crops,
        crop_size=args.crop_size,
        variants=val_variants,
        root=args.feature_root,
        extra=expert_table(
            args.val_manifest, args.experts, val_variants, args.score_root
        ),
    )

    print(
        f"\ntrain rows {len(train_set):,}  ({train_set.n_crops} crops, dim {train_set.dim})"
    )
    print(train_set.summary().to_string(index=False))
    print(f"\nval rows   {len(val_set):,}")
    print(val_set.summary().to_string(index=False))

    if not len(train_set) or not len(val_set):
        print("\nno usable rows; build the caches first", file=sys.stderr)
        return 1

    cfg = StackingConfig(
        dim=train_set.dim,
        extra_dim=train_set.extra_dim,
        head_hidden=args.head_hidden,
        head_dropout=args.dropout,
        feature_correction=not args.no_correction,
        expert_names=list(args.experts),
    )
    fit_cfg = FitConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        loss=LossWeights(),
    )

    def progress(epoch: int, train_row: dict, summary: dict) -> None:
        line = f"epoch {epoch + 1:3d}/{args.epochs}  loss {train_row.get('train_total', float('nan')):.4f}"
        if summary:
            line += (
                f"  clean {summary['clean_auc']:.4f}"
                f"  robust {summary['robust_auc']:.4f}"
                f"  worst {summary['worst_cell_auc']:.4f}"
            )
        print(line, flush=True)

    print()
    result = fit(train_set, val_set, cfg, fit_cfg, progress=progress)
    model = result["model"]
    print(f"\n{model.describe()}")

    final = evaluate(model, val_set, device=next(model.parameters()).device)
    print("\nvalidation at the selected checkpoint:")
    print(pd.Series(final).round(4).to_string())

    path = save(
        args.out,
        model,
        backbone=args.backbone,
        crops=args.crops,
        crop_size=args.crop_size,
        history=result["history"],
        best_robust_auc=result["best_robust_auc"],
        variants=train_variants,
    )
    history_frame(result["history"]).to_csv(
        path.with_suffix(".history.csv"), index=False
    )
    print(f"\nwrote {path}")
    print(f"wrote {path.with_suffix('.history.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
