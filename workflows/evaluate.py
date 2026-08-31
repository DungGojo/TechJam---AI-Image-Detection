"""Evaluate one scorer on clean and transformed images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from data_pipeline import manifest as manifest_mod
from evaluation.robustness import run_grid, save_report, to_markdown
from modeling.common.device import get_device
from modeling.registry import available, check_param_budget, get


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help=f"scorer name; one of {available()}")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--split", default=None, help="evaluate only this split")
    p.add_argument(
        "--limit", type=int, default=None, help="cap rows (for a quick check)"
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--checkpoint", default=None, help="checkpoint path for trained models"
    )
    p.add_argument(
        "--ensemble-config", default=None, help="YAML config for --model ensemble"
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/robustness"))
    p.add_argument(
        "--skip-validate",
        action="store_true",
        help="skip manifest validation (only for a manifest you have already validated)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    df = manifest_mod.load(args.manifest)
    if not args.skip_validate:
        blocklist = manifest_mod.Blocklist.load()
        manifest_mod.validate(df, blocklist=blocklist)
    if args.split:
        df = df[df["split"] == args.split].reset_index(drop=True)
    if args.limit:
        df = (
            df.groupby("label", group_keys=False)
            .head(max(1, args.limit // 2))
            .reset_index(drop=True)
        )

    if df.empty:
        print("no rows to evaluate", file=sys.stderr)
        return 1

    scorer_kwargs = {"checkpoint": args.checkpoint} if args.checkpoint else {}
    if args.ensemble_config:
        scorer_kwargs["config"] = args.ensemble_config
    scorer = get(args.model, **scorer_kwargs)
    n_params = check_param_budget(scorer)

    print(f"model      {scorer.name}  ({n_params:,} params)")
    print(f"device     {get_device()}")
    print(
        f"manifest   {args.manifest}  ->  {len(df)} rows "
        f"({int((df['label'] == 0).sum())} real / {int((df['label'] == 1).sum())} fake)"
    )
    print()

    results = run_grid(
        scorer,
        df,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    paths = save_report(results, scorer.name, args.out_dir)
    print()
    print(to_markdown(results))
    print()
    for kind, path in paths.items():
        print(f"{kind:<9} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
