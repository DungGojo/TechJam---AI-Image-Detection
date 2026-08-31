"""Cache expert probabilities by manifest row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from Data_Pipeline import manifest as manifest_mod  # noqa: E402
from Data_Pipeline.image_io import load_image_safe  # noqa: E402
from Modeling.registry import check_param_budget, get  # noqa: E402
from Workflows.cache_features import parse_variant  # noqa: E402

DEFAULT_ROOT = Path("cache/scores")


def cache_paths(
    manifest: str | Path, *, model: str, variant: str, root: str | Path = DEFAULT_ROOT
) -> tuple[Path, Path, Path]:
    """(scores, done mask, metadata) for one cached (model, manifest, variant)."""
    stem = f"{Path(manifest).stem}_{variant.replace('=', '')}"
    directory = Path(root) / model.replace("/", "_")
    return (
        directory / f"{stem}.npy",
        directory / f"{stem}.done.npy",
        directory / f"{stem}.meta.json",
    )


def load_scores(
    manifest: str | Path, *, model: str, variant: str = "clean", root: str | Path = DEFAULT_ROOT
) -> tuple[np.ndarray, np.ndarray]:
    """(probabilities, done mask) for a cached expert. Raises with the build command."""
    score_path, done_path, _ = cache_paths(manifest, model=model, variant=variant, root=root)
    if not score_path.exists() or not done_path.exists():
        raise FileNotFoundError(
            f"no cached scores at {score_path}. Build them with:\n"
            f"  python -m Workflows.cache_scores --manifest {manifest} "
            f"--model {model} --variant {variant}"
        )
    return np.load(score_path), np.load(done_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--model", required=True, help="a name from Modeling.registry.available()")
    p.add_argument("--variant", default="clean", help="'clean' or a cell like 'jpeg=50'")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    df = manifest_mod.load(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    n_rows = len(df)
    if not n_rows:
        print("empty manifest", file=sys.stderr)
        return 1

    cell = parse_variant(args.variant)
    scorer = get(args.model, **({"device": args.device} if args.device else {}))
    n_params = check_param_budget(scorer)

    score_path, done_path, meta_path = cache_paths(
        args.manifest, model=args.model, variant=args.variant, root=args.out
    )
    score_path.parent.mkdir(parents=True, exist_ok=True)

    if score_path.exists() and done_path.exists():
        scores = np.load(score_path)
        done = np.load(done_path)
        if scores.shape != (n_rows,) or done.shape != (n_rows,):
            print(
                f"cache shape {scores.shape} does not match {(n_rows,)}; "
                "delete the cache or use a different manifest",
                file=sys.stderr,
            )
            return 1
    else:
        # Missing scores carry no information.
        scores = np.full(n_rows, 0.5, dtype=np.float64)
        done = np.zeros(n_rows, dtype=bool)

    todo = np.flatnonzero(~done)
    print(f"model     {args.model}  ({n_params:,} params)")
    print(f"manifest  {args.manifest}  {n_rows:,} rows")
    print(f"variant   {cell.label}")
    print(f"cache     {score_path}")
    print(f"todo      {len(todo):,} of {n_rows:,} ({n_rows - len(todo):,} already cached)\n")

    if not len(todo):
        print("nothing to do")
        return 0

    paths = df["path"].tolist()
    n_failed = 0

    with tqdm(total=len(todo), desc="score", unit="img") as bar:
        for batch_index, start in enumerate(range(0, len(todo), args.batch_size)):
            group = todo[start : start + args.batch_size]
            images, owners = [], []

            for row in group:
                img, _error = load_image_safe(paths[row])
                if img is None:
                    n_failed += 1
                    done[row] = True  # stays at the 0.5 fallback, never retried
                    continue
                rng = np.random.default_rng(args.seed + int(row))
                images.append(cell.apply(img, rng))
                owners.append(row)

            if images:
                scores[owners] = scorer.score_checked(images)
                done[owners] = True

            bar.update(len(group))
            if batch_index % 20 == 0:
                np.save(score_path, scores)
                np.save(done_path, done)

    np.save(score_path, scores)
    np.save(done_path, done)
    meta_path.write_text(json.dumps({
        "model": args.model,
        "n_params": n_params,
        "manifest": str(args.manifest),
        "rows": n_rows,
        "variant": args.variant,
        "seed": args.seed,
        "failed": n_failed,
        "note": "rows that could not be read keep the honest fallback probability 0.5",
    }, indent=2))

    print(f"\nscored {int(done.sum()):,}/{n_rows:,} rows"
          f"{f' ({n_failed} unreadable)' if n_failed else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
