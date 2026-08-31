"""Cache frozen-backbone image features."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from data_pipeline import manifest as manifest_mod
from data_pipeline.image_io import load_image_safe
from modeling.dinov3.backbone_cache import PRIMARY, FrozenBackbone


def parse_variant(text: str):
    """'clean' or 'family=value' -> something with .apply(img, rng)."""
    from evaluation.robustness import CleanCell

    if text == "clean":
        return CleanCell()
    from modeling.common.degradations import Cell

    family, _, value = text.partition("=")
    if not value:
        raise SystemExit(f"--variant must be 'clean' or 'family=value', got {text!r}")
    return Cell(family, float(value))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--backbone", default=PRIMARY)
    p.add_argument("--crops", type=int, default=5)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument(
        "--variant",
        default="clean",
        help="'clean', or a degradation cell like 'jpeg=50' -- §9 wants both, "
        "because the degraded features are what a consistency loss needs",
    )
    p.add_argument("--batch-size", type=int, default=8, help="images per forward group")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="decode/crop threads running ahead of the forward pass. 0 keeps the "
        "original serial path. A fast backbone on MPS is starved by "
        "single-threaded PIL, so 4 is a good default for ViT-B and smaller",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("cache/features"))
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    df = manifest_mod.load(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    n_rows = len(df)
    if not n_rows:
        print("empty manifest", file=sys.stderr)
        return 1

    cell = parse_variant(args.variant)

    try:
        backbone = FrozenBackbone(args.backbone, crop_size=args.crop_size)
    except RuntimeError as exc:
        if "Unknown model" not in str(exc):
            raise
        # Distinguish an outdated timm environment from a download failure.
        import timm

        print(
            f"timm {timm.__version__} at {sys.executable} knows neither "
            f"{args.backbone!r} nor the DINOv2 fallback.\n"
            "DINOv3 backbones need timm >= 1.0; run this with the project "
            "environment at .venv/bin/python.",
            file=sys.stderr,
        )
        return 1

    if backbone.used_fallback:
        print(
            f"WARNING: {args.backbone!r} was unavailable, so features are being cached "
            f"under {backbone.name!r}. Anything that asks the cache for "
            f"{args.backbone!r} will not find them.",
            file=sys.stderr,
        )

    stem = (
        f"{args.manifest.stem}_{args.crops}x{args.crop_size}"
        f"_{args.variant.replace('=', '')}"
    )
    out_dir = args.out / backbone.name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / f"{stem}.npy"
    done_path = out_dir / f"{stem}.done.npy"
    meta_path = out_dir / f"{stem}.meta.json"

    shape = (n_rows, args.crops, backbone.dim)
    if feat_path.exists() and done_path.exists():
        features = np.lib.format.open_memmap(feat_path, mode="r+")
        done = np.load(done_path)
        if features.shape != shape or done.shape != (n_rows,):
            print(
                f"cache shape {features.shape} does not match {shape}; "
                "delete the cache or use a different manifest",
                file=sys.stderr,
            )
            return 1
    else:
        features = np.lib.format.open_memmap(
            feat_path, mode="w+", dtype=np.float16, shape=shape
        )
        done = np.zeros(n_rows, dtype=bool)

    todo = np.flatnonzero(~done)
    print(
        f"backbone  {backbone.name}  ({backbone.n_params():,} params, dim {backbone.dim})"
    )
    print(f"device    {backbone.device}")
    print(f"manifest  {args.manifest}  {n_rows:,} rows")
    print(f"variant   {cell.label}   crops {args.crops}x{args.crop_size}")
    print(f"cache     {feat_path}")
    print(
        f"todo      {len(todo):,} of {n_rows:,} ({n_rows - len(todo):,} already cached)\n"
    )

    if not len(todo):
        print("nothing to do")
        return 0

    paths = df["path"].tolist()
    n_failed = 0

    def prepare(group: np.ndarray):
        """Decode, degrade, and crop a group without touching model state."""
        crops, owners, failed = [], [], []
        for row in group:
            img, _error = load_image_safe(paths[row])
            if img is None:
                failed.append(row)
                continue
            # Keep crops stable across batch orders.
            rng = np.random.default_rng(args.seed + int(row))
            degraded = cell.apply(img, rng)
            crops += backbone.crops(degraded, args.crops, rng)
            owners.append(row)
        return group, owners, crops, failed

    groups = [
        todo[s : s + args.batch_size] for s in range(0, len(todo), args.batch_size)
    ]

    def prepared():
        """Yield prepared groups with optional threaded decoding."""
        if args.workers <= 0:
            yield from (prepare(group) for group in groups)
            return
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending: deque = deque()
            index = 0
            while index < len(groups) and len(pending) <= args.workers:
                pending.append(pool.submit(prepare, groups[index]))
                index += 1
            while pending:
                result = pending.popleft().result()
                if index < len(groups):
                    pending.append(pool.submit(prepare, groups[index]))
                    index += 1
                yield result

    with tqdm(total=len(todo), desc="extract", unit="img") as bar:
        for batch_index, (group, owners, crops, failed) in enumerate(prepared()):
            for row in failed:
                n_failed += 1
                done[row] = True  # never retried; the row stays zero-filled

            if crops:
                embeddings = backbone.embed(crops)
                for i, row in enumerate(owners):
                    features[row] = embeddings[i * args.crops : (i + 1) * args.crops]
                    done[row] = True

            bar.update(len(group))
            if batch_index % 50 == 0:
                features.flush()
                np.save(done_path, done)

    features.flush()
    np.save(done_path, done)
    meta_path.write_text(
        json.dumps(
            {
                "backbone": backbone.name,
                "requested_backbone": backbone.requested,
                "used_fallback": backbone.used_fallback,
                "dim": backbone.dim,
                "manifest": str(args.manifest),
                "rows": n_rows,
                "crops": args.crops,
                "crop_size": args.crop_size,
                "variant": args.variant,
                "seed": args.seed,
                "workers": args.workers,
                "failed": n_failed,
                "note": "crops taken at native resolution; images smaller than the crop are "
                "reflection-padded, never upscaled (hard rule 4)",
            },
            indent=2,
        )
    )

    print(
        f"\ncached {int(done.sum()):,}/{n_rows:,} rows"
        f"{f' ({n_failed} unreadable)' if n_failed else ''}"
    )
    print(f"size   {feat_path.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
