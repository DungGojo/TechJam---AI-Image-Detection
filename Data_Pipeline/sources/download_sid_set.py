"""Download a bounded SID_Set subset."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402
from tqdm import tqdm  # noqa: E402

from Data_Pipeline import manifest as manifest_mod  # noqa: E402

REPO = "saberzl/SID_Set"
CLASS_NAMES = {0: "real", 1: "full_synthetic", 2: "tampered"}
# Preserve source split boundaries to avoid near-duplicate leakage.
SPLIT_FOR_SHARD = {"train": "train", "validation": "val"}


def shard_names(prefix: str) -> list[str]:
    files = list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith(f"data/{prefix}-") and f.endswith(".parquet"))


def extract_shard(
    shard: str, out_root: Path, split: str, tampered_as: str, budget: dict[str, int]
) -> list[tuple[Path, int, str, str, str]]:
    """Download one shard, write its images, delete the parquet. Returns entries."""
    local = Path(
        hf_hub_download(REPO, shard, repo_type="dataset", cache_dir="cache/hf")
    )
    entries: list[tuple[Path, int, str, str, str]] = []

    try:
        pf = pq.ParquetFile(local)
        for group in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(group, columns=["img_id", "image", "label"])
            data = table.to_pydict()
            for img_id, image, label in zip(data["img_id"], data["image"], data["label"]):
                klass = CLASS_NAMES.get(int(label), "unknown")
                if budget.get(klass, 0) <= 0:
                    continue

                if klass == "tampered":
                    if tampered_as == "skip":
                        continue
                    row_split, row_label = "held", 1
                elif klass == "real":
                    row_split, row_label = split, 0
                else:
                    row_split, row_label = split, 1

                payload = image["bytes"] if isinstance(image, dict) else image
                if not payload:
                    continue

                dest = out_root / klass
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / f"{img_id}.png"
                if not path.exists():
                    path.write_bytes(payload)
                budget[klass] -= 1
                entries.append((path, row_label, "sid_set", klass, row_split))
    finally:
        # Remove both the snapshot link and backing blob after each shard.
        try:
            blob = local.resolve()
            local.unlink(missing_ok=True)
            if blob != local and blob.exists():
                blob.unlink()
        except OSError:
            pass

    return entries


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=20_000, help="total images to materialise")
    p.add_argument("--per-class", type=int, default=None, help="cap per class (default: limit/3)")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--out", type=Path, default=Path("data/train/images/sid_set"))
    p.add_argument("--manifest", type=Path, default=Path("data/train/source_manifests/sid_set.csv"))
    p.add_argument(
        "--tampered-as",
        choices=["held", "skip"],
        default="held",
        help="'held' keeps tampered rows out of training but in the manifest; "
             "'skip' drops them entirely",
    )
    args = p.parse_args(argv)

    per_class = args.per_class or max(1, args.limit // 3)
    budget = {"real": per_class, "full_synthetic": per_class, "tampered": per_class}
    print(f"target: {per_class:,} per class ({', '.join(budget)})")
    print(f"tampered rows -> split={args.tampered_as!r}\n")

    entries: list[tuple[Path, int, str, str, str]] = []

    for prefix in ("train", "validation"):
        split = SPLIT_FOR_SHARD[prefix]
        shards = shard_names(prefix)
        want = args.val_fraction if prefix == "validation" else (1 - args.val_fraction)
        target = {k: int(v * want) for k, v in budget.items()}
        local_budget = dict(target)
        print(f"{prefix}: {len(shards)} shards available, target {target}")

        for shard in tqdm(shards, desc=f"  {prefix}", unit="shard"):
            if all(v <= 0 for v in local_budget.values()):
                break
            entries += extract_shard(shard, args.out, split, args.tampered_as, local_budget)

    print(f"\nmaterialised {len(entries):,} images")
    df = manifest_mod.build_from_paths(entries, args.manifest, desc="  manifest")
    blocklist = manifest_mod.Blocklist.load()
    df = manifest_mod.clean(df, blocklist)
    df.to_csv(args.manifest, index=False)
    manifest_mod.validate(df, blocklist=blocklist)

    print("\n" + manifest_mod.summarise(df))
    print(f"\nmanifest {args.manifest}")
    print("\n§4.3 DECISION SURFACED, NOT MADE: tampered rows are in the manifest with")
    print("split='held' and are excluded from the binary task. Confirm or override.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
