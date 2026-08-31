"""Download the quarantined evaluation subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from data_pipeline import manifest as manifest_mod
from data_pipeline.hashing import phash, sha256_file, write_hash_list
from data_pipeline.image_io import ImageLoadError, load_image
from data_pipeline.sources.remote_zip import DEFAULT_WORKERS
from data_pipeline.sources.wildfake import (
    DEMO_FAKE,
    DEMO_REAL,
    DEMO_TOTAL,
    archive_sizes,
    image_members,
    open_archive,
)


def fetch_half(
    spec: dict,
    out_root: Path,
    sizes: dict[str, int],
    workers: int,
    limit: int | None = None,
) -> list[Path]:
    """Download one half of the demo subset. Returns the local paths."""
    dest = out_root / spec["subdir"]
    dest.mkdir(parents=True, exist_ok=True)

    archive = open_archive(spec["archive"], sizes)
    print(f"\n{spec['archive']}  ({archive.size / 1e9:.2f} GB)")
    members = image_members(archive, spec["prefix"])
    print(f"  members under {spec['prefix']!r}: {len(members):,}")

    if limit is None and len(members) != spec["expected"]:
        print(
            f"  WARNING: expected {spec['expected']:,} images, found {len(members):,}. "
            "The archive layout may have changed -- check before trusting the blocklist."
        )
    if limit is not None:
        # Apply the debug limit before downloading.
        members = sorted(members, key=lambda m: m.header_offset)[:limit]
        print(f"  --limit {limit}: fetching {len(members):,}")

    def local_path(name: str) -> Path:
        return dest / name[len(spec["prefix"]) :].replace("/", "__")

    todo = [m for m in members if not local_path(m.name).exists()]
    have = len(members) - len(todo)
    if have:
        print(f"  already on disk: {have:,} (resuming)")
    if not todo:
        return [local_path(m.name) for m in members]

    total_bytes = sum(m.compress_size for m in todo)
    print(f"  to fetch: {len(todo):,} images, {total_bytes / 1e9:.2f} GB")

    failures = 0
    with tqdm(total=len(todo), desc=f"  {spec['generator']}", unit="img") as bar:
        for member, data, error in archive.read_many(todo, workers=workers):
            if error is not None:
                failures += 1
                print(f"\n  FAILED {member.name}: {error}", file=sys.stderr)
            else:
                local_path(member.name).write_bytes(data)
            bar.update(1)

    if failures:
        print(f"  {failures:,} member(s) failed; re-run to retry them", file=sys.stderr)
    return [local_path(m.name) for m in members if local_path(m.name).exists()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=Path("data/evaluation/images/benchmark"))
    p.add_argument(
        "--manifest", type=Path, default=Path("data/evaluation/benchmark_manifest.csv")
    )
    p.add_argument(
        "--blocklist-dir", type=Path, default=Path("data/evaluation/blocklists")
    )
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--limit", type=int, default=None, help="debug: cap images per half")
    args = p.parse_args(argv)

    sizes = archive_sizes()
    paths: list[tuple[Path, dict]] = []

    for spec in (DEMO_REAL, DEMO_FAKE):
        got = fetch_half(spec, args.out, sizes, args.workers, args.limit)
        paths += [(path, spec) for path in got]

    print(f"\non disk: {len(paths):,} images (target {DEMO_TOTAL:,})")

    print("\nhashing for the blocklist...")
    sha_list, phash_list, entries = [], [], []
    unreadable = 0

    for path, spec in tqdm(paths, desc="  hashing", unit="img"):
        sha_list.append(sha256_file(path))
        try:
            phash_list.append(phash(load_image(path)))
        except ImageLoadError as exc:
            unreadable += 1
            print(f"\n  unreadable, sha256 only: {exc}", file=sys.stderr)
        entries.append((path, spec["label"], "demo", spec["generator"], "eval_only"))

    args.blocklist_dir.mkdir(parents=True, exist_ok=True)
    sha_path = args.blocklist_dir / "sha256.txt"
    phash_path = args.blocklist_dir / "phash.txt"
    write_hash_list(
        sha_path,
        sha_list,
        header=(
            "SETUP.md §4.5 -- eval-only quarantine (sha256).\n"
            f"COCO val2017 + DALL-E 3 Advanced. {len(sha_list)} entries.\n"
            "Any training or validation row matching one of these fails validate()."
        ),
    )
    write_hash_list(
        phash_path,
        phash_list,
        header=(
            "SETUP.md §4.5 -- eval-only quarantine (perceptual, 64-bit DCT).\n"
            f"{len(phash_list)} entries. Catches re-encoded near-duplicates that\n"
            "sha256 cannot see -- WildFake ships COCO train2017 and val2017 in\n"
            "the same archive."
        ),
    )

    df = manifest_mod.build_from_paths(entries, args.manifest, desc="  manifest")
    blocklist = manifest_mod.Blocklist.load(sha_path, phash_path)
    manifest_mod.validate(df, blocklist=blocklist)

    print("\n" + manifest_mod.summarise(df))
    print(f"\nsha256 blocklist   {len(sha_list):,}  -> {sha_path}")
    print(f"phash blocklist    {len(phash_list):,}  -> {phash_path}")
    if unreadable:
        print(f"unreadable images  {unreadable}")
    print(f"manifest           {args.manifest}")

    if args.limit:
        print(f"\n--limit was set, so {len(sha_list):,} != {DEMO_TOTAL:,} is expected")
        return 0
    if len(sha_list) == DEMO_TOTAL:
        print(f"\nblocklist count {len(sha_list):,} == expected {DEMO_TOTAL:,}  OK")
    else:
        print(
            f"\nMISMATCH: {len(sha_list):,} hashes, expected {DEMO_TOTAL:,}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
