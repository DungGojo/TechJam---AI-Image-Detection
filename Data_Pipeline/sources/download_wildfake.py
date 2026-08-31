"""Download a stratified WildFake subset."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from Data_Pipeline import manifest as manifest_mod  # noqa: E402
from Data_Pipeline.sources.remote_zip import DEFAULT_WORKERS  # noqa: E402
from Data_Pipeline.sources.wildfake import (  # noqa: E402
    FAKE_ARCHIVES,
    REAL_ARCHIVES,
    UNSEEN_GENERATORS,
    archive_sizes,
    generator_of,
    image_members,
    is_quarantined,
    open_archive,
)


WINDOW = 24


def sample_windows(pool: list, want: int, rng: np.random.Generator, window: int = WINDOW) -> list:
    """Sample seeded consecutive member runs to reduce HTTP range requests."""
    ordered = sorted(pool, key=lambda m: m.header_offset)
    n_windows = max(1, -(-want // window))
    max_start = max(1, len(ordered) - window + 1)
    starts = rng.choice(max_start, size=min(n_windows, max_start), replace=False)

    picked: list = []
    seen: set[int] = set()
    for start in sorted(starts.tolist()):
        for i in range(start, min(start + window, len(ordered))):
            if i not in seen:
                seen.add(i)
                picked.append(ordered[i])
            if len(picked) >= want:
                return picked
    return picked


Pick = tuple  # (archive_path, member, generator)


def plan(
    archives: list[str],
    per_generator: int,
    sizes: dict[str, int],
    rng: np.random.Generator,
    seen: collections.Counter,
) -> list[Pick]:
    """Index archives and create a reproducible per-generator sample plan."""
    picks: list[Pick] = []

    for path in archives:
        try:
            archive = open_archive(path, sizes)
            members = image_members(archive)
        except Exception as exc:  # noqa: BLE001 - one bad archive must not stop the run
            print(f"  SKIP {path}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
            continue

        by_generator: dict[str, list] = collections.defaultdict(list)
        n_quarantined = 0
        for member in members:
            if is_quarantined(member.name):
                n_quarantined += 1
                continue
            generator, _ = generator_of(member.name, path)
            by_generator[generator].append(member)

        chosen = 0
        for generator, pool in sorted(by_generator.items()):
            want = max(0, per_generator - seen[generator])
            if want == 0:
                continue
            if len(pool) > want:
                picked = sample_windows(pool, want, rng)
            else:
                picked = pool
            seen[generator] += len(picked)
            chosen += len(picked)
            picks += [(path, m, generator) for m in picked]

        note = f" (skipped {n_quarantined:,} quarantined)" if n_quarantined else ""
        print(
            f"  {Path(path).name:<24} {len(members):>8,} images, "
            f"{len(by_generator):>2} generator(s) -> picked {chosen:,}{note}"
        )

    return picks


def enforce_budget(picks: list[Pick], budget_bytes: int, floor: int = 400) -> list[Pick]:
    """Trim the largest generators first while preserving a minimum per generator."""
    by_generator: dict[str, list[Pick]] = collections.defaultdict(list)
    for pick in picks:
        by_generator[pick[2]].append(pick)

    def total_bytes() -> int:
        return sum(m.compress_size for group in by_generator.values() for _, m, _ in group)

    while total_bytes() > budget_bytes:
        weights = {
            g: sum(m.compress_size for _, m, _ in group)
            for g, group in by_generator.items()
            if len(group) > floor
        }
        if not weights:
            break
        heaviest = max(weights, key=weights.get)
        group = by_generator[heaviest]
        keep = max(floor, int(len(group) * 0.75))
        by_generator[heaviest] = group[:keep]

    return [pick for group in by_generator.values() for pick in group]


def report(picks: list[Pick], title: str) -> None:
    by_generator: dict[str, list[Pick]] = collections.defaultdict(list)
    for pick in picks:
        by_generator[pick[2]].append(pick)
    print(f"\n{title}: {len(picks):,} images across {len(by_generator)} generators")
    for generator, group in sorted(by_generator.items(), key=lambda kv: -len(kv[1])):
        size = sum(m.compress_size for _, m, _ in group)
        tag = "  <- test_unseen" if generator in UNSEEN_GENERATORS else ""
        print(f"    {generator:<28} {len(group):>6,}  {size / 1e9:6.2f} GB{tag}")


def fetch(
    picks: list[Pick],
    label: int,
    out_root: Path,
    sizes: dict[str, int],
    workers: int,
    val_fraction: float,
    rng: np.random.Generator,
) -> list[tuple[Path, int, str, str, str]]:
    entries: list[tuple[Path, int, str, str, str]] = []
    by_archive: dict[str, list[Pick]] = collections.defaultdict(list)
    for pick in picks:
        by_archive[pick[0]].append(pick)

    for path, group in by_archive.items():
        archive = open_archive(path, sizes)
        todo, lookup = [], {}
        for _, member, generator in group:
            dest = out_root / generator
            dest.mkdir(parents=True, exist_ok=True)
            local = dest / member.name.replace("/", "__")[-180:]
            if local.exists():
                entries.append((local, label, "wildfake", generator, ""))
            else:
                todo.append(member)
                lookup[member.name] = (local, generator)

        if not todo:
            continue

        total = sum(m.compress_size for m in todo)
        with tqdm(total=len(todo), desc=f"  {Path(path).name[:22]:<22}", unit="img", leave=False) as bar:
            for member, data, error in archive.read_many(todo, workers=workers):
                local, generator = lookup[member.name]
                if error is None:
                    local.write_bytes(data)
                    entries.append((local, label, "wildfake", generator, ""))
                bar.update(1)
        print(f"  {Path(path).name:<24} fetched {len(todo):,} ({total / 1e9:.2f} GB)")

    finalised = []
    for local, lbl, source, generator, _ in entries:
        # Hold out complete generator families to prevent leakage.
        if generator in UNSEEN_GENERATORS:
            split = "test_unseen"
        else:
            split = "val" if rng.random() < val_fraction else "train"
        finalised.append((local, lbl, source, generator, split))
    return finalised


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-generator", type=int, default=3000)
    p.add_argument("--budget-gb", type=float, default=15.0, help="disk ceiling for this source")
    p.add_argument("--real-reserve", type=float, default=0.12,
                   help="fraction of the budget reserved for real images")
    p.add_argument("--out", type=Path, default=Path("data/train/images/wildfake"))
    p.add_argument("--manifest", type=Path, default=Path("data/train/source_manifests/wildfake.csv"))
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--plan-only", action="store_true", help="index and report, download nothing")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    sizes = archive_sizes()
    seen: collections.Counter = collections.Counter()

    print(f"planning (cap {args.per_generator:,} per generator)\n\nFAKE:")
    fake_picks = plan(FAKE_ARCHIVES, args.per_generator, sizes, rng, seen)

    planned = sum(m.compress_size for _, m, _ in fake_picks)
    print(f"\nfake planned: {len(fake_picks):,} images, {planned / 1e9:.2f} GB")

    # Reserve space for real images before trimming the fake pool.
    budget = int(args.budget_gb * 1e9)
    real_reserve = int(budget * args.real_reserve)
    if planned > budget - real_reserve:
        fake_picks = enforce_budget(fake_picks, budget - real_reserve)
        print(
            f"trimmed fake to fit: "
            f"{sum(m.compress_size for _, m, _ in fake_picks) / 1e9:.2f} GB"
        )

    # Derive the real cap from the fake count to keep each split balanced.
    real_cap = -(-len(fake_picks) // max(1, len(REAL_ARCHIVES)))
    print(f"\nREAL (cap {real_cap:,} per source, to match {len(fake_picks):,} fake):")
    real_picks = plan(REAL_ARCHIVES, real_cap, sizes, rng, seen)

    total = sum(m.compress_size for _, m, _ in fake_picks + real_picks)
    print(f"\ntotal planned: {len(fake_picks) + len(real_picks):,} images, {total / 1e9:.2f} GB")

    report(fake_picks, "FAKE")
    report(real_picks, "REAL")
    if args.plan_only:
        return 0

    print("\nfetching fake:")
    entries = fetch(fake_picks, 1, args.out / "fake", sizes, args.workers, args.val_fraction, rng)
    print("\nfetching real:")
    entries += fetch(real_picks, 0, args.out / "real", sizes, args.workers, args.val_fraction, rng)

    print(f"\nbuilding manifest for {len(entries):,} images")
    df = manifest_mod.build_from_paths(entries, args.manifest, desc="  manifest")
    blocklist = manifest_mod.Blocklist.load()
    df = manifest_mod.clean(df, blocklist)
    df = manifest_mod.balance_test_unseen(df, seed=args.seed)
    df.to_csv(args.manifest, index=False)
    manifest_mod.validate(df, blocklist=blocklist)

    print("\n" + manifest_mod.summarise(df))
    print(f"\nheld-out generators (test_unseen): {UNSEEN_GENERATORS}")
    print(f"manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
