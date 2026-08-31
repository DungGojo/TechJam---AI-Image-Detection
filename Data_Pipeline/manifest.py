"""Manifest loading, validation, and balancing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from Data_Pipeline.hashing import (
    PHASH_MATCH_THRESHOLD,
    PHashIndex,
    phash,
    read_hash_list,
    sha256_file,
)
from Data_Pipeline.image_io import ImageLoadError, image_size, load_image, resolve_path

COLUMNS: tuple[str, ...] = (
    "path",
    "label",
    "source",
    "generator",
    "split",
    "sha256",
    "width",
    "height",
    "phash",
)

# Locally tampered images remain held outside the binary task.
SPLITS: frozenset[str] = frozenset({"train", "val", "test_unseen", "eval_only", "held"})

LABEL_AUTHENTIC = 0
LABEL_GENERATED = 1

# Fixed evaluation and held splits are exempt from class-balance validation.
_IMBALANCE_EXEMPT: frozenset[str] = frozenset({"eval_only", "held"})

MAX_CLASS_FRACTION = 0.60


class ManifestError(Exception):
    """Raised by `validate` with every problem found, not just the first."""


@dataclass(frozen=True)
class Blocklist:
    """The eval-only quarantine (§4.5), on both hash kinds."""

    sha256: frozenset[str]
    phash_index: PHashIndex

    @classmethod
    def load(
        cls,
        sha_path: str | Path = "data/evaluation/blocklists/sha256.txt",
        phash_path: str | Path = "data/evaluation/blocklists/phash.txt",
    ) -> "Blocklist":
        return cls(
            sha256=frozenset(read_hash_list(resolve_path(sha_path))),
            phash_index=PHashIndex(read_hash_list(resolve_path(phash_path))),
        )

    @classmethod
    def empty(cls) -> "Blocklist":
        return cls(sha256=frozenset(), phash_index=PHashIndex([]))

    def __len__(self) -> int:
        """Number of quarantined images, by the exact-hash list."""
        return len(self.sha256)

    @property
    def is_empty(self) -> bool:
        """Return true only when both hash collections are empty."""
        return not self.sha256 and not len(self.phash_index)


def row_from_image(
    path: str | Path,
    label: int,
    source: str,
    generator: str,
    split: str,
    *,
    with_phash: bool = True,
) -> dict[str, object]:
    """Build one manifest row, recording native dimensions before processing."""
    p = Path(path).resolve()
    width, height = image_size(p)
    row: dict[str, object] = {
        "path": str(p),
        "label": int(label),
        "source": source,
        "generator": generator,
        "split": split,
        "sha256": sha256_file(p),
        "width": int(width),
        "height": int(height),
        "phash": "",
    }
    if with_phash:
        try:
            row["phash"] = phash(load_image(p))
        except ImageLoadError:
            row["phash"] = ""
    return row


def build(
    rows: Iterable[dict[str, object]],
    out_path: str | Path | None = None,
    *,
    sort: bool = True,
) -> pd.DataFrame:
    """Assemble rows into a manifest DataFrame and optionally write the CSV."""
    df = pd.DataFrame(list(rows), columns=list(COLUMNS))
    if sort:
        df = df.sort_values(["source", "split", "path"]).reset_index(drop=True)
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def build_from_paths(
    entries: Sequence[tuple[str | Path, int, str, str, str]],
    out_path: str | Path | None = None,
    *,
    with_phash: bool = True,
    desc: str = "hashing",
) -> pd.DataFrame:
    """Convenience wrapper: (path, label, source, generator, split) tuples in."""
    rows = []
    skipped: list[str] = []
    for path, label, source, generator, split in tqdm(entries, desc=desc, unit="img"):
        try:
            rows.append(
                row_from_image(path, label, source, generator, split, with_phash=with_phash)
            )
        except Exception as exc:  # noqa: BLE001 - a bad file must not kill the build
            skipped.append(f"{path}: {exc.__class__.__name__}: {exc}")
    if skipped:
        print(f"[manifest] skipped {len(skipped)} unreadable file(s):")
        for line in skipped[:10]:
            print(f"  - {line}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    return build(rows, out_path)


def load(path: str | Path) -> pd.DataFrame:
    path = resolve_path(path)
    df = pd.read_csv(path, dtype={"sha256": str, "phash": str}, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError(f"{path}: missing column(s) {missing}")
    return df


def validate(
    df: pd.DataFrame,
    *,
    blocklist: Blocklist | None = None,
    check_files: bool = True,
    check_phash: bool = True,
    phash_threshold: int = PHASH_MATCH_THRESHOLD,
) -> pd.DataFrame:
    """Validate schema, files, leakage, quarantine, and class balance."""
    problems: list[str] = []

    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        raise ManifestError(f"manifest is missing column(s): {missing_cols}")
    if df.empty:
        raise ManifestError("manifest is empty")

    bad_labels = sorted(set(df["label"].unique()) - {LABEL_AUTHENTIC, LABEL_GENERATED})
    if bad_labels:
        problems.append(f"label must be 0 or 1; found {bad_labels}")

    bad_splits = sorted(set(df["split"].unique()) - SPLITS)
    if bad_splits:
        problems.append(f"unknown split(s) {bad_splits}; allowed: {sorted(SPLITS)}")

    non_positive = df[(df["width"] <= 0) | (df["height"] <= 0)]
    if len(non_positive):
        problems.append(f"{len(non_positive)} row(s) have a non-positive width/height")

    empty_hash = df[df["sha256"].astype(str).str.len() != 64]
    if len(empty_hash):
        problems.append(f"{len(empty_hash)} row(s) have a malformed sha256")

    if check_files:
        absent = [p for p in df["path"] if not resolve_path(p).exists()]
        if absent:
            problems.append(
                f"{len(absent)} file(s) in the manifest do not exist, e.g. {absent[:3]}"
            )

    spread = df.groupby("sha256")["split"].nunique()
    straddling = spread[spread > 1]
    if len(straddling):
        examples = list(straddling.index[:3])
        problems.append(
            f"{len(straddling)} sha256(s) appear in more than one split (leakage), "
            f"e.g. {examples}"
        )

    if blocklist is not None and not blocklist.is_empty:
        trainable = df[df["split"] != "eval_only"]

        hits = trainable[trainable["sha256"].isin(blocklist.sha256)] if blocklist.sha256 else trainable.iloc[:0]
        if len(hits):
            problems.append(
                f"CONTAMINATION: {len(hits)} training/val row(s) match the eval-only "
                f"blocklist on sha256, e.g. {list(hits['path'].head(3))}"
            )

        if check_phash and len(blocklist.phash_index):
            near = []
            for path, ph in zip(trainable["path"], trainable["phash"]):
                if not ph:
                    continue
                distance, _ = blocklist.phash_index.closest(ph)
                if distance <= phash_threshold:
                    near.append((path, distance))
            if near:
                problems.append(
                    f"CONTAMINATION: {len(near)} training/val row(s) are perceptual "
                    f"near-duplicates of eval-only images (distance <= "
                    f"{phash_threshold}), e.g. {near[:3]}"
                )

    for split_name, group in df.groupby("split"):
        if split_name in _IMBALANCE_EXEMPT:
            continue
        counts = group["label"].value_counts()
        if len(counts) < 2:
            problems.append(
                f"split '{split_name}' has only one class "
                f"(label={list(counts.index)}, n={len(group)})"
            )
            continue
        fraction = counts.max() / len(group)
        if fraction > MAX_CLASS_FRACTION:
            problems.append(
                f"split '{split_name}' is imbalanced beyond "
                f"{MAX_CLASS_FRACTION:.0%}/{1 - MAX_CLASS_FRACTION:.0%}: "
                f"{fraction:.1%} of {len(group)} rows are label={counts.idxmax()}"
            )

    if problems:
        raise ManifestError(
            f"manifest failed validation ({len(problems)} problem(s)):\n"
            + "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(problems))
        )
    return df


def drop_cross_split_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep each hash in one split, preferring the stricter split."""
    precedence = {"eval_only": 0, "test_unseen": 1, "held": 2, "val": 3, "train": 4}
    spread = df.groupby("sha256")["split"].nunique()
    offenders = set(spread[spread > 1].index)
    if not offenders:
        return df, 0

    keep_mask = []
    chosen: dict[str, str] = {}
    for sha, group in df[df["sha256"].isin(offenders)].groupby("sha256"):
        chosen[sha] = min(group["split"], key=lambda s: precedence.get(s, 99))

    for sha, split in zip(df["sha256"], df["split"]):
        keep_mask.append(sha not in offenders or split == chosen[sha])

    cleaned = df[pd.Series(keep_mask, index=df.index)].drop_duplicates("sha256")
    return cleaned.reset_index(drop=True), len(df) - len(cleaned)


def drop_contaminated(
    df: pd.DataFrame,
    blocklist: Blocklist,
    *,
    phash_threshold: int = PHASH_MATCH_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove non-evaluation rows matching quarantined hashes."""
    if blocklist.is_empty:
        return df, {"sha256": 0, "phash": 0}

    trainable = df["split"] != "eval_only"
    by_sha = trainable & df["sha256"].isin(blocklist.sha256)

    by_phash = pd.Series(False, index=df.index)
    if len(blocklist.phash_index):
        for i, (ok, ph) in enumerate(zip(trainable, df["phash"])):
            if ok and ph and blocklist.phash_index.closest(ph)[0] <= phash_threshold:
                by_phash.iloc[i] = True

    drop = by_sha | by_phash
    counts = {"sha256": int(by_sha.sum()), "phash": int((by_phash & ~by_sha).sum())}
    return df[~drop].reset_index(drop=True), counts


def clean(
    df: pd.DataFrame, blocklist: Blocklist, *, verbose: bool = True
) -> pd.DataFrame:
    """Everything a freshly built manifest needs before it can validate."""
    df, n_dupes = drop_cross_split_duplicates(df)
    df, contaminated = drop_contaminated(df, blocklist)
    if verbose:
        if n_dupes:
            print(f"[manifest] dropped {n_dupes} row(s) duplicated across splits")
        if contaminated["sha256"] or contaminated["phash"]:
            print(
                f"[manifest] QUARANTINE: dropped {contaminated['sha256']} exact and "
                f"{contaminated['phash']} perceptual match(es) against the eval-only set"
            )
    return df


def balance_test_unseen(
    df: pd.DataFrame, seed: int = 1337, *, verbose: bool = True
) -> pd.DataFrame:
    """Move stratified real rows into the unseen-generator test split."""
    unseen = df["split"] == "test_unseen"
    n_fake = int((unseen & (df["label"] == LABEL_GENERATED)).sum())
    n_real = int((unseen & (df["label"] == LABEL_AUTHENTIC)).sum())
    if n_fake == 0 or n_real >= n_fake:
        return df

    need = n_fake - n_real
    pool = df[(df["split"] == "train") & (df["label"] == LABEL_AUTHENTIC)]
    if pool.empty:
        return df
    need = min(need, len(pool) - 1)

    rng = np.random.default_rng(seed)
    sources = sorted(pool["generator"].unique())
    per_source = -(-need // len(sources))

    picked: list = []
    for source in sources:
        rows = pool[pool["generator"] == source]
        take = min(per_source, len(rows), need - len(picked))
        if take <= 0:
            continue
        idx = rng.choice(rows.index.to_numpy(), size=take, replace=False)
        picked.extend(idx.tolist())

    out = df.copy()
    out.loc[picked, "split"] = "test_unseen"
    if verbose:
        print(
            f"[manifest] moved {len(picked):,} real image(s) from train into "
            f"test_unseen across {len(sources)} source(s), so the split has "
            f"both classes ({n_fake:,} fake)"
        )
    return out


def summarise(df: pd.DataFrame) -> str:
    """Human-readable breakdown, for the STOP-checkpoint reports."""
    lines = [f"rows: {len(df)}   sources: {sorted(df['source'].unique())}"]
    table = (
        df.groupby(["source", "split", "label"]).size().rename("n").reset_index()
    )
    for _, r in table.iterrows():
        kind = "real" if r["label"] == LABEL_AUTHENTIC else "fake"
        lines.append(f"  {r['source']:<10} {r['split']:<11} {kind:<5} {r['n']:>7}")
    px = df[["width", "height"]]
    lines.append(
        f"  resolution: median {int(px['width'].median())}x{int(px['height'].median())}, "
        f"min side {int(px.min().min())}, max side {int(px.max().max())}"
    )
    lines.append(f"  generators: {df['generator'].nunique()} distinct")
    return "\n".join(lines)
