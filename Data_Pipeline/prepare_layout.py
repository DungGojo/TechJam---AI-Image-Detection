#!/usr/bin/env python3
"""Build lifecycle manifests after dataset acquisition."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PATH_REWRITES = {
    "/data/raw/sid_set/": "data/train/images/sid_set/",
    "/data/raw/wildfake/": "data/train/images/wildfake/",
    "/data/raw/wildfake_meta/": "data/train/images/wildfake_meta/",
    "/data/eval_only/": "data/evaluation/images/benchmark/",
}


def portable_path(value: str) -> str:
    normalised = value.replace("\\", "/")
    path = Path(value)
    if path.is_absolute():
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            pass
    for marker, replacement in PATH_REWRITES.items():
        if marker in normalised:
            return replacement + normalised.split(marker, 1)[1]
    if not path.is_absolute():
        return path.as_posix()
    raise ValueError(f"cannot map legacy path into the new layout: {value}")


def rewrite(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["path"] = frame["path"].astype(str).map(portable_path)
    frame.to_csv(path, index=False)
    return frame


def relocate_by_lifecycle(frame: pd.DataFrame) -> pd.DataFrame:
    """Move acquired images into their owning lifecycle without copying bytes."""
    roots = {
        "train": ROOT / "data/train/images",
        "val": ROOT / "data/evaluation/images/validation",
        "held": ROOT / "data/evaluation/images/held",
        "test_unseen": ROOT / "data/test/images",
    }
    moved = 0
    out = frame.copy()
    for index, row in out.iterrows():
        split = str(row["split"])
        source_name = str(row["source"])
        if split not in roots or source_name not in {"sid_set", "wildfake"}:
            continue
        source_path = ROOT / str(row["path"])
        parts = list(Path(row["path"]).parts)
        try:
            source_index = parts.index(source_name)
        except ValueError as exc:
            raise ValueError(f"path has no source segment {source_name!r}: {row['path']}") from exc
        suffix = Path(*parts[source_index + 1 :])
        target = roots[split] / source_name / suffix
        if source_path != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            if source_path.exists():
                if target.exists():
                    raise FileExistsError(f"both source and lifecycle target exist: {target}")
                source_path.replace(target)
                moved += 1
            elif not target.exists():
                raise FileNotFoundError(f"missing both source and lifecycle target: {source_path}")
        out.at[index, "path"] = target.relative_to(ROOT).as_posix()
    print(f"moved {moved:,} image(s) into lifecycle-owned directories")
    return out


def remove_empty_image_directories() -> None:
    for base in (ROOT / "data/train/images", ROOT / "data/evaluation/images", ROOT / "data/test/images"):
        for directory in sorted((path for path in base.rglob("*") if path.is_dir()), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass


def main() -> int:
    sid_path = ROOT / "data/train/source_manifests/sid_set.csv"
    wild_path = ROOT / "data/train/source_manifests/wildfake.csv"
    benchmark_path = ROOT / "data/evaluation/benchmark_manifest.csv"
    sid = relocate_by_lifecycle(rewrite(sid_path))
    wild = relocate_by_lifecycle(rewrite(wild_path))
    sid.to_csv(sid_path, index=False)
    wild.to_csv(wild_path, index=False)
    rewrite(benchmark_path)
    combined = pd.concat([sid, wild], ignore_index=True)
    targets = {
        ROOT / "data/train/manifest.csv": combined[combined["split"] == "train"],
        ROOT / "data/evaluation/validation_manifest.csv": combined[combined["split"] == "val"],
        ROOT / "data/evaluation/held_manifest.csv": combined[combined["split"] == "held"],
        ROOT / "data/test/manifest.csv": combined[combined["split"] == "test_unseen"],
    }
    for path, frame in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.reset_index(drop=True).to_csv(path, index=False)
        print(f"{path.relative_to(ROOT)}: {len(frame):,} rows")
    remove_empty_image_directories()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
