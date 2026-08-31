"""Read frozen-feature caches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from Data_Pipeline import manifest as manifest_mod

DEFAULT_ROOT = Path("cache/features")

# Only deterministic, size-preserving variants support paired consistency loss.
ALIGNED_FAMILIES = frozenset({"jpeg", "blur", "resize", "webp", "sharpen"})


def variant_tag(variant: str) -> str:
    """'jpeg=50' -> 'jpeg50'. Mirrors the stem built by cache_features."""
    return variant.replace("=", "")


def crops_aligned(variant: str) -> bool:
    """Are this variant's crops taken at the same positions as the clean crops?"""
    if variant == "clean":
        return True
    return variant.partition("=")[0] in ALIGNED_FAMILIES


def cache_paths(
    manifest: str | Path,
    *,
    backbone: str,
    crops: int,
    crop_size: int,
    variant: str,
    root: str | Path = DEFAULT_ROOT,
) -> tuple[Path, Path, Path]:
    """(features, done mask, metadata) for one cached variant."""
    stem = f"{Path(manifest).stem}_{crops}x{crop_size}_{variant_tag(variant)}"
    directory = Path(root) / backbone.replace("/", "_")
    return (
        directory / f"{stem}.npy",
        directory / f"{stem}.done.npy",
        directory / f"{stem}.meta.json",
    )


@dataclass
class Variant:
    """One cached degradation cell for one manifest."""

    name: str
    features: np.ndarray       # (rows, crops, dim) float16, memory-mapped
    done: np.ndarray           # (rows,) bool
    meta: dict

    @property
    def aligned(self) -> bool:
        return crops_aligned(self.name)

    @property
    def dim(self) -> int:
        return int(self.features.shape[-1])

    @property
    def crops(self) -> int:
        return int(self.features.shape[1])


def load_variant(
    manifest: str | Path,
    *,
    backbone: str,
    crops: int,
    crop_size: int,
    variant: str,
    root: str | Path = DEFAULT_ROOT,
) -> Variant:
    feature_path, done_path, meta_path = cache_paths(
        manifest, backbone=backbone, crops=crops, crop_size=crop_size,
        variant=variant, root=root,
    )
    if not feature_path.exists() or not done_path.exists():
        raise FileNotFoundError(
            f"no cached features at {feature_path}. Build them with:\n"
            f"  python -m Workflows.cache_features --manifest {manifest} "
            f"--backbone {backbone} --crops {crops} --crop-size {crop_size} "
            f"--variant {variant}"
        )
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return Variant(
        name=variant,
        features=np.load(feature_path, mmap_mode="r"),
        done=np.load(done_path),
        meta=meta,
    )


class FeatureSet:
    """Load row-aligned cached variants for one manifest."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        backbone: str,
        crops: int,
        crop_size: int = 224,
        variants: list[str] | None = None,
        root: str | Path = DEFAULT_ROOT,
        extra: np.ndarray | dict[str, np.ndarray] | None = None,
    ):
        self.manifest_path = Path(manifest)
        self.backbone = backbone
        self.frame: pd.DataFrame = manifest_mod.load(self.manifest_path)
        self.variants: dict[str, Variant] = {
            name: load_variant(
                manifest, backbone=backbone, crops=crops,
                crop_size=crop_size, variant=name, root=root,
            )
            for name in (variants or ["clean"])
        }

        rows = len(self.frame)
        for name, variant in self.variants.items():
            if variant.features.shape[0] != rows:
                raise ValueError(
                    f"variant {name!r} caches {variant.features.shape[0]} rows but "
                    f"{self.manifest_path.name} has {rows}; the cache was built from a "
                    "different manifest"
                )

        self.ready = np.ones(rows, dtype=bool)
        for variant in self.variants.values():
            self.ready &= variant.done
        # Exclude unreadable images recorded as zero features.
        for variant in self.variants.values():
            self.ready &= np.abs(np.asarray(variant.features[:, 0, :])).sum(axis=1) > 0

        self.index = np.flatnonzero(self.ready)
        self.labels = self.frame["label"].to_numpy(dtype=np.int64)[self.index]

        # Match expert scores to each degradation; bare arrays mean clean-only.
        self.extra_by_variant: dict[str, np.ndarray] = {}
        if extra is not None:
            table = extra if isinstance(extra, dict) else {"clean": extra}
            if "clean" not in table:
                raise ValueError("expert scores must include a 'clean' entry")
            for name, values in table.items():
                array = np.asarray(values, dtype=np.float32)
                if array.ndim == 1:
                    array = array[:, None]
                if array.shape[0] != rows:
                    raise ValueError(
                        f"expert scores for {name!r} have {array.shape[0]} rows but "
                        f"{self.manifest_path.name} has {rows}"
                    )
                self.extra_by_variant[name] = array[self.index]


    @property
    def dim(self) -> int:
        return next(iter(self.variants.values())).dim

    @property
    def n_crops(self) -> int:
        return next(iter(self.variants.values())).crops

    @property
    def extra_dim(self) -> int:
        if not self.extra_by_variant:
            return 0
        return int(self.extra_by_variant["clean"].shape[1])

    def extra_for(self, variant: str) -> np.ndarray | None:
        """Return expert scores for a variant, falling back to clean scores."""
        if not self.extra_by_variant:
            return None
        return self.extra_by_variant.get(variant, self.extra_by_variant["clean"])

    def __len__(self) -> int:
        return len(self.index)

    def features(self, variant: str) -> np.ndarray:
        """(n_ready, crops, dim) float32, materialised from the memmap."""
        if variant not in self.variants:
            raise KeyError(f"variant {variant!r} not loaded; have {sorted(self.variants)}")
        return np.asarray(self.variants[variant].features[self.index], dtype=np.float32)

    def degraded_names(self) -> list[str]:
        return [name for name in self.variants if name != "clean"]

    def select(self, positions: np.ndarray) -> "FeatureSet":
        """A view over a subset of the ready rows. Shares the underlying memmaps."""
        view = object.__new__(FeatureSet)
        view.__dict__.update(self.__dict__)
        view.index = self.index[positions]
        view.labels = self.labels[positions]
        view.extra_by_variant = {
            name: values[positions] for name, values in self.extra_by_variant.items()
        }
        return view

    def split(self, fraction: float = 0.5, seed: int = 1337) -> tuple["FeatureSet", "FeatureSet"]:
        """Return disjoint stratified subsets for selection and calibration."""
        rng = np.random.default_rng(seed)
        first: list[np.ndarray] = []
        second: list[np.ndarray] = []
        for label in np.unique(self.labels):
            positions = np.flatnonzero(self.labels == label)
            rng.shuffle(positions)
            cut = int(round(len(positions) * fraction))
            first.append(positions[:cut])
            second.append(positions[cut:])
        return (
            self.select(np.sort(np.concatenate(first))),
            self.select(np.sort(np.concatenate(second))),
        )

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, variant in self.variants.items():
            rows.append({
                "variant": name,
                "cached rows": int(variant.done.sum()),
                "usable rows": int(self.ready.sum()),
                "crops": variant.crops,
                "dim": variant.dim,
                "crops aligned with clean": variant.aligned,
                "expert scores": (
                    "variant-matched" if name in self.extra_by_variant
                    else ("clean fallback" if self.extra_by_variant else "none")
                ),
            })
        return pd.DataFrame(rows)
