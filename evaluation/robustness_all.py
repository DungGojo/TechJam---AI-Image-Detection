"""Streaming, resumable robustness scoring for the presentation pipeline."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from data_pipeline.image_io import load_image_safe


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _dataset_signature(paths: list[Path]) -> str:
    records = []
    for path in paths:
        path = path.resolve()
        stat = path.stat()
        records.append(f"{path}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(records).encode()).hexdigest()[:16]


def _condition_key(name: str, parameters: str) -> str:
    return hashlib.sha256(f"{name}|{parameters}".encode()).hexdigest()[:12]


def _score_condition(
    scorer,
    sources: pd.DataFrame,
    transform_spec: tuple[str, str, str],
    make_transformation,
    cache_root: Path,
    dataset_key: str,
    description: str,
    cache_name: str,
    batch_size: int,
) -> np.ndarray:
    transform_name, parameters, _ = transform_spec
    run_cache = cache_root / "robustness_all" / dataset_key / _condition_key(
        transform_name, parameters
    )
    run_cache.mkdir(parents=True, exist_ok=True)
    probability_path = run_cache / f"{cache_name}.npy"
    done_path = run_cache / f"{cache_name}.done.npy"

    cache_matches = False
    if probability_path.exists() and done_path.exists():
        probabilities = np.load(probability_path, mmap_mode="r+")
        done = np.load(done_path)
        cache_matches = len(probabilities) == len(sources) and len(done) == len(sources)

    if cache_matches:
        print(description, "- resumed", int(done.sum()), "/", len(done))
    else:
        probabilities = np.lib.format.open_memmap(
            probability_path, mode="w+", dtype=np.float64, shape=(len(sources),)
        )
        probabilities[:] = np.nan
        done = np.zeros(len(sources), dtype=bool)
        np.save(done_path, done)

    for start in tqdm(
        range(0, len(sources), batch_size), desc=description, unit="batch"
    ):
        stop = min(start + batch_size, len(sources))
        pending = [position for position in range(start, stop) if not done[position]]
        transformed_images, positions = [], []

        for position in pending:
            row = sources.iloc[position]
            loaded, error = load_image_safe(row.path)
            if loaded is None:
                print("Skipped unreadable image:", row.path, error)
                done[position] = True
                continue
            original = loaded.convert("RGB")
            loaded.close()
            transformed = make_transformation(
                original, transform_name, parameters, row.path
            )
            original.close()
            transformed_images.append(transformed)
            positions.append(position)

        if transformed_images:
            probabilities[positions] = scorer.score_checked(transformed_images)
            done[positions] = True
            for image in transformed_images:
                image.close()

        probabilities.flush()
        np.save(done_path, done)

    return np.asarray(probabilities).copy()


def run_all_image_robustness(
    *,
    sources: pd.DataFrame,
    transform_specs: list[tuple[str, str, str]],
    make_transformation,
    detector_runs: list[tuple[str, str, object, int]],
    feature_columns: list[str],
    fusion_model,
    threshold: float,
    cache_root: Path,
) -> pd.DataFrame:
    """Score every source under every transform without storing transformed images."""
    sources = sources.reset_index(drop=True).copy()
    dataset_key = _dataset_signature([Path(path) for path in sources.path])
    condition_scores = {
        index: {
            "Transform": name,
            "Parameters": parameters,
            "Real-World Analog": analog,
        }
        for index, (name, parameters, analog) in enumerate(transform_specs)
    }

    for detector_number, detector_run in enumerate(detector_runs, start=1):
        feature_name, cache_name, make_scorer, batch_size = detector_run
        scorer = make_scorer()
        for condition_index, transform_spec in enumerate(transform_specs):
            name, parameters, _ = transform_spec
            description = (
                f"Robustness {detector_number}/3 {name} {parameters}"
            )
            condition_scores[condition_index][feature_name] = _score_condition(
                scorer,
                sources,
                transform_spec,
                make_transformation,
                cache_root,
                dataset_key,
                description,
                cache_name,
                batch_size,
            )
        del scorer
        _release_memory()

    frames = []
    base_columns = ["path", "relative_path", "label"]
    for condition_index in range(len(transform_specs)):
        frame = sources[base_columns].copy()
        condition = condition_scores[condition_index]
        for column in ("Transform", "Parameters", "Real-World Analog"):
            frame[column] = condition[column]
        for feature_name in feature_columns:
            frame[feature_name] = condition[feature_name]

        valid = frame[feature_columns].notna().all(axis=1)
        frame["final_probability"] = np.nan
        frame.loc[valid, "final_probability"] = fusion_model.predict_proba(
            frame.loc[valid, feature_columns].to_numpy(dtype=float)
        )[:, 1]
        frame["prediction"] = pd.array(
            np.where(
                frame.final_probability.notna(),
                (frame.final_probability >= threshold).astype(int),
                pd.NA,
            ),
            dtype="Int64",
        )
        frames.append(
            frame[
                base_columns
                + [
                    "Transform",
                    "Parameters",
                    "Real-World Analog",
                    "final_probability",
                    "prediction",
                ]
            ]
        )

    result = pd.concat(frames, ignore_index=True)
    del condition_scores, frames
    _release_memory()
    return result
