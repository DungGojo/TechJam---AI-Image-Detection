"""Reconstruct clean and transformed Notebook 6 predictions from score caches."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/workspace/Coding")
IMAGE_DIR = PROJECT_ROOT / "testing_images"
CACHE_ROOT = PROJECT_ROOT / "outputs/full_pipeline/score_cache_sample"
FUSION_ARTIFACT = PROJECT_ROOT / "checkpoints/fusion/selected_final_fusion.joblib"
OUTPUT_PATH = PROJECT_ROOT / "outputs/notebook6/notebook6_all_image_predictions.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LABELS = {"cocoval2017(real)": 0, "dalleadvanced(fake)": 1}
LABEL_NAMES = {0: "Real", 1: "Fake"}

TRANSFORM_SPECS = [
    ("JPEG Compression", "quality=90"),
    ("JPEG Compression", "quality=70"),
    ("JPEG Compression", "quality=50"),
    ("JPEG Compression", "quality=30"),
    ("Gaussian Blur", "sigma=0.5"),
    ("Gaussian Blur", "sigma=1.0"),
    ("Gaussian Blur", "sigma=2.0"),
    ("Resize", "scale=0.5x then upscale"),
    ("Resize", "scale=0.25x then upscale"),
    ("Gaussian Noise", "sigma=0.02"),
    ("Gaussian Noise", "sigma=0.05"),
    ("Gaussian Noise", "sigma=0.10"),
    ("Color Jitter", "brightness/contrast/saturation ±20%"),
    ("Center Crop", "crop=80%"),
]

CACHE_NAMES = {
    "Community Forensics probability": "community_forensics.npy",
    "DINOv3-L probability": "dinov3_l.npy",
    "DINOv3-B probability": "dinov3_b_stacking.npy",
}


def dataset_signature(paths: list[Path]) -> str:
    records = []
    for path in paths:
        path = path.resolve()
        stat = path.stat()
        records.append(f"{path}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(records).encode()).hexdigest()[:16]


def condition_key(name: str, parameters: str) -> str:
    return hashlib.sha256(f"{name}|{parameters}".encode()).hexdigest()[:12]


def infer_label(path: Path) -> int:
    folders = [part.lower() for part in path.relative_to(IMAGE_DIR).parts[:-1]]
    return next(LABELS[folder] for folder in folders if folder in LABELS)


def load_probabilities(cache_dir: Path, expected_rows: int) -> dict[str, np.ndarray]:
    probabilities = {
        column: np.load(cache_dir / filename)
        for column, filename in CACHE_NAMES.items()
    }
    for values in probabilities.values():
        if len(values) != expected_rows:
            raise ValueError(
                f"Cache length {len(values):,} does not match {expected_rows:,}: {cache_dir}"
            )
    return probabilities


def make_prediction_frame(
    *,
    paths: list[Path],
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    transform: str,
    parameters: str,
    fusion_model,
    feature_order: list[str],
    threshold: float,
) -> pd.DataFrame:
    probability_frame = pd.DataFrame(probabilities)
    feature_mapping = {
        "Community Forensics Pr": "Community Forensics probability",
        "DINOv3-L detector Pr": "DINOv3-L probability",
        "DINOv3-B stacking detector Pr": "DINOv3-B probability",
    }
    fusion_input = probability_frame[
        [feature_mapping[feature] for feature in feature_order]
    ].to_numpy(dtype=float)
    final_probability = fusion_model.predict_proba(fusion_input)[:, 1]
    prediction = (final_probability >= threshold).astype(int)

    return pd.DataFrame(
        {
            "image path": [str(path.relative_to(PROJECT_ROOT)) for path in paths],
            "actual label": [LABEL_NAMES[int(label)] for label in labels],
            **probabilities,
            "final probability": final_probability,
            "predicted label": [LABEL_NAMES[int(value)] for value in prediction],
            "correct/incorrect": np.where(prediction == labels, "Correct", "Incorrect"),
            "transform": transform,
            "transform parameters": parameters,
        }
    )


def main() -> None:
    paths = sorted(
        path.resolve()
        for path in IMAGE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and any(
            folder.lower() in LABELS
            for folder in path.relative_to(IMAGE_DIR).parts[:-1]
        )
    )
    labels = np.asarray([infer_label(path) for path in paths], dtype=int)
    signature = dataset_signature(paths)

    artifact = joblib.load(FUSION_ARTIFACT)
    fusion_model = artifact["model"]
    feature_order = list(artifact["feature_order"])
    threshold = float(artifact["threshold"])

    frames = [
        make_prediction_frame(
            paths=paths,
            labels=labels,
            probabilities=load_probabilities(CACHE_ROOT / signature, len(paths)),
            transform="Clean",
        parameters="not applicable",
            fusion_model=fusion_model,
            feature_order=feature_order,
            threshold=threshold,
        )
    ]

    robustness_root = CACHE_ROOT / "robustness_all" / signature
    for transform, parameters in TRANSFORM_SPECS:
        frames.append(
            make_prediction_frame(
                paths=paths,
                labels=labels,
                probabilities=load_probabilities(
                    robustness_root / condition_key(transform, parameters), len(paths)
                ),
                transform=transform,
                parameters=parameters,
                fusion_model=fusion_model,
                feature_order=feature_order,
                threshold=threshold,
            )
        )

    output = pd.concat(frames, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, float_format="%.10f")
    print(f"Saved {len(output):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
