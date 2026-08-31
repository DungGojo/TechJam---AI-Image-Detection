"""Classify user-supplied images as Real or AI-generated with the full pipeline.

Three detectors score every image, and the selected fusion model turns those
three probabilities into one Real/Fake decision. This is the command-line form
of `notebooks/06_Full_Pipeline_Image_Classification_(Test Sample).ipynb`.

The two DINOv3 branches sample random crops seeded by an image's position
inside its batch, so a probability is reproducible for a given input set but
shifts slightly when the same image is scored alongside a different set. The
score cache is therefore keyed on the whole input set, never on one file.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]

# Only a source checkout needs its root on the path; an installed package is
# already importable, and prepending site-packages would shadow nothing useful.
if (_MODULE_ROOT / "pyproject.toml").exists() and str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))


def find_project_root() -> Path:
    """Locate the checkout holding weights/ and outputs/.

    The installed `classify` command runs from anywhere, so the module's own
    location is only a hint. An explicit AIGC_HOME wins, then the checkout this
    module lives in, then the nearest parent of the working directory that
    holds the fusion artifact.
    """
    override = os.environ.get("AIGC_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if (_MODULE_ROOT / "pyproject.toml").exists():
        return _MODULE_ROOT
    working = Path.cwd().resolve()
    for candidate in (working, *working.parents):
        if (candidate / "weights" / "fusion").is_dir():
            return candidate
    return working


PROJECT_ROOT = find_project_root()

# Set before any Hugging Face import so downloads land in the project cache.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "cache/hf"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from Data_Pipeline.image_io import iter_image_paths, load_image_safe  # noqa: E402
from Modeling.common.device import get_device  # noqa: E402
from Modeling.dinov3.scorer import DINOv3Scorer  # noqa: E402
from Modeling.registry import get  # noqa: E402
from Modeling.stacking.scorer import StackingScorer  # noqa: E402

DEFAULT_DINOV3_CHECKPOINT = PROJECT_ROOT / "weights/dinov3/notebook_dinov3/best_robust.pt"
DEFAULT_STACKING_CHECKPOINT = PROJECT_ROOT / "weights/stacking/notebook_vitb/best_robust.pt"
DEFAULT_FUSION_ARTIFACT = PROJECT_ROOT / "weights/fusion/selected_final_fusion.joblib"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs/cli_cache"

DEFAULT_SEED = 1337
DINOV3_CROPS = 8
STACKING_CROPS = 3

LABELS = {"real": 0, "fake": 1}
LABEL_NAMES = {0: "REAL", 1: "FAKE"}


@dataclass(frozen=True)
class Branch:
    """One detector column expected by the fusion model."""

    column: str
    cache_name: str
    label: str
    header: str
    batch_size: int


BRANCHES: dict[str, Branch] = {
    branch.column: branch
    for branch in (
        Branch("Community Forensics Pr", "community_forensics", "Community Forensics", "CF", 8),
        Branch("DINOv3-L detector Pr", "dinov3_l", "DINOv3-L", "DINO-L", 2),
        Branch("DINOv3-B stacking detector Pr", "dinov3_b_stacking", "DINOv3-B stacking", "DINO-B", 4),
    )
}


def build_scorer(column: str, args: argparse.Namespace, device: torch.device):
    """Construct the detector that produces `column`, one at a time to save memory."""
    if column == "Community Forensics Pr":
        return get("community_forensics", device=str(device))
    if column == "DINOv3-L detector Pr":
        return DINOv3Scorer(
            checkpoint=str(args.dinov3_checkpoint),
            device=str(device),
            n_crops=DINOV3_CROPS,
            seed=args.seed,
        )
    if column == "DINOv3-B stacking detector Pr":
        return StackingScorer(
            checkpoint=str(args.stacking_checkpoint),
            device=str(device),
            n_crops=STACKING_CROPS,
            seed=args.seed,
        )
    raise KeyError(f"no detector is wired for fusion feature {column!r}")


# --------------------------------------------------------------------------- input


def unescape_path(text: str) -> str:
    """Accept shell-style paths typed or dragged into the prompt."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return re.sub(r"\\(.)", r"\1", text)


def prompt_for_inputs() -> list[str]:
    """Ask for image or folder paths until a blank line."""
    print("Enter image files or folders to classify (drag and drop works).")
    print("Press Enter on an empty line when you are done.\n")
    entries: list[str] = []
    while True:
        try:
            line = input("path> ")
        except EOFError:
            print()
            break
        if not line.strip():
            break
        entries.append(unescape_path(line))
    return entries


def collect_images(inputs: list[str]) -> tuple[list[Path], list[str]]:
    """Expand files and directories into a deduplicated, sorted image list."""
    images: list[Path] = []
    seen: set[Path] = set()
    problems: list[str] = []

    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            problems.append(f"no such file or directory: {raw}")
            continue
        found = iter_image_paths(path)
        if not found:
            kind = "directory contains no supported images" if path.is_dir() else "unsupported file type"
            problems.append(f"{kind}: {raw}")
            continue
        for image in found:
            resolved = image.resolve()
            if resolved not in seen:
                seen.add(resolved)
                images.append(resolved)

    return sorted(images), problems


def label_from_path(path: Path) -> int | None:
    """Read ground truth from the nearest `real` or `fake` parent folder."""
    for part in reversed(path.parent.parts):
        if part.lower() in LABELS:
            return LABELS[part.lower()]
    return None


# --------------------------------------------------------------------------- scoring


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def input_signature(image_paths: list[Path]) -> str:
    """Identify this exact set of files by path, size, and modification time."""
    records = [f"{p}|{p.stat().st_size}|{p.stat().st_mtime_ns}" for p in image_paths]
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()[:16]


def score_images(
    scorer,
    image_paths: list[Path],
    description: str,
    batch_size: int,
    cache_dir: Path | None,
    cache_name: str,
    quiet: bool,
    reported: set[int],
) -> np.ndarray:
    """Score every image, resuming from a partial run when caching is enabled."""
    probabilities = np.full(len(image_paths), np.nan, dtype=np.float64)
    done = np.zeros(len(image_paths), dtype=bool)
    cache_path = done_path = None

    if cache_dir is not None:
        cache_path = cache_dir / f"{cache_name}.npy"
        done_path = cache_dir / f"{cache_name}.done.npy"
        if cache_path.exists() and done_path.exists():
            cached = np.load(cache_path)
            cached_done = np.load(done_path)
            if len(cached) == len(image_paths) and len(cached_done) == len(image_paths):
                probabilities, done = cached.astype(np.float64), cached_done.astype(bool)
                if not quiet and done.any():
                    print(f"{description}: reusing {done.sum()}/{len(done)} cached scores", file=sys.stderr)

    def persist() -> None:
        if cache_path is not None:
            np.save(cache_path, probabilities)
            np.save(done_path, done)

    starts = [start for start in range(0, len(image_paths), batch_size) if not done[start : start + batch_size].all()]
    for start in tqdm(starts, desc=description, unit="batch", disable=quiet, file=sys.stderr):
        stop = min(start + batch_size, len(image_paths))
        images, positions = [], []
        for position in range(start, stop):
            if done[position]:
                continue
            image, error = load_image_safe(image_paths[position])
            if image is None:
                if position not in reported:
                    reported.add(position)
                    print(f"WARNING: unreadable image, skipped: {error}", file=sys.stderr)
                done[position] = True
                continue
            images.append(image)
            positions.append(position)
        if images:
            probabilities[positions] = scorer.score_checked(images)
            done[positions] = True
            for image in images:
                image.close()
        persist()

    persist()
    return probabilities


def classify_images(
    image_paths: list[Path],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, object]]:
    """Run the three detectors, then the fusion model. Returns per-branch and final scores."""
    artifact = joblib.load(args.fusion)
    feature_columns = list(artifact["feature_order"])
    threshold = float(args.threshold if args.threshold is not None else artifact["threshold"])

    unknown = [column for column in feature_columns if column not in BRANCHES]
    if unknown:
        raise KeyError(f"fusion artifact needs features with no detector wired: {unknown}")

    device = get_device(args.device)
    cache_dir = None
    if not args.no_cache:
        cache_dir = Path(args.cache_dir) / input_signature(image_paths)
        cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"device            {device}", file=sys.stderr)
        print(f"fusion model      {artifact['model_name']} (threshold {threshold:.3f})", file=sys.stderr)
        print(f"images            {len(image_paths)}", file=sys.stderr)
        if cache_dir is not None:
            print(f"score cache       {cache_dir}", file=sys.stderr)
        print(file=sys.stderr)

    scores: dict[str, np.ndarray] = {}
    unreadable: set[int] = set()
    for index, column in enumerate(feature_columns, start=1):
        branch = BRANCHES[column]
        scorer = build_scorer(column, args, device)
        scores[column] = score_images(
            scorer,
            image_paths,
            f"{index}/{len(feature_columns)} {branch.label}",
            branch.batch_size,
            cache_dir,
            branch.cache_name,
            args.quiet,
            unreadable,
        )
        del scorer
        release_memory()

    fusion_input = np.column_stack([scores[column] for column in feature_columns])
    usable = np.isfinite(fusion_input).all(axis=1)
    final_probability = np.full(len(image_paths), np.nan, dtype=np.float64)
    if usable.any():
        final_probability[usable] = artifact["model"].predict_proba(fusion_input[usable])[:, 1]

    prediction = np.where(np.isnan(final_probability), -1, (final_probability >= threshold).astype(int))

    meta = {
        "model_name": artifact["model_name"],
        "threshold": threshold,
        "feature_order": feature_columns,
        "device": str(device),
    }
    return scores, final_probability, prediction, meta


# --------------------------------------------------------------------------- reporting


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else "..." + text[-(width - 3) :]


def _paint(text: str, prediction: int, colour: bool) -> str:
    if not colour or prediction not in (0, 1):
        return text
    return f"\033[3{2 if prediction == 0 else 1}m{text}\033[0m"


def print_table(
    image_paths: list[Path],
    scores: dict[str, np.ndarray],
    final_probability: np.ndarray,
    prediction: np.ndarray,
    labels: list[int | None],
    meta: dict[str, object],
    colour: bool,
) -> None:
    names = [display_path(path) for path in image_paths]
    path_width = max(5, min(60, max(len(name) for name in names)))
    show_truth = any(label is not None for label in labels)

    headers = ["PREDICT", "P(FAKE)"]
    headers += [BRANCHES[column].header for column in meta["feature_order"]]
    if show_truth:
        headers.append("ACTUAL")
    headers.append("IMAGE")

    widths = [7, 7] + [max(6, len(BRANCHES[column].header)) for column in meta["feature_order"]]
    if show_truth:
        widths.append(6)
    widths.append(path_width)

    line = "  ".join(header.ljust(width) for header, width in zip(headers, widths))
    print(line)
    print("-" * len(line))

    for index, name in enumerate(names):
        verdict = LABEL_NAMES.get(int(prediction[index]), "SKIPPED")
        probability = final_probability[index]
        cells = [
            _paint(verdict.ljust(widths[0]), int(prediction[index]), colour),
            ("n/a" if np.isnan(probability) else f"{probability:.3f}").ljust(widths[1]),
        ]
        for offset, column in enumerate(meta["feature_order"]):
            value = scores[column][index]
            cells.append(("n/a" if np.isnan(value) else f"{value:.3f}").ljust(widths[2 + offset]))
        if show_truth:
            truth = LABEL_NAMES.get(labels[index], "-")
            cells.append(truth.ljust(widths[-2]))
        cells.append(_shorten(name, path_width))
        print("  ".join(cells))

    fake = int((prediction == 1).sum())
    real = int((prediction == 0).sum())
    skipped = int((prediction == -1).sum())
    noun = "image" if len(names) == 1 else "images"
    summary = f"\n{len(names)} {noun}: {fake} FAKE, {real} REAL"
    print(summary + (f", {skipped} skipped" if skipped else ""))


def print_metrics(labels: list[int | None], final_probability: np.ndarray, prediction: np.ndarray) -> None:
    """Score the run against folder-derived ground truth."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    keep = [
        index
        for index, label in enumerate(labels)
        if label is not None and prediction[index] in (0, 1)
    ]
    if not keep:
        print("\nNo images had a real/fake parent folder, so no metrics were computed.")
        return

    y_true = np.array([labels[index] for index in keep], dtype=int)
    y_prediction = prediction[keep].astype(int)
    y_probability = final_probability[keep]

    if len(keep) != len(labels):
        print(f"\nMetrics use the {len(keep)} of {len(labels)} images with a known label.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_prediction, labels=[0, 1]).ravel()
    rows = [
        ("accuracy", accuracy_score(y_true, y_prediction)),
        ("precision", precision_score(y_true, y_prediction, zero_division=0)),
        ("recall_sensitivity", recall_score(y_true, y_prediction, zero_division=0)),
        ("specificity", tn / (tn + fp) if (tn + fp) else float("nan")),
        ("f1", f1_score(y_true, y_prediction, zero_division=0)),
        ("mcc", matthews_corrcoef(y_true, y_prediction)),
    ]
    if len(np.unique(y_true)) == 2:
        rows.append(("roc_auc", roc_auc_score(y_true, y_probability)))

    print("\nPerformance")
    for name, value in rows:
        print(f"  {name:<20}{value:.4f}")

    print("\nConfusion matrix")
    print(f"  {'':<14}{'pred REAL':>10}{'pred FAKE':>10}")
    print(f"  {'actual REAL':<14}{tn:>10}{fp:>10}")
    print(f"  {'actual FAKE':<14}{fn:>10}{tp:>10}")


def build_records(
    image_paths: list[Path],
    scores: dict[str, np.ndarray],
    final_probability: np.ndarray,
    prediction: np.ndarray,
    labels: list[int | None],
    meta: dict[str, object],
) -> list[dict[str, object]]:
    records = []
    for index, path in enumerate(image_paths):
        record: dict[str, object] = {"image_path": str(path)}
        for column in meta["feature_order"]:
            value = scores[column][index]
            record[column] = None if np.isnan(value) else float(value)
        probability = final_probability[index]
        record["final_probability"] = None if np.isnan(probability) else float(probability)
        record["prediction"] = int(prediction[index]) if prediction[index] in (0, 1) else None
        record["prediction_label"] = LABEL_NAMES.get(int(prediction[index]), None)
        if any(label is not None for label in labels):
            record["actual"] = labels[index]
            record["actual_label"] = LABEL_NAMES.get(labels[index], None)
        records.append(record)
    return records


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, records: list[dict[str, object]], meta: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fusion": meta, "predictions": records}
    path.write_text(json.dumps(payload, indent=2) + "\n")


# --------------------------------------------------------------------------- cli


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # prog is left to argparse so the help matches how it was invoked:
    # "classify" once installed, "classify.py" when run from the checkout.
    parser = argparse.ArgumentParser(
        description="Classify images as Real or AI-generated with the full three-detector pipeline.",
        epilog=(
            "examples:\n"
            "  classify photo.jpg\n"
            "  classify testing_images --evaluate\n"
            "  classify ~/Downloads/*.png --csv outputs/verdicts.csv\n"
            "  classify                      # prompts for paths\n"
            "\n"
            "Without an install, run the same thing as: python classify.py ...\n"
            "Model files are found next to the project. From another directory,\n"
            "set AIGC_HOME=/path/to/AI-Image-Detection.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="*", help="image files or folders (folders are searched recursively)")
    parser.add_argument("--csv", type=Path, default=None, help="write per-image results to this CSV")
    parser.add_argument("--json", type=Path, default=None, help="write per-image results to this JSON file")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="read ground truth from the nearest real/ or fake/ parent folder and report metrics",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="decision threshold on P(fake); default comes from the fusion artifact",
    )
    parser.add_argument("--device", default=None, help="force cpu/mps/cuda (default: auto)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="crop-sampling seed")
    parser.add_argument("--dinov3-checkpoint", type=Path, default=DEFAULT_DINOV3_CHECKPOINT)
    parser.add_argument("--stacking-checkpoint", type=Path, default=DEFAULT_STACKING_CHECKPOINT)
    parser.add_argument("--fusion", type=Path, default=DEFAULT_FUSION_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="resumable score cache root")
    parser.add_argument("--no-cache", action="store_true", help="score every image again, cache nothing")
    parser.add_argument("--no-color", action="store_true", help="disable coloured verdicts")
    parser.add_argument("--quiet", action="store_true", help="suppress progress bars and setup lines")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    inputs = list(args.inputs)
    if not inputs and not sys.stdin.isatty():
        inputs = [unescape_path(line) for line in sys.stdin.read().splitlines() if line.strip()]
    if not inputs:
        inputs = prompt_for_inputs()
    if not inputs:
        print("ERROR: no images given.", file=sys.stderr)
        return 2

    image_paths, problems = collect_images(inputs)
    for problem in problems:
        print(f"WARNING: {problem}", file=sys.stderr)
    if not image_paths:
        print("ERROR: none of the given paths contained a supported image.", file=sys.stderr)
        return 2

    for checkpoint in (args.dinov3_checkpoint, args.stacking_checkpoint, args.fusion):
        if not Path(checkpoint).exists():
            print(f"ERROR: missing model file: {checkpoint}", file=sys.stderr)
            print(
                "Run from the project directory, set AIGC_HOME to it, or pass the "
                "path explicitly. See weights/README.md for the expected layout.",
                file=sys.stderr,
            )
            return 2

    scores, final_probability, prediction, meta = classify_images(image_paths, args)

    labels = [label_from_path(path) for path in image_paths] if args.evaluate else [None] * len(image_paths)
    colour = not args.no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    print_table(image_paths, scores, final_probability, prediction, labels, meta, colour)
    if args.evaluate:
        print_metrics(labels, final_probability, prediction)

    records = build_records(image_paths, scores, final_probability, prediction, labels, meta)
    if args.csv:
        write_csv(args.csv, records)
        print(f"\nwrote {args.csv}", file=sys.stderr)
    if args.json:
        write_json(args.json, records, meta)
        print(f"wrote {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
