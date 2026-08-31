"""Clean and transformed evaluation grid."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib

# Select a headless backend only when the caller has not chosen one.
try:  # pragma: no cover - depends on how the process was started
    from IPython import get_ipython

    _INTERACTIVE = get_ipython() is not None
except Exception:
    _INTERACTIVE = False

if not _INTERACTIVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from data_pipeline.image_io import load_image_safe
from evaluation.metrics import compute
from modeling.common.degradations import (
    BRIEF_FAMILIES,
    FAMILIES,
    discrete_grid,
)
from modeling.common.interfaces import Scorer

CLEAN = "clean"


@dataclass(frozen=True)
class CleanCell:
    """The untransformed reference. Duck-types `Cell` so the grid is uniform."""

    family: str = CLEAN
    value: float = 0.0

    @property
    def label(self) -> str:
        return CLEAN

    def apply(self, img: Image.Image, rng=None) -> Image.Image:
        return img


def full_grid(include_clean: bool = True) -> list:
    """Build evaluation grid with clean reference and discrete brief degradation cells."""
    cells: list = [CleanCell()] if include_clean else []
    cells += discrete_grid(BRIEF_FAMILIES)
    return cells


def score_cell(
    scorer: Scorer,
    paths: Sequence[str],
    labels: Sequence[int],
    cell,
    *,
    batch_size: int = 32,
    seed: int = 1337,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply one degradation cell and score every readable image."""
    rng = np.random.default_rng(seed)  # per-cell seed -> reproducible degradations

    y_true: list[int] = []
    y_score: list[float] = []
    n_failed = 0

    iterator = range(0, len(paths), batch_size)
    if progress:
        iterator = tqdm(iterator, desc=cell.label, unit="batch", leave=False)

    for start in iterator:
        chunk_paths = paths[start : start + batch_size]
        chunk_labels = labels[start : start + batch_size]

        images, kept_labels = [], []
        for path, label in zip(chunk_paths, chunk_labels):
            img, error = load_image_safe(path)
            if img is None:
                n_failed += 1
                continue
            images.append(cell.apply(img, rng))
            kept_labels.append(int(label))

        if not images:
            continue
        scores = scorer.score_checked(images)
        y_score.extend(scores.tolist())
        y_true.extend(kept_labels)

    return np.array(y_true, dtype=int), np.array(y_score, dtype=float), n_failed


def run_grid(
    scorer: Scorer,
    df: pd.DataFrame,
    *,
    cells: Sequence | None = None,
    batch_size: int = 32,
    seed: int = 1337,
    progress: bool = True,
) -> pd.DataFrame:
    """One row per grid cell, all metrics."""
    if cells is None:
        cells = full_grid()

    paths = list(df["path"])
    labels = list(df["label"])

    rows = []
    for cell in tqdm(
        cells, desc=f"grid[{scorer.name}]", unit="cell", disable=not progress
    ):
        y_true, y_score, n_failed = score_cell(
            scorer,
            paths,
            labels,
            cell,
            batch_size=batch_size,
            seed=seed,
            progress=progress,
        )
        metrics = compute(y_true, y_score)
        rows.append(
            {
                "model": scorer.name,
                "cell": cell.label,
                "transform": cell.family,
                "severity": float(cell.value),
                "n_failed": n_failed,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _severity_rank(results: pd.DataFrame) -> pd.DataFrame:
    """Map each family’s severities to an ordinal rank."""
    ranks = []
    for _, row in results.iterrows():
        family = row["transform"]
        if family == CLEAN:
            ranks.append(0)
            continue
        discrete = list(FAMILIES[family].discrete) if family in FAMILIES else []
        if row["severity"] in discrete:
            ranks.append(discrete.index(row["severity"]) + 1)
        else:
            ranks.append(1)
    out = results.copy()
    out["severity_rank"] = ranks
    return out


def plot_grid(
    results: pd.DataFrame, out_path: str | Path, *, metric: str = "roc_auc"
) -> Path:
    """AUC vs severity, one line per family, clean as a horizontal reference."""
    ranked = _severity_rank(results)
    clean_rows = ranked[ranked["transform"] == CLEAN]
    clean_value = float(clean_rows[metric].iloc[0]) if len(clean_rows) else np.nan

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    if np.isfinite(clean_value):
        ax.axhline(
            clean_value,
            color="#111827",
            linestyle="--",
            linewidth=1.6,
            zorder=1,
            label=f"clean ({clean_value:.3f})",
        )
    ax.axhline(
        0.5,
        color="#9ca3af",
        linestyle=":",
        linewidth=1.2,
        zorder=0,
        label="chance (0.500)",
    )

    families = [f for f in ranked["transform"].unique() if f != CLEAN]
    colours = plt.get_cmap("tab10")
    for i, family in enumerate(families):
        sub = ranked[ranked["transform"] == family].sort_values("severity_rank")
        ax.plot(
            sub["severity_rank"],
            sub[metric],
            marker="o",
            linestyle="-",
            color=colours(i % 10),
            label=f"{family}",
            zorder=2,
        )
        for _, r in sub.iterrows():
            value = r["severity"]
            text = f"{int(value)}" if float(value).is_integer() else f"{value:g}"
            ax.annotate(
                text,
                (r["severity_rank"], r[metric]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color=colours(i % 10),
            )

    model = results["model"].iloc[0] if len(results) else "?"
    ax.set_xlabel("severity (ordinal within family; labels are the actual parameter)")
    ax.set_ylabel(metric)
    ax.set_title(f"{model}: {metric} under degradation")
    ax.set_xticks(range(0, int(ranked["severity_rank"].max()) + 1))
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, ncols=2, loc="best")
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def to_markdown(results: pd.DataFrame) -> str:
    """A table ready to paste into the submission."""
    cols = [
        ("cell", "cell"),
        ("roc_auc", "AUC"),
        ("accuracy", "acc"),
        ("average_precision", "AP"),
        ("tpr_at_fpr0.01", "TPR@1%FPR"),
        ("tpr_at_fpr0.05", "TPR@5%FPR"),
    ]
    model = results["model"].iloc[0] if len(results) else "?"
    clean_rows = results[results["transform"] == CLEAN]
    clean_auc = float(clean_rows["roc_auc"].iloc[0]) if len(clean_rows) else np.nan

    lines = [
        f"### `{model}` — robustness grid",
        "",
        "| " + " | ".join(h for _, h in cols) + " | Δ AUC vs clean |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for _, r in results.iterrows():
        cells = []
        for key, _ in cols:
            v = r[key]
            cells.append(
                v if key == "cell" else ("n/a" if not np.isfinite(v) else f"{v:.3f}")
            )
        delta = r["roc_auc"] - clean_auc
        delta_str = (
            "—"
            if r["transform"] == CLEAN or not np.isfinite(delta)
            else f"{delta:+.3f}"
        )
        label = f"`{cells[0]}`"
        lines.append("| " + " | ".join([label, *cells[1:], delta_str]) + " |")

    graded = results[results["transform"] != CLEAN]
    if len(graded) and np.isfinite(clean_auc):
        worst = graded.loc[graded["roc_auc"].idxmin()]
        lines += [
            "",
            f"Clean AUC **{clean_auc:.3f}**; mean across the graded grid "
            f"**{graded['roc_auc'].mean():.3f}**; worst cell **`{worst['cell']}`** "
            f"at **{worst['roc_auc']:.3f}** ({worst['roc_auc'] - clean_auc:+.3f}).",
        ]
    return "\n".join(lines)


def save_report(
    results: pd.DataFrame,
    model_name: str,
    out_dir: str | Path = "outputs/robustness",
    *,
    timestamp: str | None = None,
) -> dict[str, Path]:
    """Write the CSV / PNG / markdown triple and return the paths."""
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_{stamp}"

    csv_path, png_path, md_path = (
        out_dir / f"{stem}.{e}" for e in ("csv", "png", "md")
    )
    results.to_csv(csv_path, index=False)
    plot_grid(results, png_path)
    md_path.write_text(to_markdown(results) + "\n")
    return {"csv": csv_path, "png": png_path, "markdown": md_path}
