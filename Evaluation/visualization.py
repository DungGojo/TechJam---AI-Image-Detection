"""Shared evaluation visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SERIES = ["#0E9384", "#C1541B", "#4A5CB5", "#A07C00"]
SURFACE = "#FBFCFC"
INK = "#14191B"
INK_2 = "#4A5459"
INK_3 = "#78848A"
GRID = "#E4E9E9"
REFERENCE = "#9AA5AA"

GOOD = "#0E9384"
WARN = "#9A6407"
BAD = "#A33A3E"


def use_house_style() -> None:
    """Call once at the top of a notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    })


def _save(fig, path: str | Path | None) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        print(f"saved {path}")



def robustness_facets(
    grids: dict[str, dict[str, dict[str, float]]],
    metric: str = "auc",
    families: list[str] | None = None,
    title: str = "Detection AUC under each transform",
    path: str | Path | None = None,
):
    """Plot one robustness panel per transform family."""
    from Modeling.common.degradations import BRIEF_FAMILIES, FAMILIES

    families = families or BRIEF_FAMILIES
    models = list(grids)
    n = len(families)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow), squeeze=False)

    for idx, fam in enumerate(families):
        ax = axes[idx // ncol][idx % ncol]
        values = list(FAMILIES[fam].discrete)
        xs = np.arange(len(values))

        for m_i, model in enumerate(models):
            grid = grids[model]
            ys = [grid.get(f"{fam}={v:g}", {}).get(metric, np.nan) for v in values]
            colour = SERIES[m_i % len(SERIES)]
            ax.plot(xs, ys, marker="o", color=colour, label=model, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.4)
            clean = grid.get("clean", {}).get(metric)
            if clean is not None and m_i == 0:
                ax.axhline(clean, color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.2,
                           zorder=1)

        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in values])
        # Preserve comparable axes for single-severity families.
        ax.set_xlim(-0.6, max(0.6, len(values) - 1 + 0.6))
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{fam}  ·  {FAMILIES[fam].kwarg}", fontsize=9, color=INK,
                     loc="left", pad=6)
        ax.set_ylim(0.45, 1.02)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        if idx % ncol == 0:
            ax.set_ylabel(metric.upper())

    for idx in range(n, nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(mpl.lines.Line2D([], [], color=REFERENCE, linestyle=(0, (4, 3)),
                                    linewidth=1.2))
    labels.append(f"clean ({models[0]})")
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    _save(fig, path)
    return fig



def summary_bars(summaries: dict[str, dict[str, float]], path=None):
    """Clean vs robust AUC per model. The gap between the pair IS the story."""
    models = list(summaries)
    xs = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2.4, 3.1))

    clean = [summaries[m]["clean_auc"] for m in models]
    robust = [summaries[m]["robust_auc"] for m in models]
    ax.bar(xs - w / 2 - 0.01, clean, w, color=REFERENCE, label="clean", zorder=3)
    ax.bar(xs + w / 2 + 0.01, robust, w, color=SERIES[0], label="robust (mean of cells)",
           zorder=3)

    for x, c, r in zip(xs, clean, robust):
        ax.text(x - w / 2 - 0.01, c + 0.008, f"{c:.3f}", ha="center", va="bottom",
                fontsize=7.5, color=INK_2)
        ax.text(x + w / 2 + 0.01, r + 0.008, f"{r:.3f}", ha="center", va="bottom",
                fontsize=7.5, color=INK)

    ax.set_xticks(xs)
    ax.set_xticklabels(models)
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title("Clean vs robust AUC — the gap is the result", fontsize=10, loc="left",
                 color=INK, pad=8)
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, path)
    return fig



def training_curves(history: list[dict], path=None):
    """Plot clean and robust AUC with the selected checkpoint."""
    epochs = [h["summary"]["epoch"] for h in history]
    clean = [h["summary"]["clean_auc"] for h in history]
    robust = [h["summary"]["robust_auc"] for h in history]
    worst = [h["summary"]["worst_cell_auc"] for h in history]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(epochs, clean, color=REFERENCE, marker="o", label="clean AUC",
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
    ax.plot(epochs, robust, color=SERIES[0], marker="o", label="robust AUC",
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
    ax.plot(epochs, worst, color=SERIES[1], marker="o", label="worst cell",
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)

    best_i = int(np.nanargmax(robust))
    ax.scatter([epochs[best_i]], [robust[best_i]], s=150, facecolor="none",
               edgecolor=SERIES[0], linewidth=1.6, zorder=5)
    ax.annotate("selected", (epochs[best_i], robust[best_i]),
                textcoords="offset points", xytext=(8, 10), fontsize=8, color=INK)

    ax.set_xlabel("epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Validation AUC per epoch", fontsize=10, loc="left", color=INK, pad=8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    _save(fig, path)
    return fig



def reliability_diagram(y_true, p, bins: int = 10, path=None):
    """Are the confidences honest? A raw sigmoid usually is not."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    centres, freqs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        centres.append(p[m].mean())
        freqs.append(y_true[m].mean())
        counts.append(int(m.sum()))

    ece = float(np.sum(np.array(counts) / len(p) * np.abs(np.array(freqs) - np.array(centres))))

    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    ax.plot([0, 1], [0, 1], color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.2,
            label="perfectly calibrated", zorder=2)
    ax.plot(centres, freqs, color=SERIES[0], marker="o", label="observed",
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
    ax.set_xlabel("predicted P(AI-generated)")
    ax.set_ylabel("observed fraction")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Reliability  ·  ECE = {ece:.3f}", fontsize=10, loc="left", color=INK,
                 pad=8)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, path)
    return fig, ece



def degradation_strip(img, cells, scores=None, path=None, ncol: int = 5):
    """Display one image across degradation cells and optional scores."""
    import numpy as np

    panels = [("original", img)] + [(c.label, c.apply(img, rng=np.random.default_rng(0)))
                                    for c in cells]
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.25 * nrow), squeeze=False)
    for i, (label, im) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        ax.imshow(im)
        caption = label if scores is None or i >= len(scores) else \
            f"{label}\np={scores[i]:.3f}"
        colour = INK if scores is None else (BAD if scores[i] < 0.5 else INK)
        ax.set_title(caption, fontsize=8, color=colour, pad=4)
        ax.axis("off")
    for i in range(len(panels), nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")
    fig.tight_layout()
    _save(fig, path)
    return fig


def error_gallery(paths, scores, labels, kind: str = "fp", k: int = 8, path=None):
    """Display the most confident false positives and false negatives."""
    from PIL import Image

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if kind == "fp":
        idx = np.where(labels == 0)[0]
        idx = idx[np.argsort(-scores[idx])][:k]
        title = "Most confident false positives — authentic images called AI"
    else:
        idx = np.where(labels == 1)[0]
        idx = idx[np.argsort(scores[idx])][:k]
        title = "Most confident false negatives — AI images called authentic"

    ncol = 4
    nrow = int(np.ceil(len(idx) / ncol)) or 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.4 * nrow), squeeze=False)
    for i, j in enumerate(idx):
        ax = axes[i // ncol][i % ncol]
        try:
            ax.imshow(Image.open(paths[j]).convert("RGB"))
        except Exception as exc:                                    # noqa: BLE001
            ax.text(0.5, 0.5, f"unreadable\n{type(exc).__name__}", ha="center",
                    va="center", fontsize=7, color=INK_3)
        ax.set_title(f"p={scores[j]:.3f}", fontsize=8, color=BAD, pad=4)
        ax.axis("off")
    for i in range(len(idx), nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    _save(fig, path)
    return fig



def grid_to_markdown(grids: dict[str, dict[str, dict[str, float]]], metric="auc") -> str:
    """Paste-ready robustness table for the Devpost write-up and the README."""
    models = list(grids)
    keys = ["clean"] + [k for k in grids[models[0]] if k != "clean"]
    head = "| transform | " + " | ".join(models) + " |"
    rule = "|---" * (len(models) + 1) + "|"
    rows = []
    for k in keys:
        cells = [f"{grids[m].get(k, {}).get(metric, float('nan')):.4f}" for m in models]
        rows.append(f"| `{k}` | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])
