"""Fig. 5: layerwise M1 own-class argmax rate (absolute).

One IEEE column (3.5 in), saved at its exact size (no tight bounding box):
  (a) Dataset curves, averaged over five encoders and four perturbations.
  (b) FAIR1M curves by perturbation, averaged over five encoders.
Within each split seed the cells are averaged first; lines are 3-seed means and
bands are +/-1 sample SD across the three seeds. Dotted lines mark chance
(1/K: DOTA v1.5 K = 16, DIOR-R K = 20, FAIR1M K = 37). fit15, M1, unit,
true OBB, confirmation role.

Run from this directory:
    python fig5_layerwise.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)
STEM = OUT / "Figure 5"

LAYERS = [f"L{i:02d}" for i in range(1, 13)] + ["final"]
X = np.arange(len(LAYERS))
TICK_LABELS = {layer: ("Final" if layer == "final" else layer) for layer in LAYERS}

# Okabe-Ito colors; marker shapes keep series separable in grayscale print.
DATASET_SERIES = [
    ("dota", "DOTA v1.5", "#000000", "o"),
    ("dior", "DIOR-R", "#D55E00", "s"),
    ("fair1m", "FAIR1M", "#56B4E9", "^"),
]
PERTURBATION_SERIES = [
    ("blur", "Blur", "#0072B2", "o"),
    ("brightness", "Brightness", "#E69F00", "s"),
    ("gaussian_noise", "Noise", "#009E73", "^"),
    ("patch_mask", "Block", "#CC79A7", "D"),
]
CLASS_COUNTS = {"dota": 16, "dior": 20, "fair1m": 37}

# Layout in inches.
FIG_W = 3.5
LEFT = 0.50
RIGHT_PAD = 0.08
BOTTOM = 0.36
PANEL_A_H = 1.30
PANEL_B_H = 1.05
GAP = 0.24
TOP_PAD = 0.17
FIG_H = BOTTOM + PANEL_B_H + GAP + PANEL_A_H + TOP_PAD

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def axes_rect(bottom_in: float, left_in: float, width_in: float, height_in: float) -> list[float]:
    return [left_in / FIG_W, bottom_in / FIG_H, width_in / FIG_W, height_in / FIG_H]


def read_curves(path: Path, key: str) -> dict[str, dict[str, tuple[float, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run build_fig5_data.py first")
    table: dict[str, dict[str, tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            table.setdefault(row[key], {})[row["layer"]] = (float(row["mean"]), float(row["sd"]))
    return table


def chance_line(ax, value: float, color: str) -> None:
    ax.axhline(value, color=color, linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=1)


def draw(ax, curves, series, title: str, ylim, yticks, show_xlabels: bool, legend_cols: int):
    for key, label, color, marker in series:
        if key not in curves or any(layer not in curves[key] for layer in LAYERS):
            raise ValueError(f"incomplete curve: {key}")
        mean = np.array([curves[key][layer][0] for layer in LAYERS])
        sd = np.array([curves[key][layer][1] for layer in LAYERS])
        ax.fill_between(X, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(
            X,
            mean,
            color=color,
            linewidth=1.0,
            marker=marker,
            markersize=3.0,
            markeredgewidth=0,
            label=label,
            zorder=3,
        )

    ax.set_xlim(-0.4, len(LAYERS) - 0.6)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in yticks])
    ax.set_xticks(X)
    if show_xlabels:
        ax.set_xticklabels(
            [TICK_LABELS[layer] if index % 2 == 0 else "" for index, layer in enumerate(LAYERS)]
        )
        ax.set_xlabel("Layer", labelpad=2.0)
    else:
        ax.set_xticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2.0, width=0.5, pad=2.0)
    ax.set_title(title, loc="left", pad=3.0)
    return ax.legend(
        loc="upper left",
        ncol=legend_cols,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.2,
        bbox_to_anchor=(0.0, 1.0),
    )


def main() -> None:
    dataset_curves = read_curves(DATA / "fig5_dataset.csv", "dataset")
    fair_curves = read_curves(DATA / "fig5_fair1m.csv", "perturbation")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    width = FIG_W - LEFT - RIGHT_PAD
    ax_b = fig.add_axes(axes_rect(BOTTOM, LEFT, width, PANEL_B_H))
    ax_a = fig.add_axes(axes_rect(BOTTOM + PANEL_B_H + GAP, LEFT, width, PANEL_A_H))

    for key, _, color, _ in DATASET_SERIES:
        chance_line(ax_a, 1.0 / CLASS_COUNTS[key], color)
    draw(
        ax_a,
        dataset_curves,
        DATASET_SERIES,
        "(a) By dataset",
        (0.0, 1.0),
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        False,
        3,
    )
    ax_a.text(
        len(LAYERS) - 0.65,
        1.0 / CLASS_COUNTS["dota"] + 0.012,
        "Chance (1/K)",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#404040",
    )

    chance_line(ax_b, 1.0 / CLASS_COUNTS["fair1m"], "#56B4E9")
    draw(
        ax_b,
        fair_curves,
        PERTURBATION_SERIES,
        "(b) FAIR1M by perturbation",
        (0.0, 0.6),
        [0.0, 0.2, 0.4, 0.6],
        True,
        4,
    )
    ax_b.text(
        len(LAYERS) - 0.65,
        1.0 / CLASS_COUNTS["fair1m"] + 0.008,
        "Chance (1/37)",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#404040",
    )

    fig.text(
        0.08 / FIG_W,
        (BOTTOM + (PANEL_B_H + GAP + PANEL_A_H) / 2) / FIG_H,
        "Own-class argmax rate",
        rotation=90,
        ha="left",
        va="center",
        fontsize=8.0,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = STEM.with_suffix(suffix)
        fig.savefig(path, dpi=600)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
