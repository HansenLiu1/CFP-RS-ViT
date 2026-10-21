"""Fig. 6: mean response, class-specific residual, and cross-perturbation alignment.

One single-column layout (3.5 in), saved at its exact size (no tight bounding box):
  (a) Mean-response share ||A|| / mean||Delta|| (A is the mean response
      across classes, i.e. the common-mode axis).
  (b) Class-specific / mean-response ratio r / ||A||; the dashed line marks 1.
  (c) Same-class cross-perturbation cosine. Thick lines are six-pair means in
      raw and CMC (M1) space; thin gray lines are the individual raw pairs
      (solid: pairs without noise, dashed: pairs with noise).
Each curve is the mean of three per-seed values (5 encoders x 3 datasets
averaged within each seed); bands are +/-1 sample SD across seeds.

Run from this directory:
    python fig6_common_alignment.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)
STEM = OUT / "Figure 6"

LAYERS = [f"L{i:02d}" for i in range(1, 13)] + ["final"]
X = np.arange(len(LAYERS))
TICK_LABELS = {layer: ("Final" if layer == "final" else layer) for layer in LAYERS}

# Same perturbation encoding as Fig. 5(b).
PERTURBATION_SERIES = [
    ("blur", "Blur", "#0072B2", "o"),
    ("brightness", "Brightness", "#E69F00", "s"),
    ("gaussian_noise", "Noise", "#009E73", "^"),
    ("patch_mask", "Block", "#CC79A7", "D"),
]
SPACE_SERIES = [
    ("raw", "Raw, six-pair mean", "#000000", "o"),
    ("cmc", "CMC (M1), six-pair mean", "#D55E00", "s"),
]
PAIR_COLOR = "#9A9A9A"

# Layout in inches.
FIG_W = 3.5
LEFT = 0.50
RIGHT_PAD = 0.08
BOTTOM = 0.36
PANEL_H = {"a": 1.00, "b": 1.00, "c": 1.30}
GAP = 0.25
TOP_PAD = 0.17
FIG_H = BOTTOM + sum(PANEL_H.values()) + 2 * GAP + TOP_PAD

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


def axes_rect(bottom_in: float, height_in: float) -> list[float]:
    return [LEFT / FIG_W, bottom_in / FIG_H, (FIG_W - LEFT - RIGHT_PAD) / FIG_W, height_in / FIG_H]


def read_table(name: str) -> list[dict[str, str]]:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run build_fig6_data.py first")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def curve(rows, key_field: str, key: str, value: str, sd: str) -> tuple[np.ndarray, np.ndarray]:
    lookup = {row["layer"]: row for row in rows if row[key_field] == key}
    missing = [layer for layer in LAYERS if layer not in lookup]
    if missing:
        raise ValueError(f"{key}: missing layers {missing}")
    return (
        np.array([float(lookup[layer][value]) for layer in LAYERS]),
        np.array([float(lookup[layer][sd]) for layer in LAYERS]),
    )


def plot_series(ax, mean, sd, color, marker, label, linewidth=1.0, markersize=3.0, zorder=3):
    ax.fill_between(X, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0, zorder=zorder - 1)
    ax.plot(
        X,
        mean,
        color=color,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        markeredgewidth=0,
        label=label,
        zorder=zorder,
    )


def style(ax, ylim, yticks, ylabel: str, title: str, show_xlabels: bool) -> None:
    ax.set_xlim(-0.4, len(LAYERS) - 0.6)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in yticks])
    ax.set_ylabel(ylabel, labelpad=3.0)
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


def main() -> None:
    common = read_table("fig6_common.csv")
    alignment = read_table("fig6_alignment.csv")
    pairs = read_table("fig6_alignment_pairs.csv")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    bottom_c = BOTTOM
    bottom_b = bottom_c + PANEL_H["c"] + GAP
    bottom_a = bottom_b + PANEL_H["b"] + GAP
    ax_a = fig.add_axes(axes_rect(bottom_a, PANEL_H["a"]))
    ax_b = fig.add_axes(axes_rect(bottom_b, PANEL_H["b"]))
    ax_c = fig.add_axes(axes_rect(bottom_c, PANEL_H["c"]))

    # (a) Mean-response share.
    for key, label, color, marker in PERTURBATION_SERIES:
        mean, sd = curve(common, "perturbation", key, "a_share", "a_share_sd")
        plot_series(ax_a, mean, sd, color, marker, label)
    style(
        ax_a,
        (0.0, 1.0),
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "Share",
        r"(a) Mean-response share, $\|A\|\,/\,\mathrm{mean}\,\|\Delta\|$",
        False,
    )
    ax_a.legend(
        loc="lower left",
        ncol=4,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.2,
    )

    # (b) Class-specific / mean-response ratio.
    ax_b.axhline(1.0, color="#808080", linewidth=0.5, linestyle=(0, (4, 2)), zorder=1)
    for key, label, color, marker in PERTURBATION_SERIES:
        mean, sd = curve(common, "perturbation", key, "ratio", "ratio_sd")
        plot_series(ax_b, mean, sd, color, marker, label)
    style(
        ax_b,
        (0.0, 1.8),
        [0.0, 0.5, 1.0, 1.5],
        "Ratio",
        r"(b) Class-specific / mean-response ratio, $r\,/\,\|A\|$",
        False,
    )
    ax_b.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.2,
    )

    # (c) Cross-perturbation alignment.
    ax_c.axhline(0.0, color="#808080", linewidth=0.5, zorder=1)
    raw_pairs = sorted({row["pair"] for row in pairs if row["space"] == "raw"})
    for pair in raw_pairs:
        mean, _ = curve(
            [row for row in pairs if row["space"] == "raw"], "pair", pair, "mean", "sd"
        )
        linestyle = (0, (3, 1.5)) if "gaussian_noise" in pair else "-"
        ax_c.plot(X, mean, color=PAIR_COLOR, linewidth=0.6, linestyle=linestyle, zorder=2)
    for key, label, color, marker in SPACE_SERIES:
        mean, sd = curve(alignment, "space", key, "mean", "sd")
        plot_series(ax_c, mean, sd, color, marker, label, linewidth=1.2, markersize=3.2, zorder=4)
    style(
        ax_c,
        (-0.1, 0.8),
        [0.0, 0.2, 0.4, 0.6, 0.8],
        "Cosine",
        "(c) Same-class alignment across perturbations",
        True,
    )
    handles, labels = ax_c.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color=PAIR_COLOR, linewidth=0.6),
        Line2D([0], [0], color=PAIR_COLOR, linewidth=0.6, linestyle=(0, (3, 1.5))),
    ]
    labels += ["Raw, pairs without noise", "Raw, pairs with noise"]
    ax_c.legend(
        handles,
        labels,
        loc="upper right",
        ncol=2,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.2,
    )

    fig.align_ylabels([ax_a, ax_b, ax_c])
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = STEM.with_suffix(suffix)
        fig.savefig(path, dpi=600)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
