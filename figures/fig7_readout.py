"""Fig. 7: readout sensitivity to direction removal.

One IEEE column (3.5 in), saved at its exact size (no tight bounding box).
All panels use M1 directions, the fit15 direction budget, a centered feature
space (the readout-fit mean is subtracted), a frozen linear readout, and
percentage points; (b) and (c) are at alpha = 1. Data: EXP3 M1 recompute
(2026-10-20), fit15, centered.
  (a) Final representation: frozen linear-readout accuracy drop versus removal
      strength alpha in {0, .25, .5, .75, 1}. Four conditions are drawn: own
      class, wrong class, the common-mode (mean-response) axis, and a random
      unit direction. The dose table also carries the weight-aligned random
      condition, which is not plotted (alpha = 1 drop .0490 would rescale the
      axis).
  (b) Same intervention at alpha = 1 across L01-L12 and the final layer
      (EXP3 b3 layerwise re-fit, E07-equivalent: all readout-fit/confirmation
      objects, directions fitted on at most 15 objects per class, norm
      preserved, readout frozen).
  (c) Paired differences at alpha = 1: own class minus wrong class, and own
      class minus the common-mode axis.
Each value is the mean of three per-seed values (60 model x dataset x
perturbation cells averaged within each seed); bands are +/-1 sample SD across
the three split seeds. No alpha = 2 condition and no raw readout space.

Run from this directory:
    python fig7_readout.py
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
STEM = OUT / "Figure 7"

LAYERS = [f"L{i:02d}" for i in range(1, 13)] + ["final"]
X = np.arange(len(LAYERS))
TICK_LABELS = {layer: ("Final" if layer == "final" else layer) for layer in LAYERS}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Okabe-Ito colors with distinct markers (grayscale safe).
CONDITION_SERIES = [
    ("own", "Own class", "#0072B2", "o"),
    ("wrong", "Wrong class", "#56B4E9", "s"),
    ("common", "Common-mode axis", "#E69F00", "^"),
    ("random", "Random", "#808080", "D"),
]
CONTRAST_SERIES = [
    ("own_minus_wrong", "Own \u2212 wrong", "#0072B2", "o"),
    ("own_minus_common", "Own \u2212 common mode", "#E69F00", "^"),
]

# Layout in inches.
FIG_W = 3.5
LEFT = 0.56
RIGHT_PAD = 0.08
BOTTOM = 0.36
PANEL_H = {"a": 1.05, "b": 1.05, "c": 0.85}
GAP_AB = 0.47  # panel (a) has its own x-axis label
GAP_BC = 0.24
TOP_PAD = 0.17
FIG_H = BOTTOM + sum(PANEL_H.values()) + GAP_AB + GAP_BC + TOP_PAD

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
        raise FileNotFoundError(f"{path} not found; run build_fig7_data.py first")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def series(rows, key_field: str, key: str, x_field: str, xs) -> tuple[np.ndarray, np.ndarray]:
    lookup = {row[x_field]: row for row in rows if row[key_field] == key}
    out_mean, out_sd = [], []
    for x in xs:
        match = next(
            (row for label, row in lookup.items() if (float(label) == x if x_field == "alpha" else label == x)),
            None,
        )
        if match is None:
            raise ValueError(f"{key}: missing {x_field}={x}")
        out_mean.append(float(match["mean"]))
        out_sd.append(float(match["sd"]))
    return np.array(out_mean), np.array(out_sd)


def plot(ax, x, mean, sd, color, marker, label) -> None:
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(
        x,
        mean,
        color=color,
        linewidth=1.0,
        marker=marker,
        markersize=3.0,
        markeredgewidth=0,
        label=label,
        zorder=3,
    )


def percent_axis(ax, ylim, yticks, ylabel: str) -> None:
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick * 100:.0f}" if tick != 0 else "0" for tick in yticks])
    ax.set_ylabel(ylabel, labelpad=3.0)


def finish(ax, title: str) -> None:
    ax.axhline(0.0, color="#808080", linewidth=0.5, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2.0, width=0.5, pad=2.0)
    ax.set_title(title, loc="left", pad=3.0)


def layer_axis(ax, show_labels: bool) -> None:
    ax.set_xlim(-0.4, len(LAYERS) - 0.6)
    ax.set_xticks(X)
    if show_labels:
        ax.set_xticklabels(
            [TICK_LABELS[layer] if index % 2 == 0 else "" for index, layer in enumerate(LAYERS)]
        )
        ax.set_xlabel("Layer", labelpad=2.0)
    else:
        ax.set_xticklabels([])


def main() -> None:
    dose = read_table("fig7_dose.csv")
    layerwise = read_table("fig7_layerwise.csv")
    diff = read_table("fig7_layerwise_diff.csv")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    bottom_c = BOTTOM
    bottom_b = bottom_c + PANEL_H["c"] + GAP_BC
    bottom_a = bottom_b + PANEL_H["b"] + GAP_AB
    ax_a = fig.add_axes(axes_rect(bottom_a, PANEL_H["a"]))
    ax_b = fig.add_axes(axes_rect(bottom_b, PANEL_H["b"]))
    ax_c = fig.add_axes(axes_rect(bottom_c, PANEL_H["c"]))

    # (a) Final-layer dose response.
    alphas = np.array(ALPHAS)
    for key, label, color, marker in CONDITION_SERIES:
        mean, sd = series(dose, "condition", key, "alpha", ALPHAS)
        plot(ax_a, alphas, mean, sd, color, marker, label)
    ax_a.set_xlim(-0.04, 1.04)
    ax_a.set_xticks(ALPHAS)
    ax_a.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax_a.set_xlabel(r"Removal strength $\alpha$", labelpad=2.0)
    percent_axis(ax_a, (-0.003, 0.035), [0.0, 0.01, 0.02, 0.03], "Accuracy drop (pp)")
    finish(ax_a, "(a) Final layer, M1 dose response")
    ax_a.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.2,
    )

    # (b) Across depth at alpha = 1.
    for key, label, color, marker in CONDITION_SERIES:
        mean, sd = series(layerwise, "condition", key, "layer", LAYERS)
        plot(ax_b, X, mean, sd, color, marker, label)
    layer_axis(ax_b, False)
    percent_axis(ax_b, (-0.004, 0.055), [0.0, 0.01, 0.02, 0.03, 0.04, 0.05], "Accuracy drop (pp)")
    finish(ax_b, r"(b) Across depth, M1, $\alpha = 1$")
    ax_b.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.2,
    )

    # (c) Paired differences at alpha = 1.
    for key, label, color, marker in CONTRAST_SERIES:
        mean, sd = series(diff, "contrast", key, "layer", LAYERS)
        plot(ax_c, X, mean, sd, color, marker, label)
    layer_axis(ax_c, True)
    percent_axis(ax_c, (-0.008, 0.024), [-0.005, 0.0, 0.005, 0.01, 0.015, 0.02], "Difference (pp)")
    ax_c.set_yticklabels(["\u22120.5", "0", "0.5", "1.0", "1.5", "2.0"])
    finish(ax_c, r"(c) Own-class excess, M1, $\alpha = 1$")
    ax_c.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.9,
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
