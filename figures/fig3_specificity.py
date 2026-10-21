"""Fig. 3: encoder-resolved class specificity under four perturbations.

Three 5 x 4 heatmaps stacked in one IEEE column (3.5 in):
  (a) raw directions (M0) and (b) CMC residuals (M1) share one color scale;
  (c) the gain M1 - M0 uses its own scale.
Values are equal-weight means over three source-image-disjoint split seeds and
three datasets (DOTA v1.5, DIOR-R, FAIR1M 37 fine classes); fit15, unit
normalization, true-OBB control, confirmation role.

Run from this directory:
    python fig3_specificity.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "fig3_specificity.csv"
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)
STEM = OUT / "Figure 3"

MODELS = ["remoteclip", "clip", "siglip2", "dinov2", "mae"]
MODEL_LABELS = {
    "remoteclip": "RemoteCLIP",
    "clip": "CLIP",
    "siglip2": "SigLIP2",
    "dinov2": "DINOv2",
    "mae": "MAE",
}
PERTURBATIONS = ["blur", "brightness", "gaussian_noise", "patch_mask"]
PERTURBATION_LABELS = {
    "blur": "Blur",
    "brightness": "Brightness",
    "gaussian_noise": "Noise",
    "patch_mask": "Block",
}

# Layout in inches; the figure is saved at its exact size (no tight bbox).
FIG_W = 3.5
LEFT = 0.66
MAP_W = 2.20
ROW_H = 0.15
MAP_H = ROW_H * len(MODELS)
BOTTOM = 0.22
PANEL_GAP = 0.26
TOP_PAD = 0.18
CBAR_GAP = 0.10
CBAR_W = 0.08
FIG_H = BOTTOM + 3 * MAP_H + 2 * PANEL_GAP + TOP_PAD

RATE_NORM = Normalize(vmin=0.0, vmax=1.0)
GAIN_NORM = Normalize(vmin=0.0, vmax=0.35)

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
        "axes.linewidth": 0.5,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def truncated(name: str, low: float, high: float) -> ListedColormap:
    """Keep the light-to-medium range so black cell labels stay readable."""

    base = plt.get_cmap(name)
    return ListedColormap(base(np.linspace(low, high, 256)), name=f"{name}_truncated")


RATE_CMAP = truncated("Blues", 0.04, 0.75)
GAIN_CMAP = truncated("Oranges", 0.04, 0.68)


def load() -> dict[tuple[str, str], tuple[float, float, float]]:
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} not found; run build_data.py first")
    out: dict[tuple[str, str], tuple[float, float, float]] = {}
    with DATA.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out[(row["model"], row["perturbation"])] = (
                float(row["m0"]),
                float(row["m1"]),
                float(row["delta"]),
            )
    missing = [
        (model, perturbation)
        for model in MODELS
        for perturbation in PERTURBATIONS
        if (model, perturbation) not in out
    ]
    if missing:
        raise ValueError(f"missing cells: {missing}")
    return out


def axes_rect(bottom_in: float, left_in: float, width_in: float, height_in: float) -> list[float]:
    return [left_in / FIG_W, bottom_in / FIG_H, width_in / FIG_W, height_in / FIG_H]


def draw_map(ax, grid: np.ndarray, cmap, norm, fmt, title: str, show_xlabels: bool):
    mesh = ax.pcolormesh(
        grid,
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidth=0.8,
    )
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(grid.shape[0], 0)

    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            value = grid[row, col]
            ax.text(
                col + 0.5,
                row + 0.5,
                fmt(value),
                ha="center",
                va="center",
                fontsize=7.5,
                color="black",
            )

    ax.set_yticks(np.arange(grid.shape[0]) + 0.5)
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_xticks(np.arange(grid.shape[1]) + 0.5)
    if show_xlabels:
        ax.set_xticklabels([PERTURBATION_LABELS[p] for p in PERTURBATIONS])
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="both", which="both", length=0, pad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", pad=3.0)
    return mesh


def style_colorbar(cbar, label: str, ticks: list[float], ticklabels: list[str]) -> None:
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.set_label(label, labelpad=2.0)
    cbar.ax.tick_params(length=2.0, width=0.5, pad=1.5, labelsize=7.0)
    cbar.outline.set_linewidth(0.5)


def main() -> None:
    values = load()
    m0 = np.array([[values[(m, p)][0] for p in PERTURBATIONS] for m in MODELS])
    m1 = np.array([[values[(m, p)][1] for p in PERTURBATIONS] for m in MODELS])
    gain = np.array([[values[(m, p)][2] for p in PERTURBATIONS] for m in MODELS])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    bottom_c = BOTTOM
    bottom_b = bottom_c + MAP_H + PANEL_GAP
    bottom_a = bottom_b + MAP_H + PANEL_GAP

    ax_a = fig.add_axes(axes_rect(bottom_a, LEFT, MAP_W, MAP_H))
    ax_b = fig.add_axes(axes_rect(bottom_b, LEFT, MAP_W, MAP_H))
    ax_c = fig.add_axes(axes_rect(bottom_c, LEFT, MAP_W, MAP_H))

    rate_fmt = lambda v: f"{v:.2f}"  # noqa: E731
    gain_fmt = lambda v: "<0.01" if v < 0.005 else f"{v:.2f}"  # noqa: E731

    draw_map(ax_a, m0, RATE_CMAP, RATE_NORM, rate_fmt, "(a) Raw directions (M0)", False)
    rate_mesh = draw_map(ax_b, m1, RATE_CMAP, RATE_NORM, rate_fmt, "(b) CMC residuals (M1)", False)
    gain_mesh = draw_map(ax_c, gain, GAIN_CMAP, GAIN_NORM, gain_fmt, "(c) Gain (M1 \u2212 M0)", True)

    cbar_left = LEFT + MAP_W + CBAR_GAP
    cax_rate = fig.add_axes(axes_rect(bottom_b, cbar_left, CBAR_W, bottom_a + MAP_H - bottom_b))
    cax_gain = fig.add_axes(axes_rect(bottom_c, cbar_left, CBAR_W, MAP_H))

    style_colorbar(
        fig.colorbar(rate_mesh, cax=cax_rate),
        "Own-class argmax rate",
        [0.0, 0.25, 0.5, 0.75, 1.0],
        ["0.00", "0.25", "0.50", "0.75", "1.00"],
    )
    style_colorbar(
        fig.colorbar(gain_mesh, cax=cax_gain),
        "M1 \u2212 M0",
        [0.0, 0.1, 0.2, 0.3],
        ["0.00", "0.10", "0.20", "0.30"],
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = STEM.with_suffix(suffix)
        fig.savefig(path, dpi=600)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
