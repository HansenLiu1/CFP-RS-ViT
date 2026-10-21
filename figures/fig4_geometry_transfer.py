"""Fig. 4: mask-geometry controls and cross-perturbation transfer.

One single-column layout (3.5 in), saved at its exact size (no tight bounding box):
  (a) Own-class argmax rate under the true OBB, the shifted OBB, the random
      geometry-matched OBB, and label permutation. M1, fit15, unit, confirmation.
      Bars are 3-seed means of the 5 encoder x 3 dataset averages; error bars
      are +/-1 sample SD across the three split seeds.
  (b) M1 cross-perturbation transfer (macro OVR AUC, S5a). Rows are the
      direction-fit perturbation, columns the scored confirmation perturbation;
      outlined diagonal cells are same-perturbation transfer. The color scale
      starts at chance (0.5).

Run from this directory:
    python fig4_geometry_transfer.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
DATA_GEOMETRY = HERE / "data" / "fig4_geometry.csv"
DATA_TRANSFER = HERE / "data" / "fig4_transfer.csv"
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)
STEM = OUT / "Figure 4"

PERTURBATIONS = ["blur", "brightness", "gaussian_noise", "patch_mask"]
PERTURBATION_LABELS = {
    "blur": "Blur",
    "brightness": "Brightness",
    "gaussian_noise": "Noise",
    "patch_mask": "Block",
}

CONTROLS = [
    "true_obb",
    "shifted_obb",
    "random_geometry_matched_obb",
    "label_permutation",
]
CONTROL_LABELS = {
    "true_obb": "True OBB",
    "shifted_obb": "Shifted OBB",
    "random_geometry_matched_obb": "Random OBB",
    "label_permutation": "Label perm.",
}
CONTROL_STYLE = {
    "true_obb": {"color": "#1F4E79", "hatch": None},
    "shifted_obb": {"color": "#5B9BD5", "hatch": None},
    "random_geometry_matched_obb": {"color": "#C9DDF0", "hatch": None},
    "label_permutation": {"color": "white", "hatch": "////"},
}

# Layout in inches.
FIG_W = 3.5
LEFT = 0.66
RIGHT_PAD = 0.08
BOTTOM_B = 0.36
ROW_H_B = 0.22
MAP_H_B = ROW_H_B * len(PERTURBATIONS)
MAP_W_B = 2.20
TITLE_PAD = 0.17
GAP_AB = 0.24
HEIGHT_A = 1.25
CBAR_GAP = 0.10
CBAR_W = 0.08
FIG_H = BOTTOM_B + MAP_H_B + TITLE_PAD + GAP_AB + HEIGHT_A + TITLE_PAD

AUC_NORM = Normalize(vmin=0.5, vmax=0.9)

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
        "hatch.linewidth": 0.4,
        "hatch.color": "#555555",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def truncated(name: str, low: float, high: float) -> ListedColormap:
    """Keep the light-to-medium range so black cell labels stay readable."""

    base = plt.get_cmap(name)
    return ListedColormap(base(np.linspace(low, high, 256)), name=f"{name}_truncated")


AUC_CMAP = truncated("Greens", 0.04, 0.72)


def axes_rect(bottom_in: float, left_in: float, width_in: float, height_in: float) -> list[float]:
    return [left_in / FIG_W, bottom_in / FIG_H, width_in / FIG_W, height_in / FIG_H]


def load_geometry() -> dict[tuple[str, str], tuple[float, float]]:
    if not DATA_GEOMETRY.exists():
        raise FileNotFoundError(f"{DATA_GEOMETRY} not found; run build_data.py first")
    out = {}
    with DATA_GEOMETRY.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out[(row["control"], row["perturbation"])] = (
                float(row["estimate"]),
                float(row["sd"]),
            )
    missing = [(c, p) for c in CONTROLS for p in PERTURBATIONS if (c, p) not in out]
    if missing:
        raise ValueError(f"missing geometry cells: {missing}")
    return out


def load_transfer() -> np.ndarray:
    if not DATA_TRANSFER.exists():
        raise FileNotFoundError(f"{DATA_TRANSFER} not found; run build_data.py first")
    lookup = {}
    with DATA_TRANSFER.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            lookup[(row["source"], row["target"])] = float(row["auc"])
    return np.array(
        [[lookup[(source, target)] for target in PERTURBATIONS] for source in PERTURBATIONS]
    )


def panel_geometry(ax, values: dict[tuple[str, str], tuple[float, float]]) -> None:
    x = np.arange(len(PERTURBATIONS))
    width = 0.19
    offsets = (np.arange(len(CONTROLS)) - (len(CONTROLS) - 1) / 2) * width

    for index, control in enumerate(CONTROLS):
        means = [values[(control, p)][0] for p in PERTURBATIONS]
        sds = [values[(control, p)][1] for p in PERTURBATIONS]
        style = CONTROL_STYLE[control]
        ax.bar(
            x + offsets[index],
            means,
            width=width,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor="#333333",
            linewidth=0.4,
            yerr=sds,
            error_kw={"elinewidth": 0.5, "capsize": 1.3, "capthick": 0.5, "ecolor": "black"},
            label=CONTROL_LABELS[control],
            zorder=2,
        )

    ax.set_xlim(-0.5, len(PERTURBATIONS) - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8"])
    ax.spines["left"].set_bounds(0.0, 0.8)
    ax.set_ylabel("Own-class argmax rate")
    ax.set_xticks(x)
    ax.set_xticklabels([PERTURBATION_LABELS[p] for p in PERTURBATIONS])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.tick_params(axis="y", length=2.5, width=0.5, pad=2.0)
    ax.set_title("(a) Mask-geometry controls", loc="left", pad=3.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        handleheight=0.8,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.1,
    )


def panel_transfer(ax, cax, matrix: np.ndarray) -> None:
    mesh = ax.pcolormesh(
        matrix,
        cmap=AUC_CMAP,
        norm=AUC_NORM,
        edgecolors="white",
        linewidth=0.8,
    )
    n = matrix.shape[0]
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)

    for row in range(n):
        for col in range(n):
            ax.text(
                col + 0.5,
                row + 0.5,
                f"{matrix[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="black",
            )
        ax.add_patch(
            Rectangle(
                (row + 0.04, row + 0.06),
                0.92,
                0.88,
                fill=False,
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )
        )

    labels = [PERTURBATION_LABELS[p] for p in PERTURBATIONS]
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Scored perturbation", labelpad=2.0)
    ax.set_ylabel("Fitted perturbation", labelpad=3.0)
    ax.tick_params(axis="both", which="both", length=0, pad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("(b) Cross-perturbation transfer", loc="left", pad=3.0)

    cbar = ax.figure.colorbar(mesh, cax=cax)
    cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9])
    cbar.set_ticklabels(["0.5", "0.6", "0.7", "0.8", "0.9"])
    cbar.set_label("Macro OVR AUC", labelpad=2.0)
    cbar.ax.tick_params(length=2.0, width=0.5, pad=1.5, labelsize=7.0)
    cbar.outline.set_linewidth(0.5)


def main() -> None:
    geometry = load_geometry()
    transfer = load_transfer()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    bottom_a = BOTTOM_B + MAP_H_B + TITLE_PAD + GAP_AB
    ax_a = fig.add_axes(axes_rect(bottom_a, LEFT, FIG_W - LEFT - RIGHT_PAD, HEIGHT_A))
    ax_b = fig.add_axes(axes_rect(BOTTOM_B, LEFT, MAP_W_B, MAP_H_B))
    cax = fig.add_axes(axes_rect(BOTTOM_B, LEFT + MAP_W_B + CBAR_GAP, CBAR_W, MAP_H_B))

    panel_geometry(ax_a, geometry)
    panel_transfer(ax_b, cax, transfer)
    fig.align_ylabels([ax_a, ax_b])

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = STEM.with_suffix(suffix)
        fig.savefig(path, dpi=600)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
