"""Shared analysis primitives for the paper-level tables and figures.

The functions here are the public, path-parameterized counterpart of the
internal recomputation helpers.  They never add the source tree to
``sys.path`` and never mutate BLAS thread settings at import time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cfra.directions import OPERATORS, common_mode, fit, unit
from cfra.evaluation.metrics import macro_ovr_auc, specificity_metrics
from cfra.representations.deltas import normalize_deltas

MODELS: tuple[str, ...] = ("remoteclip", "clip", "siglip2", "dinov2", "mae")
DATASETS: tuple[str, ...] = ("dota", "dior", "fair1m")
PERTURBATIONS: tuple[str, ...] = ("blur", "brightness", "gaussian_noise", "patch_mask")
EPS = 1.0e-8
FIT_LIMIT = 15
PCA_RANK = 5
META_KEYS = (
    "dataset_id",
    "object_id",
    "source_image_id",
    "class_id",
    "class_name",
    "coarse_name",
    "role",
    "perturbation",
    "excluded_from_main",
    "background_valid",
)


@dataclass(frozen=True)
class RunLayout:
    """Location of the three paper run directories.

    Parameters
    ----------
    run_root:
        Directory containing the per-seed run folders.
    runs:
        Mapping ``seed -> run directory name``.
    """

    run_root: Path
    runs: Mapping[int, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_root", Path(self.run_root).expanduser().resolve())
        object.__setattr__(self, "runs", {int(seed): str(name) for seed, name in self.runs.items()})

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted(self.runs))

    def run_dir(self, seed: int) -> Path:
        return self.run_root / self.runs[int(seed)]

    def feature_path(self, seed: int, model: str) -> Path:
        return self.run_dir(seed) / "intermediates" / "features" / f"features_{model}.npz"

    def layerwise_path(self, seed: int, model: str, dataset: str) -> Path:
        return self.run_dir(seed) / "intermediates" / "features" / f"layerwise_{model}_{dataset}.npz"


class Features:
    """Lazy view on one run's ``features_{model}.npz``."""

    def __init__(self, layout: RunLayout, seed: int, model: str):
        self.layout = layout
        self.seed = int(seed)
        self.model = model
        self.path = layout.feature_path(seed, model)
        self._npz = np.load(self.path, allow_pickle=False)
        self.meta = {key: np.asarray(self._npz[key]) for key in META_KEYS}
        self.meta["geometry"] = np.asarray(self._npz["geometry"], dtype=np.float64)
        self._arrays: dict[str, np.ndarray] = {}

    def array(self, key: str) -> np.ndarray:
        if key not in self._arrays:
            self._arrays[key] = np.asarray(self._npz[key])
        return self._arrays[key]

    def release(self, *keys: str) -> None:
        for key in keys or tuple(self._arrays):
            self._arrays.pop(key, None)

    def close(self) -> None:
        self._arrays.clear()
        self._npz.close()

    def mask(self, dataset: str, perturbation: str, role: str) -> np.ndarray:
        meta = self.meta
        return (
            (meta["dataset_id"] == dataset)
            & (meta["perturbation"] == perturbation)
            & (meta["role"] == role)
            & ~meta["excluded_from_main"].astype(bool)
        )

    def control_valid(self, control: str) -> np.ndarray:
        if control in {"true_obb", "label_permutation", "geometry_regression"}:
            return np.ones(len(self.meta["object_id"]), dtype=bool)
        return np.asarray(self.array(f"control_valid_{control}"), dtype=bool)


def limit_fit(
    mask: np.ndarray,
    class_ids: np.ndarray,
    object_ids: np.ndarray,
    limit: int | None = FIT_LIMIT,
) -> np.ndarray:
    """Select the first ``limit`` objects per class in sha256(object_id) order."""

    if limit is None:
        return mask.copy()
    result = np.zeros(len(mask), dtype=bool)
    indices = np.flatnonzero(mask)
    labels = class_ids[indices]
    for class_id in sorted(set(labels.tolist())):
        class_indices = indices[labels == class_id]
        ordered = sorted(
            class_indices.tolist(),
            key=lambda index: (
                hashlib.sha256(str(object_ids[index]).encode("utf-8")).hexdigest(),
                str(object_ids[index]),
            ),
        )
        result[ordered[:limit]] = True
    return result


def unit_delta(
    clear: np.ndarray,
    degraded: np.ndarray,
    normalization: str = "unit",
    reference_norm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized delta rows and the validity mask, as in ``core.run_e04``."""

    batch = normalize_deltas(
        np.asarray(clear, dtype=np.float64) - np.asarray(degraded, dtype=np.float64),
        normalization=normalization,
        epsilon=EPS,
        reference_norm=reference_norm,
    )
    return batch.values[batch.valid], batch.valid, batch.norms


def cell_rows(
    feat: Features,
    dataset: str,
    perturbation: str,
    role: str,
    *,
    limit: int | None = None,
    control: str = "true_obb",
) -> np.ndarray:
    """Boolean row mask for one cell and role, after control validity and fit budget."""

    mask = feat.mask(dataset, perturbation, role) & feat.control_valid(control)
    if limit is not None:
        mask = limit_fit(mask, feat.meta["class_id"], feat.meta["object_id"], limit)
    return mask


def cell_delta(
    feat: Features,
    dataset: str,
    perturbation: str,
    role: str,
    *,
    limit: int | None = None,
    control: str = "true_obb",
    normalization: str = "unit",
    reference_norm: float | None = None,
) -> dict[str, Any]:
    """Normalized deltas of one cell/role with labels and provenance."""

    prefix = "true" if control in {"true_obb", "label_permutation", "geometry_regression"} else f"control_{control}"
    mask = cell_rows(feat, dataset, perturbation, role, limit=limit, control=control)
    clear = feat.array(f"{prefix}_clear_degraded")[mask]
    degraded = feat.array(f"{prefix}_degraded_degraded")[mask]
    values, valid, norms = unit_delta(clear, degraded, normalization, reference_norm)
    index = np.flatnonzero(mask)[valid]
    return {
        "values": values,
        "labels": feat.meta["class_id"][index].astype(np.int64),
        "images": feat.meta["source_image_id"][index],
        "objects": feat.meta["object_id"][index],
        "coarse": feat.meta["coarse_name"][index],
        "index": index,
        "norms": norms[valid],
        "row_mask": mask,
    }


def background_fit(feat: Features, fit_mask: np.ndarray, normalization: str = "unit") -> np.ndarray:
    """Background responses used only by M6."""

    mask = fit_mask & feat.meta["background_valid"].astype(bool)
    values, _valid, _norms = unit_delta(
        feat.array("background_clear")[mask],
        feat.array("background_degraded")[mask],
        normalization,
    )
    return values


def fit_operator(
    fit_values: np.ndarray,
    fit_labels: np.ndarray,
    operator: str = "M1",
    background_values: np.ndarray | None = None,
):
    """Fit a pipeline operator through the public direction API."""

    return fit(
        fit_values,
        fit_labels,
        operator=OPERATORS.get(operator, operator),
        background_values=background_values,
        pca_rank=PCA_RANK,
        epsilon=EPS,
    )


def specificity(
    operator_obj: Any,
    directions: np.ndarray,
    classes: np.ndarray,
    eval_values: np.ndarray,
    eval_labels: np.ndarray,
    eval_images: np.ndarray,
) -> dict[str, float]:
    """Own-argmax rate, margin and mean own cosine over source-image cells."""

    queries = operator_obj.transform(eval_values)
    result = specificity_metrics(directions, queries, eval_labels, classes, source_image_ids=eval_images)
    auc = macro_ovr_auc(queries, eval_labels, directions, classes)
    return {
        "own_argmax_rate": float(result["own_argmax_rate"]),
        "margin": float(result["mean_diagonal_margin"]),
        "mean_own_cosine": float(result["mean_own_cosine"]),
        "macro_ovr_auc": float(auc),
        "score_matrix": result["score_matrix"],
    }


def spectrum(rows: np.ndarray) -> dict[str, float]:
    """Gram-eigenvalue summary of a direction matrix."""

    lam = np.clip(np.linalg.eigvalsh(rows @ rows.T)[::-1], 0.0, None)
    share = lam / lam.sum()
    cumulative = np.cumsum(share)
    return {
        "top1_share": float(share[0]),
        "effective_rank": float(lam.sum() ** 2 / (lam**2).sum()),
        "n50": int(np.searchsorted(cumulative, 0.5) + 1),
        "n80": int(np.searchsorted(cumulative, 0.8) + 1),
    }


def seed_mean(frame, keys: list[str], value: str):
    """Three-seed equal-weight mean and sample SD of a per-seed metric."""

    grouped = frame.groupby(keys)[value]
    return grouped.agg(mean="mean", seed_sd="std", n_seeds="count").reset_index()


__all__ = [
    "DATASETS",
    "EPS",
    "FIT_LIMIT",
    "META_KEYS",
    "MODELS",
    "OPERATORS",
    "PERTURBATIONS",
    "Features",
    "RunLayout",
    "background_fit",
    "cell_delta",
    "cell_rows",
    "common_mode",
    "fit_operator",
    "limit_fit",
    "seed_mean",
    "specificity",
    "spectrum",
    "unit",
    "unit_delta",
]
