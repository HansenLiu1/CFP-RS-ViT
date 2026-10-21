"""Public class-direction API.

The paper uses one direction-fitting implementation for every analysis.  This
module is the single public entry point; it delegates to the pipeline's fitted
residual operators and exposes the M0--M6 names used in the manuscript.

M1 (CMC):
    u_i = unit(delta_i)
    A_c = mean_{i in class c} u_i
    w_c = unit(A_c)
    m = mean_c w_c          # not renormalized
    d_c = unit(A_c - m)
    query = unit(delta) - m
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cfra.pipeline.core import _fit_operator_and_directions

OPERATORS: dict[str, str] = {
    "M0": "raw",
    "M1": "euclidean_mean_subtraction",
    "M2": "unit_axis_subtraction",
    "M3": "orthogonal_axis_projection",
    "M4": "direction_fit_pca_removal",
    "M5": "whitened_centering",
    "M6": "background_svd_removal",
}
EPSILON = 1.0e-8
PCA_RANK = 5


def unit(values: np.ndarray, *, epsilon: float = EPSILON) -> np.ndarray:
    """Normalize rows, guarding near-zero vectors."""

    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, epsilon)


def fit(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    operator: str = "M1",
    background_values: np.ndarray | None = None,
    pca_rank: int = PCA_RANK,
    epsilon: float = EPSILON,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit a residual operator and its class directions.

    Parameters
    ----------
    values:
        Fit rows in representation-delta space (already normalized as needed).
    labels:
        Integer class id per row.
    operator:
        ``"M0"``--``"M6"`` or the internal operator id registered in
        :data:`OPERATORS`.
    background_values:
        Background responses used by M6.
    pca_rank:
        Rank used by M4/M6.
    epsilon:
        Near-zero guard.

    Returns
    -------
    (operator, directions, class_ids)
        ``directions`` has shape ``[n_classes, dim]`` and rows correspond to
        ``class_ids``.
    """

    method_id = OPERATORS.get(operator, operator)
    return _fit_operator_and_directions(
        np.asarray(values, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        method_id,
        background_fit_values=background_values,
        pca_rank=pca_rank,
        epsilon=epsilon,
    )


def common_mode(operator: Any) -> np.ndarray:
    """Return the M1 origin ``m`` (the fitted operator mean)."""

    mean = getattr(operator, "mean", None)
    if mean is None:
        raise ValueError("operator has no fitted mean; fit M1 to obtain m")
    return np.asarray(mean, dtype=np.float64)
