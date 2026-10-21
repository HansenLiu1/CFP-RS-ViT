"""Representation-delta definitions used by all formal stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class DeltaError(ValueError):
    """Raised when a delta cannot be represented under the registered formula."""


@dataclass(frozen=True)
class DeltaBatch:
    values: np.ndarray
    norms: np.ndarray
    valid: np.ndarray
    normalization: str
    epsilon: float


def _as_matrix(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise DeltaError(f"{name} must have shape [n, dimension]")
    if not np.isfinite(matrix).all():
        raise DeltaError(f"{name} contains non-finite values")
    return matrix


def representation_delta(clear_features: Any, degraded_features: Any) -> np.ndarray:
    """Compute the signed delta ``f(clear) - f(degraded)`` exactly."""

    clear = _as_matrix(clear_features, "clear_features")
    degraded = _as_matrix(degraded_features, "degraded_features")
    if clear.shape != degraded.shape:
        raise DeltaError("clear_features and degraded_features must have the same shape")
    return clear - degraded


def normalize_deltas(
    deltas: Any,
    *,
    normalization: str,
    epsilon: float = 1.0e-8,
    reference_norm: float | None = None,
) -> DeltaBatch:
    """Apply the preregistered unit/raw/magnitude-weighted delta variants.

    ``unit`` is the standard unit-normalized directional formula. ``raw`` retains the signed response
    magnitude. ``magnitude_weighted`` uses unit vectors weighted by each sample's
    norm relative to the fit-set median norm; it therefore retains relative, but
    not arbitrary absolute, feature-scale units.
    """

    values = _as_matrix(deltas, "deltas")
    if epsilon <= 0:
        raise DeltaError("epsilon must be positive")
    norms = np.linalg.norm(values, axis=1)
    valid = norms > epsilon
    output = np.zeros_like(values)
    if normalization == "raw":
        output = values.copy()
    elif normalization in {"unit", "magnitude_weighted"}:
        output[valid] = values[valid] / norms[valid, None]
        if normalization == "magnitude_weighted":
            scale = reference_norm if reference_norm is not None else float(np.median(norms[valid])) if valid.any() else 1.0
            if not np.isfinite(scale) or scale <= epsilon:
                raise DeltaError("magnitude-weighted normalization has no positive reference norm")
            output[valid] *= norms[valid, None] / scale
    else:
        raise DeltaError(f"Unknown delta normalization: {normalization}")
    return DeltaBatch(output, norms, valid, normalization, epsilon)


def factorial_deltas(
    *,
    foreground_clear_background_clean: Any,
    foreground_degraded_background_clean: Any,
    foreground_clear_background_degraded: Any,
    foreground_degraded_background_degraded: Any,
) -> dict[str, np.ndarray]:
    """Return foreground effects and the background interaction term.

    For background state ``b``:

    ``D_b = f(foreground_clear, b) - f(foreground_degraded, b)``.

    The interaction contrast is ``D_clean - D_degraded``.  A non-zero contrast
    directly records departure from the additive/superposition assumption.
    """

    clean_clean = _as_matrix(foreground_clear_background_clean, "foreground_clear_background_clean")
    degraded_clean = _as_matrix(foreground_degraded_background_clean, "foreground_degraded_background_clean")
    clean_degraded = _as_matrix(foreground_clear_background_degraded, "foreground_clear_background_degraded")
    degraded_degraded = _as_matrix(
        foreground_degraded_background_degraded,
        "foreground_degraded_background_degraded",
    )
    shape = clean_clean.shape
    if any(matrix.shape != shape for matrix in (degraded_clean, clean_degraded, degraded_degraded)):
        raise DeltaError("All factorial feature matrices must have the same shape")
    delta_clean_background = clean_clean - degraded_clean
    delta_degraded_background = clean_degraded - degraded_degraded
    return {
        "delta_clean_background": delta_clean_background,
        "delta_degraded_background": delta_degraded_background,
        "background_interaction": delta_clean_background - delta_degraded_background,
        "delta_main": delta_degraded_background,
    }


def aggregate_class_direction(values: Any, *, epsilon: float = 1.0e-8) -> np.ndarray:
    """Normalize the arithmetic mean of sample vectors, with no zero fallback."""

    matrix = _as_matrix(values, "values")
    if matrix.shape[0] == 0:
        raise DeltaError("Cannot fit a class direction from zero samples")
    direction = matrix.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= epsilon:
        raise DeltaError("Class direction mean norm is at or below epsilon")
    return direction / norm


def cosine_scores(vectors: Any, directions: Any, *, epsilon: float = 1.0e-8) -> np.ndarray:
    """Compute pairwise cosine scores ``vectors @ directions.T``."""

    left = _as_matrix(vectors, "vectors")
    right = _as_matrix(directions, "directions")
    if left.shape[1] != right.shape[1]:
        raise DeltaError("vectors and directions have incompatible dimensions")
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    if np.any(left_norm <= epsilon) or np.any(right_norm <= epsilon):
        raise DeltaError("Cosine input contains a near-zero vector")
    return (left / left_norm[:, None]) @ (right / right_norm[:, None]).T
