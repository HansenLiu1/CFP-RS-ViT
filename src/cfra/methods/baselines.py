"""Registered raw-feature and clear/degraded classifier baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class BaselineError(ValueError):
    """Raised when a baseline cannot be fit."""


def _matrix(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise BaselineError(f"{name} must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise BaselineError(f"{name} contains non-finite values")
    return matrix


def _unit(vector: np.ndarray, epsilon: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= epsilon:
        raise BaselineError("Baseline direction has a near-zero norm")
    return vector / norm


def fit_raw_prototype_directions(
    features: Any,
    class_ids: Any,
    *,
    epsilon: float = 1.0e-8,
) -> dict[int, np.ndarray]:
    """Fit class-mean-minus-global-mean prototype directions from readout-fit ROI features."""

    matrix = _matrix(features, "features")
    labels = np.asarray(class_ids)
    if labels.ndim != 1 or len(labels) != len(matrix):
        raise BaselineError("class_ids must align with features")
    global_mean = matrix.mean(axis=0)
    directions: dict[int, np.ndarray] = {}
    for class_id in sorted(set(int(value) for value in labels.tolist())):
        class_mean = matrix[labels == class_id].mean(axis=0)
        directions[class_id] = _unit(class_mean - global_mean, epsilon)
    return directions


@dataclass(frozen=True)
class LinearCAV:
    weight: np.ndarray
    bias: float
    class_id: int

    def decision_function(self, features: Any) -> np.ndarray:
        matrix = _matrix(features, "features")
        if matrix.shape[1] != self.weight.shape[0]:
            raise BaselineError("CAV feature dimension mismatch")
        return matrix @ self.weight + self.bias


def fit_clear_degraded_cav(
    clear_features: Any,
    degraded_features: Any,
    *,
    class_id: int,
    regularization: float = 1.0e-3,
    steps: int = 200,
    learning_rate: float = 0.1,
) -> LinearCAV:
    """Fit a deterministic logistic CAV normal to paired clear/degraded features.

    The classifier labels clear samples as one and degraded samples as zero.  It
    is a baseline direction, never used to fit the CFRA residual operators.
    """

    clear = _matrix(clear_features, "clear_features")
    degraded = _matrix(degraded_features, "degraded_features")
    if clear.shape != degraded.shape:
        raise BaselineError("clear_features and degraded_features must have the same shape")
    if regularization <= 0 or steps <= 0 or learning_rate <= 0:
        raise BaselineError("CAV optimization parameters must be positive")
    features = np.concatenate([clear, degraded], axis=0)
    labels = np.concatenate([np.ones(len(clear)), np.zeros(len(degraded))])
    scale = np.maximum(features.std(axis=0), 1.0e-6)
    standardized = features / scale
    weight = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(steps):
        logits = np.clip(standardized @ weight + bias, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        residual = probabilities - labels
        gradient_w = standardized.T @ residual / len(labels) + regularization * weight
        gradient_b = float(residual.mean())
        weight -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b
    # Return the normal in the original feature coordinates.
    original_weight = weight / scale
    original_bias = bias
    return LinearCAV(original_weight, original_bias, class_id)


def cav_direction(cav: LinearCAV, *, epsilon: float = 1.0e-8) -> np.ndarray:
    return _unit(cav.weight, epsilon)
