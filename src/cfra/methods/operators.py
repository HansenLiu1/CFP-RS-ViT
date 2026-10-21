"""Symmetric direction-fit residual operators M0-M6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class OperatorError(ValueError):
    """Raised when a residual operator cannot be fit or transformed."""


@dataclass(frozen=True)
class GeometryResidualizer:
    """Direction-fit OLS residualizer for registered geometry covariates."""

    coefficients: np.ndarray
    ridge: float

    def transform(self, values: Any, geometry: Any) -> np.ndarray:
        matrix = _matrix(values, "values")
        covariates = np.asarray(geometry, dtype=np.float64)
        if covariates.ndim != 2 or covariates.shape[0] != matrix.shape[0]:
            raise OperatorError("geometry covariates must align with values")
        design = np.column_stack([np.ones(covariates.shape[0]), covariates])
        if design.shape[1] != self.coefficients.shape[0]:
            raise OperatorError("geometry covariate dimension does not match the fitted residualizer")
        return matrix - design @ self.coefficients


def fit_geometry_residualizer(
    values: Any,
    geometry: Any,
    *,
    ridge: float = 1.0e-5,
) -> GeometryResidualizer:
    """Fit ``Y ~ 1 + geometry`` only on direction-fit samples."""

    matrix = _matrix(values, "values")
    covariates = np.asarray(geometry, dtype=np.float64)
    if covariates.ndim != 2 or covariates.shape[0] != matrix.shape[0]:
        raise OperatorError("geometry covariates must align with values")
    if ridge <= 0:
        raise OperatorError("geometry residualizer ridge must be positive")
    design = np.column_stack([np.ones(covariates.shape[0]), covariates])
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ matrix)
    return GeometryResidualizer(coefficients=coefficients, ridge=ridge)


def _matrix(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise OperatorError(f"{name} must be a non-empty matrix")
    if not np.isfinite(array).all():
        raise OperatorError(f"{name} contains non-finite values")
    return array


def _unit(vector: np.ndarray, epsilon: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= epsilon:
        raise OperatorError("Operator fit produced a near-zero axis")
    return vector / norm


@dataclass(frozen=True)
class ResidualOperator:
    method_id: str
    dimension: int
    epsilon: float
    mean: np.ndarray | None = None
    axis: np.ndarray | None = None
    components: np.ndarray | None = None
    whitening: np.ndarray | None = None
    background_basis: np.ndarray | None = None

    def transform(self, values: Any) -> np.ndarray:
        matrix = _matrix(values, "values")
        if matrix.shape[1] != self.dimension:
            raise OperatorError("Values have the wrong operator dimension")
        method = self.method_id
        if method == "raw":
            return matrix.copy()
        if method == "euclidean_mean_subtraction":
            if self.mean is None:
                raise OperatorError("M1 has no fitted mean")
            return matrix - self.mean
        if method == "unit_axis_subtraction":
            if self.axis is None:
                raise OperatorError("M2 has no fitted unit axis")
            return matrix - self.axis
        if method == "orthogonal_axis_projection":
            if self.axis is None:
                raise OperatorError("M3 has no fitted axis")
            return matrix - (matrix @ self.axis)[:, None] * self.axis[None, :]
        if method == "direction_fit_pca_removal":
            if self.mean is None or self.components is None:
                raise OperatorError("M4 has no fitted PCA parameters")
            centered = matrix - self.mean
            return centered - (centered @ self.components.T) @ self.components
        if method == "whitened_centering":
            if self.mean is None or self.whitening is None:
                raise OperatorError("M5 has no fitted whitening parameters")
            return (matrix - self.mean) @ self.whitening
        if method == "background_svd_removal":
            if self.background_basis is None:
                raise OperatorError("M6 has no fitted background basis")
            return matrix - (matrix @ self.background_basis.T) @ self.background_basis
        raise OperatorError(f"Unknown residual operator: {method}")


def fit_residual_operator(
    method_id: str,
    direction_fit_values: Any,
    *,
    background_fit_values: Any | None = None,
    pca_rank: int = 5,
    whitening_ridge: float = 1.0e-5,
    epsilon: float = 1.0e-8,
) -> ResidualOperator:
    """Fit an operator only from direction-fit values and return a reusable transform."""

    values = _matrix(direction_fit_values, "direction_fit_values")
    dimension = values.shape[1]
    if epsilon <= 0 or whitening_ridge <= 0:
        raise OperatorError("epsilon and whitening_ridge must be positive")
    if method_id == "raw":
        return ResidualOperator(method_id, dimension, epsilon)
    if method_id == "euclidean_mean_subtraction":
        # M1: subtract the (unnormalized) mean of the per-class unit raw
        # directions.  Note this is the mean of the class directions, not of
        # the individual deltas.  It coincides with M2 (unit-axis subtraction)
        # up to the norm of that mean when the class directions are strongly
        # aligned; both are reported so the distinction stays explicit.
        return ResidualOperator(method_id, dimension, epsilon, mean=values.mean(axis=0))
    if method_id in {"unit_axis_subtraction", "orthogonal_axis_projection"}:
        axis = _unit(values.mean(axis=0), epsilon)
        return ResidualOperator(method_id, dimension, epsilon, axis=axis)
    if method_id == "direction_fit_pca_removal":
        mean = values.mean(axis=0)
        centered = values - mean
        _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
        if len(singular_values) == 0 or singular_values[0] <= epsilon:
            raise OperatorError("Direction-fit PCA has no non-zero component")
        rank = min(max(1, pca_rank), right.shape[0], dimension)
        components = right[:rank]
        return ResidualOperator(method_id, dimension, epsilon, mean=mean, components=components)
    if method_id == "whitened_centering":
        # M5 is fitted on the class-direction matrix (one vector per class, C of
        # them), so the covariance has rank <= C-1 in a d-dimensional space with
        # C << d.  The ridge (1e-5) then amplifies the null space by ~1/sqrt(ridge)
        # and the whitened space is dominated by directions the data never
        # constrain, which is why M5 collapses to chance.  It is kept for
        # completeness but is not a usable residual operator at this C/d ratio.
        mean = values.mean(axis=0)
        centered = values - mean
        covariance = (centered.T @ centered) / max(1, values.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        whitening = (eigenvectors * (1.0 / np.sqrt(eigenvalues + whitening_ridge))) @ eigenvectors.T
        return ResidualOperator(method_id, dimension, epsilon, mean=mean, whitening=whitening)
    if method_id == "background_svd_removal":
        if background_fit_values is None:
            raise OperatorError("M6 requires independent direction-fit background deltas")
        background = _matrix(background_fit_values, "background_fit_values")
        if background.shape[1] != dimension:
            raise OperatorError("Background values have the wrong operator dimension")
        _, singular_values, right = np.linalg.svd(background, full_matrices=False)
        if len(singular_values) == 0 or singular_values[0] <= epsilon:
            raise OperatorError("Background SVD has no non-zero component")
        rank = min(max(1, pca_rank), right.shape[0], dimension)
        return ResidualOperator(method_id, dimension, epsilon, background_basis=right[:rank])
    raise OperatorError(f"Unknown residual operator: {method_id}")
