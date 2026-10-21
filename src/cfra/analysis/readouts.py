"""Readouts fitted on clean ``readout_fit`` features only."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

from cfra.directions import unit


class LinearReadout:
    """Multinomial logistic readout; the intercept makes it centering-invariant."""

    name = "linear"

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int = 1729,
        max_iter: int = 2000,
        C: float = 1.0,
    ):
        self.mean = np.asarray(X).mean(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.clf = LogisticRegression(max_iter=max_iter, random_state=seed, C=C)
            self.clf.fit(np.asarray(X) - self.mean, y)
        self.classes = np.asarray(self.clf.classes_).astype(int)

    def scores(self, X: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(np.asarray(X) - self.mean)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.scores(X), axis=1)]

    def weight_basis(self, rtol: float = 1e-6) -> np.ndarray:
        """Orthonormal basis of the span of the class weight vectors."""

        _, sing, vt = np.linalg.svd(self.clf.coef_, full_matrices=False)
        rank = int(np.sum(sing > rtol * sing[0]))
        return vt[:rank].T

    def class_directions(self) -> np.ndarray:
        """Unit weight vectors, one row per entry of ``self.classes``."""

        return unit(self.clf.coef_)


class PrototypeReadout:
    """Cosine to class centroids; ``centered`` subtracts the readout-fit mean."""

    name = "prototype"

    def __init__(self, X: np.ndarray, y: np.ndarray, centered: bool = False):
        matrix = np.asarray(X)
        self.center = matrix.mean(axis=0) if centered else np.zeros(matrix.shape[1])
        self.classes = np.unique(y).astype(int)
        shifted = matrix - self.center
        self.centroids = unit(
            np.stack([shifted[y == class_id].mean(axis=0) for class_id in self.classes])
        )

    def scores(self, X: np.ndarray) -> np.ndarray:
        return unit(np.asarray(X) - self.center) @ self.centroids.T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.scores(X), axis=1)]

    def weight_basis(self, rtol: float = 1e-6) -> np.ndarray:
        """Orthonormal basis of the span of the class centroids."""

        _, sing, vt = np.linalg.svd(self.centroids, full_matrices=False)
        rank = int(np.sum(sing > rtol * sing[0]))
        return vt[:rank].T

    def class_directions(self) -> np.ndarray:
        """Unit centroid vectors, one row per entry of ``self.classes``."""

        return self.centroids


__all__ = ["LinearReadout", "PrototypeReadout"]
