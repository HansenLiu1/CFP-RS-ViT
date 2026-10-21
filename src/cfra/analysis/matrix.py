"""Matrix helpers shared by the final-layer geometry analyses."""

from __future__ import annotations

import hashlib
import zlib

import numpy as np

from cfra.analysis.common import EPS


def stable_seed(*parts: object, tag: str = "cfra-analysis") -> int:
    """Deterministic 32-bit seed for a tuple of identifiers."""

    text = "|".join([tag, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def project_out(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project rows onto the orthogonal complement of ``basis``, preserving norm."""

    matrix = np.asarray(values, dtype=np.float64)
    rows = np.asarray(basis, dtype=np.float64)
    if rows.shape[0] == 0:
        return matrix.copy()
    modified = matrix - (matrix @ rows.T) @ rows
    original = np.linalg.norm(matrix, axis=-1, keepdims=True)
    updated = np.linalg.norm(modified, axis=-1, keepdims=True)
    return modified * np.where(updated > 1.0e-12, original / np.maximum(updated, 1.0e-12), 1.0)


def random_basis(dim: int, count: int, key: str | int) -> np.ndarray:
    """Orthonormal random rows with a deterministic seed derived from ``key``."""

    rng = np.random.default_rng(zlib.crc32(str(key).encode("utf-8")))
    q, _ = np.linalg.qr(rng.standard_normal((count, dim)).T)
    return q.T


def random_nested(dim: int, count: int, key: str | int) -> np.ndarray:
    """Nested random orthonormal rows (the first ``k`` are the same for every use)."""

    return random_basis(dim, count, key)


def orthonormalize(rows: np.ndarray, other: np.ndarray | None = None) -> np.ndarray:
    """Orthonormalise rows after projecting out an already orthonormal ``other``."""

    basis = np.asarray(rows, dtype=np.float64)
    if other is not None and np.asarray(other).shape[0]:
        basis = basis - (basis @ np.asarray(other).T) @ np.asarray(other)
    q, _ = np.linalg.qr(basis.T)
    return q.T


def paired_cos(matrix: np.ndarray) -> float:
    """Mean off-diagonal cosine of the rows of a small unit matrix."""

    rows = np.asarray(matrix, dtype=np.float64)
    if rows.shape[0] < 2:
        return float("nan")
    gram = rows @ rows.T
    return float((gram.sum() - np.trace(gram)) / (rows.shape[0] * (rows.shape[0] - 1)))


def class_unit_mean(values: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Unit mean row for every class, preserving the requested class order."""

    matrix = np.asarray(values, dtype=np.float64)
    label_values = np.asarray(labels)
    class_values = np.asarray(classes, dtype=np.int64)
    sums = np.zeros((len(class_values), matrix.shape[1]), dtype=np.float64)
    position = {int(value): index for index, value in enumerate(class_values.tolist())}
    for index, label in enumerate(label_values.tolist()):
        position_index = position.get(int(label))
        if position_index is not None:
            sums[position_index] += matrix[index]
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    return sums / np.maximum(norms, EPS)


def same_diff(matrix: np.ndarray) -> tuple[float, float]:
    """Mean diagonal and mean off-diagonal of a square class matrix."""

    values = np.asarray(matrix, dtype=np.float64)
    count = values.shape[0]
    if count < 2:
        return float("nan"), float("nan")
    return (
        float(np.trace(values) / count),
        float((values.sum() - np.trace(values)) / (count * (count - 1))),
    )


__all__ = [
    "class_unit_mean",
    "orthonormalize",
    "paired_cos",
    "project_out",
    "random_basis",
    "random_nested",
    "same_diff",
    "stable_seed",
]
