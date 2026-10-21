"""Pure-NumPy E06 layerwise diagnostics.

The layerwise stage deliberately treats every layer as a separate feature space.
Directions are fitted and scored within one layer only; the late-minus-early
summary compares scalar metrics and never compares vectors from different
dimensions.

Accepted layer input forms are intentionally small and explicit:

* ``FeatureTable``-like objects with ``arrays`` and ``metadata`` attributes;
* mappings of ``layer_name -> record``;
* a mapping with a ``layers`` member;
* a record containing ``features`` (or ``values``/``embedding``) and metadata.

Metadata may be a sequence of row mappings/objects or a column mapping.  Each
row must provide ``role``, ``source_image_id`` (``image_id`` is accepted as an
alias), ``class_id`` and ``perturbation`` by default.
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("cfra")


class LayerwiseError(ValueError):
    """Raised when an E06 layerwise input violates its analysis contract."""


@dataclass(frozen=True)
class FeatureRecord:
    """Normalized feature matrix and row metadata used by E06/E08."""

    features: np.ndarray
    metadata: tuple[Any, ...]
    representation: dict[str, Any]
    model_id: str | None = None


@dataclass(frozen=True)
class LayerwiseFeatureTable:
    """Convenience container for a mapping of named layer records."""

    layers: Mapping[str, Any]
    metadata: Any | None = None
    model_id: str | None = None
    representation: Mapping[str, Any] | None = None


def _artifact_name(name: str) -> str:
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise LayerwiseError("layerwise artifact name must be a non-empty relative filename stem")
    if name != name.strip() or any(character in name for character in ("/", "\\", "\x00")):
        raise LayerwiseError("layerwise artifact name cannot escape the current run")
    if name.endswith("."):
        raise LayerwiseError("layerwise artifact name cannot end with a dot")
    return name


def _layer_feature_arrays(
    payload: Any, layer: str, clear: np.ndarray, degraded: np.ndarray, expected_count: int
) -> dict[str, np.ndarray]:
    """Paired clear/degraded arrays, plus the optional clean ``raw`` array if stored."""

    features = {"clear": clear, "degraded": degraded}
    raw_key = f"{layer}__raw"
    if raw_key in payload:
        raw = np.asarray(payload[raw_key], dtype=np.float64)
        if raw.shape[0] == expected_count:
            features["raw"] = raw
    return features


def load_layerwise_feature_artifact(
    run_root: Path | str,
    artifact_name: str = "layerwise_features",
) -> LayerwiseFeatureTable:
    """Load one current-run paired layerwise artifact without accepting paths."""

    import json
    name = _artifact_name(artifact_name)
    try:
        root = Path(run_root).expanduser().resolve(strict=True)
        output_root = (root / "intermediates" / "features").resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LayerwiseError("layerwise artifact directory is missing or invalid") from exc
    if output_root == root or root not in output_root.parents:
        raise LayerwiseError("layerwise artifact root escapes the current run")
    metadata_path = output_root / f"{name}.json"
    arrays_path = output_root / f"{name}.npz"
    try:
        metadata_path = metadata_path.resolve(strict=True)
        arrays_path = arrays_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LayerwiseError("layerwise artifact files are missing or invalid") from exc
    for path in (metadata_path, arrays_path):
        if output_root not in path.parents or not path.is_file():
            raise LayerwiseError("layerwise artifact files must remain below the current run")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LayerwiseError(f"cannot read layerwise artifact metadata: {metadata_path}") from exc
    if not isinstance(metadata, Mapping):
        raise LayerwiseError("layerwise artifact metadata must be a JSON object")
    if metadata.get("artifact_kind") != "layerwise_paired_features":
        raise LayerwiseError("unsupported layerwise artifact kind")
    selected_layers = metadata.get("selected_layers")
    if (
        not isinstance(selected_layers, list)
        or not selected_layers
        or len(set(selected_layers)) != len(selected_layers)
        or any(
            not isinstance(layer, str)
            or not layer
            or "\x00" in layer
            or "/" in layer
            or "\\" in layer
            for layer in selected_layers
        )
    ):
        raise LayerwiseError("layerwise artifact must register selected_layers")
    layer_metadata = metadata.get("layer_metadata")
    if not isinstance(layer_metadata, Mapping):
        raise LayerwiseError("layerwise artifact must register per-layer metadata")
    rows = metadata.get("row_metadata")
    try:
        expected_count = int(metadata.get("sample_count"))
    except (TypeError, ValueError) as exc:
        raise LayerwiseError("layerwise artifact sample_count must be an integer") from exc
    if expected_count <= 0:
        raise LayerwiseError("layerwise artifact sample_count must be positive")
    if rows is not None:
        if isinstance(rows, Mapping):
            row_records = rows.get("records", rows.get("rows"))
            if row_records is not None and len(row_records) != expected_count:
                raise LayerwiseError("layerwise artifact row metadata does not match sample_count")
            if row_records is None:
                for value in rows.values():
                    if isinstance(value, (str, bytes)) or np.asarray(value).ndim == 0:
                        continue
                    if len(value) != expected_count:
                        raise LayerwiseError("layerwise artifact metadata columns do not match sample_count")
        else:
            try:
                if len(rows) != expected_count:
                    raise LayerwiseError("layerwise artifact row metadata does not match sample_count")
            except TypeError as exc:
                raise LayerwiseError("layerwise artifact row metadata must be a sequence or mapping") from exc
    layers: dict[str, dict[str, Any]] = {}
    try:
        with np.load(arrays_path, allow_pickle=False) as payload:
            for layer in selected_layers:
                clear_key = f"{layer}__clear"
                degraded_key = f"{layer}__degraded"
                if clear_key not in payload or degraded_key not in payload:
                    raise LayerwiseError(f"layerwise artifact is missing paired arrays for {layer!r}")
                clear = np.asarray(payload[clear_key], dtype=np.float64)
                degraded = np.asarray(payload[degraded_key], dtype=np.float64)
                if clear.ndim != 2 or degraded.ndim != 2 or clear.shape != degraded.shape:
                    raise LayerwiseError(f"paired arrays for {layer!r} must have equal matrix shapes")
                if clear.shape[0] != expected_count or not np.isfinite(clear).all() or not np.isfinite(degraded).all():
                    raise LayerwiseError(f"invalid paired arrays for layer {layer!r}")
                if layer not in layer_metadata:
                    raise LayerwiseError(f"layerwise artifact is missing metadata for {layer!r}")
                representation = layer_metadata[layer]
                if not isinstance(representation, Mapping):
                    raise LayerwiseError(f"layerwise artifact metadata for {layer!r} must be a mapping")
                required_metadata = {
                    "kind",
                    "block",
                    "dim",
                    "pooling",
                    "projection",
                    "text_shared_space",
                    "token",
                    "grid",
                    "interpolation",
                }
                missing_metadata = sorted(required_metadata - set(representation))
                if missing_metadata:
                    raise LayerwiseError(f"layerwise artifact metadata for {layer!r} is missing: {missing_metadata}")
                dimension = representation.get("dim", representation.get("dimension"))
                if dimension is not None:
                    try:
                        if int(dimension) != clear.shape[1]:
                            raise LayerwiseError(f"layerwise artifact dimension metadata does not match {layer!r}")
                    except (TypeError, ValueError) as exc:
                        raise LayerwiseError(f"layerwise artifact dimension metadata is invalid for {layer!r}") from exc
                layers[layer] = {
                    "features": _layer_feature_arrays(payload, layer, clear, degraded, expected_count),
                    "metadata": rows,
                    "representation": dict(representation),
                }
    except LayerwiseError:
        raise
    except (OSError, ValueError) as exc:
        raise LayerwiseError(f"cannot read layerwise artifact arrays: {arrays_path}") from exc
    return LayerwiseFeatureTable(
        layers=layers,
        metadata=rows,
        model_id=str(metadata.get("model_id")) if metadata.get("model_id") is not None else None,
        representation=metadata,
    )


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _field(row: Any, name: str, default: Any = None) -> Any:
    """Read a field from both dataclass-style rows and mapping rows."""

    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _metadata_columns(metadata: Mapping[str, Any], count: int | None) -> tuple[Any, ...]:
    if "records" in metadata:
        return _metadata_records(metadata["records"], count)
    if "rows" in metadata:
        return _metadata_records(metadata["rows"], count)
    values = list(metadata.values())
    if count == 1 and all(not isinstance(value, (list, tuple, np.ndarray)) for value in values):
        return (dict(metadata),)
    if not values:
        raise LayerwiseError("metadata must contain at least one column or row")
    columns: dict[str, Any] = {}
    lengths: list[int] = []
    for key, value in metadata.items():
        if isinstance(value, (str, bytes)) or np.asarray(value).ndim == 0:
            columns[key] = value
            continue
        try:
            column = tuple(value)
        except TypeError as exc:
            raise LayerwiseError("columnar metadata values must be sequences or scalar constants") from exc
        columns[key] = column
        lengths.append(len(column))
    inferred = count if count is not None else (lengths[0] if lengths else 1)
    if any(length != inferred for length in lengths):
        raise LayerwiseError("columnar metadata columns must have equal lengths")
    if count is not None and inferred != count:
        raise LayerwiseError("metadata columns do not align with the feature rows")
    return tuple(
        {
            key: value if isinstance(value, (str, bytes)) or np.asarray(value).ndim == 0 else value[index]
            for key, value in columns.items()
        }
        for index in range(inferred)
    )


def _metadata_records(metadata: Any, count: int | None) -> tuple[Any, ...]:
    if metadata is None:
        raise LayerwiseError("feature input must include metadata")
    if isinstance(metadata, Mapping):
        records = _metadata_columns(metadata, count)
    else:
        try:
            records = tuple(metadata)
        except TypeError as exc:
            if count == 1:
                records = (metadata,)
            else:
                raise LayerwiseError("metadata must be a sequence of rows") from exc
    if count is not None and len(records) != count:
        raise LayerwiseError("metadata row count does not match the feature matrix")
    if not records:
        raise LayerwiseError("metadata must contain at least one row")
    return records


def _source_mapping(source: Any) -> Mapping[str, Any] | None:
    if isinstance(source, Mapping):
        return source
    return None


def _arrays_from(source: Any) -> Mapping[str, Any] | None:
    mapping = _source_mapping(source)
    if mapping is not None and isinstance(mapping.get("arrays"), Mapping):
        return mapping["arrays"]
    arrays = getattr(source, "arrays", None)
    return arrays if isinstance(arrays, Mapping) else None


def _nested_feature_mapping(source: Any) -> Mapping[str, Any] | None:
    mapping = _source_mapping(source)
    if mapping is not None:
        for key in ("features", "layer_features", "embeddings"):
            value = mapping.get(key)
            if isinstance(value, Mapping):
                return value
    for key in ("features", "layer_features", "embeddings"):
        value = getattr(source, key, None)
        if isinstance(value, Mapping):
            return value
    arrays = _arrays_from(source)
    if arrays is not None:
        for key in ("features", "layer_features", "embeddings"):
            value = arrays.get(key)
            if isinstance(value, Mapping):
                return value
    return None


def _extract_features(source: Any, *, feature_key: str | None, layer_name: str | None) -> Any:
    mapping = _source_mapping(source)
    if isinstance(source, np.ndarray):
        return source
    if mapping is not None:
        nested = _nested_feature_mapping(source)
        if nested is not None:
            if feature_key is not None and feature_key in nested:
                return nested[feature_key]
            if layer_name is not None and layer_name in nested:
                return nested[layer_name]
            # Prefer the paired clear/degraded contrast over the optional clean
            # ``raw`` array: a paired layer artifact may now carry ``raw`` for the
            # readout intervention, but the class/perturbation diagnostics must
            # keep operating on the delta, not on the clean representation.
            for clear_name, degraded_name in (
                ("true_clear_degraded", "true_degraded_degraded"),
                ("clear", "degraded"),
            ):
                if clear_name in nested and degraded_name in nested:
                    clear = np.asarray(nested[clear_name], dtype=np.float64)
                    degraded = np.asarray(nested[degraded_name], dtype=np.float64)
                    if clear.shape != degraded.shape:
                        raise LayerwiseError(
                            f"paired arrays {clear_name!r} and {degraded_name!r} have different shapes"
                        )
                    return clear - degraded
            if "raw" in nested:
                return nested["raw"]
            matrix_candidates = [value for value in nested.values() if np.asarray(value).ndim == 2]
            if len(matrix_candidates) == 1:
                return matrix_candidates[0]
        if feature_key is not None and feature_key in mapping:
            return mapping[feature_key]
        # Prefer the paired clear/degraded contrast over the optional clean
        # ``raw`` array for the same reason as the nested branch above.
        for clear_name, degraded_name in (
            ("true_clear_degraded", "true_degraded_degraded"),
            ("clear", "degraded"),
        ):
            if clear_name in mapping and degraded_name in mapping:
                clear = np.asarray(mapping[clear_name], dtype=np.float64)
                degraded = np.asarray(mapping[degraded_name], dtype=np.float64)
                if clear.shape != degraded.shape:
                    raise LayerwiseError(
                        f"paired arrays {clear_name!r} and {degraded_name!r} have different shapes"
                    )
                return clear - degraded
        if feature_key is None and "raw" in mapping:
            return mapping["raw"]
        if layer_name is not None and layer_name in mapping:
            candidate = mapping[layer_name]
            if not isinstance(candidate, Mapping) or any(
                key in candidate for key in ("features", "feature", "values", "embedding", "arrays", "data", "matrix", "array")
            ):
                return candidate
        for key in ((feature_key,) if feature_key is not None else ()):
            if key is not None and key in mapping:
                return mapping[key]
        for key in ("features", "feature", "values", "embedding", "X", "x", "data", "matrix", "array", "delta"):
            if key in mapping:
                return mapping[key]
    if mapping is None:
        nested = _nested_feature_mapping(source)
        if nested is not None:
            if feature_key is not None and feature_key in nested:
                return nested[feature_key]
            if layer_name is not None and layer_name in nested:
                return nested[layer_name]
            for clear_name, degraded_name in (
                ("true_clear_degraded", "true_degraded_degraded"),
                ("clear", "degraded"),
            ):
                if clear_name in nested and degraded_name in nested:
                    clear = np.asarray(nested[clear_name], dtype=np.float64)
                    degraded = np.asarray(nested[degraded_name], dtype=np.float64)
                    if clear.shape != degraded.shape:
                        raise LayerwiseError(
                            f"paired arrays {clear_name!r} and {degraded_name!r} have different shapes"
                        )
                    return clear - degraded
            if "raw" in nested:
                return nested["raw"]
            matrix_candidates = [value for value in nested.values() if np.asarray(value).ndim == 2]
            if len(matrix_candidates) == 1:
                return matrix_candidates[0]
    arrays = _arrays_from(source)
    if arrays is not None:
        if feature_key is not None and feature_key in arrays:
            return arrays[feature_key]
        if layer_name is not None and layer_name in arrays:
            return arrays[layer_name]
        for key in ("features", "feature", "values", "embedding", "X", "x", "data", "matrix", "array", "delta"):
            if key in arrays:
                return arrays[key]
        # Prefer paired clear/degraded arrays over the optional clean ``raw``.
        for clear_name, degraded_name in (
            ("true_clear_degraded", "true_degraded_degraded"),
            ("clear", "degraded"),
        ):
            if clear_name in arrays and degraded_name in arrays:
                clear = np.asarray(arrays[clear_name], dtype=np.float64)
                degraded = np.asarray(arrays[degraded_name], dtype=np.float64)
                if clear.shape != degraded.shape:
                    raise LayerwiseError(
                        f"paired arrays {clear_name!r} and {degraded_name!r} have different shapes"
                    )
                return clear - degraded
        if feature_key is None and "raw" in arrays:
            return arrays["raw"]
        if feature_key is None:
            candidates = [
                value
                for key, value in arrays.items()
                if key not in {"geometry", "background_valid"} and not key.startswith("control_valid_")
            ]
            matrix_candidates = [value for value in candidates if np.asarray(value).ndim == 2]
            if len(matrix_candidates) == 1:
                return matrix_candidates[0]
    for key in ("features", "feature", "values", "embedding", "data", "matrix", "array"):
        candidate = getattr(source, key, None)
        if candidate is not None:
            return candidate
    raise LayerwiseError("could not find a feature matrix in the layer record")


def _representation_from(source: Any, inherited: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(inherited or {})
    mapping = _source_mapping(source)
    if mapping is not None and isinstance(mapping.get("representation"), Mapping):
        result.update(mapping["representation"])
    representation = getattr(source, "representation", None)
    if isinstance(representation, Mapping):
        result.update(representation)
    for key in ("patch_emb", "block", "pooling", "projection", "layer"):
        value = mapping.get(key) if mapping is not None else getattr(source, key, None)
        if value is not None:
            result[key] = value
    return result


def coerce_feature_record(
    source: Any,
    *,
    feature_key: str | None = None,
    layer_name: str | None = None,
    metadata: Any | None = None,
    representation: Mapping[str, Any] | None = None,
) -> FeatureRecord:
    """Normalize a feature record without importing optional model packages."""

    try:
        raw_features = _extract_features(source, feature_key=feature_key, layer_name=layer_name)
        features = np.asarray(raw_features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LayerwiseError("feature values must be a numeric matrix") from exc
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise LayerwiseError("features must be a non-empty matrix with shape [n, dimension]")
    if not np.isfinite(features).all():
        raise LayerwiseError("features contain non-finite values")

    mapping = _source_mapping(source)
    own_metadata = mapping.get("metadata") if mapping is not None else getattr(source, "metadata", None)
    rows = _metadata_records(own_metadata if own_metadata is not None else metadata, len(features))
    own_model_id = mapping.get("model_id") if mapping is not None else getattr(source, "model_id", None)
    return FeatureRecord(
        features=features,
        metadata=rows,
        representation=_representation_from(source, representation),
        model_id=None if own_model_id is None else str(own_model_id),
    )


def _layer_sources(table: Any, selected_layers: Sequence[str]) -> tuple[dict[str, Any], Any, Mapping[str, Any], str | None]:
    mapping = _source_mapping(table)
    inherited_metadata = mapping.get("metadata") if mapping is not None else getattr(table, "metadata", None)
    inherited_representation = _representation_from(table, None)
    model_id = mapping.get("model_id") if mapping is not None else getattr(table, "model_id", None)
    if model_id is not None:
        model_id = str(model_id)

    if isinstance(table, LayerwiseFeatureTable):
        return dict(table.layers), table.metadata, inherited_representation, table.model_id
    if mapping is not None and isinstance(mapping.get("layers"), Mapping):
        return dict(mapping["layers"]), inherited_metadata, inherited_representation, model_id
    nested = _nested_feature_mapping(table)
    if nested is not None and all(layer in nested for layer in selected_layers):
        return {layer: nested[layer] for layer in selected_layers}, inherited_metadata, inherited_representation, model_id
    layers = getattr(table, "layers", None)
    if isinstance(layers, Mapping):
        return dict(layers), inherited_metadata, inherited_representation, model_id
    if mapping is not None and all(layer in mapping for layer in selected_layers):
        return {layer: mapping[layer] for layer in selected_layers}, inherited_metadata, inherited_representation, model_id
    arrays = _arrays_from(table)
    if arrays is not None and all(layer in arrays for layer in selected_layers):
        return {layer: arrays[layer] for layer in selected_layers}, inherited_metadata, inherited_representation, model_id
    if len(selected_layers) == 1:
        return {selected_layers[0]: table}, inherited_metadata, inherited_representation, model_id
    raise LayerwiseError(
        "layerwise input must expose every selected layer as a mapping key, a layers mapping, or an arrays key"
    )


def _ordered_unique(values: Any) -> list[Any]:
    unique: list[Any] = []
    for value in np.asarray(values).tolist():
        if not any(value == existing for existing in unique):
            unique.append(value)
    try:
        return sorted(unique)
    except (TypeError, ValueError):
        return unique


def _labels(metadata: Sequence[Any], indices: np.ndarray, field: str, *, alias: str | None = None) -> np.ndarray:
    values = []
    for index in indices.tolist():
        value = _field(metadata[index], field)
        if value is None and alias is not None:
            value = _field(metadata[index], alias)
        if value is None:
            raise LayerwiseError(f"metadata field {field!r} is required")
        values.append(value)
    return np.asarray(values)


def _source_ids(metadata: Sequence[Any], indices: np.ndarray) -> np.ndarray:
    values = []
    for index in indices.tolist():
        value = _field(metadata[index], "source_image_id")
        if value is None:
            value = _field(metadata[index], "image_id")
        if value is None:
            raise LayerwiseError("metadata must provide source_image_id (or image_id) for image-disjoint analysis")
        values.append(value)
    return np.asarray(values)


def _assert_disjoint(left: np.ndarray, right: np.ndarray, message: str) -> None:
    overlap = set(left.tolist()) & set(right.tolist())
    if overlap:
        preview = sorted(overlap, key=str)[:3]
        raise LayerwiseError(f"{message}; overlapping source images include {preview}")


def _unit(vector: np.ndarray, epsilon: float, message: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= epsilon:
        raise LayerwiseError(message)
    return vector / norm


def _fit_directions(values: np.ndarray, labels: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    classes = _ordered_unique(labels)
    directions = []
    kept_classes = []
    for class_id in classes:
        direction = values[labels == class_id].mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= epsilon:
            continue
        directions.append(direction / norm)
        kept_classes.append(class_id)
    if not directions:
        raise LayerwiseError("no class direction could be fitted")
    return np.stack(directions), np.asarray(kept_classes)


def _cosine_scores(values: np.ndarray, directions: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    value_norms = np.linalg.norm(values, axis=1)
    direction_norms = np.linalg.norm(directions, axis=1)
    if np.any(direction_norms <= epsilon):
        raise LayerwiseError("fitted directions contain a near-zero vector")
    valid = value_norms > epsilon
    scores = np.full((len(values), len(directions)), np.nan, dtype=np.float64)
    if valid.any():
        scores[valid] = (values[valid] / value_norms[valid, None]) @ (directions / direction_norms[:, None]).T
    return scores, valid


def _nanmean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2.0
        position = end
    count_positive = int(positive.sum())
    count_negative = int(negative.sum())
    return float(
        (ranks[positive].sum() - count_positive * (count_positive + 1) / 2.0)
        / (count_positive * count_negative)
    )


def _binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    if not positive.any():
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_positive = positive[order]
    cumulative = np.cumsum(sorted_positive, dtype=np.int64)
    total = int(positive.sum())
    result = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        count = int(sorted_positive[start:end].sum())
        if count:
            result += float(cumulative[end - 1]) / end * count / total
        start = end
    return float(result)


def _predictability(
    values: np.ndarray,
    labels: np.ndarray,
    directions: np.ndarray,
    direction_labels: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    missing = [label for label in _ordered_unique(labels) if not np.any(direction_labels == label)]
    exclude_mask = np.ones(len(labels), dtype=bool)
    for label in missing:
        exclude_mask &= labels != label
    excluded_count = int(exclude_mask.size - int(exclude_mask.sum()))
    values = values[exclude_mask]
    labels = labels[exclude_mask]
    if len(values) == 0:
        return {
            "class_ids": direction_labels.tolist(),
            "macro_ovr_auc": float("nan"),
            "macro_auc": float("nan"),
            "auc": float("nan"),
            "macro_average_precision": float("nan"),
            "macro_ap": float("nan"),
            "accuracy": float("nan"),
            "own_argmax_rate": float("nan"),
            "mean_own_cosine": float("nan"),
            "mean_diagonal_margin": float("nan"),
            "score_matrix": np.full((len(direction_labels), len(direction_labels)), np.nan, dtype=np.float64),
            "scores": np.full((0, len(direction_labels)), np.nan, dtype=np.float64),
            "valid_row_count": 0,
            "near_zero_row_count": 0,
            "excluded_missing_class_row_count": excluded_count,
        }
    scores, valid = _cosine_scores(values, directions, epsilon)
    if not valid.any():
        return {
            "class_ids": direction_labels.tolist(),
            "macro_ovr_auc": float("nan"),
            "macro_auc": float("nan"),
            "auc": float("nan"),
            "macro_average_precision": float("nan"),
            "macro_ap": float("nan"),
            "accuracy": float("nan"),
            "own_argmax_rate": float("nan"),
            "mean_own_cosine": float("nan"),
            "mean_diagonal_margin": float("nan"),
            "score_matrix": np.full((len(direction_labels), len(direction_labels)), np.nan, dtype=np.float64),
            "scores": scores,
            "valid_row_count": 0,
            "near_zero_row_count": len(values),
            "excluded_missing_class_row_count": excluded_count,
        }
    scores = scores[valid]
    active_labels = labels[valid]
    predicted = direction_labels[np.argmax(scores, axis=1)]
    aucs = []
    aps = []
    score_matrix = np.full((len(direction_labels), len(direction_labels)), np.nan, dtype=np.float64)
    for direction_index, label in enumerate(direction_labels.tolist()):
        binary = (active_labels == label).astype(np.int64)
        aucs.append(_binary_auc(binary, scores[:, direction_index]))
        aps.append(_binary_average_precision(binary, scores[:, direction_index]))
        for actual_index, actual_label in enumerate(direction_labels.tolist()):
            selected = active_labels == actual_label
            if selected.any():
                score_matrix[direction_index, actual_index] = float(np.mean(scores[selected, direction_index]))
    margins = []
    own_scores = []
    for row_index, label in enumerate(active_labels.tolist()):
        label_index = next(index for index, value in enumerate(direction_labels.tolist()) if value == label)
        own_scores.append(float(scores[row_index, label_index]))
        alternatives = np.delete(scores[row_index], label_index)
        margins.append(float(scores[row_index, label_index] - np.max(alternatives)) if len(alternatives) else float("nan"))
    return {
        "class_ids": direction_labels.tolist(),
        "macro_ovr_auc": _nanmean(aucs),
        "macro_auc": _nanmean(aucs),
        "auc": _nanmean(aucs),
        "macro_average_precision": _nanmean(aps),
        "macro_ap": _nanmean(aps),
        "accuracy": float(np.mean(predicted == active_labels)),
        "own_argmax_rate": float(np.mean(predicted == active_labels)),
        "mean_own_cosine": _nanmean(own_scores),
        "mean_diagonal_margin": _nanmean(margins),
        "score_matrix": score_matrix,
        "scores": scores,
        "valid_row_count": len(active_labels),
        "near_zero_row_count": len(values) - int(valid.sum()),
        "excluded_missing_class_row_count": excluded_count,
    }


def _aggregate_predictability(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of per-perturbation predictability results (legacy layer summary).

    Scalar metrics are averaged across perturbation groups; matrix/array outputs
    are dropped from the aggregate, since the per-perturbation detail is kept in
    ``by_perturbation``.  The number of aggregated groups is recorded.
    """

    scalar_names = (
        "macro_ovr_auc",
        "macro_auc",
        "auc",
        "macro_average_precision",
        "macro_ap",
        "accuracy",
        "own_argmax_rate",
        "mean_own_cosine",
        "mean_diagonal_margin",
        "valid_row_count",
        "near_zero_row_count",
        "excluded_missing_class_row_count",
    )
    aggregated: dict[str, Any] = {}
    for name in scalar_names:
        values = [r.get(name) for r in results]
        numeric = [float(v) for v in values if v is not None and np.isfinite(v)]
        aggregated[name] = float(np.mean(numeric)) if numeric else float("nan")
    class_ids = results[0].get("class_ids") if results else []
    aggregated["class_ids"] = list(class_ids) if class_ids is not None else []
    aggregated["aggregated_group_count"] = len(results)
    aggregated["per_group_own_argmax_rate"] = [
        float(r.get("own_argmax_rate", float("nan"))) for r in results
    ]
    return aggregated


def _layer_metadata(layer_name: str, record: FeatureRecord) -> dict[str, Any]:
    metadata = dict(record.representation)
    explicit_patch = metadata.get("patch_emb")
    patch_emb = bool(explicit_patch) if explicit_patch is not None else layer_name.lower() in {
        "patch_emb",
        "patch_embedding",
        "patch-emb",
    }
    block = metadata.get("block")
    if block is None:
        match = re.fullmatch(r"(?:L|block[_-]?)(\d+)", layer_name, flags=re.IGNORECASE)
        block = int(match.group(1)) if match else None
    block_index = block if isinstance(block, (int, np.integer)) else None
    return {
        "layer": layer_name,
        "patch_emb": patch_emb,
        "block": block,
        "block_index": None if block_index is None else int(block_index),
        "pooling": metadata.get("pooling", "unspecified"),
        "projection": metadata.get("projection", "unspecified"),
        "model_id": record.model_id,
    }


def _scalar_late_minus_early(early: Mapping[str, Any], late: Mapping[str, Any]) -> dict[str, float]:
    excluded = {"scores", "score_matrix", "class_ids", "status"}
    result: dict[str, float] = {}
    for key, early_value in early.items():
        late_value = late.get(key)
        if key in excluded or not isinstance(early_value, (float, int, np.number)):
            continue
        if isinstance(late_value, (float, int, np.number)):
            result[key] = float(late_value) - float(early_value)
    return result


_BLOCK_LAYER_RE = re.compile(r"^L(\d+)$")


def _block_layer_index(name: Any) -> int | None:
    """Return the transformer block index for a layer name like ``L03``."""

    match = _BLOCK_LAYER_RE.match(str(name))
    return int(match.group(1)) if match else None


def _ordered_block_layers(layers: Sequence[str]) -> list[str]:
    """Transformer-block layers (``L<n>``), ordered by block index.

    ``patch_emb`` and the final pooling/projection output are excluded because
    they live in different representation spaces and cannot anchor a depth trend.
    """

    blocks = [(index, layer) for layer in layers if (index := _block_layer_index(layer)) is not None]
    blocks.sort()
    return [layer for _, layer in blocks]


def _rank(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values)).astype(np.float64)


def _spearman(x: Any, y: Any) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    rx = _rank(x[finite])
    ry = _rank(y[finite])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _metric_trend(layers: Sequence[str], results: Mapping[str, Any], group: str) -> dict[str, Any]:
    """Spearman trend of every scalar metric across an ordered layer sequence."""

    keys = sorted(
        {
            key
            for layer in layers
            for key, value in results[layer].get(group, {}).items()
            if isinstance(value, (float, int, np.number)) and np.isfinite(float(value))
        }
    )
    index = np.arange(len(layers), dtype=np.float64)
    spearman = {}
    early_mean = {}
    late_mean = {}
    for key in keys:
        values = np.asarray([float(results[layer].get(group, {}).get(key, np.nan)) for layer in layers])
        spearman[key] = _spearman(index, values)
        early_mean[key] = float(values[0]) if values.size else float("nan")
        late_mean[key] = float(values[-1]) if values.size else float("nan")
    return {
        "layers": list(layers),
        "spearman_vs_layer_index": spearman,
        "early_layer_mean": early_mean,
        "late_layer_mean": late_mean,
    }


def run_e06(
    layerwise: Any,
    selected_layers: Sequence[str],
    *,
    epsilon: float = 1.0e-8,
    feature_key: str | None = None,
    fit_role: str = "direction_fit",
    confirmation_role: str = "confirmation",
    class_field: str = "class_id",
    perturbation_field: str = "perturbation",
    early_layer: str | None = None,
    late_layer: str | None = None,
) -> dict[str, Any]:
    """Run E06 class and perturbation predictability for selected layers.

    ``features`` are expected to be one row per ROI and one matrix per layer.
    If a layer record only exposes paired ``true_clear_degraded`` and
    ``true_degraded_degraded`` arrays, their signed difference is used.  Both
    class and perturbation directions are fitted from ``fit_role`` rows and
    evaluated only on ``confirmation_role`` rows.  Every selected layer is
    normalized and fitted independently, so dimensions may differ.
    """

    if not isinstance(epsilon, (int, float, np.number)) or not np.isfinite(epsilon) or float(epsilon) <= 0:
        raise LayerwiseError("epsilon must be a finite positive number")
    if isinstance(selected_layers, (str, bytes)):
        raise LayerwiseError("selected_layers must be a non-empty sequence of layer names")
    try:
        selected = [str(layer) for layer in selected_layers]
    except TypeError as exc:
        raise LayerwiseError("selected_layers must be a non-empty sequence of layer names") from exc
    if not selected or len(set(selected)) != len(selected):
        raise LayerwiseError("selected_layers must contain at least one unique layer name")

    logger.info(f"[E06] 开始: layers={selected} fit_role={fit_role}")
    e06_started = time.time()
    layer_sources, inherited_metadata, inherited_representation, table_model_id = _layer_sources(layerwise, selected)
    missing = [layer for layer in selected if layer not in layer_sources]
    if missing:
        raise LayerwiseError(f"selected layers are missing from the input: {missing}")
    normalized: dict[str, FeatureRecord] = {}
    for layer in selected:
        layer_representation = dict(inherited_representation)
        layer_metadata = layer_representation.get(layer)
        if isinstance(layer_metadata, Mapping):
            layer_representation = dict(layer_metadata)
        metadata_for_layer = inherited_metadata
        if isinstance(inherited_metadata, Mapping) and all(name in inherited_metadata for name in selected):
            metadata_for_layer = inherited_metadata[layer]
        normalized[layer] = coerce_feature_record(
            layer_sources[layer],
            feature_key=feature_key,
            layer_name=layer,
            metadata=metadata_for_layer,
            representation=layer_representation,
        )
        if normalized[layer].model_id is None and table_model_id is not None:
            normalized[layer] = FeatureRecord(
                normalized[layer].features,
                normalized[layer].metadata,
                normalized[layer].representation,
                table_model_id,
            )

    first_metadata = normalized[selected[0]].metadata
    for layer in selected[1:]:
        if len(normalized[layer].metadata) != len(first_metadata):
            raise LayerwiseError("all selected layers must have the same metadata row count")
        for index, (left, right) in enumerate(zip(first_metadata, normalized[layer].metadata)):
            for field_name, alias in (
                ("object_id", None),
                ("source_image_id", "image_id"),
                ("class_id", None),
                ("role", None),
                ("perturbation", None),
            ):
                left_value = _field(left, field_name)
                if left_value is None and alias is not None:
                    left_value = _field(left, alias)
                right_value = _field(right, field_name)
                if right_value is None and alias is not None:
                    right_value = _field(right, alias)
                if left_value is not None and right_value is not None and left_value != right_value:
                    raise LayerwiseError(f"layer metadata are not row-aligned at row {index} ({field_name})")

    results: dict[str, dict[str, Any]] = {}
    for layer in selected:
        layer_started = time.time()
        record = normalized[layer]
        metadata = record.metadata
        roles = np.asarray([_field(row, "role") for row in metadata])
        if any(value is None for value in roles.tolist()):
            raise LayerwiseError("every metadata row must provide role")
        fit_indices = np.flatnonzero(roles == fit_role)
        confirmation_indices = np.flatnonzero(roles == confirmation_role)
        if len(fit_indices) == 0:
            raise LayerwiseError(f"layer {layer!r} has no {fit_role!r} rows")
        if len(confirmation_indices) == 0:
            raise LayerwiseError(f"layer {layer!r} has no {confirmation_role!r} rows")
        fit_images = _source_ids(metadata, fit_indices)
        confirmation_images = _source_ids(metadata, confirmation_indices)
        _assert_disjoint(fit_images, confirmation_images, f"layer {layer!r} fit and confirmation roles must be image-disjoint")

        fit_labels = _labels(metadata, fit_indices, class_field)
        confirmation_labels = _labels(metadata, confirmation_indices, class_field)
        fit_perturbations = _labels(metadata, fit_indices, perturbation_field)
        confirmation_perturbations = _labels(metadata, confirmation_indices, perturbation_field)
        fit_features = record.features[fit_indices]
        confirmation_features = record.features[confirmation_indices]
        perturbation_group = [str(value) for value in _ordered_unique(confirmation_perturbations)]
        by_perturbation: dict[str, dict[str, Any]] = {}
        class_results: list[dict[str, Any]] = []
        fit_features_grouped: list[np.ndarray] = []
        fit_labels_grouped: list[np.ndarray] = []
        for pert in perturbation_group:
            fsel = np.asarray([str(value) == pert for value in fit_perturbations.tolist()])
            csel = np.asarray([str(value) == pert for value in confirmation_perturbations.tolist()])
            if not fsel.any() or not csel.any():
                continue
            group_fit = fit_features[fsel]
            group_confirm = confirmation_features[csel]
            group_confirm_labels = confirmation_labels[csel]
            group_class_directions, group_class_ids = _fit_directions(
                group_fit, fit_labels[fsel], float(epsilon)
            )
            group_class_result = _predictability(
                group_confirm,
                group_confirm_labels,
                group_class_directions,
                group_class_ids,
                float(epsilon),
            )
            by_perturbation[pert] = {
                "class_predictability": group_class_result,
                "fit_object_count": int(fsel.sum()),
                "confirmation_object_count": int(csel.sum()),
                "fit_source_image_count": len(set(fit_images[fsel].tolist())),
                "confirmation_source_image_count": len(set(confirmation_images[csel].tolist())),
            }
            class_results.append(group_class_result)
            fit_features_grouped.append(group_fit)
            fit_labels_grouped.append(fit_labels[fsel])
        if not class_results:
            raise LayerwiseError(f"layer {layer!r} has no perturbation group with fitted directions")
        class_result = _aggregate_predictability(class_results)
        class_directions, class_ids = _fit_directions(fit_features, fit_labels, float(epsilon))
        perturbation_directions, perturbation_ids = _fit_directions(
            fit_features,
            fit_perturbations,
            float(epsilon),
        )
        perturbation_result = _predictability(
            confirmation_features,
            confirmation_perturbations,
            perturbation_directions,
            perturbation_ids,
            float(epsilon),
        )
        layer_result = _layer_metadata(layer, record)
        layer_result["metadata"] = dict(layer_result)
        layer_result.update(
            {
                "dimension": int(record.features.shape[1]),
                "fit_object_count": len(fit_indices),
                "fit_source_image_count": len(set(fit_images.tolist())),
                "confirmation_object_count": len(confirmation_indices),
                "confirmation_source_image_count": len(set(confirmation_images.tolist())),
                "class_directions": class_directions,
                "direction_class_ids": class_ids,
                "perturbation_directions": perturbation_directions,
                "direction_perturbation_ids": perturbation_ids,
                "class_predictability": class_result,
                "perturbation_predictability": perturbation_result,
                "by_perturbation": by_perturbation,
            }
        )
        results[layer] = layer_result
        logger.info(
            f"[E06] layer={layer} 完成 (耗时 {time.time() - layer_started:.1f}s): "
            f"class_auc={class_result.get('macro_ovr_auc', float('nan')):.4f} "
            f"pert_auc={perturbation_result.get('macro_ovr_auc', float('nan')):.4f} "
            f"(fit={len(fit_indices)}, confirm={len(confirmation_indices)})"
        )

    block_layers = _ordered_block_layers(selected)
    if len(block_layers) >= 2:
        trend_basis = block_layers
        trend_basis_kind = "blocks"
    else:
        trend_basis = list(selected)
        trend_basis_kind = "selected"
    early_name = early_layer or trend_basis[0]
    late_name = late_layer or trend_basis[-1]
    if early_name not in results or late_name not in results:
        raise LayerwiseError("early_layer and late_layer must be selected layers")
    early_class = results[early_name]["class_predictability"]
    late_class = results[late_name]["class_predictability"]
    early_perturbation = results[early_name]["perturbation_predictability"]
    late_perturbation = results[late_name]["perturbation_predictability"]
    class_delta = _scalar_late_minus_early(early_class, late_class)
    perturbation_delta = _scalar_late_minus_early(early_perturbation, late_perturbation)
    late_minus_early = {
        "early_layer": early_name,
        "late_layer": late_name,
        "basis": trend_basis_kind,
        "class_predictability": class_delta,
        "perturbation_predictability": perturbation_delta,
        "class_macro_ovr_auc": class_delta.get("macro_ovr_auc", float("nan")),
        "perturbation_macro_ovr_auc": perturbation_delta.get("macro_ovr_auc", float("nan")),
    }
    block_trend = {
        "layers": trend_basis,
        "basis": trend_basis_kind,
        "class_predictability": _metric_trend(trend_basis, results, "class_predictability"),
        "perturbation_predictability": _metric_trend(trend_basis, results, "perturbation_predictability"),
    }
    return {
        "stage": "E06",
        "model_id": table_model_id,
        "selected_layers": selected,
        "block_layers": block_layers,
        "patch_emb_layer": next((layer for layer in selected if results[layer].get("patch_emb")), None),
        "final_layer": next((layer for layer in selected if layer not in block_layers and not results[layer].get("patch_emb")), None),
        "layers": results,
        "results": results,
        "metadata": {layer: {key: value for key, value in result.items() if key in {
            "layer", "patch_emb", "block", "block_index", "pooling", "projection", "model_id", "dimension"
        }} for layer, result in results.items()},
        "layer_metadata": {layer: {key: value for key, value in result.items() if key in {
            "layer", "patch_emb", "block", "block_index", "pooling", "projection", "model_id", "dimension"
        }} for layer, result in results.items()},
        "class_predictability": {layer: result["class_predictability"] for layer, result in results.items()},
        "perturbation_predictability": {
            layer: result["perturbation_predictability"] for layer, result in results.items()
        },
        "late_minus_early": late_minus_early,
        "block_trend": block_trend,
        "image_disjoint": True,
    }
    logger.info(
        f"[E06] 完成 (耗时 {time.time() - e06_started:.1f}s): "
        f"{len(selected)} 层, class_delta={class_delta.get('macro_ovr_auc', float('nan')):.4f}, "
        f"pert_delta={perturbation_delta.get('macro_ovr_auc', float('nan')):.4f}"
    )


def _decision_matrix(classifier: Any, features: np.ndarray) -> np.ndarray:
    """Per-class decision scores as an ``(n, n_classes)`` matrix (handles binary)."""

    scores = np.asarray(classifier.decision_function(features), dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores


def _margin_accuracy(scores: np.ndarray, classes: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Overall accuracy and mean own-minus-strongest-other margin for a score matrix."""

    classes = np.asarray(classes)
    index = {int(c): i for i, c in enumerate(classes.tolist())}
    predicted = classes[np.argmax(scores, axis=1)]
    accuracy = float(np.mean(predicted == labels))
    own = np.array([scores[i, index[int(labels[i])]] for i in range(len(labels))])
    mask = np.ones_like(scores, dtype=bool)
    for i in range(len(labels)):
        mask[i, index[int(labels[i])]] = False
    other = np.where(mask, scores, -np.inf).max(axis=1)
    return accuracy, float(np.mean(own - other))


def _unit_rows(values: np.ndarray, epsilon: float = 1.0e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.clip(norms, epsilon, None)


def _readout_class_directions(
    unit_deltas: np.ndarray, labels: np.ndarray, epsilon: float
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    """Raw (M0), CMC (M1) class directions and the across-class mean axis."""

    labels = np.asarray(labels).astype(np.int64)
    classes = sorted(set(labels.tolist()))
    class_mean = {c: unit_deltas[labels == c].mean(axis=0) for c in classes}
    unit_class_mean = {c: _unit_rows(class_mean[c][None, :], epsilon)[0] for c in classes}
    axis = _unit_rows(np.stack([unit_class_mean[c] for c in classes]).mean(axis=0)[None, :], epsilon)[0]
    raw_dirs = {c: unit_class_mean[c] for c in classes}
    cmc_dirs = {c: _unit_rows((class_mean[c] - axis)[None, :], epsilon)[0] for c in classes}
    return raw_dirs, cmc_dirs, axis


def _nearest_wrong(centroids_unit: np.ndarray, classes: np.ndarray) -> dict[int, int]:
    similarity = centroids_unit @ centroids_unit.T
    np.fill_diagonal(similarity, -np.inf)
    return {int(c): int(classes[int(np.argmax(similarity[i]))]) for i, c in enumerate(classes.tolist())}


def _removal_drops(
    eval_raw: np.ndarray,
    eval_labels: np.ndarray,
    direction_map: Mapping[int, np.ndarray],
    alpha: float,
    preserve_norm: bool,
    *,
    classifier: Any,
    classes: np.ndarray,
    centroids_unit: np.ndarray,
    base_linear_accuracy: float,
    base_linear_margin: float,
    base_prototype_accuracy: float,
    base_prototype_margin: float,
    epsilon: float,
) -> dict[str, float]:
    """Accuracy and margin drops after removing ``direction_map`` from centered features."""

    directions = np.stack([direction_map[int(c)] for c in eval_labels.tolist()])
    directions = _unit_rows(directions, epsilon)
    projection = np.sum(eval_raw * directions, axis=1, keepdims=True)
    transformed = eval_raw - alpha * projection * directions
    if preserve_norm:
        original = np.linalg.norm(eval_raw, axis=1, keepdims=True)
        updated = np.linalg.norm(transformed, axis=1, keepdims=True)
        transformed = transformed * (original / np.clip(updated, epsilon, None))
    lin_acc, lin_margin = _margin_accuracy(_decision_matrix(classifier, transformed), classes, eval_labels)
    proto_acc, proto_margin = _margin_accuracy(
        _unit_rows(transformed, epsilon) @ centroids_unit.T, classes, eval_labels
    )
    return {
        "linear_accuracy_drop": base_linear_accuracy - lin_acc,
        "linear_margin_drop": base_linear_margin - lin_margin,
        "prototype_accuracy_drop": base_prototype_accuracy - proto_acc,
        "prototype_margin_drop": base_prototype_margin - proto_margin,
    }


def _mean_readout_across_perturbations(
    per_perturbation: Mapping[str, Any], resolved: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if resolved is None:
        resolved = per_perturbation
    keys = sorted({key for result in resolved.values() for key in result["alpha"]})
    out: dict[str, Any] = {"perturbations": list(resolved.keys()), "alpha": {}}
    for key in keys:
        metrics = sorted({metric for result in resolved.values() for metric in result["alpha"].get(key, {})})
        out["alpha"][key] = {
            metric: float(np.mean([result["alpha"][key][metric] for result in resolved.values() if key in result["alpha"]]))
            for metric in metrics
        }
    return out


def _readout_own_wrong_trend(block_layers: Sequence[str], layers_result: Mapping[str, Any]) -> dict[str, Any]:
    own_minus_wrong = {}
    for layer in block_layers:
        own = layers_result[layer]["mean"]["alpha"].get("own@1", {})
        wrong = layers_result[layer]["mean"]["alpha"].get("wrong@1", {})
        own_minus_wrong[layer] = float(own.get("linear_margin_drop", np.nan)) - float(wrong.get("linear_margin_drop", np.nan))
    values = np.asarray([own_minus_wrong[layer] for layer in block_layers], dtype=np.float64)
    return {
        "layers": list(block_layers),
        "own_minus_wrong_linear_margin_drop": own_minus_wrong,
        "spearman_vs_layer_index": _spearman(np.arange(len(block_layers)), values),
    }


def run_e06_readout(
    layer_arrays: Mapping[str, Mapping[str, np.ndarray]],
    metadata: Sequence[Any],
    selected_layers: Sequence[str],
    *,
    epsilon: float = 1.0e-8,
    alphas: Sequence[float] = (0.25, 0.5, 1.0),
    direction_fit_role: str = "direction_fit",
    readout_fit_role: str = "readout_fit",
    confirmation_role: str = "confirmation",
    class_field: str = "class_id",
    perturbation_field: str = "perturbation",
    summary_perturbations: Sequence[str] = ("blur", "gaussian_noise", "patch_mask"),
    seed: int = 1729,
) -> dict[str, Any]:
    """Layerwise, centered-space readout intervention on CFP directions.

    For every layer and perturbation the evaluation-only linear readout is fitted
    on ``readout_fit`` RAW features after subtracting their global mean (the
    CMC-style origin shift), then the own/wrong/random/raw/class-mean-axis CFP
    directions are removed from the centered ``confirmation`` features.  Each
    layer is an independent feature space; vectors are never transported across
    layers.  ``layer_arrays[layer]`` must expose ``raw``, ``clear`` and
    ``degraded`` matrices (the latter two give the CFP deltas).
    """

    from sklearn.linear_model import LogisticRegression

    from cfra.pipeline.features import _stable_seed

    selected = [str(layer) for layer in selected_layers]
    rows = list(metadata)
    all_index = np.arange(len(rows))
    roles = np.asarray([_field(row, "role") for row in rows])
    labels_all = np.asarray(_labels(rows, all_index, class_field)).astype(np.int64)
    perturbations_all = np.asarray(_labels(rows, all_index, perturbation_field)).astype(str)

    layers_result: dict[str, Any] = {}
    for layer in selected:
        arrays = layer_arrays[layer]
        raw = np.asarray(arrays["raw"], dtype=np.float64)
        clear = np.asarray(arrays["clear"], dtype=np.float64)
        degraded = np.asarray(arrays["degraded"], dtype=np.float64)
        if raw.shape[0] != len(rows) or clear.shape[0] != len(rows):
            raise LayerwiseError(f"layer {layer!r} arrays do not align with metadata")

        per_perturbation: dict[str, Any] = {}
        for perturbation in [str(value) for value in dict.fromkeys(perturbations_all.tolist())]:
            direction_fit = np.flatnonzero((roles == direction_fit_role) & (perturbations_all == perturbation))
            readout_fit = np.flatnonzero((roles == readout_fit_role) & (perturbations_all == perturbation))
            confirmation = np.flatnonzero((roles == confirmation_role) & (perturbations_all == perturbation))
            if min(len(direction_fit), len(readout_fit), len(confirmation)) < 3:
                continue
            deltas = _unit_rows(clear[direction_fit] - degraded[direction_fit], epsilon)
            raw_dirs, cmc_dirs, axis = _readout_class_directions(deltas, labels_all[direction_fit], epsilon)

            fit_raw = raw[readout_fit]
            global_mean = fit_raw.mean(axis=0)
            classifier = LogisticRegression(max_iter=5000, random_state=seed)
            # The frozen-feature readout is fitted with a fixed iteration budget,
            # so lbfgs emits a ConvergenceWarning on most layers; it is expected
            # and not actionable, so it is silenced rather than spamming the log.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                classifier.fit(fit_raw - global_mean, labels_all[readout_fit])
            classes = np.asarray(classifier.classes_).astype(np.int64)
            centroids = np.stack([(fit_raw - global_mean)[labels_all[readout_fit] == c].mean(axis=0) for c in classes.tolist()])
            centroids_unit = _unit_rows(centroids, epsilon)

            eval_raw = raw[confirmation] - global_mean
            eval_labels = labels_all[confirmation]
            keep = np.isin(eval_labels, classes) & np.isin(eval_labels, list(raw_dirs.keys()))
            eval_raw, eval_labels = eval_raw[keep], eval_labels[keep]
            if len(eval_labels) < 3:
                continue
            common = [int(c) for c in classes.tolist() if int(c) in raw_dirs]
            common_positions = [index for index, c in enumerate(classes.tolist()) if int(c) in raw_dirs]

            base_lin_acc, base_lin_margin = _margin_accuracy(_decision_matrix(classifier, eval_raw), classes, eval_labels)
            base_proto_acc, base_proto_margin = _margin_accuracy(
                _unit_rows(eval_raw, epsilon) @ centroids_unit.T, classes, eval_labels
            )

            # Nearest wrong class is chosen among the comparable classes only, so
            # every entry is guaranteed to exist in the raw and CMC direction maps.
            wrong = _nearest_wrong(centroids_unit[common_positions], classes[common_positions])
            # A per-layer, per-perturbation random control: reusing one seed
            # everywhere would hand every layer and perturbation the same unit
            # vector, which is not an independent null.
            rng = np.random.default_rng(_stable_seed("E06_readout_random", seed, layer, perturbation))
            random_dirs = {c: _unit_rows(rng.standard_normal((1, eval_raw.shape[1])), epsilon)[0] for c in common}
            conditions = {
                "own": {c: cmc_dirs[c] for c in common},
                "wrong": {c: cmc_dirs[wrong[c]] for c in common},
                "wrong_raw": {c: raw_dirs[wrong[c]] for c in common},
                "random": random_dirs,
                "own_raw": {c: raw_dirs[c] for c in common},
                "class_mean_axis": {c: axis for c in common},
            }

            drop_kwargs = {
                "classifier": classifier,
                "classes": classes,
                "centroids_unit": centroids_unit,
                "base_linear_accuracy": base_lin_acc,
                "base_linear_margin": base_lin_margin,
                "base_prototype_accuracy": base_proto_acc,
                "base_prototype_margin": base_proto_margin,
                "epsilon": epsilon,
            }

            perturbation_result: dict[str, Any] = {
                "comparable_classes": common,
                "confirmation_objects": len(eval_labels),
                "base_linear_accuracy": base_lin_acc,
                "base_linear_margin": base_lin_margin,
                "base_prototype_accuracy": base_proto_acc,
                "base_prototype_margin": base_proto_margin,
                "alpha": {},
            }
            for name, direction_map in conditions.items():
                if not direction_map:
                    continue
                alphas_here = [float(a) for a in alphas] if name in ("own", "wrong") else [float(alphas[-1])]
                for alpha in alphas_here:
                    perturbation_result["alpha"][f"{name}@{alpha:g}"] = _removal_drops(
                        eval_raw, eval_labels, direction_map, alpha, True, **drop_kwargs
                    )
            perturbation_result["alpha"]["own@1_norm_off"] = _removal_drops(
                eval_raw, eval_labels, conditions["own"], 1.0, False, **drop_kwargs
            )
            per_perturbation[perturbation] = perturbation_result

        layers_result[layer] = {
            "layer": layer,
            "representation": dict(layer_arrays[layer].get("representation", {})),
            "dimension": int(np.asarray(layer_arrays[layer]["raw"]).shape[1]),
            "by_perturbation": per_perturbation,
            "mean": _mean_readout_across_perturbations(
                per_perturbation,
                {key: value for key, value in per_perturbation.items() if key in summary_perturbations}
                or per_perturbation,
            ),
            "mean_all_perturbations": _mean_readout_across_perturbations(per_perturbation),
        }

    block_layers = [layer for layer in _ordered_block_layers(selected) if layer in layers_result]
    trend = _readout_own_wrong_trend(block_layers, layers_result)
    return {
        "stage": "E06_readout",
        "selected_layers": selected,
        "block_layers": block_layers,
        "layers": layers_result,
        "summary": {"own_minus_wrong_linear_margin_drop_by_layer": trend},
    }


__all__ = [
    "FeatureRecord",
    "LayerwiseError",
    "LayerwiseFeatureTable",
    "coerce_feature_record",
    "load_layerwise_feature_artifact",
    "run_e06",
    "run_e06_readout",
]
