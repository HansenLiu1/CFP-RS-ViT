"""Pure-NumPy E08 image-disjoint ROI retrieval.

The retrieval implementation has three independent data boundaries:

* ``direction_fit`` rows fit CFP class directions;
* ``readout_fit`` rows fit raw prototypes and CAV-like readouts;
* confirmation ``query`` and ``candidate`` rows are scored once, with class
  equality used only to build evaluation ground truth.

Candidate selection never consults a GT field.  Query and candidate source
images must be disjoint, as must both fit roles and confirmation images.  A
text baseline is represented by the small ``TextBaseline`` protocol below; the
pipeline does not import or call a model text encoder.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from cfra.evaluation import retrieval_metrics, retrieval_metrics_per_query
from cfra.methods.operators import fit_residual_operator
from cfra.statistics import clustered_bootstrap_multi

logger = logging.getLogger("cfra")

from .layerwise import FeatureRecord, LayerwiseError, _field, _labels, _source_ids, coerce_feature_record


class RetrievalError(ValueError):
    """Raised when an E08 retrieval input violates its split or metric contract."""


class TextBaseline(Protocol):
    """Canonical pluggable text-baseline interface.

    ``fit`` receives only the readout-fit rows and may return arbitrary frozen
    state.  ``score`` must return a finite ``[query_count, candidate_count]``
    matrix.  Implementations may instead expose only a callable scorer, in
    which case no text encoder is created by this module.
    """

    def fit(self, features: np.ndarray, labels: np.ndarray, metadata: Sequence[Any]) -> Any:
        ...

    def score(
        self,
        query_features: np.ndarray,
        candidate_features: np.ndarray,
        query_metadata: Sequence[Any],
        candidate_metadata: Sequence[Any],
        fitted: Any,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class TextBaselineSpec:
    """Function-based adapter for inserting an external text baseline."""

    score: Callable[..., Any]
    fit: Callable[..., Any] | None = None


_METHOD_ALIASES = {
    "raw": "raw_prototype",
    "prototype": "raw_prototype",
    "raw_prototype": "raw_prototype",
    "cfp": "cfp_direction",
    "cfp_direction": "cfp_direction",
    "direction": "cfp_direction",
    "cfp_direction_raw": "cfp_direction_raw",
    "cfp_raw_direction": "cfp_direction_raw",
    "cfp_matched": "cfp_direction_matched",
    "cfp_direction_matched": "cfp_direction_matched",
    "cfp_raw": "cfp_direction_matched",
    "cav": "cav_like",
    "cav_like": "cav_like",
    "random": "random",
    "random_baseline": "random",
    "text_baseline": "text_baseline",
    "text": "text_baseline",
}


def _as_bool_mask(value: Any, count: int, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 1 or len(mask) != count or mask.dtype.kind not in "biu":
        raise RetrievalError(f"{name} must be a one-dimensional boolean mask aligned with metadata")
    return mask.astype(bool, copy=False)


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RetrievalError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise RetrievalError(f"{name} must be a finite positive number")
    return result


def _array_from(table: Any, key: str, count: int) -> np.ndarray | None:
    mapping = table if isinstance(table, Mapping) else None
    arrays = mapping.get("arrays") if mapping is not None else getattr(table, "arrays", None)
    candidate: Any = None
    if mapping is not None and key in mapping:
        candidate = mapping[key]
    elif isinstance(arrays, Mapping) and key in arrays:
        candidate = arrays[key]
    else:
        nested = None
        if mapping is not None:
            for container_name in ("features", "layer_features", "embeddings"):
                if isinstance(mapping.get(container_name), Mapping):
                    nested = mapping[container_name]
                    break
        if nested is None and isinstance(arrays, Mapping):
            for container_name in ("features", "layer_features", "embeddings"):
                if isinstance(arrays.get(container_name), Mapping):
                    nested = arrays[container_name]
                    break
        if nested is not None and key in nested:
            candidate = nested[key]
    if candidate is None:
        return None
    array = np.asarray(candidate, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != count:
        raise RetrievalError(f"feature array {key!r} must have shape [n, dimension] aligned with metadata")
    if not np.isfinite(array).all():
        raise RetrievalError(f"feature array {key!r} contains non-finite values")
    return array


def _record_with_fallback(table: Any, feature_key: str) -> FeatureRecord:
    try:
        return coerce_feature_record(table, feature_key=feature_key)
    except LayerwiseError as first_error:
        if feature_key != "raw":
            raise RetrievalError(str(first_error)) from first_error
        try:
            return coerce_feature_record(table, feature_key=None)
        except LayerwiseError as exc:
            raise RetrievalError(str(exc)) from exc


def _labels_for_rows(metadata: Sequence[Any], indices: np.ndarray, field: str) -> np.ndarray:
    try:
        return _labels(metadata, indices, field)
    except LayerwiseError as exc:
        raise RetrievalError(str(exc)) from exc


def _source_ids_for_rows(metadata: Sequence[Any], indices: np.ndarray) -> np.ndarray:
    try:
        return _source_ids(metadata, indices)
    except LayerwiseError as exc:
        raise RetrievalError(str(exc)) from exc


def _unique(values: np.ndarray) -> list[Any]:
    unique: list[Any] = []
    for value in values.tolist():
        if not any(value == existing for existing in unique):
            unique.append(value)
    try:
        return sorted(unique)
    except (TypeError, ValueError):
        return unique


def _fit_unit_directions(features: np.ndarray, labels: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    classes = _unique(labels)
    directions = []
    for class_id in classes:
        direction = features[labels == class_id].mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= epsilon:
            raise RetrievalError(f"cannot fit a CFP direction for class {class_id!r}")
        directions.append(direction / norm)
    return np.stack(directions), np.asarray(classes)


def _fit_raw_prototype_directions(
    features: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the repository's centered raw-feature prototype baseline."""

    classes = _unique(labels)
    global_mean = features.mean(axis=0)
    directions = []
    for class_id in classes:
        direction = features[labels == class_id].mean(axis=0) - global_mean
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= epsilon:
            raise RetrievalError(f"cannot fit a raw prototype direction for class {class_id!r}")
        directions.append(direction / norm)
    return np.stack(directions), np.asarray(classes)


def _cosine_profiles(features: np.ndarray, directions: np.ndarray, epsilon: float) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1)
    direction_norms = np.linalg.norm(directions, axis=1)
    if np.any(direction_norms <= epsilon):
        raise RetrievalError("readout directions contain a near-zero vector")
    valid = norms > epsilon
    profiles = np.full((len(features), len(directions)), np.nan, dtype=np.float64)
    if valid.any():
        profiles[valid] = (features[valid] / norms[valid, None]) @ (directions / direction_norms[:, None]).T
    return profiles


def _profile_similarity(query_profiles: np.ndarray, candidate_profiles: np.ndarray, epsilon: float) -> np.ndarray:
    query_norms = np.linalg.norm(query_profiles, axis=1)
    candidate_norms = np.linalg.norm(candidate_profiles, axis=1)
    result = np.zeros((len(query_profiles), len(candidate_profiles)), dtype=np.float64)
    query_valid = query_norms > epsilon
    candidate_valid = candidate_norms > epsilon
    if query_valid.any() and candidate_valid.any():
        result[np.ix_(query_valid, candidate_valid)] = (
            query_profiles[query_valid] / query_norms[query_valid, None]
        ) @ (candidate_profiles[candidate_valid] / candidate_norms[candidate_valid, None]).T
    return result


def _fit_ridge_cav(features: np.ndarray, labels: np.ndarray, epsilon: float, regularization: float) -> tuple[np.ndarray, np.ndarray]:
    classes = _unique(labels)
    augmented = np.column_stack([features, np.ones(len(features), dtype=np.float64)])
    gram = augmented.T @ augmented
    penalty = np.eye(gram.shape[0], dtype=np.float64) * regularization
    penalty[-1, -1] = 0.0
    weights = []
    for class_id in classes:
        target = np.where(labels == class_id, 1.0, -1.0)
        try:
            coefficient = np.linalg.solve(gram + penalty, augmented.T @ target)
        except np.linalg.LinAlgError:
            coefficient = np.linalg.lstsq(gram + penalty, augmented.T @ target, rcond=None)[0]
        if not np.isfinite(coefficient).all():
            raise RetrievalError(f"CAV-like readout for class {class_id!r} is non-finite")
        weights.append(coefficient)
    coefficient_matrix = np.stack(weights)
    return coefficient_matrix[:, :-1], coefficient_matrix[:, -1]


def _readout_scores(features: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return features @ weights.T + bias[None, :]


def _retrieval_role(row: Any) -> str | None:
    for key in ("retrieval_role", "retrieval_split", "gallery_role", "partition", "split", "set"):
        value = _field(row, key)
        if value is not None:
            return str(value).strip().lower()
    role = _field(row, "role")
    if role is not None:
        return str(role).strip().lower()
    return None


def _role_mask(
    metadata: Sequence[Any],
    *,
    target: str,
    explicit: Any | None,
    count: int,
) -> np.ndarray:
    if explicit is not None:
        return _as_bool_mask(explicit, count, f"{target}_mask")
    if target == "candidate":
        flag_names = ("is_candidate", "candidate", "is_gallery", "gallery")
        names = {"candidate", "candidates", "gallery", "database", "confirmation_candidate", "confirmation_candidates"}
    else:
        flag_names = ("is_query", "query", "is_probe")
        names = {"query", "queries", "probe", "confirmation_query", "confirmation_queries"}
    flags = []
    has_flag = False
    for row in metadata:
        flag = None
        for key in flag_names:
            value = _field(row, key)
            if value is not None:
                flag = bool(value)
                has_flag = True
                break
        flags.append(flag)
    if has_flag:
        return np.asarray(
            [
                bool(value) if value is not None else _retrieval_role(row) in names
                for row, value in zip(metadata, flags)
            ],
            dtype=bool,
        )
    return np.asarray([_retrieval_role(row) in names for row in metadata], dtype=bool)


def _confirmation_mask(metadata: Sequence[Any], indices: np.ndarray, fit_roles: set[str]) -> np.ndarray:
    values = []
    for index in indices.tolist():
        role = _field(metadata[index], "role")
        if role is None:
            values.append(True)
        else:
            values.append(str(role).strip().lower() not in fit_roles and str(role).strip().lower() != "validation")
    return np.asarray(values, dtype=bool)


def _validate_disjoint(
    metadata: Sequence[Any],
    direction_indices: np.ndarray,
    readout_indices: np.ndarray,
    query_indices: np.ndarray,
    candidate_indices: np.ndarray,
) -> dict[str, Any]:
    if len(direction_indices) == 0:
        raise RetrievalError("no direction_fit rows are available")
    if len(readout_indices) == 0:
        raise RetrievalError("no readout_fit rows are available")
    if len(query_indices) == 0:
        raise RetrievalError("no confirmation query rows are available")
    if len(candidate_indices) == 0:
        raise RetrievalError("no confirmation candidate rows are available")
    if np.intersect1d(query_indices, candidate_indices).size:
        raise RetrievalError("query and candidate rows must be disjoint")
    direction_images = _source_ids_for_rows(metadata, direction_indices)
    readout_images = _source_ids_for_rows(metadata, readout_indices)
    query_images = _source_ids_for_rows(metadata, query_indices)
    candidate_images = _source_ids_for_rows(metadata, candidate_indices)
    for left, right, message in (
        (direction_images, readout_images, "direction_fit and readout_fit must be image-disjoint"),
        (direction_images, query_images, "direction_fit and confirmation must be image-disjoint"),
        (direction_images, candidate_images, "direction_fit and confirmation must be image-disjoint"),
        (readout_images, query_images, "readout_fit and confirmation must be image-disjoint"),
        (readout_images, candidate_images, "readout_fit and confirmation must be image-disjoint"),
        (query_images, candidate_images, "query and candidate source images must be disjoint"),
    ):
        overlap = set(left.tolist()) & set(right.tolist())
        if overlap:
            preview = sorted(overlap, key=str)[:3]
            raise RetrievalError(f"{message}; overlapping images include {preview}")
    return {
        "direction_fit_source_image_count": len(set(direction_images.tolist())),
        "readout_fit_source_image_count": len(set(readout_images.tolist())),
        "query_source_image_count": len(set(query_images.tolist())),
        "candidate_source_image_count": len(set(candidate_images.tolist())),
        "image_disjoint": True,
    }


def _get_text_member(baseline: Any, name: str) -> Callable[..., Any] | None:
    if isinstance(baseline, TextBaselineSpec):
        value = getattr(baseline, name)
    elif isinstance(baseline, Mapping):
        value = baseline.get(name)
    else:
        value = getattr(baseline, name, None)
    return value if callable(value) else None


def _invoke_options(function: Callable[..., Any], options: Sequence[tuple[Any, ...]]) -> Any:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        for option in options:
            try:
                signature.bind(*option)
            except TypeError:
                continue
            return function(*option)
    last_error: TypeError | None = None
    for option in options:
        try:
            return function(*option)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RetrievalError("baseline function has no supported call signature")


def _text_score_matrix(
    baseline: Any,
    *,
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    fit_metadata: Sequence[Any],
    query_features: np.ndarray,
    candidate_features: np.ndarray,
    query_metadata: Sequence[Any],
    candidate_metadata: Sequence[Any],
) -> np.ndarray:
    fit_function = _get_text_member(baseline, "fit")
    score_function = _get_text_member(baseline, "score")
    if score_function is None and callable(baseline):
        score_function = baseline
    if score_function is None:
        raise RetrievalError("text_baseline must be callable or expose a score method")
    fitted = None
    if fit_function is not None:
        fitted = _invoke_options(
            fit_function,
            (
                (fit_features, fit_labels, fit_metadata),
                (fit_features, fit_labels),
                (),
            ),
        )
    scores = _invoke_options(
        score_function,
        (
            (query_features, candidate_features, query_metadata, candidate_metadata, fitted),
            (query_features, candidate_features, query_metadata, candidate_metadata),
            (query_features, candidate_features, fitted),
            (query_features, candidate_features),
        ),
    )
    if isinstance(scores, Mapping) and "scores" in scores:
        scores = scores["scores"]
    matrix = np.asarray(scores, dtype=np.float64)
    expected = (len(query_features), len(candidate_features))
    if matrix.shape != expected:
        raise RetrievalError(f"text baseline scores must have shape {expected}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise RetrievalError("text baseline scores contain non-finite values")
    return matrix


def _bootstrap_metrics(
    relevance: np.ndarray,
    scores: np.ndarray,
    query_source_images: np.ndarray,
    *,
    k: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    positions = np.arange(len(relevance), dtype=np.int64)
    metric_names = (
        "macro_ap",
        "ndcg_at_k",
        "recall_at_k",
        "precision_at_k",
    )
    # Every macro metric is a ``nanmean`` over independent per-query values, so
    # the per-row metrics are computed once and each replicate only resamples
    # and averages fixed scalars.  This is bit-identical to recomputing
    # ``retrieval_metrics`` on the resampled matrix, but avoids re-sorting the
    # full query x candidate matrix once per replicate per method.
    per_query = retrieval_metrics_per_query(relevance, scores, k)

    if per_query is not None:

        def statistic(
            sampled_positions: np.ndarray,
            _sampled_labels: np.ndarray,
            _sampled_images: np.ndarray,
        ) -> dict[str, float]:
            result: dict[str, float] = {}
            for name in metric_names:
                sampled = per_query[name][sampled_positions]
                result[name] = (
                    float(np.nanmean(sampled))
                    if np.isfinite(sampled).any()
                    else float("nan")
                )
            return result

    else:

        def statistic(
            sampled_positions: np.ndarray,
            _sampled_labels: np.ndarray,
            _sampled_images: np.ndarray,
        ) -> dict[str, float]:
            values = retrieval_metrics(
                relevance[sampled_positions], scores[sampled_positions], k=k
            )
            return {name: float(values[name]) for name in metric_names}

    multi = clustered_bootstrap_multi(
        positions,
        query_source_images,
        statistic,
        replicates=replicates,
        seed=seed,
        relabel_clusters=True,
    )
    output: dict[str, Any] = {}
    for output_name, metric_name in zip(metric_names, metric_names):
        item = multi["metrics"][metric_name]
        output[output_name] = {
            "estimate": item["estimate"],
            "ci_low": item["ci_low"],
            "ci_high": item["ci_high"],
            "confidence_level": item["confidence_level"],
            "replicates": item["replicates"],
            "source_image_count": item["source_image_count"],
            "observation_count": item["observation_count"],
            "seed": item["seed"],
        }
    return output


def _method_metrics(
    method: str,
    relevance: np.ndarray,
    scores: np.ndarray,
    *,
    k: int,
    query_source_images: np.ndarray,
    bootstrap_replicates: int | None,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values = retrieval_metrics(relevance, scores, k=k)
    result: dict[str, Any] = {
        "method": method,
        **values,
        "scores": scores,
        "relevance": relevance,
    }
    if bootstrap_replicates is not None and bootstrap_replicates > 0:
        bootstrap = _bootstrap_metrics(
            relevance,
            scores,
            query_source_images,
            k=k,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        result["bootstrap"] = bootstrap
        for metric_name, bootstrap_result in bootstrap.items():
            result[f"{metric_name}_ci"] = (bootstrap_result["ci_low"], bootstrap_result["ci_high"])
            result[f"{metric_name}_ci_low"] = bootstrap_result["ci_low"]
            result[f"{metric_name}_ci_high"] = bootstrap_result["ci_high"]
    return result


def run_e08(
    table: Any,
    *,
    dataset_id: str | None = None,
    perturbation: str | None = None,
    feature_key: str = "raw",
    direction_feature_key: str | None = None,
    readout_feature_key: str | None = None,
    candidate_mask: Any | None = None,
    query_mask: Any | None = None,
    class_field: str = "class_id",
    methods: Sequence[str] = ("raw_prototype", "cfp_direction", "cav_like", "random"),
    k: int = 10,
    epsilon: float = 1.0e-8,
    cav_regularization: float = 1.0e-3,
    random_seed: int = 1729,
    text_baseline: Any | None = None,
    bootstrap_ci: bool | None = None,
    bootstrap_replicates: int | None = None,
    bootstrap_seed: int = 1729,
    bootstrap_unit: str = "source_image",
) -> dict[str, Any]:
    """Run image-disjoint ROI retrieval against registered NumPy baselines.

    ``candidate_mask`` and ``query_mask`` are preferred for machine-generated
    tables.  Without them, rows are selected from ``retrieval_role`` (or the
    boolean ``is_candidate``/``is_query`` fields), accepting ``candidate`` and
    ``query`` aliases.  Confirmation GT is derived from ``class_field`` after
    both masks have been fixed and is never used to choose candidates.

    ``direction_feature_key`` may point to a precomputed delta.  When omitted,
    ``delta`` is used if present, then ``true_clear_degraded -
    true_degraded_degraded`` is derived, and finally the raw feature matrix is
    used.  This fallback makes synthetic and existing ``FeatureTable`` inputs
    work without a model text encoder or sklearn.
    """

    e08_started = time.time()
    logger.info(
        f"[E08] 开始: dataset={dataset_id} perturbation={perturbation} methods={list(methods)} k={k}"
    )
    epsilon = _positive_float(epsilon, "epsilon")
    cav_regularization = _positive_float(cav_regularization, "cav_regularization")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or int(k) <= 0:
        raise RetrievalError("k must be a positive integer")
    if bootstrap_unit not in {"source_image", "image", "image_cluster"}:
        raise RetrievalError("bootstrap_unit must be source_image")
    if bootstrap_replicates is not None and (
        not isinstance(bootstrap_replicates, (int, np.integer)) or isinstance(bootstrap_replicates, bool) or int(bootstrap_replicates) < 0
    ):
        raise RetrievalError("bootstrap_replicates must be a non-negative integer or None")
    if bootstrap_ci is not None and not isinstance(bootstrap_ci, (bool, np.bool_)):
        raise RetrievalError("bootstrap_ci must be boolean when provided")
    if bootstrap_ci and bootstrap_replicates is None:
        bootstrap_replicates = 5000
    if isinstance(methods, (str, bytes)):
        raise RetrievalError("methods must be a sequence of method names")
    canonical_methods: list[str] = []
    for method in methods:
        try:
            canonical = _METHOD_ALIASES[str(method).lower()]
        except KeyError as exc:
            raise RetrievalError(f"unknown retrieval method: {method!r}") from exc
        if canonical not in canonical_methods:
            canonical_methods.append(canonical)
    if text_baseline is not None and "text_baseline" not in canonical_methods:
        canonical_methods.append("text_baseline")
    if not canonical_methods:
        raise RetrievalError("methods must contain at least one method")

    record = _record_with_fallback(table, feature_key)
    metadata = record.metadata
    count = len(metadata)
    raw_features = record.features
    direction_features = raw_features
    if direction_feature_key is not None:
        selected = _array_from(table, direction_feature_key, count)
        if selected is None:
            raise RetrievalError(f"direction feature array {direction_feature_key!r} was not found")
        direction_features = selected
    else:
        selected = _array_from(table, "delta", count)
        if selected is not None:
            direction_features = selected
        else:
            clear = _array_from(table, "true_clear_degraded", count)
            degraded = _array_from(table, "true_degraded_degraded", count)
            if clear is not None or degraded is not None:
                if clear is None or degraded is None or clear.shape != degraded.shape:
                    raise RetrievalError("paired direction arrays must both be present with matching shapes")
                direction_features = clear - degraded
    readout_features = raw_features
    if readout_feature_key is not None:
        selected = _array_from(table, readout_feature_key, count)
        if selected is None:
            raise RetrievalError(f"readout feature array {readout_feature_key!r} was not found")
        readout_features = selected

    dataset_values = []
    for row in metadata:
        value = _field(row, "dataset_id")
        dataset_values.append(None if value is None else str(value))
    dataset_mask = np.asarray(
        [dataset_id is None or value == str(dataset_id) for value in dataset_values],
        dtype=bool,
    )
    if dataset_id is not None and not dataset_mask.any():
        raise RetrievalError(f"no rows are available for dataset_id={dataset_id!r}")

    perturbation_values = []
    for row in metadata:
        value = _field(row, "perturbation")
        perturbation_values.append(None if value is None else str(value))
    perturbation_mask = np.asarray(
        [perturbation is None or value == str(perturbation) for value in perturbation_values],
        dtype=bool,
    )
    if perturbation is not None and not perturbation_mask.any():
        raise RetrievalError(f"no rows are available for perturbation={perturbation!r}")
    analysis_mask = dataset_mask & perturbation_mask

    role_values = []
    for row in metadata:
        role = _field(row, "role")
        role_values.append("" if role is None else str(role).strip().lower())
    roles = np.asarray(role_values)
    direction_indices = np.flatnonzero((roles == "direction_fit") & analysis_mask)
    readout_indices = np.flatnonzero((roles == "readout_fit") & analysis_mask)
    fit_roles = {"direction_fit", "readout_fit", "validation"}
    candidate_all = _role_mask(metadata, target="candidate", explicit=candidate_mask, count=count)
    query_all = _role_mask(metadata, target="query", explicit=query_mask, count=count)
    confirmation_all = _confirmation_mask(metadata, np.arange(count), fit_roles)
    candidate_indices = np.flatnonzero(candidate_all & confirmation_all & analysis_mask)
    query_indices = np.flatnonzero(query_all & confirmation_all & analysis_mask)
    if len(query_indices) == 0 and len(candidate_indices) > 0:
        query_indices = np.flatnonzero((~candidate_all) & confirmation_all & analysis_mask)
    split_summary = _validate_disjoint(metadata, direction_indices, readout_indices, query_indices, candidate_indices)

    direction_fit_labels = _labels_for_rows(metadata, direction_indices, class_field)
    readout_fit_labels = _labels_for_rows(metadata, readout_indices, class_field)
    query_labels = _labels_for_rows(metadata, query_indices, class_field)
    candidate_labels = _labels_for_rows(metadata, candidate_indices, class_field)
    # This is the only GT construction point; candidate indices were already fixed above.
    relevance = (query_labels[:, None] == candidate_labels[None, :]).astype(np.float64)
    query_source_images = _source_ids_for_rows(metadata, query_indices)

    direction_classes = np.asarray(_unique(direction_fit_labels))
    readout_classes = np.asarray(_unique(readout_fit_labels))
    if not np.array_equal(direction_classes, readout_classes):
        raise RetrievalError("direction_fit and readout_fit must contain the same class set")

    direction_directions: np.ndarray | None = None
    raw_direction_directions: np.ndarray | None = None
    cfp_operator = None
    prototype_directions: np.ndarray | None = None
    cav_weights: np.ndarray | None = None
    cav_bias: np.ndarray | None = None
    if "cfp_direction" in canonical_methods or "cfp_direction_raw" in canonical_methods:
        direction_fit_deltas = direction_features[direction_indices]
        raw_direction_directions, direction_classes = _fit_unit_directions(
            direction_fit_deltas, direction_fit_labels, epsilon
        )
    if "cfp_direction" in canonical_methods:
        # CMC/M1-residualized delta direction.  The origin is the mean of the
        # per-class unit directions (exactly as the runner fits
        # ``euclidean_mean_subtraction``), not the mean of the per-object deltas;
        # the same origin subtracts from the query/candidate delta profiles.
        assert raw_direction_directions is not None
        cfp_operator = fit_residual_operator(
            "euclidean_mean_subtraction", raw_direction_directions, epsilon=epsilon
        )
        transformed_fit = cfp_operator.transform(direction_fit_deltas)
        direction_directions, direction_classes = _fit_unit_directions(
            transformed_fit, direction_fit_labels, epsilon
        )
    if "raw_prototype" in canonical_methods:
        prototype_directions, prototype_classes = _fit_raw_prototype_directions(
            readout_features[readout_indices], readout_fit_labels, epsilon
        )
        if not np.array_equal(np.asarray(direction_classes), np.asarray(prototype_classes)):
            raise RetrievalError("direction_fit and readout_fit must contain the same class set")
    if "cav_like" in canonical_methods:
        cav_weights, cav_bias = _fit_ridge_cav(
            readout_features[readout_indices],
            readout_fit_labels,
            epsilon,
            cav_regularization,
        )

    score_matrices: dict[str, np.ndarray] = {}
    fit_class_count = len(direction_classes)
    if "raw_prototype" in canonical_methods:
        assert prototype_directions is not None
        query_profile = _cosine_profiles(readout_features[query_indices], prototype_directions, epsilon)
        candidate_profile = _cosine_profiles(readout_features[candidate_indices], prototype_directions, epsilon)
        score_matrices["raw_prototype"] = _profile_similarity(query_profile, candidate_profile, epsilon)
    if "cfp_direction" in canonical_methods:
        assert direction_directions is not None and cfp_operator is not None
        query_profile = _cosine_profiles(
            cfp_operator.transform(direction_features[query_indices]), direction_directions, epsilon
        )
        candidate_profile = _cosine_profiles(
            cfp_operator.transform(direction_features[candidate_indices]), direction_directions, epsilon
        )
        score_matrices["cfp_direction"] = _profile_similarity(query_profile, candidate_profile, epsilon)
    if "cfp_direction_raw" in canonical_methods:
        assert raw_direction_directions is not None
        query_profile = _cosine_profiles(direction_features[query_indices], raw_direction_directions, epsilon)
        candidate_profile = _cosine_profiles(direction_features[candidate_indices], raw_direction_directions, epsilon)
        score_matrices["cfp_direction_raw"] = _profile_similarity(query_profile, candidate_profile, epsilon)
    # Matched raw-feature CFP scoring: directions are still fitted on the
    # uncentered perturbation-delta direction set, but query and gallery profiles
    # are both computed in the shared raw ROI feature space, so the three
    # direction families (CFP / raw prototype / raw-feature CAV) are compared
    # under one scoring space instead of a mismatched delta-profile vs
    # raw-profile mix.
    if raw_direction_directions is not None and len(readout_features):
        query_profile = _cosine_profiles(readout_features[query_indices], raw_direction_directions, epsilon)
        candidate_profile = _cosine_profiles(readout_features[candidate_indices], raw_direction_directions, epsilon)
        score_matrices["cfp_direction_matched"] = _profile_similarity(query_profile, candidate_profile, epsilon)
    if "cav_like" in canonical_methods:
        assert cav_weights is not None and cav_bias is not None
        query_profile = _readout_scores(readout_features[query_indices], cav_weights, cav_bias)
        candidate_profile = _readout_scores(readout_features[candidate_indices], cav_weights, cav_bias)
        score_matrices["cav_like"] = _profile_similarity(query_profile, candidate_profile, float(epsilon))
    if "random" in canonical_methods:
        rng = np.random.default_rng(random_seed)
        random_projection = rng.normal(size=(readout_features.shape[1], fit_class_count))
        random_projection /= np.maximum(np.linalg.norm(random_projection, axis=0, keepdims=True), float(epsilon))
        query_profile = readout_features[query_indices] @ random_projection
        candidate_profile = readout_features[candidate_indices] @ random_projection
        score_matrices["random"] = _profile_similarity(query_profile, candidate_profile, float(epsilon))
    if "text_baseline" in canonical_methods:
        if text_baseline is None:
            raise RetrievalError("text_baseline method was requested but no text_baseline was supplied")
        expected_space = getattr(text_baseline, "shared_space_id", None)
        actual_space = None
        if isinstance(record.representation, Mapping):
            actual_space = record.representation.get("shared_space_id")
        if expected_space is not None and actual_space != expected_space:
            raise RetrievalError(
                f"text/image shared-space mismatch: image={actual_space!r}, text={expected_space!r}"
            )
        score_matrices["text_baseline"] = _text_score_matrix(
            text_baseline,
            fit_features=readout_features[readout_indices],
            fit_labels=readout_fit_labels,
            fit_metadata=tuple(metadata[index] for index in readout_indices.tolist()),
            query_features=readout_features[query_indices],
            candidate_features=readout_features[candidate_indices],
            query_metadata=tuple(metadata[index] for index in query_indices.tolist()),
            candidate_metadata=tuple(metadata[index] for index in candidate_indices.tolist()),
        )

    method_results: dict[str, dict[str, Any]] = {}
    for method in canonical_methods:
        method_results[method] = _method_metrics(
            method,
            relevance,
            score_matrices[method],
            k=int(k),
            query_source_images=query_source_images,
            bootstrap_replicates=None if bootstrap_replicates is None else int(bootstrap_replicates),
            bootstrap_seed=int(bootstrap_seed),
        )
    if "text_baseline" in method_results:
        # The baseline reports whether its shared-space prototypes/features were
        # modality-centered; the exporter can separate centered from raw rows.
        method_results["text_baseline"]["space"] = getattr(text_baseline, "space", None)
    # The matched raw-scored CFP variant is derived gate-free from the same
    # fitted directions; it completes the retrieval panel and is exported
    # alongside the requested methods whenever its direction bank exists.
    if "cfp_direction_matched" in score_matrices:
        method_results["cfp_direction_matched"] = _method_metrics(
            "cfp_direction_matched",
            relevance,
            score_matrices["cfp_direction_matched"],
            k=int(k),
            query_source_images=query_source_images,
            bootstrap_replicates=None if bootstrap_replicates is None else int(bootstrap_replicates),
            bootstrap_seed=int(bootstrap_seed),
        )
    rows = [
        {
            key: value
            for key, value in result.items()
            if key not in {"scores", "relevance", "bootstrap"}
        }
        for result in method_results.values()
    ]
    logger.info(
        f"[E08] 完成 (耗时 {time.time() - e08_started:.1f}s): "
        f"query={len(query_indices)} candidate={len(candidate_indices)} "
        f"methods={sorted(method_results)}"
    )
    method_results_compact = {
        method: {
            key: value
            for key, value in result.items()
            if key not in {"scores", "relevance"}
        }
        for method, result in method_results.items()
    }
    return {
        "stage": "E08",
        "model_id": record.model_id,
        "dataset_id": dataset_id,
        "perturbation": perturbation,
        "feature_key": feature_key,
        "methods": method_results_compact,
        "metrics": method_results_compact,
        "results": method_results_compact,
        "rows": rows,
        "k": int(k),
        "query_count": len(query_indices),
        "candidate_count": len(candidate_indices),
        "fit_class_count": int(fit_class_count),
        "direction_class_ids": direction_classes,
        "fitted": {
            "direction": None
            if direction_directions is None
            else {"class_ids": direction_classes, "directions": direction_directions,
                  "dir_space": "perturbation_delta_M1", "score_space": "perturbation_delta_M1"},
            "direction_raw": None
            if raw_direction_directions is None
            else {"class_ids": direction_classes, "directions": raw_direction_directions,
                  "dir_space": "perturbation_delta", "score_space": "perturbation_delta"},
            "direction_matched": None
            if raw_direction_directions is None
            else {"class_ids": direction_classes, "directions": raw_direction_directions,
                  "dir_space": "perturbation_delta", "score_space": "raw_roi"},
            "raw_prototype": None
            if prototype_directions is None
            else {"class_ids": prototype_classes, "directions": prototype_directions},
            "cav_like": None
            if cav_weights is None or cav_bias is None
            else {"class_ids": readout_classes, "weights": cav_weights, "bias": cav_bias},
        },
        "candidate_indices": candidate_indices,
        "query_indices": query_indices,
        "candidate_selection_uses_ground_truth": False,
        "ground_truth": {
            "field": class_field,
            "source": "confirmation_metadata",
            "used_for": "evaluation_only",
        },
        "bootstrap_unit": "source_image",
        "bootstrap_enabled": bool(bootstrap_replicates),
        **split_summary,
    }


__all__ = ["RetrievalError", "TextBaseline", "TextBaselineSpec", "run_e08"]
