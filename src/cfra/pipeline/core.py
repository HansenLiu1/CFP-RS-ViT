"""E03-E05 model-backed readability, geometry and transfer analyses."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("cfra")

from cfra.evaluation import (
    binary_auc,
    intervene,
    macro_ovr_auc,
    macro_ovr_auc_from_scores,
    near_zero_audit,
    specificity_metrics,
    specificity_summary_metrics,
    summary_from_score_matrix,
)
from cfra.evaluation.metrics import _auc_from_scores_fast
from cfra.methods import fit_geometry_residualizer, fit_residual_operator
from cfra.methods.baselines import cav_direction, fit_clear_degraded_cav, fit_raw_prototype_directions
from cfra.pipeline.features import FeatureTable
from cfra.representations import aggregate_class_direction, cosine_scores, factorial_deltas, normalize_deltas
from cfra.runs import write_manifest_json
from cfra.statistics import benjamini_hochberg, clustered_bootstrap, paired_difference_ci, paired_sign_flip
from cfra.statistics.clustered import (
    _bootstrap_block_path,
    _load_shared_payload,
    _replicate_blocks,
    _shared_payload_map,
)


class CoreStageError(RuntimeError):
    """Raised when a core experiment cannot produce a complete result."""


_CONTROL_REGISTRY: tuple[str, ...] = (
    "shifted_obb",
    "random_geometry_matched_obb",
    "label_permutation",
    "background_only",
    "boundary_only",
    "interior_eroded",
    "mask_eroded",
    "mask_dilated",
    "geometry_regression",
)
_REGISTERED_CONTROL_NAMES: tuple[str, ...] = ("true_obb", *_CONTROL_REGISTRY)
_DEFAULT_METHOD_IDS: tuple[str, ...] = (
    "raw",
    "euclidean_mean_subtraction",
    "unit_axis_subtraction",
    "orthogonal_axis_projection",
    "direction_fit_pca_removal",
    "whitened_centering",
    "background_svd_removal",
)
_DEFAULT_NORMALIZATIONS: tuple[str, ...] = ("unit", "raw", "magnitude_weighted")


def _json_safe(value: Any) -> Any:
    """Convert NumPy-rich stage results into strict JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _statistics_settings(statistics_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve the shared source-image bootstrap contract from configuration."""

    config = dict(statistics_config or {})
    replicates = int(config.get("bootstrap_replicates", 0) or 0)
    if replicates < 0:
        raise CoreStageError("statistics.bootstrap_replicates must be non-negative")
    confidence = float(config.get("confidence_level", 0.95))
    if not 0.0 < confidence < 1.0:
        raise CoreStageError("statistics.confidence_level must be between zero and one")
    cluster_unit = str(config.get("cluster_unit", "source_image"))
    if cluster_unit not in {"source_image", "image", "image_cluster"}:
        raise CoreStageError("core bootstrap requires source-image clusters")
    permutation_replicates = int(config.get("permutation_replicates", replicates or 1))
    if permutation_replicates <= 0:
        raise CoreStageError("statistics.permutation_replicates must be positive")
    fdr_q = float(config.get("fdr_q", 0.05))
    if not 0.0 < fdr_q <= 1.0:
        raise CoreStageError("statistics.fdr_q must be in (0, 1]")
    # Bootstrap workers are read from the statistics config first (cell workers
    # pin this to 1) and otherwise from the environment.  They are deliberately
    # NOT persisted into the resolved config, so enabling parallelism cannot
    # change ``config_sha256`` / the run parameter signature and invalidate a
    # resumable offline run.
    bootstrap_workers = config.get("bootstrap_workers")
    if bootstrap_workers is None:
        bootstrap_workers = os.environ.get("CFRA_BOOTSTRAP_WORKERS", "1")
    bootstrap_workers = max(1, int(bootstrap_workers))
    return {
        "bootstrap_enabled": bool(config.get("bootstrap_ci", replicates > 0)) and replicates > 0,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": int(config.get("bootstrap_seed", 1729)),
        "bootstrap_workers": bootstrap_workers,
        "confidence_level": confidence,
        "cluster_unit": "source_image",
        "permutation_replicates": permutation_replicates,
        "fdr_method": str(config.get("fdr_method", "benjamini_hochberg")),
        "fdr_q": fdr_q,
    }


def _seed_for(*parts: Any, seed: int) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int((seed + int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")) % (2**32 - 1))


def _fit_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int,
    random_state: int,
) -> Any:
    """Fit an evaluation readout while silencing the expected lbfgs notice.

    The probes are fitted on frozen features with a fixed iteration budget, so
    scikit-learn's "iterations reached limit" ``ConvergenceWarning`` fires on
    every fit.  It is a deterministic property of the frozen features rather
    than an actionable error, and E07 refits the readout many times per cell, so
    the notice is suppressed at the call site to keep the run log readable.  The
    fitted coefficients are unaffected.
    """

    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(max_iter=max_iter, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(features, labels)
    return classifier


def _bootstrap_scalar(
    values: np.ndarray,
    source_images: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    settings: Mapping[str, Any],
    seed_parts: Sequence[Any] = (),
) -> dict[str, Any]:
    """Bootstrap a scalar with complete source-image clusters, or return an explicit disabled result."""

    if not bool(settings["bootstrap_enabled"]):
        return {
            "status": "disabled",
            "estimate": float(statistic(values, source_images)) if len(values) else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "confidence_level": float(settings["confidence_level"]),
            "replicates": 0,
            "source_image_count": len(set(source_images.tolist())) if len(source_images) else 0,
        }
    return clustered_bootstrap(
        values,
        source_image_ids=source_images,
        statistic=statistic,
        replicates=int(settings["bootstrap_replicates"]),
        seed=_seed_for(*seed_parts, seed=int(settings["bootstrap_seed"])),
        confidence=float(settings["confidence_level"]),
        relabel_clusters=True,
        workers=int(settings.get("bootstrap_workers", 1)),
    )


def _near_zero_fields(values: np.ndarray, epsilon: float) -> dict[str, Any]:
    array = np.asarray(values)
    if array.size == 0:
        return {
            "near_zero_audit_count": 0,
            "near_zero_audit_total_count": 0,
            "near_zero_audit_fraction": float("nan"),
            "near_zero_nonfinite_count": 0,
        }
    audit = near_zero_audit(values, epsilon=epsilon)
    return {
        "near_zero_audit_count": int(audit["near_zero_count"]),
        "near_zero_audit_total_count": int(audit["total_count"]),
        "near_zero_audit_fraction": float(audit["fraction"]),
        "near_zero_nonfinite_count": int(audit["nonfinite_count"]),
    }


def _source_metric_values(
    directions: np.ndarray,
    direction_class_ids: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    source_images: np.ndarray,
) -> dict[str, list[Any]]:
    """Keep image-level observations for paired control effects in the public statistics file."""

    scores = cosine_scores(values, directions)
    class_to_index = {int(class_id): index for index, class_id in enumerate(direction_class_ids.tolist())}
    image_ids = list(dict.fromkeys(source_images.tolist()))

    # Per-row own-argmax and margin, vectorized exactly like the historical
    # per-object loop: own compares the true-class score with the full-row
    # maximum, margin compares it with the row maximum after removing the true
    # class (NaN is ignored by nanmax).  Rows whose label has no direction
    # column stay NaN and are dropped by nanmean, matching the old ``continue``.
    label_values = np.asarray(labels)
    row_class_index = np.asarray(
        [class_to_index.get(int(label), -1) for label in label_values.tolist()], dtype=np.int64
    )
    valid_rows = row_class_index >= 0
    own_values = np.full(len(scores), np.nan, dtype=np.float64)
    margin_values = np.full(len(scores), np.nan, dtype=np.float64)
    if valid_rows.any():
        valid_scores = scores[valid_rows]
        valid_class_index = row_class_index[valid_rows]
        own_score = valid_scores[np.arange(len(valid_class_index)), valid_class_index]
        masked = valid_scores.copy()
        masked[np.arange(len(valid_class_index)), valid_class_index] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row_max = np.nanmax(valid_scores, axis=1)
            hardest = np.nanmax(masked, axis=1)
        own_values[valid_rows] = (own_score >= row_max).astype(np.float64)
        margin_values[valid_rows] = own_score - hardest

    output: dict[str, list[Any]] = {"source_image_id": [], "own_argmax_rate": [], "mean_diagonal_margin": [], "macro_ovr_auc": []}
    for image_id in image_ids:
        selected = source_images == image_id
        selected_labels = labels[selected]
        selected_scores = scores[selected]
        own = own_values[selected]
        margins = margin_values[selected]
        aucs = _auc_from_scores_fast(selected_scores, selected_labels, direction_class_ids)
        output["source_image_id"].append(str(image_id))
        output["own_argmax_rate"].append(float(np.nanmean(own)) if np.isfinite(own).any() else float("nan"))
        output["mean_diagonal_margin"].append(float(np.nanmean(margins)) if np.isfinite(margins).any() else float("nan"))
        output["macro_ovr_auc"].append(float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan"))
    return output


def _attach_metric_ci(
    result: dict[str, Any],
    *,
    metric_name: str,
    values: np.ndarray,
    labels: np.ndarray,
    source_images: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    settings: Mapping[str, Any],
    seed_parts: Sequence[Any],
    fast_auc_classes: Sequence[Any] | None = None,
) -> None:
    if len(values) == 0:
        ci_result = {
            "status": "empty",
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "source_image_count": 0,
        }
    elif not bool(settings["bootstrap_enabled"]):
        point = float(statistic(values, labels, source_images))
        ci_result = {
            "status": "disabled",
            "estimate": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "source_image_count": len(set(source_images.tolist())),
        }
    elif fast_auc_classes is not None:
        ci_result = _fast_score_auc_ci(
            values,
            labels,
            source_images,
            np.asarray(list(fast_auc_classes), dtype=np.int64),
            replicates=int(settings["bootstrap_replicates"]),
            seed=_seed_for(*seed_parts, metric_name, seed=int(settings["bootstrap_seed"])),
            confidence=float(settings["confidence_level"]),
            workers=int(settings.get("bootstrap_workers", 1)),
        )
    else:
        ci_result = clustered_bootstrap(
            values,
            labels=labels,
            source_image_ids=source_images,
            statistic=statistic,
            replicates=int(settings["bootstrap_replicates"]),
            seed=_seed_for(*seed_parts, metric_name, seed=int(settings["bootstrap_seed"])),
            confidence=float(settings["confidence_level"]),
            relabel_clusters=True,
            workers=int(settings.get("bootstrap_workers", 1)),
        )
    result.setdefault("bootstrap", {})[metric_name] = ci_result
    result[f"{metric_name}_ci_low"] = ci_result["ci_low"]
    result[f"{metric_name}_ci_high"] = ci_result["ci_high"]
    result[f"{metric_name}_ci"] = (ci_result["ci_low"], ci_result["ci_high"])


def _mask(table: FeatureTable, *, dataset_id: str, perturbation: str, role: str) -> np.ndarray:
    return np.asarray(
        [
            sample.dataset_id == dataset_id
            and sample.perturbation == perturbation
            and sample.role == role
            and not sample.excluded_from_main
            for sample in table.metadata
        ],
        dtype=bool,
    )


def _mask_all_classes(table: FeatureTable, *, dataset_id: str, perturbation: str, role: str) -> np.ndarray:
    """Mask that keeps classes excluded from the main experiments (used by E04-2)."""
    return np.asarray(
        [
            sample.dataset_id == dataset_id and sample.perturbation == perturbation and sample.role == role
            for sample in table.metadata
        ],
        dtype=bool,
    )


def _class_ids(table: FeatureTable, mask: np.ndarray) -> np.ndarray:
    return np.asarray([sample.class_id for sample, keep in zip(table.metadata, mask) if keep], dtype=np.int64)


def _source_images(table: FeatureTable, mask: np.ndarray) -> np.ndarray:
    return np.asarray([sample.source_image_id for sample, keep in zip(table.metadata, mask) if keep], dtype="U")


def _coarse_names(table: FeatureTable, mask: np.ndarray) -> np.ndarray:
    return np.asarray([sample.coarse_name for sample, keep in zip(table.metadata, mask) if keep], dtype="U")


def _encode_labels(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    names = sorted({str(value) for value in values.tolist()})
    mapping = {name: index for index, name in enumerate(names)}
    return np.asarray([mapping[str(value)] for value in values.tolist()], dtype=np.int64), mapping


def _limit_fit_mask(table: FeatureTable, mask: np.ndarray, limit: int | None) -> np.ndarray:
    """Limit each class to a deterministic object budget without mixing roles."""

    if limit is None:
        return mask
    if limit <= 0:
        raise CoreStageError("fit_size must be positive")
    result = np.zeros(len(mask), dtype=bool)
    labels = _class_ids(table, mask)
    indices = np.flatnonzero(mask)
    for class_id in sorted(set(labels.tolist())):
        class_indices = indices[labels == class_id]
        ordered = sorted(
            class_indices.tolist(),
            key=lambda index: (
                hashlib.sha256(table.metadata[index].object_id.encode("utf-8")).hexdigest(),
                table.metadata[index].object_id,
            ),
        )
        result[ordered[:limit]] = True
    return result


def _control_valid_mask(table: FeatureTable, control: str) -> np.ndarray:
    if control in {"true_obb", "label_permutation", "geometry_regression"}:
        return np.ones(len(table.metadata), dtype=bool)
    return np.asarray(table.arrays.get(f"control_valid_{control}", np.zeros(len(table.metadata), dtype=bool)), dtype=bool)


def _control_switches(control_config: Mapping[str, Any] | Sequence[str] | None) -> dict[str, bool]:
    """Resolve registered E04 controls from the configuration table."""

    if control_config is None:
        return {name: True for name in _CONTROL_REGISTRY}
    if isinstance(control_config, Mapping):
        configured = dict(control_config)
        return {
            name: bool(
                configured[name]
                if name in configured
                else configured.get("mask_dilation_erosion", False)
                if name in {"mask_eroded", "mask_dilated"}
                else configured.get(name, False)
            )
            for name in _CONTROL_REGISTRY
        }
    enabled = {str(name) for name in control_config}
    return {name: name in enabled for name in _CONTROL_REGISTRY}


def _control_row(
    *,
    dataset_id: str,
    model_id: str,
    perturbation: str,
    control: str,
    method: str,
    normalization: str,
    status: str,
    reason: str,
    control_enabled: bool,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "perturbation": perturbation,
        "stage": "E04",
        "control": control,
        "control_registered": control in _REGISTERED_CONTROL_NAMES,
        "control_enabled": control_enabled,
        "method": method,
        "normalization": normalization,
        "status": status,
        "reason": reason,
    }


def _fit_directions(values: np.ndarray, labels: np.ndarray, operator: Any, *, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    transformed = operator.transform(values)
    class_ids = sorted({int(value) for value in labels.tolist()})
    directions = []
    kept_classes = []
    for class_id in class_ids:
        class_values = transformed[labels == class_id]
        try:
            direction = aggregate_class_direction(class_values, epsilon=epsilon)
        except ValueError:
            continue
        directions.append(direction)
        kept_classes.append(class_id)
    if not directions:
        raise CoreStageError("No class direction could be fitted")
    return np.stack(directions), np.asarray(kept_classes, dtype=np.int64)


def _fit_operator_and_directions(
    values: np.ndarray,
    labels: np.ndarray,
    method_id: str,
    *,
    background_fit_values: np.ndarray | None,
    pca_rank: int,
    epsilon: float,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit M1-M5 on equal-weight class direction summaries."""

    raw_operator = fit_residual_operator("raw", values, epsilon=epsilon)
    raw_directions, _ = _fit_directions(values, labels, raw_operator, epsilon=epsilon)
    operator_fit_values = values if method_id == "background_svd_removal" else raw_directions
    operator = fit_residual_operator(
        method_id,
        operator_fit_values,
        background_fit_values=background_fit_values if method_id == "background_svd_removal" else None,
        pca_rank=pca_rank,
        epsilon=epsilon,
    )
    directions, direction_class_ids = _fit_directions(values, labels, operator, epsilon=epsilon)
    return operator, directions, direction_class_ids


def _valid_normalized_delta(clear: np.ndarray, degraded: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    batch = normalize_deltas(clear - degraded, normalization="unit", epsilon=epsilon)
    return batch.values[batch.valid], batch.valid


def _clustered_ids(source_images: np.ndarray) -> list[Any]:
    values = source_images.tolist()
    try:
        return sorted(set(values))
    except (TypeError, ValueError):
        return list(dict.fromkeys(str(value) for value in values))


def _unit_cell_values(
    score_matrix: np.ndarray,
    labels: np.ndarray,
    image_codes: np.ndarray,
    query_class: Any,
    code_values: list[Any],
    direction_count: int,
) -> np.ndarray:
    """Per-cluster mean score rows for one query class (bit-identical cell means).

    Cell means use ``np.mean`` on the original row-mask element sequence, so
    every unit value equals the historical per-mask cell mean exactly.
    ``image_codes`` are integer encodings of the source image ids, so the
    comparisons run on int64 instead of strings.  A stable argsort groups the
    class rows per cluster while preserving each cell's original row order, so
    the ``np.mean`` reduction stays bit-identical without rescanning every
    cluster for every query class.
    """

    units = np.full((len(code_values), direction_count), np.nan, dtype=np.float64)
    class_rows = np.flatnonzero(labels == query_class)
    if not len(class_rows):
        return units
    index_of = {int(code): index for index, code in enumerate(code_values)}
    codes = image_codes[class_rows]
    order = np.argsort(codes, kind="stable")
    sorted_rows = class_rows[order]
    sorted_codes = codes[order]
    starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    ends = np.r_[starts[1:], len(sorted_codes)]
    for start, end in zip(starts.tolist(), ends.tolist()):
        cluster_index = index_of.get(int(sorted_codes[start]))
        if cluster_index is None:
            continue
        units[cluster_index, :] = np.asarray(
            np.mean(score_matrix[sorted_rows[start:end], :], axis=0), dtype=np.float64
        )
    return units


def _summary_ci_block(payload: dict[str, Any]) -> np.ndarray:
    """Replicate block for :func:`_summary_cis_from_units` in a worker process.

    The arithmetic and per-replicate order match the serial loop exactly, so
    the concatenated block samples are bit-identical to ``workers=1``.
    """

    context, arrays = _load_shared_payload(payload["shared"])
    draws = arrays["draws"]
    units = arrays["units"]
    presence = arrays["presence"]
    classes_count = int(context["classes_count"])
    metric_name = str(context["metric_name"])
    start, end = payload["block"]
    samples = np.empty(end - start, dtype=np.float64)
    for position, row in enumerate(range(start, end)):
        selected = draws[row]
        rep_scores = np.full((classes_count, classes_count), np.nan, dtype=np.float64)
        for query_index in range(classes_count):
            picked = selected[presence[query_index][selected]]
            if picked.size == 0:
                continue
            rep_scores[:, query_index] = np.mean(units[query_index][picked], axis=0)
        samples[position] = float(summary_from_score_matrix(rep_scores)[metric_name])
    return samples


def _summary_cis_from_units(
    *,
    score_matrix: np.ndarray,
    labels: np.ndarray,
    source_images: np.ndarray,
    classes: Sequence[Any],
    settings: Mapping[str, Any],
    seed_parts: Sequence[Any],
) -> dict[str, dict[str, Any]]:
    """CI bundle for the specificity-summary metrics from per-cluster units.

    Every (query class, source image) cell mean is a fixed value under cluster
    resampling, so each replicate only takes the drawn unit rows and averages
    them in draw order.  This is bit-identical to resampling rows and
    re-summarizing every replicate, while costing one row-take plus one mean
    per query class instead of a full cell aggregation.  With ``bootstrap_workers
    > 1`` the draws are still made on the caller RNG in the serial sequence and
    the replicate loop runs in blocks over a shared segment, so the sample
    vector stays bit-identical to the serial path.
    """

    outcome: dict[str, dict[str, Any]] = {}
    confidence = float(settings["confidence_level"])
    replicates = int(settings["bootstrap_replicates"])
    workers = max(1, int(settings.get("bootstrap_workers", 1)))
    code_values, image_codes = np.unique(source_images, return_inverse=True)
    clusters = [str(value) for value in code_values.tolist()]
    direction_count = score_matrix.shape[1]
    values: dict[int, np.ndarray] = {}
    presence: dict[int, np.ndarray] = {}
    for query_index, query_class in enumerate(classes):
        units = _unit_cell_values(score_matrix, labels, image_codes, query_class, list(range(len(clusters))), direction_count)
        values[query_index] = units
        presence[query_index] = ~np.isnan(units.sum(axis=1))
    classes_list = list(classes)
    stacked = None
    if classes_list:
        stacked = (
            np.stack([values[index] for index in range(len(classes_list))]),
            np.stack([presence[index] for index in range(len(classes_list))]),
        )
    point = specificity_summary_metrics(score_matrix, labels, classes_list, source_images)
    enabled = bool(settings["bootstrap_enabled"]) and replicates > 0
    for metric_name in ("own_argmax_rate", "mean_diagonal_margin"):
        if not enabled:
            outcome[metric_name] = {
                "status": "disabled",
                "estimate": float(point.get(metric_name, float("nan"))) if np.isfinite(point.get(metric_name, float("nan"))) else float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "source_image_count": len(set(source_images.tolist())),
                "confidence_level": confidence,
                "replicates": 0,
            }
            continue
        rng = np.random.default_rng(_seed_for(*seed_parts, metric_name, seed=int(settings["bootstrap_seed"])))
        if workers > 1 and len(clusters) > 1 and replicates > 1 and stacked is not None:
            draws = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
            blocks = _replicate_blocks(replicates, workers)
            context = {"classes_count": len(classes_list), "metric_name": metric_name}
            arrays = {"draws": draws, "units": stacked[0], "presence": stacked[1]}
            parts = _shared_payload_map(_summary_ci_block, context, arrays, blocks, workers)
            samples = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        else:
            samples = np.empty(replicates, dtype=np.float64)
            for rep in range(replicates):
                draws = rng.integers(0, len(clusters), size=len(clusters))
                rep_scores = np.full((len(classes_list), len(classes_list)), np.nan, dtype=np.float64)
                for query_index in range(len(classes_list)):
                    picked = draws[presence[query_index][draws]]
                    if picked.size == 0:
                        continue
                    rep_scores[:, query_index] = np.mean(values[query_index][picked], axis=0)
                summary = summary_from_score_matrix(rep_scores)
                samples[rep] = float(summary[metric_name])
        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            low = high = float("nan")
        else:
            tail = (1.0 - confidence) / 2.0
            low, high = np.quantile(finite, [tail, 1.0 - tail])
        outcome[metric_name] = {
            "status": "completed",
            "estimate": float(point.get(metric_name, float("nan"))),
            "ci_low": float(low),
            "ci_high": float(high),
            "source_image_count": len(clusters),
            "confidence_level": confidence,
            "replicates": replicates,
            "valid_replicates": int(finite.size),
            "seed": _seed_for(*seed_parts, metric_name, seed=int(settings["bootstrap_seed"])),
        }
    return outcome


def _safe_metrics(
    directions: np.ndarray,
    direction_class_ids: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    source_images: np.ndarray,
    *,
    epsilon: float = 1.0e-8,
    statistics_config: Mapping[str, Any] | None = None,
    seed_parts: Sequence[Any] = (),
    near_zero_values: np.ndarray | None = None,
) -> dict[str, Any]:
    audit_values = values if near_zero_values is None else np.asarray(near_zero_values)
    if len(values) == 0:
        return {
            "own_argmax_rate": float("nan"),
            "mean_diagonal_margin": float("nan"),
            "mean_own_cosine": float("nan"),
            "macro_ovr_auc": float("nan"),
            "class_ids": [],
            "bootstrap": {},
            **_near_zero_fields(audit_values, epsilon),
        }
    settings = _statistics_settings(statistics_config)
    values64 = np.asarray(values, dtype=np.float64)
    directions64 = np.asarray(directions, dtype=np.float64)
    score_matrix = cosine_scores(values64, directions64)
    result = specificity_metrics(
        directions,
        values,
        labels,
        direction_class_ids,
        source_image_ids=source_images,
        scores=score_matrix,
    )
    result["macro_ovr_auc"] = macro_ovr_auc(
        values,
        labels,
        directions,
        direction_class_ids,
        scores=score_matrix,
    )
    classes = [int(value) for value in np.asarray(direction_class_ids).tolist()]
    unit_cis = _summary_cis_from_units(
        score_matrix=score_matrix,
        labels=labels,
        source_images=source_images,
        classes=classes,
        settings=settings,
        seed_parts=seed_parts,
    )
    for metric_name, ci_result in unit_cis.items():
        result.setdefault("bootstrap", {})[metric_name] = ci_result
        result[f"{metric_name}_ci_low"] = ci_result["ci_low"]
        result[f"{metric_name}_ci_high"] = ci_result["ci_high"]
        result[f"{metric_name}_ci"] = (ci_result["ci_low"], ci_result["ci_high"])
    _attach_metric_ci(
        result,
        metric_name="macro_ovr_auc",
        values=score_matrix,
        labels=labels,
        source_images=source_images,
        statistic=lambda sampled_scores, sampled_labels, _sampled_images: macro_ovr_auc_from_scores(
            sampled_scores,
            sampled_labels,
            classes,
        ),
        settings=settings,
        seed_parts=seed_parts,
        fast_auc_classes=classes,
    )
    for alias, source in (
        ("own_argmax", "own_argmax_rate"),
        ("margin", "mean_diagonal_margin"),
        ("auc", "macro_ovr_auc"),
    ):
        result[f"{alias}_ci_low"] = result.get(f"{source}_ci_low")
        result[f"{alias}_ci_high"] = result.get(f"{source}_ci_high")
        result[f"{alias}_ci"] = result.get(f"{source}_ci")
    result["source_image_count"] = len(set(source_images.tolist()))
    result["confirmation_object_count"] = len(values)
    result.update(_near_zero_fields(audit_values, epsilon))
    result["source_metric_values"] = _source_metric_values(
        directions,
        direction_class_ids,
        values,
        labels,
        source_images,
    )
    return result


def _metric_row(
    *,
    dataset_id: str,
    model_id: str,
    perturbation: str,
    stage: str,
    method: str,
    normalization: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "perturbation": perturbation,
        "stage": stage,
        "method": method,
        "normalization": normalization,
        "own_argmax_rate": metrics.get("own_argmax_rate"),
        "mean_diagonal_margin": metrics.get("mean_diagonal_margin"),
        "mean_own_cosine": metrics.get("mean_own_cosine"),
        "macro_ovr_auc": metrics.get("macro_ovr_auc"),
        "own_argmax_rate_ci": metrics.get("own_argmax_rate_ci"),
        "mean_diagonal_margin_ci": metrics.get("mean_diagonal_margin_ci"),
        "macro_ovr_auc_ci": metrics.get("macro_ovr_auc_ci"),
        "own_argmax_ci": metrics.get("own_argmax_ci"),
        "margin_ci": metrics.get("margin_ci"),
        "auc_ci": metrics.get("auc_ci"),
        "bootstrap": metrics.get("bootstrap", {}),
        "source_image_count": metrics.get("source_image_count"),
        "near_zero_audit_count": metrics.get("near_zero_audit_count"),
        "near_zero_audit_total_count": metrics.get("near_zero_audit_total_count"),
        "near_zero_audit_fraction": metrics.get("near_zero_audit_fraction"),
        "source_metric_values": metrics.get("source_metric_values", {}),
        "class_count": len(metrics.get("class_ids", [])),
    }


def _metric_fields(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return the shared point, CI, count and audit fields without stage labels."""

    return {
        "own_argmax_rate": metrics.get("own_argmax_rate"),
        "mean_diagonal_margin": metrics.get("mean_diagonal_margin"),
        "mean_own_cosine": metrics.get("mean_own_cosine"),
        "macro_ovr_auc": metrics.get("macro_ovr_auc"),
        "own_argmax_rate_ci": metrics.get("own_argmax_rate_ci"),
        "mean_diagonal_margin_ci": metrics.get("mean_diagonal_margin_ci"),
        "macro_ovr_auc_ci": metrics.get("macro_ovr_auc_ci"),
        "own_argmax_ci": metrics.get("own_argmax_ci"),
        "margin_ci": metrics.get("margin_ci"),
        "auc_ci": metrics.get("auc_ci"),
        "bootstrap": metrics.get("bootstrap", {}),
        "source_image_count": metrics.get("source_image_count"),
        "near_zero_audit_count": metrics.get("near_zero_audit_count"),
        "near_zero_audit_total_count": metrics.get("near_zero_audit_total_count"),
        "near_zero_audit_fraction": metrics.get("near_zero_audit_fraction"),
        "source_metric_values": metrics.get("source_metric_values", {}),
    }


def _perturbation_audit_fields(table: FeatureTable, mask: np.ndarray) -> dict[str, Any]:
    """Summarize actual pixel changes for a result row when the feature stage recorded them."""

    fields: dict[str, Any] = {}
    for name in ("actual_changed_fraction", "requested_region_fraction", "overlap_pixels"):
        values = table.arrays.get(name)
        if values is None:
            continue
        selected = np.asarray(values)[mask]
        if len(selected) == 0:
            continue
        finite = selected[np.isfinite(selected)] if np.issubdtype(selected.dtype, np.floating) else selected
        fields[f"{name}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
        fields[f"{name}_min"] = float(np.min(finite)) if len(finite) else float("nan")
        fields[f"{name}_max"] = float(np.max(finite)) if len(finite) else float("nan")
    object_ids = [sample.object_id for sample, keep in zip(table.metadata, mask) if keep]
    fields["paired_object_count"] = len(object_ids)
    fields["paired_object_ids_sha256"] = hashlib.sha256(
        "\n".join(sorted(object_ids)).encode("utf-8")
    ).hexdigest()
    return fields


def run_e03(
    table: FeatureTable,
    *,
    dataset_ids: list[str],
    perturbations: list[str],
    statistics_config: Mapping[str, Any] | None = None,
    random_seed: int = 1729,
) -> list[dict[str, Any]]:
    """Raw ROI readability using prototype cosine and a frozen-feature linear probe."""

    logger.info(f"[E03] {table.model_id} 开始: datasets={dataset_ids} perturbations={perturbations}")
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        for perturbation in perturbations:
            started = time.time()
            fit_mask = _mask(table, dataset_id=dataset_id, perturbation=perturbation, role="readout_fit")
            eval_mask = _mask(table, dataset_id=dataset_id, perturbation=perturbation, role="confirmation")
            fit_features = table.arrays["raw"][fit_mask]
            eval_features = table.arrays["raw"][eval_mask]
            fit_labels = _class_ids(table, fit_mask)
            eval_labels = _class_ids(table, eval_mask)
            eval_images = _source_images(table, eval_mask)
            if len(set(fit_labels.tolist())) < 2 or len(eval_features) == 0:
                continue
            class_ids = np.asarray(sorted(set(fit_labels.tolist())), dtype=np.int64)
            prototype_centroids = np.stack([fit_features[fit_labels == class_id].mean(axis=0) for class_id in class_ids.tolist()])
            scores = cosine_scores(eval_features, prototype_centroids)
            prototype_auc = float(np.nanmean([binary_auc((eval_labels == class_id).astype(np.int64), scores[:, index]) for index, class_id in enumerate(class_ids)]))
            prototype_row = {
                "dataset_id": dataset_id,
                "model_id": table.model_id,
                "perturbation": perturbation,
                "readout": "raw_feature_prototype",
                "space": "raw",
                "macro_ovr_auc": prototype_auc,
                "multiclass_accuracy": float(np.mean(class_ids[np.argmax(scores, axis=1)] == eval_labels)),
                "confirmation_object_count": len(eval_labels),
                "confirmation_source_image_count": len(set(eval_images.tolist())),
            }
            rows.append(prototype_row)
            readout_mean = fit_features.mean(axis=0)
            centered_fit = fit_features - readout_mean
            centered_eval = eval_features - readout_mean
            centered_centroids = np.stack([centered_fit[fit_labels == class_id].mean(axis=0) for class_id in class_ids.tolist()])
            centered_scores = cosine_scores(centered_eval, centered_centroids)
            centered_prototype_row = {
                "dataset_id": dataset_id,
                "model_id": table.model_id,
                "perturbation": perturbation,
                "readout": "raw_feature_prototype_centered",
                "space": "centered",
                "macro_ovr_auc": float(np.nanmean([binary_auc((eval_labels == class_id).astype(np.int64), centered_scores[:, index]) for index, class_id in enumerate(class_ids)])),
                "multiclass_accuracy": float(np.mean(class_ids[np.argmax(centered_scores, axis=1)] == eval_labels)),
                "confirmation_object_count": len(eval_labels),
                "confirmation_source_image_count": len(set(eval_images.tolist())),
            }
            rows.append(centered_prototype_row)
            try:
                classifier = _fit_logistic_regression(
                    fit_features, fit_labels, max_iter=1000, random_state=random_seed
                )
            except ImportError as exc:
                raise CoreStageError("E03 linear probe requires scikit-learn") from exc
            decision = classifier.decision_function(eval_features)
            if decision.ndim == 1:
                decision = np.stack([-decision, decision], axis=1)
            linear_auc = float(np.nanmean([binary_auc((eval_labels == class_id).astype(np.int64), decision[:, index]) for index, class_id in enumerate(classifier.classes_)]))
            linear_row = {
                "dataset_id": dataset_id,
                "model_id": table.model_id,
                "perturbation": perturbation,
                "readout": "frozen_feature_linear_probe",
                "space": "raw",
                "macro_ovr_auc": linear_auc,
                "multiclass_accuracy": float(np.mean(classifier.classes_[np.argmax(decision, axis=1)] == eval_labels)),
                "confirmation_object_count": len(eval_labels),
                "confirmation_source_image_count": len(set(eval_images.tolist())),
            }
            rows.append(linear_row)
            if statistics_config is not None:
                statistics_settings = _statistics_settings(statistics_config)
                prototype_ci = _attach_auc_bootstrap(
                    eval_features,
                    eval_labels,
                    eval_images,
                    prototype_centroids,
                    class_ids,
                    settings=statistics_settings,
                    seed_parts=(table.model_id, dataset_id, perturbation, "E03", "prototype", random_seed),
                )
                centered_prototype_ci = _attach_auc_bootstrap(
                    centered_eval,
                    eval_labels,
                    eval_images,
                    centered_centroids,
                    class_ids,
                    settings=statistics_settings,
                    seed_parts=(table.model_id, dataset_id, perturbation, "E03", "prototype_centered", random_seed),
                )
                linear_ci = _score_metric_ci(
                    decision,
                    eval_labels,
                    eval_images,
                    np.asarray(classifier.classes_),
                    "auc",
                    settings=statistics_settings,
                    seed_parts=(table.model_id, dataset_id, perturbation, "E03", "linear", random_seed),
                )
                prototype_row["macro_ovr_auc_ci"] = (prototype_ci["ci_low"], prototype_ci["ci_high"])
                prototype_row["macro_ovr_auc_bootstrap"] = prototype_ci
                centered_prototype_row["macro_ovr_auc_ci"] = (centered_prototype_ci["ci_low"], centered_prototype_ci["ci_high"])
                centered_prototype_row["macro_ovr_auc_bootstrap"] = centered_prototype_ci
                linear_row["macro_ovr_auc_ci"] = (linear_ci["ci_low"], linear_ci["ci_high"])
                linear_row["macro_ovr_auc_bootstrap"] = linear_ci
            logger.info(
                f"[E03] {table.model_id} {dataset_id}/{perturbation} 完成 (耗时 {time.time() - started:.1f}s): "
                f"prototype_auc={prototype_auc:.4f} linear_auc={linear_auc:.4f} "
                f"confirm={len(eval_labels)} objects"
            )
    logger.info(f"[E03] {table.model_id} 完成: {len(rows)} 行")
    return rows


def run_e04(
    table: FeatureTable,
    *,
    dataset_ids: list[str],
    perturbations: list[str],
    method_ids: list[str],
    normalization_variants: list[str],
    epsilon: float,
    pca_rank: int = 5,
    fit_size: int | None = None,
    controls: Mapping[str, Any] | Sequence[str] | None = None,
    statistics_config: Mapping[str, Any] | None = None,
    random_seed: int = 1729,
    include_true_obb: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    logger.info(
        f"[E04] {table.model_id} 开始: datasets={dataset_ids} perturbations={perturbations} "
        f"methods={list(method_ids)} normalizations={list(normalization_variants)} "
        f"include_true_obb={include_true_obb}"
    )
    rows: list[dict[str, Any]] = []
    switches = _control_switches(controls)
    control_names = (("true_obb",) if include_true_obb else ()) + tuple(_CONTROL_REGISTRY)
    total_groups = len(dataset_ids) * len(perturbations) * len(normalization_variants) * len(control_names)
    groups_done = 0
    for dataset_id in dataset_ids:
        for perturbation in perturbations:
            cell_started = time.time()
            cell_rows = 0
            cell_failures = 0
            if progress is not None:
                progress(f'  [{table.model_id}] {dataset_id}/{perturbation}: controls start')
            for normalization in normalization_variants:
                for control in control_names:
                    control_enabled = control == "true_obb" or switches[control]
                    if not control_enabled:
                        for method_id in method_ids:
                            rows.append(
                                _control_row(
                                    dataset_id=dataset_id,
                                    model_id=table.model_id,
                                    perturbation=perturbation,
                                    control=control,
                                    method=method_id,
                                    normalization=normalization,
                                    status="disabled",
                                    reason="disabled by [controls] configuration",
                                    control_enabled=False,
                                )
                            )
                        groups_done += 1
                        if progress is not None:
                            progress(
                                f'    [{table.model_id}] {dataset_id}/{perturbation} {normalization}/{control} '
                                f'({groups_done}/{total_groups} groups, disabled)'
                            )
                        continue
                    prefix = "true" if control in {"true_obb", "label_permutation", "geometry_regression"} else f"control_{control}"
                    fit_mask = _mask(table, dataset_id=dataset_id, perturbation=perturbation, role="direction_fit")
                    eval_mask = _mask(table, dataset_id=dataset_id, perturbation=perturbation, role="confirmation")
                    valid_control = _control_valid_mask(table, control)
                    fit_mask &= valid_control
                    eval_mask &= valid_control
                    fit_mask = _limit_fit_mask(table, fit_mask, fit_size)
                    for method_id in method_ids:
                        try:
                            fit_delta = table.arrays[f"{prefix}_clear_degraded"][fit_mask] - table.arrays[f"{prefix}_degraded_degraded"][fit_mask]
                            eval_delta = table.arrays[f"{prefix}_clear_degraded"][eval_mask] - table.arrays[f"{prefix}_degraded_degraded"][eval_mask]
                            fit_batch = normalize_deltas(fit_delta, normalization=normalization, epsilon=epsilon)
                            reference_norm = None
                            if normalization == "magnitude_weighted" and fit_batch.valid.any():
                                reference_norm = float(np.median(fit_batch.norms[fit_batch.valid]))
                            eval_batch = normalize_deltas(
                                eval_delta,
                                normalization=normalization,
                                epsilon=epsilon,
                                reference_norm=reference_norm,
                            )
                            fit_valid = fit_batch.valid
                            eval_valid = eval_batch.valid
                            fit_values = fit_batch.values[fit_valid]
                            eval_values = eval_batch.values[eval_valid]
                            fit_labels = _class_ids(table, fit_mask)[fit_valid]
                            eval_labels = _class_ids(table, eval_mask)[eval_valid]
                            eval_images = _source_images(table, eval_mask)[eval_valid]
                            fit_images = _source_images(table, fit_mask)[fit_valid]
                            if len(fit_values) == 0 or len(eval_values) == 0:
                                raise CoreStageError("control has no valid direction-fit or confirmation vectors")
                            if control == "label_permutation":
                                permutation_seed = int.from_bytes(
                                    hashlib.sha256(
                                        f"label-permutation:{random_seed}:{table.model_id}:{dataset_id}:{perturbation}".encode()
                                    ).digest()[:4],
                                    "big",
                                )
                                fit_labels = np.random.default_rng(permutation_seed).permutation(fit_labels)
                            if control == "geometry_regression":
                                if "geometry" not in table.arrays:
                                    raise CoreStageError("Geometry regression requires indexed geometry covariates")
                                fit_geometry = table.arrays["geometry"][fit_mask][fit_valid]
                                eval_geometry = table.arrays["geometry"][eval_mask][eval_valid]
                                geometry_residualizer = fit_geometry_residualizer(fit_values, fit_geometry)
                                fit_values = geometry_residualizer.transform(fit_values, fit_geometry)
                                eval_values = geometry_residualizer.transform(eval_values, eval_geometry)
                            background_fit = None
                            if "background_clear" in table.arrays:
                                background_valid = table.arrays.get("background_valid", np.ones(len(table.metadata), dtype=bool))
                                background_mask = fit_mask & background_valid
                                background_delta = table.arrays["background_clear"][background_mask] - table.arrays["background_degraded"][background_mask]
                                background_batch = normalize_deltas(background_delta, normalization=normalization, epsilon=epsilon)
                                background_fit = background_batch.values[background_batch.valid]
                            operator, directions, direction_class_ids = _fit_operator_and_directions(
                                fit_values,
                                fit_labels,
                                method_id,
                                background_fit_values=background_fit,
                                pca_rank=pca_rank,
                                epsilon=epsilon,
                            )
                            metrics = _safe_metrics(
                                directions,
                                direction_class_ids,
                                operator.transform(eval_values),
                                eval_labels,
                                eval_images,
                                epsilon=epsilon,
                                statistics_config=statistics_config,
                                seed_parts=(table.model_id, dataset_id, perturbation, control, method_id, normalization),
                                near_zero_values=eval_delta,
                            )
                        except (ValueError, CoreStageError, KeyError, IndexError) as exc:
                            cell_failures += 1
                            logger.warning(
                                f"[E04] {table.model_id} {dataset_id}/{perturbation} {method_id}/{normalization}/{control} 失败: {exc}"
                            )
                            rows.append(
                                _control_row(
                                    dataset_id=dataset_id,
                                    model_id=table.model_id,
                                    perturbation=perturbation,
                                    control=control,
                                    method=method_id,
                                    normalization=normalization,
                                    status="failed",
                                    reason=str(exc),
                                    control_enabled=True,
                                )
                            )
                            continue
                        rows.append({
                            **_metric_row(
                                dataset_id=dataset_id,
                                model_id=table.model_id,
                                perturbation=perturbation,
                                stage="E04",
                                method=method_id,
                                normalization=normalization,
                                metrics=metrics,
                            ),
                            "control": control,
                            "control_registered": control in _REGISTERED_CONTROL_NAMES,
                            "control_enabled": True,
                            "status": "completed",
                            "fit_object_count": len(fit_values),
                            "fit_source_image_count": len(set(fit_images.tolist())),
                            "fit_size_limit": fit_size,
                            "confirmation_object_count": len(eval_values),
                            "confirmation_source_image_count": len(set(eval_images.tolist())),
                            "control_valid_object_count": int(np.count_nonzero(valid_control)),
                            "fit_near_zero_audit_count": int(np.count_nonzero(~fit_valid)),
                            "confirmation_near_zero_audit_count": int(np.count_nonzero(~eval_valid)),
                        })
                        cell_rows += 1
                    groups_done += 1
                    if progress is not None:
                        progress(
                            f'    [{table.model_id}] {dataset_id}/{perturbation} {normalization}/{control} '
                            f'({groups_done}/{total_groups} groups)'
                        )
            logger.info(
                f"[E04] {table.model_id} {dataset_id}/{perturbation} 完成 (耗时 {time.time() - cell_started:.1f}s): "
                f"{cell_rows} 行, {cell_failures} 失败"
            )
    logger.info(f"[E04] {table.model_id} 完成: {len(rows)} 行")
    return rows


def run_e04_1(
    table: FeatureTable,
    *,
    dataset_ids: list[str],
    perturbations: list[str],
    epsilon: float,
    fit_size: int | None = None,
    statistics_config: Mapping[str, Any] | None = None,
    random_seed: int = 1729,
) -> list[dict[str, Any]]:
    logger.info(f"[E04-1] {table.model_id} 开始: datasets={dataset_ids} perturbations={perturbations}")
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        for perturbation in perturbations:
            started = time.time()
            fit_mask = _mask(table, dataset_id=dataset_id, perturbation=perturbation, role="readout_fit")
            fit_mask = _limit_fit_mask(table, fit_mask, fit_size)
            eval_mask = _mask(table, dataset_id=dataset_id, perturbation=perturbation, role="confirmation")
            fit_raw = table.arrays["raw"][fit_mask]
            eval_raw = table.arrays["raw"][eval_mask]
            fit_labels = _class_ids(table, fit_mask)
            eval_labels = _class_ids(table, eval_mask)
            eval_images = _source_images(table, eval_mask)
            class_ids = np.asarray(sorted(set(fit_labels.tolist())), dtype=np.int64)
            prototype_dict = fit_raw_prototype_directions(fit_raw, fit_labels, epsilon=epsilon)
            prototype_directions = np.stack([prototype_dict[int(class_id)] for class_id in class_ids])
            readout_mean = fit_raw.mean(axis=0)
            eval_raw_centered = eval_raw - readout_mean
            prototype_scores = cosine_scores(eval_raw, prototype_directions)
            prototype_row = {
                "dataset_id": dataset_id,
                "model_id": table.model_id,
                "perturbation": perturbation,
                "baseline": "raw_roi_prototype_direction",
                "space": "raw",
                "macro_ovr_auc": float(np.nanmean([binary_auc((eval_labels == class_id).astype(np.int64), prototype_scores[:, index]) for index, class_id in enumerate(class_ids)])),
                "confirmation_source_image_count": len(set(eval_images.tolist())),
            }
            rows.append(prototype_row)
            centered_prototype_scores = cosine_scores(eval_raw_centered, prototype_directions)
            centered_prototype_row = {
                "dataset_id": dataset_id,
                "model_id": table.model_id,
                "perturbation": perturbation,
                "baseline": "raw_roi_prototype_direction",
                "space": "centered",
                "macro_ovr_auc": float(np.nanmean([binary_auc((eval_labels == class_id).astype(np.int64), centered_prototype_scores[:, index]) for index, class_id in enumerate(class_ids)])),
                "confirmation_source_image_count": len(set(eval_images.tolist())),
            }
            rows.append(centered_prototype_row)
            cav_directions = []
            for class_id in class_ids:
                target = fit_labels == class_id
                clear = table.arrays["true_clear_degraded"][fit_mask][target]
                degraded = table.arrays["true_degraded_degraded"][fit_mask][target]
                cav = fit_clear_degraded_cav(clear, degraded, class_id=int(class_id))
                cav_directions.append(cav_direction(cav, epsilon=epsilon))
            cav_matrix = np.stack(cav_directions)
            cav_scores = cosine_scores(eval_raw, cav_matrix)
            cav_row = {
                "dataset_id": dataset_id,
                "model_id": table.model_id,
                "perturbation": perturbation,
                "baseline": "clear_degraded_cav",
                "space": "raw",
                "macro_ovr_auc": float(np.nanmean([binary_auc((eval_labels == class_id).astype(np.int64), cav_scores[:, index]) for index, class_id in enumerate(class_ids)])),
                "confirmation_source_image_count": len(set(eval_images.tolist())),
            }
            rows.append(cav_row)
            centered_cav_directions = []
            for class_id in class_ids:
                target = fit_labels == class_id
                clear = table.arrays["true_clear_degraded"][fit_mask][target] - readout_mean
                degraded = table.arrays["true_degraded_degraded"][fit_mask][target] - readout_mean
                cav = fit_clear_degraded_cav(clear, degraded, class_id=int(class_id))
                centered_cav_directions.append(cav_direction(cav, epsilon=epsilon))
            centered_cav_matrix = np.stack(centered_cav_directions)
            centered_cav_scores = cosine_scores(eval_raw_centered, centered_cav_matrix)
            centered_cav_row = {
                "dataset_id": dataset_id,
                "model_id": table.model_id,
                "perturbation": perturbation,
                "baseline": "clear_degraded_cav",
                "space": "centered",
                "analysis": "cav_centered_query",
                "macro_ovr_auc": float(np.nanmean([binary_auc((eval_labels == class_id).astype(np.int64), centered_cav_scores[:, index]) for index, class_id in enumerate(class_ids)])),
                "confirmation_source_image_count": len(set(eval_images.tolist())),
            }
            rows.append(centered_cav_row)
            if statistics_config is not None:
                settings = _statistics_settings(statistics_config)
                for row, values, directions, baseline_name in (
                    (prototype_row, eval_raw, prototype_directions, "prototype"),
                    (centered_prototype_row, eval_raw_centered, prototype_directions, "prototype_centered"),
                    (cav_row, eval_raw, cav_matrix, "cav"),
                    (centered_cav_row, eval_raw_centered, centered_cav_matrix, "cav_centered"),
                ):
                    ci = _attach_auc_bootstrap(
                        values,
                        eval_labels,
                        eval_images,
                        directions,
                        class_ids,
                        settings=settings,
                        seed_parts=(table.model_id, dataset_id, perturbation, "E04-1", baseline_name, random_seed),
                    )
                    row["macro_ovr_auc_ci"] = (ci["ci_low"], ci["ci_high"])
                    row["macro_ovr_auc_bootstrap"] = ci
            logger.info(
                f"[E04-1] {table.model_id} {dataset_id}/{perturbation} 完成 (耗时 {time.time() - started:.1f}s): "
                f"prototype_auc={prototype_row.get('macro_ovr_auc'):.4f} cav_auc={cav_row.get('macro_ovr_auc'):.4f}"
            )
    logger.info(f"[E04-1] {table.model_id} 完成: {len(rows)} 行")
    return rows


def run_e04_2(
    table: FeatureTable,
    *,
    dataset_ids: list[str],
    perturbations: list[str],
    support_sizes: list[int],
    epsilon: float,
    method_id: str = "euclidean_mean_subtraction",
    statistics_config: Mapping[str, Any] | None = None,
    random_seed: int = 1729,
) -> list[dict[str, Any]]:
    """FAIR1M support-budget scaling with a fixed confirmation set."""

    if set(dataset_ids) != {"fair1m"}:
        raise CoreStageError("E04-2 is registered only for the FAIR1M boundary dataset")
    logger.info(f"[E04-2] {table.model_id} 开始: fair1m support_sizes={list(support_sizes)}")
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        for perturbation in perturbations:
            cell_started = time.time()
            direction_eval_mask = _mask_all_classes(table, dataset_id=dataset_id, perturbation=perturbation, role="confirmation")
            direction_eval = table.arrays["true_clear_degraded"][direction_eval_mask] - table.arrays["true_degraded_degraded"][direction_eval_mask]
            direction_eval_batch = normalize_deltas(direction_eval, normalization="unit", epsilon=epsilon)
            eval_labels = _class_ids(table, direction_eval_mask)[direction_eval_batch.valid]
            eval_images = _source_images(table, direction_eval_mask)[direction_eval_batch.valid]
            eval_values = direction_eval_batch.values[direction_eval_batch.valid]
            for support_size in support_sizes:
                fit_mask = _mask_all_classes(table, dataset_id=dataset_id, perturbation=perturbation, role="direction_fit")
                fit_mask = _limit_fit_mask(table, fit_mask, support_size)
                fit_delta = table.arrays["true_clear_degraded"][fit_mask] - table.arrays["true_degraded_degraded"][fit_mask]
                fit_batch = normalize_deltas(fit_delta, normalization="unit", epsilon=epsilon)
                fit_labels = _class_ids(table, fit_mask)[fit_batch.valid]
                try:
                    _operator, directions, direction_class_ids = _fit_operator_and_directions(
                        fit_batch.values[fit_batch.valid],
                        fit_labels,
                        method_id,
                        background_fit_values=None,
                        pca_rank=5,
                        epsilon=epsilon,
                    )
                    metrics = _safe_metrics(
                        directions,
                        direction_class_ids,
                        _operator.transform(eval_values),
                        eval_labels,
                        eval_images,
                        statistics_config=statistics_config,
                        seed_parts=(table.model_id, dataset_id, perturbation, "E04-2", support_size, random_seed),
                    )
                    rows.append({
                        "dataset_id": dataset_id,
                        "model_id": table.model_id,
                        "perturbation": perturbation,
                        "support_size": support_size,
                        "analysis": "direction_specificity",
                        "method": method_id,
                        "normalization": "unit",
                        "confirmation_object_count": len(eval_values),
                        "fit_object_count": int(fit_batch.valid.sum()),
                        "fit_source_image_count": len(set(_source_images(table, fit_mask)[fit_batch.valid].tolist())),
                        **_metric_fields(metrics),
                        "own_argmax_rate": metrics["own_argmax_rate"],
                        "mean_diagonal_margin": metrics["mean_diagonal_margin"],
                        "confirmation_source_image_count": len(set(eval_images.tolist())),
                    })
                    coarse_fit_names = _coarse_names(table, fit_mask)[fit_batch.valid]
                    coarse_eval_names = _coarse_names(table, direction_eval_mask)[direction_eval_batch.valid]
                    _, coarse_mapping = _encode_labels(np.concatenate([coarse_fit_names, coarse_eval_names]))
                    coarse_fit_labels = np.asarray(
                        [coarse_mapping[str(value)] for value in coarse_fit_names.tolist()], dtype=np.int64
                    )
                    coarse_eval_labels = np.asarray(
                        [coarse_mapping[str(value)] for value in coarse_eval_names.tolist()], dtype=np.int64
                    )
                    coarse_classes = sorted(set(coarse_fit_labels.tolist()))
                    coarse_operator, coarse_directions_array, coarse_direction_ids = _fit_operator_and_directions(
                        fit_batch.values[fit_batch.valid], coarse_fit_labels, method_id,
                        background_fit_values=None, pca_rank=5, epsilon=epsilon,
                    )
                    coarse_metrics = _safe_metrics(
                        coarse_directions_array,
                        coarse_direction_ids,
                        coarse_operator.transform(eval_values),
                        coarse_eval_labels,
                        eval_images,
                        statistics_config=statistics_config,
                        seed_parts=(table.model_id, dataset_id, perturbation, "E04-2", support_size, "coarse", random_seed),
                    )
                    chance = 1.0 / max(1, len(coarse_classes))
                    rows.append({
                        "dataset_id": dataset_id,
                        "model_id": table.model_id,
                        "perturbation": perturbation,
                        "support_size": support_size,
                        "analysis": "coarse_direction_specificity",
                        "method": method_id,
                        "normalization": "unit",
                        "confirmation_object_count": len(eval_values),
                        "fit_object_count": int(fit_batch.valid.sum()),
                        "coarse_class_count": len(coarse_classes),
                        "coarse_class_mapping": coarse_mapping,
                        "chance_level": chance,
                        "chance_adjusted_own_argmax_rate": coarse_metrics["own_argmax_rate"] - chance,
                        **_metric_fields(coarse_metrics),
                        "confirmation_source_image_count": len(set(eval_images.tolist())),
                    })
                except (ValueError, CoreStageError) as exc:
                    rows.append({
                        "dataset_id": dataset_id,
                        "model_id": table.model_id,
                        "perturbation": perturbation,
                        "support_size": support_size,
                        "analysis": "direction_specificity",
                        "status": "failed",
                        "reason": str(exc),
                    })

                readout_fit_mask = _limit_fit_mask(
                    table,
                    _mask_all_classes(table, dataset_id=dataset_id, perturbation=perturbation, role="readout_fit"),
                    support_size,
                )
                readout_eval_mask = _mask_all_classes(table, dataset_id=dataset_id, perturbation=perturbation, role="confirmation")
                fit_features = table.arrays["raw"][readout_fit_mask]
                fit_class_ids = _class_ids(table, readout_fit_mask)
                eval_features = table.arrays["raw"][readout_eval_mask]
                readout_labels = _class_ids(table, readout_eval_mask)
                classes = np.asarray(sorted(set(fit_class_ids.tolist())), dtype=np.int64)
                if len(classes) < 2 or len(eval_features) == 0:
                    continue
                centroids = np.stack([fit_features[fit_class_ids == class_id].mean(axis=0) for class_id in classes])
                scores = cosine_scores(eval_features, centroids)
                raw_readout_row = {
                    "dataset_id": dataset_id,
                    "model_id": table.model_id,
                    "perturbation": perturbation,
                    "support_size": support_size,
                    "analysis": "prototype_readout",
                    "readout": "raw_prototype",
                    "space": "raw",
                    "confirmation_object_count": len(eval_features),
                    "readout_fit_object_count": len(fit_features),
                    "macro_ovr_auc": float(np.nanmean([binary_auc((readout_labels == class_id).astype(np.int64), scores[:, index]) for index, class_id in enumerate(classes)])),
                    "multiclass_accuracy": float(np.mean(classes[np.argmax(scores, axis=1)] == readout_labels)),
                    "confirmation_source_image_count": len(set(_source_images(table, readout_eval_mask).tolist())),
                }
                rows.append(raw_readout_row)
                readout_mean = fit_features.mean(axis=0)
                centered_fit = fit_features - readout_mean
                centered_eval = eval_features - readout_mean
                centered_centroids = np.stack([centered_fit[fit_class_ids == class_id].mean(axis=0) for class_id in classes])
                centered_scores = cosine_scores(centered_eval, centered_centroids)
                centered_readout_row = {
                    "dataset_id": dataset_id,
                    "model_id": table.model_id,
                    "perturbation": perturbation,
                    "support_size": support_size,
                    "analysis": "prototype_readout",
                    "readout": "raw_prototype",
                    "space": "centered",
                    "confirmation_object_count": len(eval_features),
                    "readout_fit_object_count": len(fit_features),
                    "macro_ovr_auc": float(np.nanmean([binary_auc((readout_labels == class_id).astype(np.int64), centered_scores[:, index]) for index, class_id in enumerate(classes)])),
                    "multiclass_accuracy": float(np.mean(classes[np.argmax(centered_scores, axis=1)] == readout_labels)),
                    "confirmation_source_image_count": len(set(_source_images(table, readout_eval_mask).tolist())),
                }
                rows.append(centered_readout_row)
                if statistics_config is not None:
                    raw_ci = _attach_auc_bootstrap(
                        eval_features,
                        readout_labels,
                        _source_images(table, readout_eval_mask),
                        centroids,
                        classes,
                        settings=_statistics_settings(statistics_config),
                        seed_parts=(table.model_id, dataset_id, perturbation, "E04-2", support_size, "raw", random_seed),
                    )
                    centered_ci = _attach_auc_bootstrap(
                        centered_eval,
                        readout_labels,
                        _source_images(table, readout_eval_mask),
                        centered_centroids,
                        classes,
                        settings=_statistics_settings(statistics_config),
                        seed_parts=(table.model_id, dataset_id, perturbation, "E04-2", support_size, "raw_centered", random_seed),
                    )
                    raw_readout_row["macro_ovr_auc_ci"] = (raw_ci["ci_low"], raw_ci["ci_high"])
                    raw_readout_row["macro_ovr_auc_bootstrap"] = raw_ci
                    centered_readout_row["macro_ovr_auc_ci"] = (centered_ci["ci_low"], centered_ci["ci_high"])
                    centered_readout_row["macro_ovr_auc_bootstrap"] = centered_ci
            logger.info(
                f"[E04-2] {table.model_id} {dataset_id}/{perturbation} 完成 (耗时 {time.time() - cell_started:.1f}s): "
                f"{len(support_sizes)} 个 fit-budget 档"
            )
    logger.info(f"[E04-2] {table.model_id} 完成: {len(rows)} 行")
    return rows


def run_e04_4(
    table: FeatureTable,
    *,
    dataset_ids: list[str],
    perturbations: list[str],
    normalization_variants: list[str],
    epsilon: float,
    fit_size: int | None = None,
    controls: Mapping[str, Any] | Sequence[str] | None = None,
    statistics_config: Mapping[str, Any] | None = None,
    random_seed: int = 1729,
) -> list[dict[str, Any]]:
    """Explicit normalization sensitivity bundle for the reviewer response."""

    logger.info(
        f"[E04-4] {table.model_id} 开始: normalizations={list(normalization_variants)} "
        f"datasets={dataset_ids} perturbations={perturbations}"
    )
    rows = run_e04(
        table,
        dataset_ids=dataset_ids,
        perturbations=perturbations,
        method_ids=list(_DEFAULT_METHOD_IDS),
        normalization_variants=normalization_variants,
        epsilon=epsilon,
        fit_size=fit_size,
        controls=controls,
        statistics_config=statistics_config,
        random_seed=random_seed,
    )
    for row in rows:
        row["stage"] = "E04-4"
    logger.info(f"[E04-4] {table.model_id} 完成: {len(rows)} 行")
    return rows


def _e04_5_sensitivity_range(
    *,
    strength_variant: str,
    table: FeatureTable,
    perturbations: Sequence[str],
    method_ids: Sequence[str],
    normalization_variants: Sequence[str],
) -> dict[str, Any]:
    """Build the per-variant sensitivity metadata attached to every E04-5 row."""

    overrides = table.representation.get("perturbation_overrides", {})
    perturbation_parameters = table.representation.get("perturbation_parameters", {})
    sensitivity_range: dict[str, Any] = {
        "strength_variant": strength_variant,
        "perturbation_overrides": overrides,
        "perturbation_parameters": perturbation_parameters,
        "perturbations": list(perturbations),
        "method_ids": list(method_ids),
        "normalization_variants": list(normalization_variants),
    }
    if isinstance(overrides, Mapping) and "patch_mask" in overrides:
        patch_values = overrides["patch_mask"]
        if isinstance(patch_values, Mapping):
            sensitivity_range["patch_budget"] = patch_values.get("patch_fraction", patch_values.get("target_fraction"))
            sensitivity_range["patch_side_fraction"] = patch_values.get("patch_side_fraction")
    return sensitivity_range


def _tag_e04_5_rows(
    rows: Sequence[dict[str, Any]],
    strength_variant: str,
    sensitivity_range: Mapping[str, Any],
) -> None:
    for row in rows:
        row["stage"] = "E04-5"
        row["strength_variant"] = strength_variant
        row["sensitivity_range"] = sensitivity_range
        row["sensitivity_grid_complete"] = True


def _reusable_e04_5_perturbations(
    table: FeatureTable,
    nominal_table: FeatureTable | None,
    perturbations: Sequence[str],
) -> set[str] | None:
    """Perturbations whose variant parameters equal the nominal table's.

    ``None`` means the comparison is not possible (missing nominal table or
    missing parameter metadata), so the caller must run the full grid.  A
    perturbation whose configured parameters are unchanged produces the same
    features for the same sample seeds, so its ``(dataset, perturbation)``
    cells can be copied from the nominal result instead of being recomputed.
    """

    if nominal_table is None:
        return None
    variant_parameters = table.representation.get("perturbation_parameters")
    nominal_parameters = nominal_table.representation.get("perturbation_parameters")
    if not isinstance(variant_parameters, Mapping) or not isinstance(nominal_parameters, Mapping):
        return None
    if not variant_parameters or not nominal_parameters:
        return None
    reusable: set[str] = set()
    for perturbation in perturbations:
        if perturbation not in variant_parameters or perturbation not in nominal_parameters:
            return None
        if variant_parameters[perturbation] == nominal_parameters[perturbation]:
            reusable.add(str(perturbation))
    return reusable


def run_e04_5(
    tables: Mapping[str, FeatureTable] | Iterable[tuple[str, FeatureTable]],
    *,
    dataset_ids: list[str],
    perturbations: list[str],
    epsilon: float,
    method_ids: Sequence[str] = _DEFAULT_METHOD_IDS,
    normalization_variants: Sequence[str] = _DEFAULT_NORMALIZATIONS,
    fit_size: int | None = None,
    controls: Mapping[str, Any] | Sequence[str] | None = None,
    statistics_config: Mapping[str, Any] | None = None,
    random_seed: int = 1729,
) -> list[dict[str, Any]]:
    """Run the complete registered sensitivity grid over every enabled perturbation.

    ``tables`` maps variant names to feature tables (a ``Mapping``) or streams
    ``(name, table)`` pairs in order (an ``Iterable``), so callers can release
    each variant table right after its rows are computed.  The ``"nominal"``
    table is used as the reuse reference: a variant whose perturbation
    parameters equal the nominal parameters for some perturbations reuses the
    nominal rows for those ``(dataset, perturbation)`` cells.  Features, seeds
    and statistics settings are identical for reused cells, so the copied rows
    are bit-identical to a full recomputation while skipping the expensive
    bootstrap/AUC work.  A missing nominal table or missing parameter metadata
    falls back to the full grid for every variant.
    """

    variant_names: list[str] | None = None
    if isinstance(tables, Mapping):
        variant_names = [str(name) for name in tables]
        table_items: Iterable[tuple[str, FeatureTable]] = tables.items()
    else:
        table_items = tables
    logger.info(
        f"[E04-5] 开始: variants={variant_names if variant_names is not None else 'streamed'} "
        f"datasets={dataset_ids} perturbations={perturbations}"
    )
    rows: list[dict[str, Any]] = []
    nominal_table: FeatureTable | None = None
    nominal_rows_by_cell: dict[tuple[str, str], list[dict[str, Any]]] | None = None

    def cell_key(row: Mapping[str, Any]) -> tuple[str, str]:
        return (str(row["dataset_id"]), str(row["perturbation"]))

    for strength_variant, table in table_items:
        variant_started = time.time()
        logger.info(f"[E04-5] variant={strength_variant} 开始")
        sensitivity_range = _e04_5_sensitivity_range(
            strength_variant=strength_variant,
            table=table,
            perturbations=perturbations,
            method_ids=method_ids,
            normalization_variants=normalization_variants,
        )
        if strength_variant == "nominal":
            strength_rows = run_e04(
                table,
                dataset_ids=dataset_ids,
                perturbations=perturbations,
                method_ids=list(method_ids),
                normalization_variants=list(normalization_variants),
                epsilon=epsilon,
                fit_size=fit_size,
                controls=controls,
                statistics_config=statistics_config,
                random_seed=random_seed,
            )
            nominal_table = table
            nominal_rows_by_cell = {}
            for row in strength_rows:
                nominal_rows_by_cell.setdefault(cell_key(row), []).append(copy.deepcopy(row))
        else:
            reusable = _reusable_e04_5_perturbations(table, nominal_table, perturbations)
            if reusable is None or nominal_rows_by_cell is None:
                strength_rows = run_e04(
                    table,
                    dataset_ids=dataset_ids,
                    perturbations=perturbations,
                    method_ids=list(method_ids),
                    normalization_variants=list(normalization_variants),
                    epsilon=epsilon,
                    fit_size=fit_size,
                    controls=controls,
                    statistics_config=statistics_config,
                    random_seed=random_seed,
                )
            else:
                changed = [item for item in perturbations if str(item) not in reusable]
                computed_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
                if changed:
                    for row in run_e04(
                        table,
                        dataset_ids=dataset_ids,
                        perturbations=changed,
                        method_ids=list(method_ids),
                        normalization_variants=list(normalization_variants),
                        epsilon=epsilon,
                        fit_size=fit_size,
                        controls=controls,
                        statistics_config=statistics_config,
                        random_seed=random_seed,
                    ):
                        computed_by_cell.setdefault(cell_key(row), []).append(row)
                strength_rows = []
                for dataset_id in dataset_ids:
                    for perturbation in perturbations:
                        key = (str(dataset_id), str(perturbation))
                        if str(perturbation) in reusable:
                            strength_rows.extend(
                                copy.deepcopy(item) for item in nominal_rows_by_cell[key]
                            )
                        else:
                            strength_rows.extend(computed_by_cell[key])
        _tag_e04_5_rows(strength_rows, strength_variant, sensitivity_range)
        rows.extend(strength_rows)
        logger.info(
            f"[E04-5] variant={strength_variant} 完成: {len(strength_rows)} 行 "
            f"(耗时 {time.time() - variant_started:.1f}s)"
        )
    logger.info(f"[E04-5] 完成: {len(rows)} 行")
    return rows


def _class_index_map(classes: np.ndarray) -> dict[int, int]:
    """First-occurrence class-id to column index, matching ``flatnonzero``."""

    mapping: dict[int, int] = {}
    for index, class_id in enumerate(np.asarray(classes).tolist()):
        mapping.setdefault(int(class_id), index)
    return mapping


def _row_margin_values(matrix: np.ndarray, class_index: np.ndarray) -> np.ndarray:
    """Per-row margin = score[true_class] - max(score with true_class removed).

    ``class_index`` holds each row's true-class column (or -1 to skip).  The
    historical per-row ``np.delete`` + ``np.max`` is reproduced by masking the
    true-class entry to ``-inf`` and taking the row maximum, which is exact for
    finite score matrices; single-column rows have no alternative and stay NaN.
    """

    margins = np.full(len(matrix), np.nan, dtype=np.float64)
    valid = class_index >= 0
    if not valid.any() or matrix.shape[1] <= 1:
        return margins
    rows = np.flatnonzero(valid)
    cols = class_index[valid]
    masked = matrix[rows].copy()
    masked[np.arange(len(rows)), cols] = -np.inf
    margins[valid] = matrix[rows, cols] - np.max(masked, axis=1)
    return margins


def _readout_scores(
    features: np.ndarray,
    labels: np.ndarray,
    classifier: Any,
    centroids: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    decision = classifier.decision_function(features)
    if decision.ndim == 1:
        decision = np.stack([-decision, decision], axis=1)
    classifier_classes = np.asarray(classifier.classes_)
    predicted = classifier_classes[np.argmax(decision, axis=1)]
    label_values = np.asarray(labels)
    classifier_index_map = _class_index_map(classifier_classes)
    classifier_index = np.asarray(
        [classifier_index_map.get(int(label), -1) for label in label_values.tolist()], dtype=np.int64
    )
    linear_margins = _row_margin_values(decision, classifier_index)

    prototype_scores = cosine_scores(features, centroids)
    prototype_predicted = classes[np.argmax(prototype_scores, axis=1)]
    prototype_index_map = _class_index_map(classes)
    prototype_index = np.asarray(
        [prototype_index_map.get(int(label), -1) for label in label_values.tolist()], dtype=np.int64
    )
    prototype_margin_values = _row_margin_values(prototype_scores, prototype_index)
    linear_correct = (predicted == label_values).astype(np.float64)
    prototype_correct = (prototype_predicted == label_values).astype(np.float64)
    linear_aucs = [
        binary_auc((labels == class_id).astype(np.int64), decision[:, index])
        for index, class_id in enumerate(classifier_classes.tolist())
    ]
    prototype_aucs = [
        binary_auc((labels == class_id).astype(np.int64), prototype_scores[:, index])
        for index, class_id in enumerate(classes.tolist())
    ]
    return {
        "linear_accuracy": float(np.mean(predicted == labels)),
        "linear_margin": float(np.nanmean(linear_margins)),
        "prototype_accuracy": float(np.mean(prototype_predicted == labels)),
        "prototype_margin": float(np.nanmean(prototype_margin_values)),
        "linear_auc": float(np.nanmean(linear_aucs)) if np.isfinite(linear_aucs).any() else float("nan"),
        "prototype_auc": float(np.nanmean(prototype_aucs)) if np.isfinite(prototype_aucs).any() else float("nan"),
        "_linear_correct": linear_correct,
        "_prototype_correct": prototype_correct,
        "_linear_margins": linear_margins,
        "_prototype_margins": prototype_margin_values,
        "_linear_scores": decision,
        "_prototype_scores": prototype_scores,
    }


def _paired_ci(
    differences: np.ndarray,
    source_images: np.ndarray,
    *,
    settings: Mapping[str, Any],
    seed_parts: Sequence[Any],
    clusters: Any | None = None,
) -> dict[str, Any]:
    """Return a source-image paired bootstrap result for an intervention drop."""

    values = np.asarray(differences, dtype=np.float64)
    images = np.asarray(source_images)
    if len(values) == 0:
        return {"status": "empty", "estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "source_image_count": 0}
    if not bool(settings["bootstrap_enabled"]):
        return {
            "status": "disabled",
            "estimate": float(np.mean(values)),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "source_image_count": len(set(images.tolist())),
        }
    result = paired_difference_ci(
        values,
        images,
        replicates=int(settings["bootstrap_replicates"]),
        seed=_seed_for(*seed_parts, seed=int(settings["bootstrap_seed"])),
        confidence=float(settings["confidence_level"]),
        clusters=clusters,
    )
    result["status"] = "completed"
    return result


def _add_ci_fields(row: dict[str, Any], prefix: str, result: Mapping[str, Any]) -> None:
    row[f"{prefix}_ci"] = (result.get("ci_low"), result.get("ci_high"))
    row[f"{prefix}_ci_low"] = result.get("ci_low")
    row[f"{prefix}_ci_high"] = result.get("ci_high")


def _source_auc_differences(
    labels: np.ndarray,
    source_images: np.ndarray,
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    differences: list[float] = []
    image_ids: list[Any] = []
    for image_id in dict.fromkeys(source_images.tolist()):
        selected = source_images == image_id
        selected_labels = labels[selected]
        baseline_auc = [
            binary_auc((selected_labels == class_id).astype(np.int64), baseline_scores[selected, index])
            for index, class_id in enumerate(classes.tolist())
        ]
        current_auc = [
            binary_auc((selected_labels == class_id).astype(np.int64), current_scores[selected, index])
            for index, class_id in enumerate(classes.tolist())
        ]
        if np.isfinite(baseline_auc).any() and np.isfinite(current_auc).any():
            differences.append(float(np.nanmean(baseline_auc) - np.nanmean(current_auc)))
            image_ids.append(image_id)
    return np.asarray(differences, dtype=np.float64), np.asarray(image_ids)


def _score_macro_auc(scores: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> float:
    aucs = [
        binary_auc((labels == class_id).astype(np.int64), scores[:, index])
        for index, class_id in enumerate(classes.tolist())
    ]
    return float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan")


def _vectorized_class_auc(
    score_matrix: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Per-class one-vs-rest AUC computed by one shared sort.

    Bit-identical to the per-column ``binary_auc`` loop in
    :func:`_score_macro_auc`: all columns are sorted together, and tie segments
    use the same averaged-rank formula.  NaN is returned for classes without
    positive or negative examples, matching ``binary_auc``.
    """

    matrix = np.asarray(score_matrix, dtype=np.float64)
    n_rows, direction_count = matrix.shape
    label_values = np.asarray(labels, dtype=np.int64)
    positive = np.stack(
        [(label_values == int(class_id)) for class_id in classes.tolist()], axis=1
    )
    n_positive = positive.sum(axis=0, dtype=np.int64)
    n_negative = n_rows - n_positive
    valid = (n_positive > 0) & (n_negative > 0)
    order = np.argsort(matrix, axis=0, kind="quicksort")
    sorted_matrix = np.take_along_axis(matrix, order, axis=0)
    equals_prev = np.zeros((n_rows, direction_count), dtype=bool)
    equals_prev[1:] = sorted_matrix[1:] == sorted_matrix[:-1]
    ranks = np.empty((n_rows, direction_count), dtype=np.float64)
    for column in range(direction_count):
        column_starts = np.flatnonzero(~equals_prev[:, column])
        column_lengths = np.diff(np.append(column_starts, n_rows))
        average_ranks = column_starts + (1.0 + column_lengths) / 2.0
        column_ranks = np.repeat(average_ranks, column_lengths)
        ranks[order[:, column], column] = column_ranks
    positive_rank_sum = np.where(positive, ranks, 0.0).sum(axis=0)
    scores = np.full(direction_count, np.nan, dtype=np.float64)
    denom = n_positive[valid] * n_negative[valid]
    scores[valid] = (
        positive_rank_sum[valid] - n_positive[valid] * (n_positive[valid] + 1) / 2.0
    ) / denom
    return np.asarray(scores, dtype=np.float64)


def _fast_score_auc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    source_images: np.ndarray,
    classes: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
    workers: int = 1,
    cluster_indices: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Cluster-bootstrap CI for macro one-vs-rest AUC with a vectorized statistic.

    Draws exactly one ``rng.integers`` per replicate on the caller-provided
    seed, so the random sequence matches :func:`clustered_bootstrap` exactly;
    the returned CI is therefore identical to the generic path.  The per-replicate
    statistic is computed in one vectorized pass (shared sort over all classes)
    instead of dispatching through the generic statistic wrapper, which removes
    most of the per-replicate Python dispatch overhead.

    With ``workers > 1`` the replicate loop runs in blocks inside a shared
    process pool (see :func:`_bootstrap_block_path`); the draw sequence and the
    per-replicate math are unchanged, so results remain bit-identical to the
    single-process path.
    """

    matrix = np.asarray(scores, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    if cluster_indices is None:
        image_ids = np.asarray(source_images)
        clusters = [
            np.flatnonzero(image_ids == image_id)
            for image_id in sorted(set(image_ids.tolist()))
        ]
    else:
        clusters = list(cluster_indices)
    cluster_count = len(clusters)
    if clusters and int(workers) > 1 and int(replicates) > 1:

        def statistic(sampled_values: np.ndarray, sampled_labels: np.ndarray, _sampled_images: np.ndarray) -> float:
            class_aucs = _vectorized_class_auc(sampled_values, sampled_labels, classes)
            return float(np.nanmean(class_aucs)) if np.isfinite(class_aucs).any() else float("nan")

        samples = _bootstrap_block_path(
            array=matrix,
            labels_array=label_values,
            image_ids=np.asarray(source_images),
            clusters=clusters,
            cluster_indices=clusters,
            statistic=statistic,
            replicates=int(replicates),
            seed=seed,
            confidence=confidence,
            # The macro-AUC statistic consumes only (values, labels); it never
            # reads source-image ids, so relabelling the duplicated clusters is
            # pure per-replicate overhead here.  Disabling it leaves the sampled
            # values, labels and RNG draw sequence unchanged, so the returned CI
            # is bit-identical to the relabelled path.
            relabel_clusters=False,
            workers=int(workers),
        )
        return _auc_ci_result(matrix, label_values, clusters, samples, confidence, replicates, seed, classes)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for rep in range(replicates):
        selected = rng.integers(0, cluster_count, size=cluster_count)
        indices = np.concatenate([clusters[position] for position in selected])
        sampled = matrix[indices]
        sampled_labels = label_values[indices]
        class_aucs = _vectorized_class_auc(sampled, sampled_labels, classes)
        samples[rep] = (
            float(np.nanmean(class_aucs)) if np.isfinite(class_aucs).any() else float("nan")
        )
    return _auc_ci_result(matrix, label_values, clusters, samples, confidence, replicates, seed, classes)


def _auc_ci_result(
    matrix: np.ndarray,
    label_values: np.ndarray,
    clusters: list[np.ndarray],
    samples: np.ndarray,
    confidence: float,
    replicates: int,
    seed: int,
    classes: np.ndarray,
) -> dict[str, Any]:
    """Shared result assembly for the vectorized macro-AUC CI paths."""

    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        low = high = float("nan")
    else:
        tail = (1.0 - confidence) / 2.0
        low, high = np.quantile(finite, [tail, 1.0 - tail])
    observed_values = _vectorized_class_auc(matrix, label_values, classes)
    observed = (
        float(np.nanmean(observed_values))
        if np.isfinite(observed_values).any()
        else float("nan")
    )
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence_level": confidence,
        "replicates": replicates,
        "source_image_count": len(clusters),
        "observation_count": len(matrix),
        "seed": seed,
    }


def _score_metric(scores: np.ndarray, labels: np.ndarray, classes: np.ndarray, metric: str) -> float:
    matrix = np.asarray(scores, dtype=np.float64)
    values = np.asarray(labels)
    predicted = classes[np.argmax(matrix, axis=1)]
    if metric == "own_argmax":
        return float(np.mean(predicted == values))
    if metric == "margin":
        index_map = _class_index_map(classes)
        class_index = np.asarray(
            [index_map.get(int(label), -1) for label in values.tolist()], dtype=np.int64
        )
        margins = _row_margin_values(matrix, class_index)
        return float(np.nanmean(margins)) if np.isfinite(margins).any() else float("nan")
    if metric == "auc":
        return _score_macro_auc(matrix, values, classes)
    raise CoreStageError(f"Unknown readout metric: {metric}")


def _score_metric_rows(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Per-row metric values for row-level metrics (``own_argmax``/``margin``).

    Each row's value depends only on that row's scores and label (argmax
    correctness, or the row's margin), so a bootstrap replicate can average
    the resampled row values instead of recomputing the metric per replicate.
    The per-row formulas are identical to :func:`_score_metric`, so the values
    are bit-identical to per-replicate recomputation.
    """

    matrix = np.asarray(scores, dtype=np.float64)
    values = np.asarray(labels)
    if metric == "own_argmax":
        predicted = classes[np.argmax(matrix, axis=1)]
        return (predicted == values).astype(np.float64)
    if metric == "margin":
        index_map = _class_index_map(classes)
        class_index = np.asarray(
            [index_map.get(int(label), -1) for label in values.tolist()], dtype=np.int64
        )
        return _row_margin_values(matrix, class_index)
    raise CoreStageError(f"Unknown readout metric: {metric}")


def _score_metric_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    source_images: np.ndarray,
    classes: np.ndarray,
    metric: str,
    *,
    settings: Mapping[str, Any],
    seed_parts: Sequence[Any],
    clusters: Any | None = None,
    cluster_indices: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    point = _score_metric(scores, labels, classes, metric)
    if not bool(settings["bootstrap_enabled"]):
        return {
            "status": "disabled",
            "estimate": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "source_image_count": len(set(source_images.tolist())),
        }
    if metric in ("own_argmax", "margin"):
        row_values = _score_metric_rows(scores, labels, classes, metric)
        result = clustered_bootstrap(
            row_values,
            labels=labels,
            source_image_ids=source_images,
            statistic=lambda sampled_values, _sampled_labels, _sampled_images: float(np.nanmean(sampled_values)),
            replicates=int(settings["bootstrap_replicates"]),
            seed=_seed_for(*seed_parts, metric, seed=int(settings["bootstrap_seed"])),
            confidence=float(settings["confidence_level"]),
            # This statistic is the plain mean of the resampled per-object values
            # and never inspects labels or source-image ids, so relabelling the
            # duplicated clusters is unnecessary per-replicate work; disabling it
            # is bit-identical.
            relabel_clusters=False,
            workers=int(settings.get("bootstrap_workers", 1)),
            clusters=clusters,
            cluster_indices=cluster_indices,
        )
    else:
        result = _fast_score_auc_ci(
            np.asarray(scores, dtype=np.float64),
            labels,
            source_images,
            classes,
            replicates=int(settings["bootstrap_replicates"]),
            seed=_seed_for(*seed_parts, metric, seed=int(settings["bootstrap_seed"])),
            confidence=float(settings["confidence_level"]),
            workers=int(settings.get("bootstrap_workers", 1)),
            cluster_indices=cluster_indices,
        )
    result["status"] = "completed"
    return result


def _paired_auc_drop_ci(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    labels: np.ndarray,
    source_images: np.ndarray,
    classes: np.ndarray,
    *,
    settings: Mapping[str, Any],
    seed_parts: Sequence[Any],
    cluster_indices: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Bootstrap a global paired AUC drop while resampling complete source images."""

    baseline = np.asarray(baseline_scores, dtype=np.float64)
    current = np.asarray(current_scores, dtype=np.float64)
    labels_array = np.asarray(labels)
    images = np.asarray(source_images)
    if baseline.shape != current.shape or baseline.ndim != 2 or len(baseline) != len(labels_array) or len(baseline) != len(images):
        raise CoreStageError("AUC drop score matrices and metadata must be aligned")
    width = baseline.shape[1]

    def statistic(values: np.ndarray, sampled_labels: np.ndarray, _sampled_images: np.ndarray) -> float:
        baseline_aucs = _vectorized_class_auc(values[:, :width], sampled_labels, classes)
        current_aucs = _vectorized_class_auc(values[:, width:], sampled_labels, classes)
        return float(
            (float(np.nanmean(baseline_aucs)) if np.isfinite(baseline_aucs).any() else float("nan"))
            - (float(np.nanmean(current_aucs)) if np.isfinite(current_aucs).any() else float("nan"))
        )

    combined = np.concatenate([baseline, current], axis=1)
    if len(combined) == 0:
        return {
            "status": "empty",
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "source_image_count": 0,
        }
    if not bool(settings["bootstrap_enabled"]):
        return {
            "status": "disabled",
            "estimate": statistic(combined, labels_array, images),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "source_image_count": len(set(images.tolist())),
        }
    result = _fast_paired_auc_drop_ci(
        combined,
        labels_array,
        images,
        classes,
        width=width,
        replicates=int(settings["bootstrap_replicates"]),
        seed=_seed_for(*seed_parts, seed=int(settings["bootstrap_seed"])),
        confidence=float(settings["confidence_level"]),
        workers=int(settings.get("bootstrap_workers", 1)),
        cluster_indices=cluster_indices,
    )
    result["status"] = "completed"
    return result


def _fast_paired_auc_drop_ci(
    combined: np.ndarray,
    labels: np.ndarray,
    source_images: np.ndarray,
    classes: np.ndarray,
    *,
    width: int,
    replicates: int,
    seed: int,
    confidence: float,
    workers: int = 1,
    cluster_indices: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Vectorized cluster-bootstrap for a paired macro-AUC drop.

    Keeps the exact per-replicate ``rng.integers`` sequence of
    :func:`clustered_bootstrap` and computes both AUC halves with one shared
    vectorized pass, so the CI matches the generic statistic path while
    avoiding per-replicate statistic dispatch.  With ``workers > 1`` the
    replicate loop runs in blocks inside the shared process pool; the draw
    sequence and per-replicate math are unchanged.
    """

    matrix = np.asarray(combined, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    if cluster_indices is None:
        image_ids = np.asarray(source_images)
        clusters = [
            np.flatnonzero(image_ids == image_id)
            for image_id in sorted(set(image_ids.tolist()))
        ]
    else:
        clusters = list(cluster_indices)
    cluster_count = len(clusters)
    if clusters and int(workers) > 1 and int(replicates) > 1:

        def statistic(sampled_values: np.ndarray, sampled_labels: np.ndarray, _sampled_images: np.ndarray) -> float:
            baseline_aucs = _vectorized_class_auc(sampled_values[:, :width], sampled_labels, classes)
            current_aucs = _vectorized_class_auc(sampled_values[:, width:], sampled_labels, classes)
            return float(
                (float(np.nanmean(baseline_aucs)) if np.isfinite(baseline_aucs).any() else float("nan"))
                - (float(np.nanmean(current_aucs)) if np.isfinite(current_aucs).any() else float("nan"))
            )

        samples = _bootstrap_block_path(
            array=matrix,
            labels_array=label_values,
            image_ids=np.asarray(source_images),
            clusters=clusters,
            cluster_indices=clusters,
            statistic=statistic,
            replicates=int(replicates),
            seed=seed,
            confidence=confidence,
            # As in :func:`_fast_score_auc_ci`, the paired macro-AUC statistic
            # ignores source-image ids, so relabelling only adds a per-replicate
            # Python loop; disabling it is bit-identical and much faster.
            relabel_clusters=False,
            workers=int(workers),
        )
        return _paired_auc_drop_result(matrix, label_values, clusters, samples, confidence, replicates, seed, width, classes)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for rep in range(replicates):
        selected = rng.integers(0, cluster_count, size=cluster_count)
        indices = np.concatenate([clusters[position] for position in selected])
        sampled = matrix[indices]
        sampled_labels = label_values[indices]
        baseline_aucs = _vectorized_class_auc(sampled[:, :width], sampled_labels, classes)
        current_aucs = _vectorized_class_auc(sampled[:, width:], sampled_labels, classes)
        baseline_value = float(np.nanmean(baseline_aucs)) if np.isfinite(baseline_aucs).any() else float("nan")
        current_value = float(np.nanmean(current_aucs)) if np.isfinite(current_aucs).any() else float("nan")
        samples[rep] = baseline_value - current_value
    return _paired_auc_drop_result(matrix, label_values, clusters, samples, confidence, replicates, seed, width, classes)


def _paired_auc_drop_result(
    matrix: np.ndarray,
    label_values: np.ndarray,
    clusters: list[np.ndarray],
    samples: np.ndarray,
    confidence: float,
    replicates: int,
    seed: int,
    width: int,
    classes: np.ndarray,
) -> dict[str, Any]:
    """Shared result assembly for the paired macro-AUC drop CI paths."""

    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        low = high = float("nan")
    else:
        tail = (1.0 - confidence) / 2.0
        low, high = np.quantile(finite, [tail, 1.0 - tail])
    baseline_aucs = _vectorized_class_auc(matrix[:, :width], label_values, classes)
    current_aucs = _vectorized_class_auc(matrix[:, width:], label_values, classes)
    observed = (
        (float(np.nanmean(baseline_aucs)) if np.isfinite(baseline_aucs).any() else float("nan"))
        - (float(np.nanmean(current_aucs)) if np.isfinite(current_aucs).any() else float("nan"))
    )
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence_level": confidence,
        "replicates": replicates,
        "source_image_count": len(clusters),
        "observation_count": len(matrix),
        "seed": seed,
    }


def _direction_map(directions: np.ndarray, class_ids: np.ndarray) -> dict[int, np.ndarray]:
    return {int(class_id): directions[index] for index, class_id in enumerate(class_ids.tolist())}


def _nearest_wrong_classes(centroids: np.ndarray, classes: np.ndarray) -> dict[int, int]:
    similarities = cosine_scores(centroids, centroids)
    result: dict[int, int] = {}
    for index, class_id in enumerate(classes.tolist()):
        alternatives = np.delete(similarities[index], index)
        alternative_indices = np.delete(np.arange(len(classes)), index)
        result[int(class_id)] = int(classes[alternative_indices[int(np.argmax(alternatives))]])
    return result


def _class_random_directions(classes: np.ndarray, dimension: int, *, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[int, np.ndarray] = {}
    for class_id in classes.tolist():
        vector = rng.normal(size=dimension)
        result[int(class_id)] = vector / max(np.linalg.norm(vector), 1.0e-12)
    return result


def _projection_matched_random(features: np.ndarray, own_directions: np.ndarray, *, seed: int) -> np.ndarray:
    """Make random unit controls with the same own-axis projection magnitude."""

    rng = np.random.default_rng(seed)
    output = np.empty_like(features, dtype=np.float64)
    for index, (feature, direction) in enumerate(zip(features, own_directions)):
        direction = direction / max(np.linalg.norm(direction), 1.0e-12)
        random = rng.normal(size=feature.shape)
        random -= np.dot(random, direction) * direction
        random_norm = np.linalg.norm(random)
        if random_norm <= 1.0e-12:
            random = np.roll(direction, 1)
            random -= np.dot(random, direction) * direction
            random_norm = np.linalg.norm(random)
        orthogonal = random / max(random_norm, 1.0e-12)
        feature_norm = max(np.linalg.norm(feature), 1.0e-12)
        coefficient = min(1.0, abs(float(np.dot(feature, direction))) / feature_norm)
        output[index] = coefficient * direction + np.sqrt(max(0.0, 1.0 - coefficient**2)) * orthogonal
    return output


def _class_projection_matched_random(
    classes: np.ndarray,
    own_map: dict[int, np.ndarray],
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    *,
    seed: int,
) -> dict[int, np.ndarray]:
    """Fit one random direction per class with matched own-axis projection energy."""

    rng = np.random.default_rng(seed)
    result: dict[int, np.ndarray] = {}
    for class_id in classes.tolist():
        own = own_map[int(class_id)]
        own = own / max(np.linalg.norm(own), 1.0e-12)
        class_features = fit_features[fit_labels == class_id]
        class_norms = np.linalg.norm(class_features, axis=1)
        valid = class_norms > 1.0e-12
        coefficient = 0.0
        if valid.any():
            coefficient = float(
                np.mean(
                    np.abs(np.sum(class_features[valid] * own[None, :], axis=1) / class_norms[valid])
                )
            )
        coefficient = min(1.0, max(0.0, coefficient))
        random = rng.normal(size=own.shape)
        random -= np.dot(random, own) * own
        random /= max(np.linalg.norm(random), 1.0e-12)
        direction = coefficient * own + np.sqrt(max(0.0, 1.0 - coefficient**2)) * random
        result[int(class_id)] = direction / max(np.linalg.norm(direction), 1.0e-12)
    return result


def _directions_for_labels(
    labels: np.ndarray,
    direction_map: dict[int, np.ndarray],
    *,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    missing = sorted({int(label) for label in labels.tolist()} - direction_map.keys())
    if missing:
        raise CoreStageError(f"No direction was fitted for classes: {missing}")
    return np.stack([direction_map[int(label)] for label in labels.tolist()])


def _mean_abs_projection_cosine(
    features: np.ndarray,
    labels: np.ndarray,
    direction_map: Mapping[int, np.ndarray],
    classes: np.ndarray,
    epsilon: float,
) -> float:
    """Mean over classes of the mean ``|cos(feature, own-class direction)|``."""

    values: list[float] = []
    label_values = np.asarray(labels)
    feature_values = np.asarray(features, dtype=np.float64)
    for class_id in np.asarray(classes).tolist():
        direction = direction_map.get(int(class_id))
        if direction is None:
            continue
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= epsilon:
            continue
        direction = direction / norm
        class_features = feature_values[label_values == class_id]
        feature_norms = np.linalg.norm(class_features, axis=1)
        valid = feature_norms > epsilon
        if valid.any():
            values.append(float(np.mean(np.abs((class_features[valid] @ direction) / feature_norms[valid]))))
    return float(np.mean(values)) if values else float("nan")


def _mean_direction_alignment(
    direction_map: Mapping[int, np.ndarray],
    axis_map: Mapping[int, np.ndarray],
    classes: np.ndarray,
) -> float:
    """Mean per-class cosine between own-class CFP directions and a readout axis map."""

    values: list[float] = []
    for class_id in np.asarray(classes).tolist():
        left = direction_map.get(int(class_id))
        right = axis_map.get(int(class_id))
        if left is None or right is None:
            continue
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
            continue
        values.append(float(np.dot(left, right) / (left_norm * right_norm)))
    return float(np.mean(values)) if values else float("nan")


def _rowwise_cosine(left: np.ndarray, right: np.ndarray, *, epsilon: float = 1.0e-12) -> np.ndarray:
    """Per-row cosine similarity with a safe denominator for near-zero rows."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.sum(left * right, axis=1) / np.maximum(denominator, epsilon)


def _class_mean_pair_cosine(
    left: np.ndarray, right: np.ndarray, labels: np.ndarray, *, epsilon: float = 1.0e-12
) -> float:
    """Mean over classes of the cosine between the class-mean left and right vectors.

    This is the class-level counterpart of :func:`_rowwise_cosine`: the per-object
    cosine is dominated by object-level noise, whereas the class-mean cosine
    measures whether the two response families share a class-structured component.
    """

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    labels = np.asarray(labels)
    values: list[float] = []
    for class_id in sorted(set(labels.tolist())):
        mask = labels == class_id
        if not mask.any():
            continue
        class_left = left[mask].mean(axis=0)
        class_right = right[mask].mean(axis=0)
        denominator = float(np.linalg.norm(class_left)) * float(np.linalg.norm(class_right))
        if denominator > epsilon:
            values.append(float(np.dot(class_left, class_right) / denominator))
    return float(np.mean(values)) if values else float("nan")


def _e07_readout_space_rows(
    *,
    space: str,
    dataset_id: str,
    perturbation: str,
    model_id: str,
    fit_features: np.ndarray,
    eval_features: np.ndarray,
    fit_labels: np.ndarray,
    eval_labels: np.ndarray,
    eval_images: np.ndarray,
    eval_cluster_ids: list[Any],
    eval_cluster_indices: list[np.ndarray],
    classes: np.ndarray,
    direction_map: dict[int, np.ndarray],
    raw_direction_map: dict[int, np.ndarray],
    class_mean_axis: Any,
    clear_fit_features: np.ndarray,
    degraded_fit_features: np.ndarray,
    removal_alphas: list[float],
    reflection_alpha: float,
    preserve_norm: bool,
    epsilon: float,
    statistics_config: Mapping[str, Any] | None,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Evaluate the E07 removal controls for one readout feature space.

    ``space`` is ``centered`` (the primary CMC-style origin, subtracting the
    readout-fit global mean) or ``raw`` (the uncentered M0 reference).  The
    readout features, prototype/CAV comparators and removal projections are all
    built from the same ``fit_features``/``eval_features`` passed in, so the two
    spaces differ only by their fitted origin.
    """

    settings = _statistics_settings(statistics_config)
    rows: list[dict[str, Any]] = []
    try:
        classifier = _fit_logistic_regression(
            fit_features, fit_labels, max_iter=1000, random_state=random_seed
        )
    except ImportError as exc:
        raise CoreStageError("E07 readout intervention requires scikit-learn") from exc
    centroids = np.stack([fit_features[fit_labels == class_id].mean(axis=0) for class_id in classes])
    baseline = _readout_scores(eval_features, eval_labels, classifier, centroids, classes)
    wrong_classes = _nearest_wrong_classes(centroids, classes)
    classifier_weights = np.asarray(classifier.coef_, dtype=np.float64)
    if classifier_weights.shape[0] == 1 and len(classes) == 2:
        classifier_weight_map = {
            int(classes[0]): -classifier_weights[0] / max(np.linalg.norm(classifier_weights[0]), 1.0e-12),
            int(classes[1]): classifier_weights[0] / max(np.linalg.norm(classifier_weights[0]), 1.0e-12),
        }
    elif classifier_weights.shape[0] == len(classes):
        classifier_weight_map = {
            int(class_id): classifier_weights[index] / max(np.linalg.norm(classifier_weights[index]), 1.0e-12)
            for index, class_id in enumerate(classes.tolist())
        }
    else:
        raise CoreStageError("Linear probe coefficient shape does not match its class set")
    # Prototype-derived and CAV-derived removal comparators are fitted from the
    # readout-fit features in the same feature space and role as the readout
    # evaluation, not from the delta-space CFP bank.
    prototype_map: dict[int, np.ndarray] = {}
    cav_map: dict[int, np.ndarray] = {}
    try:
        raw_prototype_map = fit_raw_prototype_directions(fit_features, fit_labels, epsilon=epsilon)
        prototype_map = {int(key): value for key, value in raw_prototype_map.items()}
        for class_id in classes.tolist():
            target = fit_labels == class_id
            if not target.any():
                continue
            cav = fit_clear_degraded_cav(
                clear_fit_features[target],
                degraded_fit_features[target],
                class_id=int(class_id),
            )
            cav_map[int(class_id)] = cav_direction(cav, epsilon=epsilon)
    except (ValueError, CoreStageError) as exc:
        logger.warning(
            f"[E07] {model_id} {dataset_id}/{perturbation} {space} "
            f"prototype/CAV-derived comparators skipped: {exc}"
        )
    condition_maps: dict[str, dict[int, np.ndarray]] = {
        "own_class": {
            int(class_id): direction_map[int(class_id)]
            for class_id in classes.tolist()
            if int(class_id) in direction_map
        },
        "random_unit": _class_random_directions(classes, fit_features.shape[1], seed=random_seed),
        "linear_weight": dict(classifier_weight_map),
    }
    condition_maps["wrong_class"] = {
        int(class_id): direction_map[int(wrong)]
        for class_id, wrong in wrong_classes.items()
        if int(wrong) in direction_map
    }
    condition_maps["wrong_raw"] = {
        int(class_id): raw_direction_map[int(wrong)]
        for class_id, wrong in wrong_classes.items()
        if int(wrong) in raw_direction_map
    }
    condition_maps["own_raw"] = {
        int(class_id): raw_direction_map[int(class_id)]
        for class_id in classes.tolist()
        if int(class_id) in raw_direction_map
    }
    # The across-class mean axis is the M1 operator's fitted origin
    # (mean of the per-class unit directions), used as its own control.
    if class_mean_axis is not None:
        axis_norm = float(np.linalg.norm(class_mean_axis))
        if axis_norm > epsilon:
            axis_unit = np.asarray(class_mean_axis, dtype=np.float64) / axis_norm
            condition_maps["class_mean_axis"] = {int(class_id): axis_unit for class_id in classes.tolist()}
    try:
        condition_maps["projection_matched_random"] = _class_projection_matched_random(
            classes,
            direction_map,
            fit_features,
            fit_labels,
            seed=random_seed + 1,
        )
    except (KeyError, ValueError, CoreStageError) as exc:
        logger.warning(
            f"[E07] {model_id} {dataset_id}/{perturbation} {space} "
            f"projection_matched_random skipped: {exc}"
        )
    try:
        condition_maps["classifier_weight_aligned_random"] = _class_projection_matched_random(
            classes,
            classifier_weight_map,
            fit_features,
            fit_labels,
            seed=random_seed + 2,
        )
    except (KeyError, ValueError, CoreStageError) as exc:
        logger.warning(
            f"[E07] {model_id} {dataset_id}/{perturbation} {space} "
            f"classifier_weight_aligned_random skipped: {exc}"
        )
    if prototype_map:
        condition_maps["prototype_derived"] = dict(prototype_map)
    if cav_map:
        condition_maps["cav_derived"] = dict(cav_map)
    mean_abs_cos_h_w = _mean_abs_projection_cosine(
        fit_features, fit_labels, direction_map, classes, epsilon
    )
    align_prototype = _mean_direction_alignment(direction_map, prototype_map, classes)
    align_linear_weight = _mean_direction_alignment(direction_map, classifier_weight_map, classes)
    alphas = list(removal_alphas) + [float(reflection_alpha)]
    canonical_controls = {
        "own_class": "own",
        "wrong_class": "wrong",
        "wrong_raw": "wrong_raw",
        "own_raw": "own_raw",
        "class_mean_axis": "class_mean_axis",
        "linear_weight": "linear_weight",
        "random_unit": "random",
        "projection_matched_random": "projection_matched_random",
        "classifier_weight_aligned_random": "classifier_weight_aligned_random",
        "prototype_derived": "prototype_derived",
        "cav_derived": "cav_derived",
    }
    for condition, direction_map_condition in condition_maps.items():
        missing_row = {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "perturbation": perturbation,
            "condition": condition,
            "control_id": canonical_controls.get(condition, condition),
            "space": space,
            "status": "missing",
        }
        if not direction_map_condition:
            logger.warning(
                f"[E07] {model_id} {dataset_id}/{perturbation} {space} condition {condition} "
                f"has no class directions; recorded as missing"
            )
            rows.append({**missing_row, "missing_reason": "no class directions fitted for this condition"})
            continue
        try:
            fit_directions = _directions_for_labels(fit_labels, direction_map_condition)
            eval_directions = _directions_for_labels(eval_labels, direction_map_condition)
        except (CoreStageError, KeyError) as exc:
            logger.warning(
                f"[E07] {model_id} {dataset_id}/{perturbation} {space} condition {condition} "
                f"skipped: {exc}"
            )
            rows.append({**missing_row, "missing_reason": str(exc)})
            continue
        canonical_control = canonical_controls.get(condition, condition)
        for alpha in alphas:
            try:
                transformed_fit = intervene(
                    fit_features,
                    fit_directions,
                    alpha=alpha,
                    preserve_norm=preserve_norm,
                    epsilon=epsilon,
                )
                transformed_eval = intervene(
                    eval_features,
                    eval_directions,
                    alpha=alpha,
                    preserve_norm=preserve_norm,
                    epsilon=epsilon,
                )
                frozen = _readout_scores(transformed_eval, eval_labels, classifier, centroids, classes)
                retrained_classifier = _fit_logistic_regression(
                    transformed_fit, fit_labels, max_iter=1000, random_state=random_seed
                )
                retrained_centroids = np.stack(
                    [transformed_fit[fit_labels == class_id].mean(axis=0) for class_id in classes]
                )
                retrained = _readout_scores(
                    transformed_eval,
                    eval_labels,
                    retrained_classifier,
                    retrained_centroids,
                    classes,
                )
            except ValueError as exc:
                logger.warning(
                    f"[E07] {model_id} {dataset_id}/{perturbation} {space} condition {condition} "
                    f"alpha={alpha} skipped: {exc}"
                )
                continue
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "model_id": model_id,
                    "perturbation": perturbation,
                    "condition": condition,
                    "control_id": "reflection" if alpha > 1.0 else canonical_control,
                    "retrained_probe": True,
                    "rebuilt_prototype": True,
                    "alpha": alpha,
                    "operation": "reflection_sign_reversal" if alpha > 1.0 else "removal",
                    "reflection_control": alpha > 1.0,
                    "preserve_feature_norm": preserve_norm,
                    "space": space,
                    "mean_abs_cos_h_w": mean_abs_cos_h_w,
                    "align_prototype": align_prototype,
                    "align_linear_weight": align_linear_weight,
                    "frozen_linear_accuracy": frozen["linear_accuracy"],
                    "frozen_linear_margin": frozen["linear_margin"],
                    "frozen_prototype_accuracy": frozen["prototype_accuracy"],
                    "frozen_prototype_margin": frozen["prototype_margin"],
                    "retrained_linear_accuracy": retrained["linear_accuracy"],
                    "retrained_linear_margin": retrained["linear_margin"],
                    "retrained_prototype_accuracy": retrained["prototype_accuracy"],
                    "retrained_prototype_margin": retrained["prototype_margin"],
                    "frozen_linear_accuracy_drop": baseline["linear_accuracy"] - frozen["linear_accuracy"],
                    "frozen_prototype_accuracy_drop": baseline["prototype_accuracy"] - frozen["prototype_accuracy"],
                    "retrained_linear_accuracy_drop": baseline["linear_accuracy"] - retrained["linear_accuracy"],
                    "retrained_prototype_accuracy_drop": baseline["prototype_accuracy"] - retrained["prototype_accuracy"],
                    "prototype_priority": float(frozen["prototype_accuracy"]),
                    "prototype_priority_metric": "frozen_prototype_accuracy",
                    "prototype_priority_over_linear": float(frozen["prototype_accuracy"] - frozen["linear_accuracy"]),
                    "retrained_recovery": float(retrained["linear_accuracy"] - frozen["linear_accuracy"]),
                    "retrained_prototype_recovery": float(retrained["prototype_accuracy"] - frozen["prototype_accuracy"]),
                    "retrained_recovery_fraction": float(
                        (retrained["linear_accuracy"] - frozen["linear_accuracy"])
                        / max(abs(baseline["linear_accuracy"] - frozen["linear_accuracy"]), epsilon)
                    ),
                    "confirmation_source_image_count": len(set(eval_images.tolist())),
                    "near_zero_audit": _near_zero_fields(transformed_eval, epsilon),
                }
            )
            row = rows[-1]
            readout_specs = {
                "frozen_linear": (frozen["_linear_scores"], np.asarray(classifier.classes_)),
                "frozen_prototype": (frozen["_prototype_scores"], classes),
                "retrained_linear": (retrained["_linear_scores"], np.asarray(retrained_classifier.classes_)),
                "retrained_prototype": (retrained["_prototype_scores"], classes),
            }
            for score_name, (scores, score_classes) in readout_specs.items():
                row[f"{score_name}_own_argmax"] = _score_metric(scores, eval_labels, score_classes, "own_argmax")
                row[f"{score_name}_margin"] = _score_metric(scores, eval_labels, score_classes, "margin")
                row[f"{score_name}_auc"] = _score_metric(scores, eval_labels, score_classes, "auc")
            priority_ci = _paired_ci(
                frozen["_prototype_correct"],
                eval_images,
                settings=settings,
                seed_parts=(model_id, dataset_id, perturbation, condition, alpha, "prototype_priority"),
                clusters=eval_cluster_ids,
            )
            row["prototype_priority_ci"] = (priority_ci.get("ci_low"), priority_ci.get("ci_high"))
            row["prototype_priority_ci_low"] = priority_ci.get("ci_low")
            row["prototype_priority_ci_high"] = priority_ci.get("ci_high")
            row["prototype_priority_bootstrap"] = priority_ci
            drop_arrays = {
                "frozen_linear_accuracy_drop": baseline["_linear_correct"] - frozen["_linear_correct"],
                "frozen_prototype_accuracy_drop": baseline["_prototype_correct"] - frozen["_prototype_correct"],
                "retrained_linear_accuracy_drop": baseline["_linear_correct"] - retrained["_linear_correct"],
                "retrained_prototype_accuracy_drop": baseline["_prototype_correct"] - retrained["_prototype_correct"],
                "retrained_recovery": retrained["_linear_correct"] - frozen["_linear_correct"],
                "retrained_prototype_recovery": retrained["_prototype_correct"] - frozen["_prototype_correct"],
            }
            row["bootstrap"] = {"prototype_priority": priority_ci}
            for metric_name, differences in drop_arrays.items():
                ci_result = _paired_ci(
                    np.asarray(differences, dtype=np.float64),
                    eval_images,
                    settings=settings,
                    seed_parts=(model_id, dataset_id, perturbation, condition, alpha, metric_name),
                    clusters=eval_cluster_ids,
                )
                row["bootstrap"][metric_name] = ci_result
                _add_ci_fields(row, metric_name, ci_result)
            readout_scores = {
                "frozen_linear": (frozen["_linear_scores"], np.asarray(classifier.classes_)),
                "frozen_prototype": (frozen["_prototype_scores"], classes),
                "retrained_linear": (retrained["_linear_scores"], np.asarray(retrained_classifier.classes_)),
                "retrained_prototype": (retrained["_prototype_scores"], classes),
            }
            for score_name, (scores, score_classes) in readout_scores.items():
                for metric in ("own_argmax", "margin", "auc"):
                    metric_ci = _score_metric_ci(
                        scores,
                        eval_labels,
                        eval_images,
                        score_classes,
                        metric,
                        settings=settings,
                        seed_parts=(model_id, dataset_id, perturbation, condition, alpha, score_name),
                        clusters=eval_cluster_ids,
                        cluster_indices=eval_cluster_indices,
                    )
                    row.setdefault("bootstrap", {})[f"{score_name}_{metric}"] = metric_ci
                    _add_ci_fields(row, f"{score_name}_{metric}", metric_ci)
            metric_drop_specs = (
                ("linear", "frozen", baseline, frozen),
                ("linear", "retrained", baseline, retrained),
                ("prototype", "frozen", baseline, frozen),
                ("prototype", "retrained", baseline, retrained),
            )
            for readout_name, current_name, baseline_scores, current_scores in metric_drop_specs:
                score_key = "_linear_scores" if readout_name == "linear" else "_prototype_scores"
                score_classes = np.asarray(classifier.classes_) if readout_name == "linear" else classes
                baseline_predicted = score_classes[np.argmax(baseline_scores[score_key], axis=1)]
                current_predicted = score_classes[np.argmax(current_scores[score_key], axis=1)]
                own_difference = (baseline_predicted == eval_labels).astype(np.float64) - (current_predicted == eval_labels).astype(np.float64)
                margin_difference = baseline_scores["_linear_margins" if readout_name == "linear" else "_prototype_margins"] - current_scores["_linear_margins" if readout_name == "linear" else "_prototype_margins"]
                prefix = f"{current_name}_{readout_name}"
                for metric_name, differences in (("own_argmax_drop", own_difference), ("margin_drop", margin_difference)):
                    ci_result = _paired_ci(
                        differences,
                        eval_images,
                        settings=settings,
                        seed_parts=(model_id, dataset_id, perturbation, condition, alpha, prefix, metric_name),
                        clusters=eval_cluster_ids,
                    )
                    row.setdefault("removal_drops", {})[f"{prefix}_{metric_name}"] = ci_result
                    _add_ci_fields(row, f"{prefix}_{metric_name}", ci_result)
                auc_ci = _paired_auc_drop_ci(
                    baseline_scores[score_key],
                    current_scores[score_key],
                    eval_labels,
                    eval_images,
                    score_classes,
                    settings=settings,
                    seed_parts=(model_id, dataset_id, perturbation, condition, alpha, prefix, "auc_drop"),
                    cluster_indices=eval_cluster_indices,
                )
                row.setdefault("removal_drops", {})[f"{prefix}_auc_drop"] = auc_ci
                _add_ci_fields(row, f"{prefix}_auc_drop", auc_ci)
            row.update(_near_zero_fields(transformed_eval, epsilon))
    return rows


def run_e07(
    table: FeatureTable,
    *,
    dataset_ids: list[str],
    perturbations: list[str],
    removal_alphas: list[float],
    reflection_alpha: float,
    preserve_norm: bool,
    epsilon: float,
    statistics_config: Mapping[str, Any] | None = None,
    random_seed: int = 1729,
) -> list[dict[str, Any]]:
    """Evaluate frozen and retrained readouts under alpha<=1 removal controls."""

    logger.info(
        f"[E07] {table.model_id} 开始: datasets={dataset_ids} perturbations={perturbations} "
        f"alphas={list(removal_alphas)} reflection_alpha={reflection_alpha} preserve_norm={preserve_norm}"
    )
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        for perturbation in perturbations:
            cell_started = time.time()
            cell_rows = 0
            direction_fit_mask = _mask(table, dataset_id=dataset_id, perturbation=perturbation, role="direction_fit")
            readout_fit_mask = _mask(table, dataset_id=dataset_id, perturbation=perturbation, role="readout_fit")
            confirmation_mask = _mask(table, dataset_id=dataset_id, perturbation=perturbation, role="confirmation")
            deltas = table.arrays["true_clear_degraded"][direction_fit_mask] - table.arrays["true_degraded_degraded"][direction_fit_mask]
            delta_batch = normalize_deltas(deltas, normalization="unit", epsilon=epsilon)
            fit_delta_values = delta_batch.values[delta_batch.valid]
            fit_delta_labels = _class_ids(table, direction_fit_mask)[delta_batch.valid]
            if len(fit_delta_values) == 0:
                continue
            _operator, direction_values, direction_classes = _fit_operator_and_directions(
                fit_delta_values,
                fit_delta_labels,
                "euclidean_mean_subtraction",
                background_fit_values=None,
                pca_rank=5,
                epsilon=epsilon,
            )
            direction_map = _direction_map(direction_values, direction_classes)
            try:
                raw_operator = fit_residual_operator("raw", fit_delta_values, epsilon=epsilon)
                raw_direction_values, raw_direction_classes = _fit_directions(
                    fit_delta_values, fit_delta_labels, raw_operator, epsilon=epsilon
                )
                raw_direction_map = _direction_map(raw_direction_values, raw_direction_classes)
            except (ValueError, CoreStageError) as exc:
                logger.warning(
                    f"[E07] {table.model_id} {dataset_id}/{perturbation} raw/M0 directions unavailable: {exc}"
                )
                raw_direction_map = {}

            base_fit_features = table.arrays["raw"][readout_fit_mask]
            base_eval_features = table.arrays["raw"][confirmation_mask]
            fit_labels = _class_ids(table, readout_fit_mask)
            eval_labels = _class_ids(table, confirmation_mask)
            eval_images = _source_images(table, confirmation_mask)
            eval_cluster_ids = sorted(set(eval_images.tolist()))
            eval_cluster_indices = [np.flatnonzero(eval_images == cluster_id) for cluster_id in eval_cluster_ids]
            if len(base_fit_features) == 0 or len(base_eval_features) == 0 or len(set(fit_labels.tolist())) < 2:
                continue
            classes = np.asarray(sorted(set(fit_labels.tolist())), dtype=np.int64)
            class_mean_axis = getattr(_operator, "mean", None)
            clear_fit_features = table.arrays["true_clear_degraded"][readout_fit_mask]
            degraded_fit_features = table.arrays["true_degraded_degraded"][readout_fit_mask]
            # Two readout feature spaces: the primary CMC-style centered space
            # (subtract the readout-fit global mean) and the uncentered M0
            # reference.  The direction banks are shared; the fitted origin is
            # the only difference between the two passes.
            space_specs = (
                ("centered", base_fit_features.mean(axis=0)),
                ("raw", np.zeros(base_fit_features.shape[1], dtype=np.float64)),
            )
            for space, readout_mean in space_specs:
                space_rows = _e07_readout_space_rows(
                    space=space,
                    dataset_id=dataset_id,
                    perturbation=perturbation,
                    model_id=table.model_id,
                    fit_features=base_fit_features - readout_mean,
                    eval_features=base_eval_features - readout_mean,
                    fit_labels=fit_labels,
                    eval_labels=eval_labels,
                    eval_images=eval_images,
                    eval_cluster_ids=eval_cluster_ids,
                    eval_cluster_indices=eval_cluster_indices,
                    classes=classes,
                    direction_map=direction_map,
                    raw_direction_map=raw_direction_map,
                    class_mean_axis=class_mean_axis,
                    clear_fit_features=clear_fit_features - readout_mean,
                    degraded_fit_features=degraded_fit_features - readout_mean,
                    removal_alphas=removal_alphas,
                    reflection_alpha=reflection_alpha,
                    preserve_norm=preserve_norm,
                    epsilon=epsilon,
                    statistics_config=statistics_config,
                    random_seed=random_seed,
                )
                rows.extend(space_rows)
                cell_rows += len(space_rows)
            logger.info(
                f"[E07] {table.model_id} {dataset_id}/{perturbation} 完成 (耗时 {time.time() - cell_started:.1f}s): "
                f"{cell_rows} 行"
            )
    logger.info(f"[E07] {table.model_id} 完成: {len(rows)} 行")
    return rows


def run_e04_3(
    table: FeatureTable,
    *,
    dataset_ids: list[str],
    perturbations: list[str],
    statistics_config: Mapping[str, Any] | None = None,
    random_seed: int = 1729,
) -> list[dict[str, Any]]:
    logger.info(f"[E04-3] {table.model_id} 开始: datasets={dataset_ids} perturbations={perturbations}")
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        for perturbation in perturbations:
            started = time.time()
            mask = _mask(table, dataset_id=dataset_id, perturbation=perturbation, role="confirmation")
            if not mask.any():
                logger.warning(f"[E04-3] {table.model_id} {dataset_id}/{perturbation} 无有效 confirmation 样本，跳过")
                continue
            direction_fit_mask = _mask(table, dataset_id=dataset_id, perturbation=perturbation, role="direction_fit")
            state_keys = {
                "clean_clean": "true_clear_clean",
                "degraded_clean": "true_degraded_clean",
                "clear_degraded": "true_clear_degraded",
                "degraded_degraded": "true_degraded_degraded",
            }
            raw_states = {name: table.arrays[key][mask] for name, key in state_keys.items()}
            centered_states = {}
            for name, key in state_keys.items():
                fit_values = table.arrays[key][direction_fit_mask]
                center = fit_values.mean(axis=0) if len(fit_values) else np.zeros(table.arrays[key].shape[1])
                centered_states[name] = table.arrays[key][mask] - center
            effects_by_space = {
                "raw": factorial_deltas(
                    foreground_clear_background_clean=raw_states["clean_clean"],
                    foreground_degraded_background_clean=raw_states["degraded_clean"],
                    foreground_clear_background_degraded=raw_states["clear_degraded"],
                    foreground_degraded_background_degraded=raw_states["degraded_degraded"],
                ),
                "centered": factorial_deltas(
                    foreground_clear_background_clean=centered_states["clean_clean"],
                    foreground_degraded_background_clean=centered_states["degraded_clean"],
                    foreground_clear_background_degraded=centered_states["clear_degraded"],
                    foreground_degraded_background_degraded=centered_states["degraded_degraded"],
                ),
            }
            images = _source_images(table, mask)
            settings = _statistics_settings(statistics_config) if statistics_config is not None else None
            raw_row: dict[str, Any] = {}
            for space, effects in effects_by_space.items():
                delta_clean = effects["delta_clean_background"]
                delta_degraded = effects["delta_degraded_background"]
                row = {
                    "dataset_id": dataset_id,
                    "model_id": table.model_id,
                    "perturbation": perturbation,
                    "space": space,
                    "background_interaction_mean_l2": float(np.linalg.norm(effects["background_interaction"], axis=1).mean()),
                    "clean_background_delta_mean_l2": float(np.linalg.norm(delta_clean, axis=1).mean()),
                    "degraded_background_delta_mean_l2": float(np.linalg.norm(delta_degraded, axis=1).mean()),
                    "clean_degraded_delta_cosine_mean": float(np.mean(_rowwise_cosine(delta_clean, delta_degraded))),
                    "clean_degraded_class_mean_delta_cosine": _class_mean_pair_cosine(
                        delta_clean, delta_degraded, _class_ids(table, mask)
                    ),
                    "confirmation_source_image_count": len(set(images.tolist())),
                }
                if settings is not None:
                    for name, values in (
                        ("background_interaction_l2", np.linalg.norm(effects["background_interaction"], axis=1)),
                        ("clean_background_delta_l2", np.linalg.norm(delta_clean, axis=1)),
                        ("degraded_background_delta_l2", np.linalg.norm(delta_degraded, axis=1)),
                        ("clean_degraded_delta_cosine", _rowwise_cosine(delta_clean, delta_degraded)),
                    ):
                        ci = _paired_ci(
                            values,
                            images,
                            settings=settings,
                            seed_parts=(table.model_id, dataset_id, perturbation, "E04-3", name, random_seed),
                        )
                        row[f"{name}_ci"] = (ci.get("ci_low"), ci.get("ci_high"))
                        row[f"{name}_bootstrap"] = ci
                rows.append(row)
                if space == "raw":
                    raw_row = row
            logger.info(
                f"[E04-3] {table.model_id} {dataset_id}/{perturbation} 完成 (耗时 {time.time() - started:.1f}s): "
                f"interaction_l2={raw_row['background_interaction_mean_l2']:.4f} "
                f"clean_bg={raw_row['clean_background_delta_mean_l2']:.4f} "
                f"degraded_bg={raw_row['degraded_background_delta_mean_l2']:.4f}"
            )
    logger.info(f"[E04-3] {table.model_id} 完成: {len(rows)} 行")
    return rows


def _attach_auc_bootstrap(
    values: np.ndarray,
    labels: np.ndarray,
    source_images: np.ndarray,
    directions: np.ndarray,
    direction_class_ids: np.ndarray,
    *,
    settings: Mapping[str, Any],
    seed_parts: Sequence[Any],
) -> dict[str, Any]:
    """Bootstrap macro one-vs-rest AUC while keeping labels paired to image draws."""

    if len(values) == 0:
        return {"status": "empty", "estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "source_image_count": 0}
    score_matrix = cosine_scores(np.asarray(values, dtype=np.float64), np.asarray(directions, dtype=np.float64))
    classes = [int(value) for value in np.asarray(direction_class_ids).tolist()]
    statistic = lambda sampled_scores, sampled_labels, _sampled_images: macro_ovr_auc_from_scores(
        sampled_scores,
        sampled_labels,
        classes,
    )
    if not bool(settings["bootstrap_enabled"]):
        return {
            "status": "disabled",
            "estimate": float(statistic(score_matrix, labels, source_images)),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "source_image_count": len(set(source_images.tolist())),
        }
    result = clustered_bootstrap(
        score_matrix,
        labels=labels,
        source_image_ids=source_images,
        statistic=statistic,
        replicates=int(settings["bootstrap_replicates"]),
        seed=_seed_for(*seed_parts, seed=int(settings["bootstrap_seed"])),
        confidence=float(settings["confidence_level"]),
        relabel_clusters=True,
        workers=int(settings.get("bootstrap_workers", 1)),
    )
    result["status"] = "completed"
    return result


def run_e05(
    table: FeatureTable,
    *,
    dataset_ids: list[str],
    perturbations: list[str] | None = None,
    epsilon: float,
    statistics_config: Mapping[str, Any] | None = None,
    source_perturbations: Sequence[str] | None = None,
    target_perturbations: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every source-perturbation to target-perturbation transfer pair."""

    source_names = list(source_perturbations if source_perturbations is not None else (perturbations or []))
    target_names = list(target_perturbations if target_perturbations is not None else (perturbations or []))
    logger.info(
        f"[E05] {table.model_id} 开始: datasets={dataset_ids} "
        f"source={source_names} target={target_names}"
    )
    rows: list[dict[str, Any]] = []
    settings = _statistics_settings(statistics_config)
    for dataset_id in dataset_ids:
        for source_perturbation in source_names:
            source_started = time.time()
            source_rows = 0
            source_failures = 0
            direction_fit_mask = _mask(
                table,
                dataset_id=dataset_id,
                perturbation=source_perturbation,
                role="direction_fit",
            )
            readout_fit_mask = _mask(
                table,
                dataset_id=dataset_id,
                perturbation=source_perturbation,
                role="readout_fit",
            )
            fit_delta = table.arrays["true_clear_degraded"][direction_fit_mask] - table.arrays["true_degraded_degraded"][direction_fit_mask]
            fit_batch = normalize_deltas(fit_delta, normalization="unit", epsilon=epsilon)
            fit_values = fit_batch.values[fit_batch.valid]
            fit_labels = _class_ids(table, direction_fit_mask)[fit_batch.valid]
            source_fit_images = _source_images(table, direction_fit_mask)[fit_batch.valid]
            try:
                operator, directions, direction_class_ids = _fit_operator_and_directions(
                    fit_values,
                    fit_labels,
                    "euclidean_mean_subtraction",
                    background_fit_values=None,
                    pca_rank=5,
                    epsilon=epsilon,
                )
            except (ValueError, CoreStageError) as exc:
                for target_perturbation in target_names:
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "model_id": table.model_id,
                            "source_perturbation": source_perturbation,
                            "target_perturbation": target_perturbation,
                            "transfer": "source_direction",
                            "status": "failed",
                            "reason": str(exc),
                            "source_fit_source_image_count": len(set(source_fit_images.tolist())) if len(source_fit_images) else 0,
                            "negative_control": bool(
                                source_perturbation == "brightness" or target_perturbation == "brightness"
                            ),
                            "negative_control_type": "brightness" if source_perturbation == "brightness" or target_perturbation == "brightness" else None,
                            "cross_perturbation": source_perturbation != target_perturbation,
                        }
                    )
                continue

            for target_perturbation in target_names:
                target_started = time.time()
                eval_mask = _mask(
                    table,
                    dataset_id=dataset_id,
                    perturbation=target_perturbation,
                    role="confirmation",
                )
                eval_raw = table.arrays["raw"][eval_mask]
                eval_labels = _class_ids(table, eval_mask)
                eval_images = _source_images(table, eval_mask)
                negative_control = source_perturbation == "brightness" or target_perturbation == "brightness"
                shared_fields = {
                    "dataset_id": dataset_id,
                    "model_id": table.model_id,
                    "source_perturbation": source_perturbation,
                    "target_perturbation": target_perturbation,
                    "source_fit_source_image_count": len(set(source_fit_images.tolist())) if len(source_fit_images) else 0,
                    "confirmation_source_image_count": len(set(eval_images.tolist())),
                    "negative_control": negative_control,
                    "negative_control_type": "brightness" if negative_control else None,
                    "is_negative_control": negative_control,
                    "cross_perturbation": source_perturbation != target_perturbation,
                }
                if len(eval_raw) == 0:
                    source_failures += 1
                    logger.warning(
                        f"[E05] {table.model_id} {dataset_id}/{source_perturbation}->{target_perturbation} "
                        f"无 confirmation 样本，跳过"
                    )
                    rows.append({**shared_fields, "transfer": "raw_roi", "status": "failed", "reason": "empty target confirmation set"})
                    continue

                raw_row = {
                    **shared_fields,
                    "transfer": "raw_roi",
                    "space": "raw",
                    "status": "completed",
                    "macro_ovr_auc": macro_ovr_auc(eval_raw, eval_labels, directions, direction_class_ids),
                }
                raw_ci = _attach_auc_bootstrap(
                    eval_raw,
                    eval_labels,
                    eval_images,
                    directions,
                    direction_class_ids,
                    settings=settings,
                    seed_parts=(table.model_id, dataset_id, source_perturbation, target_perturbation, "raw_roi"),
                )
                _add_ci_fields(raw_row, "macro_ovr_auc", raw_ci)
                raw_row["bootstrap"] = raw_ci
                raw_row.update(_near_zero_fields(eval_raw, epsilon))
                rows.append(raw_row)

                target_readout_fit_mask = _mask(
                    table,
                    dataset_id=dataset_id,
                    perturbation=target_perturbation,
                    role="readout_fit",
                )
                readout_fit_raw = table.arrays["raw"][target_readout_fit_mask]
                # The centered origin must come from the target role's readout-fit
                # rows.  Falling back to a confirmation-derived mean would fit a
                # statistic on the evaluation split, so an absent target
                # readout-fit is recorded as missing instead of imputed.
                if len(readout_fit_raw) == 0:
                    rows.append(
                        {
                            **shared_fields,
                            "transfer": "raw_roi",
                            "space": "centered",
                            "status": "failed",
                            "reason": "target readout_fit empty; centered origin unavailable",
                        }
                    )
                else:
                    centered_eval_raw = eval_raw - readout_fit_raw.mean(axis=0)
                    centered_raw_row = {
                        **shared_fields,
                        "transfer": "raw_roi",
                        "space": "centered",
                        "status": "completed",
                        "macro_ovr_auc": macro_ovr_auc(centered_eval_raw, eval_labels, directions, direction_class_ids),
                    }
                    centered_raw_ci = _attach_auc_bootstrap(
                        centered_eval_raw,
                        eval_labels,
                        eval_images,
                        directions,
                        direction_class_ids,
                        settings=settings,
                        seed_parts=(table.model_id, dataset_id, source_perturbation, target_perturbation, "raw_roi_centered"),
                    )
                    _add_ci_fields(centered_raw_row, "macro_ovr_auc", centered_raw_ci)
                    centered_raw_row["bootstrap"] = centered_raw_ci
                    centered_raw_row.update(_near_zero_fields(centered_eval_raw, epsilon))
                    rows.append(centered_raw_row)

                eval_delta = table.arrays["true_clear_degraded"][eval_mask] - table.arrays["true_degraded_degraded"][eval_mask]
                eval_batch = normalize_deltas(eval_delta, normalization="unit", epsilon=epsilon)
                if eval_batch.valid.any():
                    target_labels = eval_labels[eval_batch.valid]
                    target_images = eval_images[eval_batch.valid]
                    transformed_target = operator.transform(eval_batch.values[eval_batch.valid])
                    metrics = _safe_metrics(
                        directions,
                        direction_class_ids,
                        transformed_target,
                        target_labels,
                        target_images,
                        epsilon=epsilon,
                        statistics_config=statistics_config,
                        seed_parts=(table.model_id, dataset_id, source_perturbation, target_perturbation, "delta"),
                    )
                    rows.append(
                        {
                            **shared_fields,
                            "transfer": "cross_perturbation_delta",
                            "space": "unit_delta_M1_source_operator",
                            "status": "completed",
                            **_metric_row(
                                dataset_id=dataset_id,
                                model_id=table.model_id,
                                perturbation=target_perturbation,
                                stage="E05",
                                method="euclidean_mean_subtraction",
                                normalization="unit",
                                metrics=metrics,
                            ),
                        }
                    )
                    # The target-operator variant only encodes a source->target
                    # perturbation shift.  When source == target the target
                    # operator equals the source operator and the row would
                    # duplicate the same-perturbation delta row above, so it is
                    # skipped rather than emitted as an independent observation.
                    if source_perturbation != target_perturbation:
                        target_direction_fit_mask = _mask(
                            table,
                            dataset_id=dataset_id,
                            perturbation=target_perturbation,
                            role="direction_fit",
                        )
                        target_fit_delta = table.arrays["true_clear_degraded"][target_direction_fit_mask] - table.arrays["true_degraded_degraded"][target_direction_fit_mask]
                        target_fit_batch = normalize_deltas(target_fit_delta, normalization="unit", epsilon=epsilon)
                        if target_fit_batch.valid.any():
                            try:
                                # Only the target's residual origin is refitted here; the
                                # class directions must stay the source-perturbation
                                # directions fitted above.  Otherwise this row would
                                # evaluate the target against itself and carry no
                                # source->target transfer information.
                                target_operator, _target_directions, _target_direction_class_ids = _fit_operator_and_directions(
                                    target_fit_batch.values[target_fit_batch.valid],
                                    _class_ids(table, target_direction_fit_mask)[target_fit_batch.valid],
                                    "euclidean_mean_subtraction",
                                    background_fit_values=None,
                                    pca_rank=5,
                                    epsilon=epsilon,
                                )
                                target_metrics = _safe_metrics(
                                    directions,
                                    direction_class_ids,
                                    target_operator.transform(eval_batch.values[eval_batch.valid]),
                                    target_labels,
                                    target_images,
                                    epsilon=epsilon,
                                    statistics_config=statistics_config,
                                    seed_parts=(table.model_id, dataset_id, source_perturbation, target_perturbation, "target_operator"),
                                )
                                rows.append(
                                    {
                                        **shared_fields,
                                        "transfer": "cross_perturbation_delta",
                                        "space": "unit_delta_M1_target_operator",
                                        "analysis": "cross_perturbation_target_operator",
                                        "status": "completed",
                                        **_metric_row(
                                            dataset_id=dataset_id,
                                            model_id=table.model_id,
                                            perturbation=target_perturbation,
                                            stage="E05",
                                            method="euclidean_mean_subtraction",
                                            normalization="unit",
                                            metrics=target_metrics,
                                        ),
                                    }
                                )
                            except (ValueError, CoreStageError) as exc:
                                rows.append(
                                    {
                                        **shared_fields,
                                        "transfer": "cross_perturbation_delta",
                                        "space": "unit_delta_M1_target_operator",
                                        "analysis": "cross_perturbation_target_operator",
                                        "status": "failed",
                                        "reason": str(exc),
                                    }
                                )

                source_rows += 1
                # Strong readout baselines are fitted from source perturbation readout-fit rows only.
                fit_raw = table.arrays["raw"][readout_fit_mask]
                fit_raw_labels = _class_ids(table, readout_fit_mask)
                classes = np.asarray(sorted(set(fit_raw_labels.tolist())), dtype=np.int64)
                if len(fit_raw) == 0 or len(classes) < 2:
                    rows.append({**shared_fields, "transfer": "raw_prototype", "status": "failed", "reason": "source readout-fit lacks two classes"})
                    rows.append({**shared_fields, "transfer": "clear_degraded_cav", "status": "failed", "reason": "source readout-fit lacks two classes"})
                    continue
                try:
                    prototype_dict = fit_raw_prototype_directions(fit_raw, fit_raw_labels, epsilon=epsilon)
                    prototype_directions = np.stack([prototype_dict[int(class_id)] for class_id in classes])
                    # The prototype direction is a class-mean-minus-global-mean
                    # vector fitted on the source readout-fit rows, so its matched
                    # query is the evaluation set centred by that same origin.
                    baseline_origin = fit_raw.mean(axis=0)
                    centered_eval_baseline = eval_raw - baseline_origin
                    prototype_auc = macro_ovr_auc(eval_raw, eval_labels, prototype_directions, classes)
                    prototype_row = {
                        **shared_fields,
                        "transfer": "raw_prototype",
                        "baseline": "prototype",
                        "baseline_id": "raw_roi_prototype",
                        "baseline_type": "prototype",
                        "space": "raw",
                        "status": "completed",
                        "macro_ovr_auc": prototype_auc,
                        "baseline_source": "source_perturbation_readout_fit",
                    }
                    prototype_ci = _attach_auc_bootstrap(
                        eval_raw,
                        eval_labels,
                        eval_images,
                        prototype_directions,
                        classes,
                        settings=settings,
                        seed_parts=(table.model_id, dataset_id, source_perturbation, target_perturbation, "prototype"),
                    )
                    _add_ci_fields(prototype_row, "macro_ovr_auc", prototype_ci)
                    prototype_row["bootstrap"] = prototype_ci
                    prototype_row.update(_near_zero_fields(eval_raw, epsilon))
                    rows.append(prototype_row)

                    centered_prototype_row = {
                        **shared_fields,
                        "transfer": "raw_prototype",
                        "baseline": "prototype",
                        "baseline_id": "raw_roi_prototype",
                        "baseline_type": "prototype",
                        "space": "centered",
                        "status": "completed",
                        "macro_ovr_auc": macro_ovr_auc(centered_eval_baseline, eval_labels, prototype_directions, classes),
                        "baseline_source": "source_perturbation_readout_fit",
                    }
                    centered_prototype_ci = _attach_auc_bootstrap(
                        centered_eval_baseline,
                        eval_labels,
                        eval_images,
                        prototype_directions,
                        classes,
                        settings=settings,
                        seed_parts=(table.model_id, dataset_id, source_perturbation, target_perturbation, "prototype_centered"),
                    )
                    _add_ci_fields(centered_prototype_row, "macro_ovr_auc", centered_prototype_ci)
                    centered_prototype_row["bootstrap"] = centered_prototype_ci
                    centered_prototype_row.update(_near_zero_fields(centered_eval_baseline, epsilon))
                    rows.append(centered_prototype_row)

                    cav_directions = []
                    for class_id in classes:
                        target_class = fit_raw_labels == class_id
                        clear = table.arrays["true_clear_degraded"][readout_fit_mask][target_class]
                        degraded = table.arrays["true_degraded_degraded"][readout_fit_mask][target_class]
                        cav_directions.append(cav_direction(fit_clear_degraded_cav(clear, degraded, class_id=int(class_id)), epsilon=epsilon))
                    cav_matrix = np.stack(cav_directions)
                    cav_row = {
                        **shared_fields,
                        "transfer": "clear_degraded_cav",
                        "baseline": "cav",
                        "baseline_id": "clear_degraded_cav",
                        "baseline_type": "cav",
                        "space": "raw",
                        "status": "completed",
                        "macro_ovr_auc": macro_ovr_auc(eval_raw, eval_labels, cav_matrix, classes),
                        "baseline_source": "source_perturbation_readout_fit",
                    }
                    cav_ci = _attach_auc_bootstrap(
                        eval_raw,
                        eval_labels,
                        eval_images,
                        cav_matrix,
                        classes,
                        settings=settings,
                        seed_parts=(table.model_id, dataset_id, source_perturbation, target_perturbation, "cav"),
                    )
                    _add_ci_fields(cav_row, "macro_ovr_auc", cav_ci)
                    cav_row["bootstrap"] = cav_ci
                    cav_row.update(_near_zero_fields(eval_raw, epsilon))
                    rows.append(cav_row)

                    # The CAV weight is (approximately) translation invariant, so
                    # the same direction is reused; only the centred query differs.
                    centered_cav_row = {
                        **shared_fields,
                        "transfer": "clear_degraded_cav",
                        "baseline": "cav",
                        "baseline_id": "clear_degraded_cav",
                        "baseline_type": "cav",
                        "space": "centered",
                        "status": "completed",
                        "macro_ovr_auc": macro_ovr_auc(centered_eval_baseline, eval_labels, cav_matrix, classes),
                        "baseline_source": "source_perturbation_readout_fit",
                    }
                    centered_cav_ci = _attach_auc_bootstrap(
                        centered_eval_baseline,
                        eval_labels,
                        eval_images,
                        cav_matrix,
                        classes,
                        settings=settings,
                        seed_parts=(table.model_id, dataset_id, source_perturbation, target_perturbation, "cav_centered"),
                    )
                    _add_ci_fields(centered_cav_row, "macro_ovr_auc", centered_cav_ci)
                    centered_cav_row["bootstrap"] = centered_cav_ci
                    centered_cav_row.update(_near_zero_fields(centered_eval_baseline, epsilon))
                    rows.append(centered_cav_row)
                except (ValueError, CoreStageError) as exc:
                    source_failures += 1
                    logger.warning(
                        f"[E05] {table.model_id} {dataset_id}/{source_perturbation}->{target_perturbation} "
                        f"基线失败: {exc}"
                    )
                    rows.append({**shared_fields, "transfer": "raw_prototype", "status": "failed", "reason": str(exc)})
                    rows.append({**shared_fields, "transfer": "clear_degraded_cav", "status": "failed", "reason": str(exc)})
                logger.info(
                    f"[E05] {table.model_id} {dataset_id}/{source_perturbation}->{target_perturbation} "
                    f"完成 (耗时 {time.time() - target_started:.1f}s)"
                )
            logger.info(
                f"[E05] {table.model_id} {dataset_id}/{source_perturbation} 完成 (耗时 {time.time() - source_started:.1f}s): "
                f"{source_rows} 行, {source_failures} 基线失败"
            )
    logger.info(f"[E05] {table.model_id} 完成: {len(rows)} 行")
    return rows


def _iter_result_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, model_result in result.get("models", {}).items():
        if not isinstance(model_result, Mapping):
            continue
        for stage, stage_rows in model_result.items():
            if isinstance(stage_rows, list):
                for row in stage_rows:
                    if isinstance(row, Mapping):
                        rows.append({"model_id": model_id, "stage": stage, **dict(row)})
            elif isinstance(stage_rows, Mapping) and stage in {"E06", "E08"}:
                rows.append({"model_id": model_id, "stage": stage, **dict(stage_rows)})
    return rows


def summarize_core_statistics(
    result: Mapping[str, Any],
    *,
    statistics_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build public source-image counts, paired effects and FDR corrections."""

    settings = _statistics_settings(statistics_config)
    rows = _iter_result_rows(result)
    confidence_intervals: list[dict[str, Any]] = []
    source_image_counts: list[dict[str, Any]] = []
    near_zero_audits: list[dict[str, Any]] = []
    stage_statuses: list[dict[str, Any]] = []
    _CI_SUFFIXES = ("_ci", "_ci_low", "_ci_high")
    for row in rows:
        if any(key.endswith(_CI_SUFFIXES) for key in row):
            confidence_intervals.append(
                {
                    "model_id": row.get("model_id"),
                    "stage": row.get("stage"),
                    "keys": {key: row[key] for key in row if key.endswith(_CI_SUFFIXES)},
                    "bootstrap": row.get("bootstrap", {}),
                }
            )
        if "source_image_count" in row or "confirmation_source_image_count" in row:
            source_image_counts.append(
                {
                    "model_id": row.get("model_id"),
                    "stage": row.get("stage"),
                    "dataset_id": row.get("dataset_id"),
                    "source_image_count": row.get("source_image_count", row.get("confirmation_source_image_count")),
                    "fit_source_image_count": row.get("fit_source_image_count"),
                    "source_fit_source_image_count": row.get("source_fit_source_image_count"),
                }
            )
        if any(
            key in row
            for key in (
                "query_source_image_count",
                "candidate_source_image_count",
                "fit_source_image_count",
            )
        ):
            source_image_counts.append(
                {
                    "model_id": row.get("model_id"),
                    "stage": row.get("stage"),
                    "dataset_id": row.get("dataset_id"),
                    "source_image_count": row.get("source_image_count"),
                    "fit_source_image_count": row.get("fit_source_image_count"),
                    "query_source_image_count": row.get("query_source_image_count"),
                    "candidate_source_image_count": row.get("candidate_source_image_count"),
                }
            )
        if "near_zero_audit_count" in row or "near_zero_audit" in row:
            near_zero_audits.append(
                {
                    "model_id": row.get("model_id"),
                    "stage": row.get("stage"),
                    "dataset_id": row.get("dataset_id"),
                    "perturbation": row.get("perturbation", row.get("target_perturbation")),
                    "audit": row.get("near_zero_audit", {key: row[key] for key in row if key.startswith("near_zero_")}),
                }
            )
        if row.get("stage") in {"E06", "E08"}:
            stage_summary = {
                "model_id": row.get("model_id"),
                "stage": row.get("stage"),
                "status": row.get("status"),
                "execution_status": row.get("execution_status"),
                "configured_enabled": row.get("configured_enabled"),
                "interface_called": row.get("interface_called"),
                "input_source": row.get("input_source"),
                "layer_output_status": row.get("layer_output_status"),
                "query_count": row.get("query_count"),
                "candidate_count": row.get("candidate_count"),
                "query_source_image_count": row.get("query_source_image_count"),
                "candidate_source_image_count": row.get("candidate_source_image_count"),
            }
            if row.get("stage") == "E06" and isinstance(row.get("layers"), Mapping):
                stage_summary["layer_count"] = len(row["layers"])
                stage_summary["layer_source_image_counts"] = {
                    str(layer): {
                        "fit": value.get("fit_source_image_count"),
                        "confirmation": value.get("confirmation_source_image_count"),
                    }
                    for layer, value in row["layers"].items()
                    if isinstance(value, Mapping)
                }
            stage_statuses.append(stage_summary)

    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("stage") != "E04" or row.get("status") != "completed":
            continue
        source_values = row.get("source_metric_values")
        control = row.get("control")
        if not isinstance(source_values, Mapping) or not control:
            continue
        key = (
            row.get("model_id"),
            row.get("dataset_id"),
            row.get("perturbation"),
            row.get("method"),
            row.get("normalization"),
        )
        grouped[key][str(control)] = row

    paired_effects: list[dict[str, Any]] = []
    metric_names = ("own_argmax_rate", "mean_diagonal_margin", "macro_ovr_auc")
    for key, controls in grouped.items():
        true_row = controls.get("true_obb")
        if true_row is None:
            continue
        true_values = true_row["source_metric_values"]
        for control_name, control_row in controls.items():
            if control_name == "true_obb":
                continue
            control_values = control_row["source_metric_values"]
            for metric_name in metric_names:
                left = {
                    str(image): value
                    for image, value in zip(
                        true_values.get("source_image_id", []),
                        true_values.get(metric_name, []),
                    )
                }
                right = {
                    str(image): value
                    for image, value in zip(
                        control_values.get("source_image_id", []),
                        control_values.get(metric_name, []),
                    )
                }
                common = [image for image in left if image in right and np.isfinite(left[image]) and np.isfinite(right[image])]
                if not common:
                    continue
                differences = np.asarray([left[image] - right[image] for image in common], dtype=np.float64)
                image_ids = np.asarray(common, dtype="U")
                ci = _paired_ci(
                    differences,
                    image_ids,
                    settings=settings,
                    seed_parts=(*key, control_name, metric_name),
                )
                sign_flip = paired_sign_flip(
                    differences,
                    image_ids,
                    replicates=max(1, int(settings["permutation_replicates"])),
                    seed=_seed_for(*key, control_name, metric_name, "sign_flip", seed=int(settings["bootstrap_seed"])),
                    confidence=float(settings["confidence_level"]),
                )
                paired_effects.append(
                    {
                        "model_id": key[0],
                        "dataset_id": key[1],
                        "perturbation": key[2],
                        "method": key[3],
                        "normalization": key[4],
                        "control": control_name,
                        "metric": metric_name,
                        "estimate": float(np.mean(differences)),
                        "source_image_count": len(common),
                        "ci": ci,
                        "p_value_two_sided": sign_flip["p_value_two_sided"],
                    }
                )

    p_values = [float(row["p_value_two_sided"]) for row in paired_effects]
    fdr_result = benjamini_hochberg(p_values, q=float(settings["fdr_q"])) if p_values else {
        "q_values": [],
        "reject": [],
        "q": float(settings["fdr_q"]),
        "tested_count": 0,
    }
    for row, q_value, reject in zip(
        paired_effects,
        np.asarray(fdr_result.get("q_values", []), dtype=np.float64).tolist(),
        np.asarray(fdr_result.get("reject", []), dtype=bool).tolist(),
    ):
        row["fdr_q_value"] = q_value
        row["fdr_reject"] = reject
    return {
        "schema_version": 1,
        "statistics_contract": {
            **settings,
            "paired_effect": "true_obb minus registered control at source-image level",
            "fdr": settings["fdr_method"],
        },
        "source_image_counts": source_image_counts,
        "near_zero_audits": near_zero_audits,
        "stage_statuses": stage_statuses,
        "confidence_intervals": confidence_intervals,
        "paired_effects": paired_effects,
        "fdr": _json_safe(fdr_result),
    }


def write_core_statistics(
    run_root: Path | str,
    result: Mapping[str, Any],
    *,
    statistics_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the shared public statistics helper output below ``run_root/statistics``."""

    logger.info(f"[statistics] 开始汇总: {len(result.get('models', {}))} 个模型")
    root = Path(run_root).resolve(strict=True) / "statistics"
    root.mkdir(parents=True, exist_ok=True)
    summary = summarize_core_statistics(result, statistics_config=statistics_config)
    payload = _json_safe(summary)
    path = root / "core_statistics.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    (root / "paired_effects.json").write_text(
        json.dumps({"paired_effects": payload["paired_effects"], "fdr": payload["fdr"]}, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (root / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_manifest_json(
        run_root,
        "manifest/core_statistics.json",
        {
            "stage": "E03-E08",
            "result_path": str(path),
            "paired_effect_count": len(payload["paired_effects"]),
            "confidence_interval_count": len(payload["confidence_intervals"]),
        },
    )
    logger.info(f"[statistics] 完成: {len(payload['paired_effects'])} paired effects, {len(payload['confidence_intervals'])} CIs")
    return payload


def write_core_results(run_root: Path | str, result: dict[str, Any]) -> None:
    logger.info(f"[results] 开始写出: {len(result.get('models', {}))} 个模型")
    root = Path(run_root).resolve(strict=True) / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "core_results.json"
    path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    write_manifest_json(
        run_root,
        "manifest/core_results.json",
        {"stage": "E03-E07", "result_path": str(path), "models": sorted(result.get("models", {}))},
    )
    logger.info(f"[results] 完成: {path}")


_CELL_WORKER_GLOBALS: dict[str, Any] = {}


def _init_cell_worker(run_root: str, model_id: str, stage_config: dict[str, Any]) -> None:
    from cfra.pipeline.features import load_feature_table

    _CELL_WORKER_GLOBALS["table"] = load_feature_table(run_root, model_id)
    _CELL_WORKER_GLOBALS.update(stage_config)


def _cell_bootstrap_once(statistics_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a statistics config that keeps bootstrap math inside the cell.

    Cell workers run with ``bootstrap_workers=1`` so the bootstrap computation
    stays in the worker process and no nested process pool is spawned inside a
    cell worker (which would oversubscribe the machine and slow every cell).
    """

    resolved = dict(statistics_config or {})
    resolved["bootstrap_workers"] = 1
    return resolved


def _cell_e04(cell: tuple[str, str]) -> list[dict[str, Any]]:
    return run_e04(
        _CELL_WORKER_GLOBALS["table"],
        dataset_ids=[cell[0]],
        perturbations=[cell[1]],
        method_ids=_CELL_WORKER_GLOBALS["method_ids"],
        normalization_variants=_CELL_WORKER_GLOBALS["normalization_variants"],
        epsilon=_CELL_WORKER_GLOBALS["epsilon"],
        pca_rank=_CELL_WORKER_GLOBALS["pca_rank"],
        fit_size=_CELL_WORKER_GLOBALS["fit_size"],
        controls=_CELL_WORKER_GLOBALS["controls"],
        statistics_config=_CELL_WORKER_GLOBALS["statistics_config"],
        random_seed=_CELL_WORKER_GLOBALS["random_seed"],
    )


def _cell_e07(cell: tuple[str, str]) -> list[dict[str, Any]]:
    return run_e07(
        _CELL_WORKER_GLOBALS["table"],
        dataset_ids=[cell[0]],
        perturbations=[cell[1]],
        removal_alphas=_CELL_WORKER_GLOBALS["removal_alphas"],
        reflection_alpha=_CELL_WORKER_GLOBALS["reflection_alpha"],
        preserve_norm=_CELL_WORKER_GLOBALS["preserve_norm"],
        epsilon=_CELL_WORKER_GLOBALS["epsilon"],
        statistics_config=_CELL_WORKER_GLOBALS["statistics_config"],
        random_seed=_CELL_WORKER_GLOBALS["random_seed"],
    )


def _cell_dispatch(
    cells: Sequence[tuple[str, str]],
    workers: int,
    table: Any,
    run_root: str,
    model_id: str,
    stage_config: dict[str, Any],
    task: Callable[[tuple[str, str]], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Execute one ``(dataset, perturbation)`` cell per worker process.

    Each worker loads the feature table once in ``_init_cell_worker`` and runs
    ``cells`` through ``task``; outputs are collected in submission order via
    ``executor.map`` so row ordering is identical to the serial path.
    """

    if len(cells) <= 1 or workers <= 1:
        _CELL_WORKER_GLOBALS["table"] = table
        _CELL_WORKER_GLOBALS.update(stage_config)
        return [row for cell in cells for row in task(cell)]
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=get_context("spawn"),
        initializer=_init_cell_worker,
        initargs=(run_root, model_id, stage_config),
    ) as executor:
        partial_results = list(executor.map(task, cells))
    return [row for part in partial_results for row in part]


def run_e04_parallel(
    table: FeatureTable,
    *,
    run_root: Path | str,
    dataset_ids: list[str],
    perturbations: list[str],
    method_ids: list[str],
    normalization_variants: list[str],
    epsilon: float,
    pca_rank: int = 5,
    fit_size: int | None = None,
    controls: Mapping[str, Any] | Sequence[str] | None = None,
    statistics_config: Mapping[str, Any] | None = None,
    random_seed: int = 1729,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    """Cell-parallel variant of :func:`run_e04`.

    Every ``(dataset_id, perturbation)`` cell runs in its own worker process
    with the same cell function as the serial path; results are concatenated in
    the original cell order, so the output rows are identical to
    :func:`run_e04` for the same inputs ad seeds.
    """

    if workers is None:
        workers = 1
    stage_config = {
        "method_ids": list(method_ids),
        "normalization_variants": list(normalization_variants),
        "epsilon": epsilon,
        "pca_rank": pca_rank,
        "fit_size": fit_size,
        "controls": controls,
        "statistics_config": _cell_bootstrap_once(statistics_config),
        "random_seed": random_seed,
    }
    return _cell_dispatch(
        list(product(dataset_ids, perturbations)),
        workers,
        table,
        str(run_root),
        table.model_id,
        stage_config,
        _cell_e04,
    )


def run_e07_parallel(
    table: FeatureTable,
    *,
    run_root: Path | str,
    dataset_ids: list[str],
    perturbations: list[str],
    removal_alphas: list[float],
    reflection_alpha: float,
    preserve_norm: bool,
    epsilon: float,
    statistics_config: Mapping[str, Any] | None = None,
    random_seed: int = 1729,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    """Cell-parallel variant of :func:`run_e07`.

    Every ``(dataset_id, perturbation)`` cell runs in its own worker process
    with the same cell function as the serial path; results are concatenated in
    the original cell order, so the output rows are identical to
    :func:`run_e07` for the same inputs and seeds.
    """

    if workers is None:
        workers = 1
    stage_config = {
        "removal_alphas": list(removal_alphas),
        "reflection_alpha": reflection_alpha,
        "preserve_norm": preserve_norm,
        "epsilon": epsilon,
        "statistics_config": _cell_bootstrap_once(statistics_config),
        "random_seed": random_seed,
    }
    return _cell_dispatch(
        list(product(dataset_ids, perturbations)),
        workers,
        table,
        str(run_root),
        table.model_id,
        stage_config,
        _cell_e07,
    )
