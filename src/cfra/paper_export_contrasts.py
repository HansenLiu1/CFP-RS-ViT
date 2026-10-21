"""Paired contrasts over persisted per-source-image statistics.

E04-4/E04-2 rows already carry ``source_metric_values``: for every source image
and metric the image-level own-argmax / diagonal-margin / OVR-AUC.  Because the
estimates come from one shared paired measurement design, a paired difference
between two cells (control vs true-obb, normalization switch, operator switch,
support budget) is reconstructed by intersecting the source-image ids and
differencing image-level values.  Paired uncertainty is then a source-image
cluster bootstrap (same replicates as the main export), and significance uses a
two-sided sign-flip permutation over the paired differences.

This never fabricates: cells without a shared image set are emitted as missing,
and every row carries state_a/state_b, the pairing key, and the hypothesis
family.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cfra.statistics import paired_difference_ci

_PAIRED_METRICS = ("own_argmax_rate", "mean_diagonal_margin", "macro_ovr_auc")
_IMAGE_METRIC_KEY = "source_image_id"

# Main-analysis contrast grid shared with the manuscript: the three headline
# operators (M0/M1/M3) and the three response definitions.
_MAIN_OPERATORS = ("raw", "euclidean_mean_subtraction", "orthogonal_axis_projection")
_NORMALIZATIONS = ("unit", "raw", "magnitude_weighted")
_SUPPORT_SIZES = ("15", "30", "50")
_SPATIAL_CONTROLS = ("shifted_obb", "random_geometry_matched_obb", "background_only",
                     "boundary_only", "interior_eroded", "mask_eroded", "mask_dilated",
                     "label_permutation")


def _paired_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _image_metric(row: dict[str, Any], metric: str) -> dict[str, float]:
    smv = row.get("source_metric_values")
    if not isinstance(smv, dict) or not smv:
        return {}
    ids = smv.get(_IMAGE_METRIC_KEY)
    values = smv.get(metric)
    if not isinstance(ids, list) or not isinstance(values, list) or len(ids) != len(values):
        return {}
    result: dict[str, float] = {}
    for image_id, value in zip(ids, values):
        if value is None or not np.isfinite(value):
            continue
        result[str(image_id)] = float(value)
    return result


def _sign_flip_permutation(image_values: np.ndarray, replicates: int, seed: int,
                           block: int = 1024) -> float:
    """Two-sided sign-flip null over image-level paired differences."""
    rng = np.random.default_rng(seed)
    observed = float(np.mean(image_values))
    exceed = 0
    # Blocked draws keep the serial RNG stream and bound the temporary
    # ``replicates x images`` allocation; the exceedance count is unchanged.
    for start in range(0, replicates, block):
        size = min(block, replicates - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(image_values)))
        null = (signs @ image_values) / len(image_values)
        exceed += int(np.count_nonzero(np.abs(null) >= abs(observed)))
    return float((exceed + 1) / (replicates + 1))


def _emit_paired(export: Any, table: str, context: dict[str, Any], metric: str,
                 values_a: dict[str, float], values_b: dict[str, float],
                 seed_parts: tuple[Any, ...]) -> None:
    shared = [image for image in values_a if image in values_b]
    valid = [image for image in shared
             if np.isfinite(values_a[image]) and np.isfinite(values_b[image])]
    row = dict(context)
    row["n_source_images"] = len(valid)
    if len(valid) < 2:
        row.update(status="missing", missing_reason="fewer than two shared source images",
                   statistics_status="paired_units_required")
        export.emit(table, row, "delta_" + metric, None)
        return
    a = np.asarray([values_a[image] for image in valid], dtype=np.float64)
    b = np.asarray([values_b[image] for image in valid], dtype=np.float64)
    difference = b - a
    seed = _paired_seed(*seed_parts, *sorted(valid))
    ci = paired_difference_ci(
        difference,
        np.asarray(valid),
        replicates=int(export.bootstrap_replicates),
        seed=seed,
    )
    p_value = _sign_flip_permutation(difference, int(export.permutation_replicates), seed=seed + 1)
    row.update(
        estimate=float(np.mean(difference)),
        CI_low=float(ci["ci_low"]),
        CI_high=float(ci["ci_high"]),
        bootstrap_replicates=int(ci["replicates"]),
        bootstrap_valid_replicates=int(ci["replicates"]),
        p_value=p_value,
        permutation_replicates=int(export.permutation_replicates),
        statistics_status="paired_cluster_bootstrap_and_sign_flip",
        permutation_null="two-sided sign flip over paired source-image differences",
    )
    export.emit(table, row, "delta_" + metric, row["estimate"], row["CI_low"], row["CI_high"])
    export.units.write(json.dumps({"context": row, "metric": metric,
                                       "source_image": valid, "difference": difference.tolist()},
                                  default=str) + "\n")


def _factory_context(export: Any, item: dict[str, Any], path: str) -> dict[str, Any]:
    p = item.get("perturbation")
    strength = export.config.get("perturbations", {}).get(p, {})
    model = item.get("model_id")
    return {
        "dataset": item.get("dataset_id"), "model": model,
        "checkpoint": export.models.get(model, {}).get("checkpoint_sha256"),
        "layer": "final", "perturbation": p,
        "perturbation_strength": item.get("strength_value", strength),
        "operator": item.get("method", item.get("baseline", item.get("transfer"))),
        "normalization": item.get("normalization"),
        "support_size": item.get("support_size", item.get("fit_size_limit")),
        "direction_fit_role": "direction_fit", "readout_fit_role": "readout_fit",
        "evaluation_role": "confirmation",
        "split_seed": export.config["split"]["seed"],
        "run_id": getattr(export, "active_run", None),
        "stage": "paired_contrast_recomputed", "artifact": path,
        "valid_n_objects": None, "n_source_images": None,
        "source_perturbation": item.get("source_perturbation"),
        "target_perturbation": item.get("target_perturbation"),
        "aggregation": "object -> source image -> paired difference -> source image bootstrap",
        "status": "observed",
    }


def _features(item: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {metric: _image_metric(item, metric) for metric in _PAIRED_METRICS}


def _emit_pair(export: Any, table: str, path: str, item: dict[str, Any],
               state_a: str, state_b: str, kind: str, family: str,
               features_a: dict[str, dict[str, float]],
               features_b: dict[str, dict[str, float]]) -> None:
    context = _factory_context(export, item, path)
    context.update(analysis="paired_" + kind, family=family, state_a=state_a, state_b=state_b)
    for metric in _PAIRED_METRICS:
        _emit_paired(export, table, context, metric,
                     features_a.get(metric, {}), features_b.get(metric, {}),
                     ("paired", kind, item.get("dataset_id"), item.get("model_id"),
                      item.get("perturbation"), item.get("method"),
                      item.get("normalization"), item.get("support_size"),
                      item.get("analysis"), state_a, state_b, metric, path))


def _emit_missing(export: Any, table: str, state_a: str | None, state_b: str | None,
                  family: str, missing_value: str, item: dict[str, Any] | None) -> None:
    context = _factory_context(export, item or {}, "")
    context.update(
        dataset=None if item is None else item.get("dataset_id"),
        model=None if item is None else item.get("model_id"),
        perturbation=None if item is None else item.get("perturbation"),
        normalization=None if item is None else item.get("normalization"),
        operator=None, support_size=None, artifact=None,
        status="missing", family=family, state_a=state_a, state_b=state_b,
        statistics_status="missing",
        missing_reason=f"no persisted cell for comparison state {missing_value!r}",
    )
    for metric in _PAIRED_METRICS:
        export.emit(table, context, "delta_" + metric, None)


def apply_contrasts(export: Any, stage: str, path: Path, rows: list[dict[str, Any]]) -> None:
    """Emit paired contrasts from the persisted stage file currently in memory."""
    # E04 predates the row provenance fields used to key single-source rows and
    # duplicates E04-4 cells; the paired grid uses E04-4 and E04-2 only.
    if stage == "E04-4":
        _e04_contrasts(export, path, rows)
    elif stage == "E04-2":
        _e04_2_contrasts(export, path, rows)
    elif stage == "E04-5":
        _e04_5_contrasts(export, path, rows)


def _e04_contrasts(export: Any, path: Path, rows: list[dict[str, Any]]) -> None:
    active = [
        row for row in rows
        if row.get("status") == "completed"
        and row.get("method") in _MAIN_OPERATORS
        and row.get("normalization") in _NORMALIZATIONS
    ]

    def partition(key_parts: tuple[str, ...], state_field: str,
                  filters: tuple[tuple[str, Any], ...] = ()) -> dict[tuple[str, ...], dict[str, dict[str, Any]]]:
        groups: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in active:
            if any(row.get(k) != v for k, v in filters):
                continue
            key = tuple(str(row.get(part)) for part in key_parts)
            state_value = row.get(state_field)
            if state_value is None:
                continue
            groups[key][str(state_value)] = row
        return groups

    # 1) Control pairing: true-OBB vs each spatial/permutation control,
    #    restricted to the headline M1 unit cell per the manuscript.
    control_cells = partition(("dataset_id", "model_id", "perturbation"), "control",
                              filters=(("normalization", "unit"),
                                       ("method", "euclidean_mean_subtraction")))
    for key, states in control_cells.items():
        base = states.get("true_obb")
        if base is None:
            continue
        base_features = _features(base)
        for control_name in _SPATIAL_CONTROLS:
            other = states.get(control_name)
            family = f"paired_control:{key[0]}:{key[1]}:{key[2]}:euclidean_mean_subtraction:unit"
            if other is None:
                _emit_missing(export, "specificity_full", "true_obb", control_name,
                              family, control_name, base)
                continue
            _emit_pair(export, "specificity_full", str(path), base,
                       "true_obb", control_name, "control", family,
                       base_features, _features(other))

    # 2) Normalization pairing on the true-OBB cell for each headline operator.
    norm_cells = partition(("dataset_id", "model_id", "perturbation", "method"), "normalization",
                           filters=(("control", "true_obb"),))
    for key, states in norm_cells.items():
        for norm_a, norm_b in (("unit", "raw"), ("unit", "magnitude_weighted"),
                               ("raw", "magnitude_weighted")):
            ra, rb = states.get(norm_a), states.get(norm_b)
            family = f"paired_normalization:{key[0]}:{key[1]}:{key[2]}:{key[3]}:true_obb"
            if ra is not None and rb is not None:
                _emit_pair(export, "specificity_full", str(path), rb,
                           norm_a, norm_b, "normalization", family,
                           _features(ra), _features(rb))
            elif (ra is None) != (rb is None):
                _emit_missing(export, "specificity_full", norm_a, norm_b,
                              family, norm_a if ra is None else norm_b, rb if rb is not None else ra)

    # 3) Operator pairing (M0 vs M1, M1 vs M3, M0 vs M3) on true-OBB / unit.
    op_cells = partition(("dataset_id", "model_id", "perturbation"), "method",
                         filters=(("control", "true_obb"), ("normalization", "unit")))
    for key, states in op_cells.items():
        for op_a, op_b in (("raw", "euclidean_mean_subtraction"),
                           ("euclidean_mean_subtraction", "orthogonal_axis_projection"),
                           ("raw", "orthogonal_axis_projection")):
            ra, rb = states.get(op_a), states.get(op_b)
            family = f"paired_operator:{key[0]}:{key[1]}:{key[2]}:unit:true_obb"
            if ra is not None and rb is not None:
                _emit_pair(export, "specificity_full", str(path), rb,
                           op_a, op_b, "operator", family,
                           _features(ra), _features(rb))
            elif (ra is None) != (rb is None):
                _emit_missing(export, "specificity_full", op_a, op_b,
                              family, op_a if ra is None else op_b, rb if rb is not None else ra)


def _e04_2_contrasts(export: Any, path: Path, rows: list[dict[str, Any]]) -> None:
    support_rows = [
        row for row in rows
        if row.get("analysis") == "direction_specificity"
        and "failed" not in (row.get("status") or "")
    ]
    groups: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in support_rows:
        key = tuple(str(row.get(part)) for part in ("dataset_id", "model_id", "perturbation"))
        state = row.get("support_size")
        if state is None:
            continue
        groups[key][str(state)] = row
    for key, states in groups.items():
        for size_a, size_b in (("15", "30"), ("30", "50"), ("15", "50")):
            ra, rb = states.get(size_a), states.get(size_b)
            family = f"paired_support:{key[0]}:{key[1]}:{key[2]}"
            if ra is not None and rb is not None:
                _emit_pair(export, "support_scaling", str(path), rb,
                           size_a, size_b, "support", family,
                           _features(ra), _features(rb))
            elif (ra is None) != (rb is None):
                ref = rb if rb is not None else ra
                _emit_missing(export, "support_scaling", size_a, size_b,
                              family, size_a if ra is None else size_b, ref)


def _e04_5_contrasts(export: Any, path: Path, rows: list[dict[str, Any]]) -> None:
    """Paired variant-vs-nominal perturbation-strength contrasts.

    Rows are grouped by (dataset, model, perturbation, method, normalization);
    every non-nominal ``variant_id`` is paired against the run's nominal cell
    over the shared source-image units, so the paired cluster bootstrap and the
    two-sided sign-flip null reuse the same machinery as the other recomputed
    contrasts.  Cells without a shared image set stay explicitly missing.
    """
    active = [
        row for row in rows
        if row.get("status") == "completed"
        and row.get("method") in _MAIN_OPERATORS
        and row.get("normalization") in _NORMALIZATIONS
        and (row.get("variant_id") or row.get("strength_variant")) is not None
    ]
    groups: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in active:
        key = tuple(str(row.get(part)) for part in
                    ("dataset_id", "model_id", "perturbation", "method", "normalization"))
        # Prefer the explicit sweep variant id; the offline runner only records
        # ``strength_variant``, so fall back to it rather than dropping the row.
        groups[key][str(row.get("variant_id") or row.get("strength_variant"))] = row
    for key, states in groups.items():
        base = states.get("nominal")
        if base is None:
            continue
        base_features = _features(base)
        for variant_id, other in states.items():
            if variant_id == "nominal":
                continue
            family = f"paired_strength:{key[0]}:{key[1]}:{key[2]}:{key[3]}:{key[4]}"
            _emit_pair(export, "strength_sweep", str(path), other, "nominal", variant_id,
                       "strength", family, base_features, _features(other))
