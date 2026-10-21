"""Unified source-image cluster intervals (S13.1-S13.4).

The final paper uses one shared source-image draw per dataset and replicate,
shared by all seeds, models, perturbations, operators and controls.  This task
implements that rule for the four interval tables:

``rank``      Table I / S13.1 (M1, blur, fit15)
``operators`` Table II / S13.2 (M0-M6 and the M1 label permutation)
``readout``   Table IV / S13.3 (alpha=1 centred readout removal)
``regions``   Table V / S13.4 (region and geometry controls)

The point estimates are the same as the corresponding non-interval task.  The
interval is a two-level percentile bootstrap: an inner source-image draw shared
across a dataset, and an outer cell bootstrap whose drawn cells use the same
inner replicate index.  The no-outer variant is also written.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cfra.analysis.common import (
    DATASETS,
    EPS,
    FIT_LIMIT,
    MODELS,
    PERTURBATIONS,
    Features,
    RunLayout,
    cell_delta,
    cell_rows,
    fit_operator,
)
from cfra.analysis.tasks import readout_removal
from cfra.evaluation.metrics import specificity_metrics
from cfra.methods import fit_geometry_residualizer
from cfra.pipeline import core

MODELS4 = ("remoteclip", "clip", "siglip2", "dinov2")
METRICS = ("own_argmax_rate", "rank2_rate", "mean_diagonal_margin")
CONTROLS = (
    "true_obb",
    "mask_dilated",
    "geometry_regression",
    "background_only",
    "mask_eroded",
    "boundary_only",
    "shifted_obb",
    "random_geometry_matched_obb",
    "label_permutation",
)
CONTROL_LABELS = {
    "true_obb": "true_obb",
    "mask_dilated": "Dilated (3 px)",
    "geometry_regression": "Geometry regression",
    "background_only": "Background region",
    "mask_eroded": "Eroded (3 px)",
    "boundary_only": "Boundary ring (3 px)",
    "shifted_obb": "Shifted OBB",
    "random_geometry_matched_obb": "Random geometry-matched",
    "label_permutation": "Label permutation",
}
OPERATORS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M1_perm")
READOUT_CONDITIONS = (
    ("own_class", "Own class"),
    ("wrong_class", "Wrong class"),
    ("class_mean_axis", "Common-mode axis"),
    ("projection_matched_random", "Projection-matched"),
    ("random_unit", "Random"),
    ("classifier_weight_aligned_random", "Weight-aligned"),
)
READOUT_REFERENCE = ("linear_weight", "Own readout weight")


def _seed(prefix: str, *parts: object) -> int:
    text = prefix + "|" + "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def _b6_seed(*parts: object) -> int:
    return _seed("b6", *parts)


def _b7_seed(*parts: object) -> int:
    return _seed("b7", *parts)


def _pool(layout: RunLayout, dataset: str, models: tuple[str, ...], seeds: tuple[int, ...], cache: Path) -> np.ndarray:
    path = cache / f"pool_{dataset}.pkl"
    if path.exists():
        return np.asarray(pickle.loads(path.read_bytes()), dtype=str)
    model = "remoteclip" if "remoteclip" in models else models[0]
    values: set[str] = set()
    for seed in seeds:
        feat = Features(layout, seed, model)
        try:
            for perturbation in PERTURBATIONS:
                mask = feat.mask(dataset, perturbation, "confirmation")
                values.update(str(value) for value in np.asarray(feat.meta["source_image_id"])[mask].tolist())
        finally:
            feat.close()
    pool = np.asarray(sorted(values), dtype=str)
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(pool.tolist()))
    return pool


def _counts(dataset: str, pool: np.ndarray, replicates: int, cache: Path) -> np.ndarray:
    path = cache / f"counts_{dataset}_{replicates}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    rng = np.random.default_rng(_b7_seed("pool", dataset, replicates))
    n = len(pool)
    draws = rng.integers(0, n, size=(replicates, n), dtype=np.int32)
    counts = np.zeros((replicates, n), dtype=np.int32)
    chunk = max(1, min(256, int(2_000_000 // max(1, n))))
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        block = draws[start : start + size].astype(np.int64)
        rows = np.arange(size, dtype=np.int64)[:, None]
        flat = (block + rows * n).ravel()
        counts[start : start + size] = np.bincount(flat, minlength=size * n).reshape(size, n)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(path, counts)
    return counts


def _cell_counts(pool: np.ndarray, counts: np.ndarray, own_ids: np.ndarray) -> np.ndarray:
    own = np.asarray(sorted({str(value) for value in own_ids}), dtype=str)
    index = np.searchsorted(pool, own)
    if index.size and (index.max() >= len(pool) or not np.array_equal(pool[index], own)):
        raise KeyError("cell source image is outside the dataset confirmation pool")
    return np.asarray(counts[:, index])


def _cluster_units(row_scores: np.ndarray, labels: np.ndarray, images: np.ndarray, classes: np.ndarray):
    score = np.asarray(row_scores, dtype=np.float64)
    label_values = np.asarray(labels)
    class_values = [int(value) for value in np.asarray(classes).tolist()]
    code_values, image_codes = np.unique(np.asarray(images), return_inverse=True)
    units = np.full((len(class_values), len(code_values), score.shape[1]), np.nan, dtype=np.float64)
    for query_index, query_class in enumerate(class_values):
        units[query_index] = core._unit_cell_values(
            score, label_values, image_codes, query_class, list(range(len(code_values))), score.shape[1]
        )
    return units, np.isfinite(units).all(axis=2), code_values


def _point_metrics(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    own = np.diag(values)
    row_max = np.nanmax(values, axis=1)
    off = values.copy()
    np.fill_diagonal(off, np.nan)
    with np.errstate(all="ignore"):
        hardest = np.nanmax(off, axis=1)
    better = np.nansum(values > own[:, None], axis=1)
    valid = np.isfinite(own) & np.isfinite(hardest)
    return {
        "own_argmax_rate": float(np.mean((own >= row_max)[valid])),
        "rank2_rate": float(np.mean((better <= 1)[valid])),
        "mean_diagonal_margin": float(np.mean((own - hardest)[valid])),
    }


def _replicate_metrics(
    counts: np.ndarray,
    units: np.ndarray,
    presence: np.ndarray,
    *,
    metrics: tuple[str, ...] = METRICS,
    chunk: int = 250,
) -> dict[str, np.ndarray]:
    counts = np.asarray(counts)
    k, _n, _ = units.shape
    out = {name: np.empty(counts.shape[0], dtype=np.float64) for name in metrics}
    for start in range(0, counts.shape[0], chunk):
        size = min(chunk, counts.shape[0] - start)
        block = counts[start : start + size]
        rep = np.full((size, k, k), np.nan, dtype=np.float64)
        for query in range(k):
            keep = presence[query]
            if not keep.any():
                continue
            cp = block[:, keep]
            denominator = cp.sum(axis=1, keepdims=True)
            ok = denominator[:, 0] > 0
            rep[ok, :, query] = (cp[ok] @ units[query][keep]) / denominator[ok]
        own = np.diagonal(rep, axis1=1, axis2=2)
        row_max = np.nanmax(rep, axis=2)
        valid = np.isfinite(own) & np.isfinite(row_max)
        off = rep.copy()
        index = np.arange(k)
        off[:, index, index] = np.nan
        with np.errstate(all="ignore"):
            hardest = np.nanmax(off, axis=2)
        better = np.sum(rep > own[:, :, None], axis=2)
        values = {
            "own_argmax_rate": np.nanmean(np.where(valid, own >= row_max, np.nan), axis=1),
            "rank2_rate": np.nanmean(np.where(valid, better <= 1, np.nan), axis=1),
            "mean_diagonal_margin": np.nanmean(np.where(valid, own - hardest, np.nan), axis=1),
        }
        for name in metrics:
            out[name][start : start + size] = values[name]
    return out


def _seed_average(arrays: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(array, dtype=np.float64) for array in arrays], axis=0).mean(axis=0)


def _percentile(samples: np.ndarray) -> tuple[float, float]:
    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _outer_shared(tensor: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    tensor = np.asarray(tensor, dtype=np.float64)
    cells = tensor.shape[0]
    if cells == 1:
        return tensor[0]
    rng = np.random.default_rng(seed)
    out = np.empty(replicates, dtype=np.float64)
    chunk = 500
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        inner = np.arange(start, start + size)
        draws = rng.integers(0, cells, size=(size, cells))
        out[start : start + size] = tensor[draws, inner[:, None]].mean(axis=1)
    return out


def _interval(tensor: np.ndarray, replicates: int, seed: int, outer: bool) -> dict:
    tensor = np.asarray(tensor, dtype=np.float64)
    no_outer = tensor.mean(axis=0)
    low_no, high_no = _percentile(no_outer)
    if outer and tensor.shape[0] > 1:
        samples = _outer_shared(tensor, replicates, seed)
        low, high = _percentile(samples)
        mean = float(np.nanmean(samples))
    else:
        low, high = low_no, high_no
        mean = float(np.nanmean(no_outer))
    return {
        "ci_low": low,
        "ci_high": high,
        "ci_low_noouter": low_no,
        "ci_high_noouter": high_no,
        "replicate_mean": mean,
    }


def _bh(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(values, np.nan)
    ok = np.isfinite(values)
    if not ok.any():
        return out
    order = np.argsort(values[ok])
    ranked = values[ok][order]
    count = len(ranked)
    q = ranked * count / np.arange(1, count + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    temp = np.empty_like(q)
    temp[order] = q
    out[ok] = temp
    return out


def _sign_flip(differences: np.ndarray, *, greater: bool, replicates: int, seed: int) -> float:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    means = (rng.choice([-1.0, 1.0], size=(replicates, len(values))) * values[None, :]).mean(axis=1)
    exceed = np.count_nonzero(means >= observed - 1e-12) if greater else np.count_nonzero(means <= observed + 1e-12)
    return float((exceed + 1.0) / (replicates + 1.0))


def _permutation_rates(matrix: np.ndarray, permutations: int, seed: int) -> dict[str, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    count = values.shape[0]
    rng = np.random.default_rng(seed)
    orders = np.argsort(rng.random((permutations, count)), axis=1)
    rows = np.arange(count)
    own = values[rows[None, :], orders]
    own_rate = (own >= values.max(axis=1)[None, :]).mean(axis=1)
    better = (values[None, :, :] > own[:, :, None]).sum(axis=2)
    rank2 = (better <= 1).mean(axis=1)
    return {"own": own_rate, "rank2": rank2}


def _exact_rates(matrix: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    order = np.asarray(list(itertools.permutations(range(values.shape[0]))), dtype=np.int64)
    rows = np.arange(values.shape[0])
    own = values[rows[None, :], order]
    own_rate = (own >= values.max(axis=1)[None, :]).mean(axis=1)
    better = (values[None, :, :] > own[:, :, None]).sum(axis=2)
    rank2 = (better <= 1).mean(axis=1)
    return {"own": own_rate, "rank2": rank2}


def _rank_inputs(feat: Features, dataset: str):
    if dataset == "fair1m":
        fit = cell_delta(feat, dataset, "blur", "direction_fit", limit=FIT_LIMIT)
        evaluation = cell_delta(feat, dataset, "blur", "confirmation")
        names = np.concatenate([feat.meta["coarse_name"][fit["index"]], evaluation["coarse"]])
        mapping = {name: index for index, name in enumerate(sorted({str(value) for value in names.tolist()}))}
        fit_labels = np.asarray([mapping[str(value)] for value in feat.meta["coarse_name"][fit["index"]].tolist()], dtype=np.int64)
        eval_labels = np.asarray([mapping[str(value)] for value in evaluation["coarse"].tolist()], dtype=np.int64)
        return fit, evaluation, fit_labels, eval_labels
    fit = cell_delta(feat, dataset, "blur", "direction_fit", limit=FIT_LIMIT)
    evaluation = cell_delta(feat, dataset, "blur", "confirmation")
    return fit, evaluation, fit["labels"], evaluation["labels"]


def _rank(
    layout: RunLayout,
    out: Path,
    models: tuple[str, ...],
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    replicates: int,
    permutations: int,
) -> None:
    cache = out / "cache"
    pools = {dataset: _pool(layout, dataset, models, seeds, cache) for dataset in datasets}
    counts = {dataset: _counts(dataset, pools[dataset], replicates, cache) for dataset in datasets}
    cells: dict[tuple[str, str, int], dict] = {}
    for model in models:
        for seed in seeds:
            feat = Features(layout, seed, model)
            try:
                for dataset in datasets:
                    fit, evaluation, fit_labels, eval_labels = _rank_inputs(feat, dataset)
                    operator, directions, classes = fit_operator(fit["values"], fit_labels, "M1")
                    queries = operator.transform(evaluation["values"])
                    row_scores = (queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), EPS)) @ directions.T
                    matrix = np.asarray(
                        specificity_metrics(
                            directions,
                            evaluation["values"],
                            eval_labels,
                            classes,
                            source_image_ids=evaluation["images"],
                            scores=row_scores,
                        )["score_matrix"],
                        dtype=np.float64,
                    )
                    units, presence, clusters = _cluster_units(row_scores, eval_labels, evaluation["images"], classes)
                    own_counts = _cell_counts(pools[dataset], counts[dataset], clusters)
                    reps = _replicate_metrics(own_counts, units, presence)
                    null = _exact_rates(matrix) if dataset == "fair1m" else _permutation_rates(
                        matrix, permutations, _b6_seed("s131perm", model, dataset, seed)
                    )
                    cells[(model, dataset, seed)] = {
                        "point": _point_metrics(matrix),
                        "reps": reps,
                        "null": null,
                        "k": int(matrix.shape[0]),
                        "n_images": int(units.shape[1]),
                    }
            finally:
                feat.close()
            print(f"  rank {model} {seed}", flush=True)

    rows = []
    for model in models:
        for dataset in datasets:
            for metric in METRICS:
                point = float(np.mean([cells[(model, dataset, seed)]["point"][metric] for seed in seeds]))
                tensor = _seed_average([cells[(model, dataset, seed)]["reps"][metric] for seed in seeds])[None, :]
                bundle = _interval(tensor, replicates, _b7_seed("rank", model, dataset, metric), outer=False)
                null_key = "own" if metric == "own_argmax_rate" else "rank2" if metric == "rank2_rate" else None
                if null_key:
                    null_arrays = [cells[(model, dataset, seed)]["null"][null_key] for seed in seeds]
                    if dataset == "fair1m" and len(null_arrays) == 3:
                        a, b, c = null_arrays
                        null = ((a[:, None, None] + b[None, :, None] + c[None, None, :]) / 3.0).ravel()
                        p = float(np.mean(null >= point - 1e-12))
                    else:
                        null = np.mean(null_arrays, axis=0)
                        p = float((1 + np.count_nonzero(null >= point - 1e-12)) / (len(null) + 1))
                else:
                    p = float("nan")
                rows.append(
                    dict(
                        table="S13.1",
                        row=model,
                        column=dataset,
                        metric=metric,
                        estimate=point,
                        ci_low=bundle["ci_low"],
                        ci_high=bundle["ci_high"],
                        ci_low_noouter=bundle["ci_low_noouter"],
                        ci_high_noouter=bundle["ci_high_noouter"],
                        p=p,
                        n_cells=1,
                        replicates=replicates,
                        pool_size=int(len(pools[dataset])),
                    )
                )
        for metric in METRICS:
            point = float(np.mean([np.mean([cells[(model, dataset, seed)]["point"][metric] for seed in seeds]) for dataset in datasets]))
            tensor = np.stack(
                [_seed_average([cells[(model, dataset, seed)]["reps"][metric] for seed in seeds]) for dataset in datasets],
                axis=0,
            )
            bundle = _interval(tensor, replicates, _b7_seed("outer", "S13.1", model, "Mean", metric), outer=True)
            null_key = "own" if metric == "own_argmax_rate" else "rank2" if metric == "rank2_rate" else None
            if null_key:
                null = np.mean(
                    [
                        np.mean([cells[(model, dataset, seed)]["null"][null_key] for seed in seeds], axis=0)
                        for dataset in datasets
                    ],
                    axis=0,
                )
                p = float((1 + np.count_nonzero(null >= point - 1e-12)) / (len(null) + 1))
            else:
                p = float("nan")
            rows.append(
                dict(
                    table="S13.1", row=model, column="Mean", metric=metric, estimate=point,
                    ci_low=bundle["ci_low"], ci_high=bundle["ci_high"],
                    ci_low_noouter=bundle["ci_low_noouter"], ci_high_noouter=bundle["ci_high_noouter"],
                    p=p, n_cells=len(datasets), replicates=replicates, pool_size=float("nan"),
                )
            )

    frame = pd.DataFrame(rows)
    frame["q"] = np.nan
    for metric, group in frame.groupby("metric"):
        index = group.index[group.p.notna()]
        if len(index):
            frame.loc[index, "q"] = _bh(frame.loc[index, "p"].astype(float).values)
    frame.to_csv(out / "intervals_rank.csv", index=False)


def _region_cell(feat: Features, dataset: str, perturbation: str, control: str, seed: int, model: str):
    fit = cell_delta(feat, dataset, perturbation, "direction_fit", limit=FIT_LIMIT, control=control)
    evaluation = cell_delta(feat, dataset, perturbation, "confirmation", control=control)
    fit_values, eval_values = fit["values"], evaluation["values"]
    if control == "geometry_regression":
        fit_geometry = np.asarray(feat.meta["geometry"], dtype=np.float64)[fit["index"]]
        eval_geometry = np.asarray(feat.meta["geometry"], dtype=np.float64)[evaluation["index"]]
        residualizer = fit_geometry_residualizer(fit_values, fit_geometry)
        fit_values = residualizer.transform(fit_values, fit_geometry)
        eval_values = residualizer.transform(eval_values, eval_geometry)
    fit_labels = fit["labels"]
    if control == "label_permutation":
        rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(f"label-permutation:{seed}:{model}:{dataset}:{perturbation}".encode()).digest()[:4],
                "big",
            )
        )
        fit_labels = rng.permutation(fit_labels)
    operator, directions, classes = fit_operator(fit_values, fit_labels, "M1")
    queries = operator.transform(eval_values)
    row_scores = (queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), EPS)) @ directions.T
    matrix = np.asarray(
        specificity_metrics(
            directions,
            eval_values,
            evaluation["labels"],
            classes,
            source_image_ids=evaluation["images"],
            scores=row_scores,
        )["score_matrix"],
        dtype=np.float64,
    )
    units, presence, clusters = _cluster_units(row_scores, evaluation["labels"], evaluation["images"], classes)
    return {
        "point": _point_metrics(matrix)["own_argmax_rate"],
        "units": units,
        "presence": presence,
        "clusters": clusters,
        "k": int(matrix.shape[0]),
        "n_images": int(units.shape[1]),
    }


def _regions(
    layout: RunLayout,
    out: Path,
    models: tuple[str, ...],
    datasets: tuple[str, ...],
    perturbations: tuple[str, ...],
    seeds: tuple[int, ...],
    replicates: int,
) -> None:
    cache = out / "cache"
    pools = {dataset: _pool(layout, dataset, models, seeds, cache) for dataset in datasets}
    counts = {dataset: _counts(dataset, pools[dataset], replicates, cache) for dataset in datasets}
    results: dict[tuple, dict] = {}
    for model in models:
        for seed in seeds:
            feat = Features(layout, seed, model)
            try:
                for dataset in datasets:
                    for perturbation in perturbations:
                        entries = {
                            control: _region_cell(feat, dataset, perturbation, control, seed, model)
                            for control in CONTROLS
                        }
                        anchor_ids = np.asarray(
                            sorted({str(value) for entry in entries.values() for value in entry["clusters"].tolist()}),
                            dtype=str,
                        )
                        anchor_counts = _cell_counts(pools[dataset], counts[dataset], anchor_ids)
                        for control, entry in entries.items():
                            own_ids = np.asarray(sorted({str(value) for value in entry["clusters"].tolist()}), dtype=str)
                            index = np.searchsorted(anchor_ids, own_ids)
                            if index.size and (index.max() >= len(anchor_ids) or not np.array_equal(anchor_ids[index], own_ids)):
                                raise KeyError("region control image is outside the anchor set")
                            current_counts = np.asarray(anchor_counts[:, index])
                            reps = _replicate_metrics(
                                current_counts, entry["units"], entry["presence"], metrics=("own_argmax_rate",)
                            )["own_argmax_rate"]
                            results[(seed, model, dataset, perturbation, control)] = {
                                "point": entry["point"],
                                "reps": reps,
                                "k": entry["k"],
                                "n_images": entry["n_images"],
                            }
            finally:
                feat.close()
            print(f"  regions {model} {seed}", flush=True)

    cells = [(model, dataset, perturbation) for model in models for dataset in datasets for perturbation in perturbations]
    rows = []
    cell_points = {}
    for control in CONTROLS:
        points = np.asarray(
            [np.mean([results[(seed, model, dataset, perturbation, control)]["point"] for seed in seeds]) for model, dataset, perturbation in cells]
        )
        tensor = np.stack(
            [_seed_average([results[(seed, model, dataset, perturbation, control)]["reps"] for seed in seeds]) for model, dataset, perturbation in cells],
            axis=0,
        )
        estimate = float(points.mean())
        bundle = _interval(tensor, replicates, _b7_seed("outer", "S13.4", control), outer=True)
        cell_points[control] = points
        rows.append(
            dict(
                table="S13.4", row=CONTROL_LABELS[control], column="mean", metric="own_argmax_rate",
                estimate=estimate, ci_low=bundle["ci_low"], ci_high=bundle["ci_high"],
                ci_low_noouter=bundle["ci_low_noouter"], ci_high_noouter=bundle["ci_high_noouter"],
                n_cells=len(cells), replicates=replicates, p=float("nan"), q=float("nan"), pool_size=float("nan"),
            )
        )
    frame = pd.DataFrame(rows)
    true = cell_points["true_obb"]
    for index in range(len(frame)):
        control = CONTROLS[index]
        if control != "true_obb":
            frame.loc[index, "p"] = _sign_flip(
                cell_points[control] - true, greater=False, replicates=replicates,
                seed=_b6_seed("s134p", control),
            )
    index = frame.index[frame.row != CONTROL_LABELS["true_obb"]]
    frame.loc[index, "q"] = _bh(frame.loc[index, "p"].astype(float).values)
    frame.to_csv(out / "intervals_regions.csv", index=False)


def _operator_cell(feat: Features, dataset: str, perturbation: str, seed: int, model: str) -> dict:
    fit = cell_delta(feat, dataset, perturbation, "direction_fit", limit=FIT_LIMIT)
    evaluation = cell_delta(feat, dataset, perturbation, "confirmation")
    fit_mask = cell_rows(feat, dataset, perturbation, "direction_fit", limit=FIT_LIMIT)
    background = None
    if "M6" in OPERATORS:
        from cfra.analysis.common import background_fit

        background = background_fit(feat, fit_mask, "unit")
    out = {}
    for operator_name in OPERATORS:
        if operator_name == "M1_perm":
            rng = np.random.default_rng(
                int.from_bytes(
                    hashlib.sha256(
                        f"label-permutation:{seed}:{model}:{dataset}:{perturbation}".encode()
                    ).digest()[:4],
                    "big",
                )
            )
            labels = rng.permutation(fit["labels"])
            operator, directions, classes = fit_operator(fit["values"], labels, "M1")
        else:
            operator, directions, classes = fit_operator(
                fit["values"], fit["labels"], operator_name, background if operator_name == "M6" else None
            )
        queries = operator.transform(evaluation["values"])
        row_scores = (queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), EPS)) @ directions.T
        matrix = np.asarray(
            specificity_metrics(
                directions,
                evaluation["values"],
                evaluation["labels"],
                classes,
                source_image_ids=evaluation["images"],
                scores=row_scores,
            )["score_matrix"],
            dtype=np.float64,
        )
        units, presence, clusters = _cluster_units(row_scores, evaluation["labels"], evaluation["images"], classes)
        out[operator_name] = {
            "matrix": matrix,
            "units": units,
            "presence": presence,
            "clusters": clusters,
            "k": int(matrix.shape[0]),
            "point": _point_metrics(matrix)["own_argmax_rate"],
        }
    return out


def _operators(
    layout: RunLayout,
    out: Path,
    models: tuple[str, ...],
    datasets: tuple[str, ...],
    perturbations: tuple[str, ...],
    seeds: tuple[int, ...],
    replicates: int,
    permutations: int,
) -> None:
    cache = out / "cache"
    pools = {dataset: _pool(layout, dataset, models, seeds, cache) for dataset in datasets}
    counts = {dataset: _counts(dataset, pools[dataset], replicates, cache) for dataset in datasets}
    results: dict[tuple, dict] = {}
    for model in models:
        for seed in seeds:
            feat = Features(layout, seed, model)
            try:
                for dataset in datasets:
                    for perturbation in perturbations:
                        entries = _operator_cell(feat, dataset, perturbation, seed, model)
                        anchor_ids = np.asarray(
                            sorted({str(value) for entry in entries.values() for value in entry["clusters"].tolist()}),
                            dtype=str,
                        )
                        anchor_counts = _cell_counts(pools[dataset], counts[dataset], anchor_ids)
                        for operator_name, entry in entries.items():
                            own_ids = np.asarray(sorted({str(value) for value in entry["clusters"].tolist()}), dtype=str)
                            index = np.searchsorted(anchor_ids, own_ids)
                            if index.size and (index.max() >= len(anchor_ids) or not np.array_equal(anchor_ids[index], own_ids)):
                                raise KeyError("operator cell image is outside the anchor set")
                            current_counts = np.asarray(anchor_counts[:, index])
                            reps = _replicate_metrics(
                                current_counts, entry["units"], entry["presence"], metrics=("own_argmax_rate",)
                            )["own_argmax_rate"]
                            null = _permutation_rates(
                                entry["matrix"], permutations,
                                _b6_seed("s132perm", model, dataset, perturbation, seed, operator_name),
                            )
                            results[(seed, model, dataset, perturbation, operator_name)] = {
                                "point": entry["point"], "reps": reps, "null": null,
                                "k": entry["k"], "n_images": int(entry["units"].shape[1]),
                            }
            finally:
                feat.close()
            print(f"  operators {model} {seed}", flush=True)

    rows = []
    for operator_name in OPERATORS:
        for perturbation in perturbations:
            cells = [(model, dataset) for model in models for dataset in datasets]
            points = np.asarray(
                [np.mean([results[(seed, model, dataset, perturbation, operator_name)]["point"] for seed in seeds]) for model, dataset in cells]
            )
            tensor = np.stack(
                [_seed_average([results[(seed, model, dataset, perturbation, operator_name)]["reps"] for seed in seeds]) for model, dataset in cells],
                axis=0,
            )
            estimate = float(points.mean())
            bundle = _interval(tensor, replicates, _b7_seed("outer", "S13.2", operator_name, perturbation), outer=True)
            null = np.mean(
                [
                    np.mean([results[(seed, model, dataset, perturbation, operator_name)]["null"]["own"] for seed in seeds], axis=0)
                    for model, dataset in cells
                ],
                axis=0,
            )
            p = float((1 + np.count_nonzero(null >= estimate - 1e-12)) / (len(null) + 1))
            rows.append(
                dict(
                    table="S13.2", row=operator_name, column=perturbation.title(), metric="own_argmax_rate",
                    estimate=estimate, ci_low=bundle["ci_low"], ci_high=bundle["ci_high"],
                    ci_low_noouter=bundle["ci_low_noouter"], ci_high_noouter=bundle["ci_high_noouter"],
                    p=p, n_cells=len(cells), replicates=replicates, pool_size=float("nan"),
                )
            )
        cells = [(model, dataset, perturbation) for model in models for dataset in datasets for perturbation in perturbations]
        points = np.asarray(
            [np.mean([results[(seed, model, dataset, perturbation, operator_name)]["point"] for seed in seeds]) for model, dataset, perturbation in cells]
        )
        tensor = np.stack(
            [_seed_average([results[(seed, model, dataset, perturbation, operator_name)]["reps"] for seed in seeds]) for model, dataset, perturbation in cells],
            axis=0,
        )
        estimate = float(points.mean())
        bundle = _interval(tensor, replicates, _b7_seed("outer", "S13.2", operator_name, "Mean"), outer=True)
        null = np.mean(
            [
                np.mean([results[(seed, model, dataset, perturbation, operator_name)]["null"]["own"] for seed in seeds], axis=0)
                for model, dataset, perturbation in cells
            ],
            axis=0,
        )
        p = float((1 + np.count_nonzero(null >= estimate - 1e-12)) / (len(null) + 1))
        rows.append(
            dict(
                table="S13.2", row=operator_name, column="Mean", metric="own_argmax_rate",
                estimate=estimate, ci_low=bundle["ci_low"], ci_high=bundle["ci_high"],
                ci_low_noouter=bundle["ci_low_noouter"], ci_high_noouter=bundle["ci_high_noouter"],
                p=p, n_cells=len(cells), replicates=replicates, pool_size=float("nan"),
            )
        )
    frame = pd.DataFrame(rows)
    frame["q"] = np.nan
    for operator_name, group in frame.groupby("row"):
        index = group.index[group.p.notna()]
        if len(index):
            frame.loc[index, "q"] = _bh(frame.loc[index, "p"].astype(float).values)
    frame.to_csv(out / "intervals_operators.csv", index=False)


def _readout_capture(feat: Features, seed: int, model: str, dataset: str, perturbation: str) -> dict:
    table = readout_removal.build_table(feat, seed, model, dataset, perturbation, "fit15")
    captured: dict[tuple[str, str, str], dict] = {}
    original_space = core._e07_readout_space_rows
    original_paired = core._paired_ci

    def space_wrapper(*args, **kwargs):
        if kwargs.get("space") != "centered":
            return []
        kwargs["removal_alphas"] = []
        kwargs["reflection_alpha"] = 1.0
        return original_space(*args, **kwargs)

    def paired(differences, source_images, *, settings, seed_parts, clusters=None):
        result = original_paired(differences, source_images, settings=settings, seed_parts=seed_parts, clusters=clusters)
        if len(seed_parts) >= 7:
            condition = str(seed_parts[3])
            alpha = float(seed_parts[4])
            prefix = str(seed_parts[5])
            metric = str(seed_parts[6])
            if (
                abs(alpha - 1.0) < 1e-9
                and condition in {name for name, _ in READOUT_CONDITIONS} | {READOUT_REFERENCE[0]}
                and prefix in ("frozen_linear", "frozen_prototype")
                and metric in ("own_argmax_drop", "margin_drop")
            ):
                values = np.asarray(differences, dtype=np.float64)
                images = np.asarray(source_images)
                cluster_list = list(clusters) if clusters is not None else sorted(set(images.tolist()))
                image_values = np.asarray([values[images == cluster].mean() for cluster in cluster_list], dtype=np.float64)
                captured[(condition, prefix, metric)] = {
                    "image_values": image_values,
                    "image_ids": np.asarray(cluster_list),
                    "object_mean": float(values.mean()),
                }
        return result

    core._e07_readout_space_rows = space_wrapper
    core._paired_ci = paired
    try:
        core.run_e07(
            table,
            dataset_ids=[dataset],
            perturbations=[perturbation],
            removal_alphas=[],
            reflection_alpha=1.0,
            preserve_norm=True,
            epsilon=EPS,
            statistics_config={
                "bootstrap_replicates": 0,
                "bootstrap_ci": False,
                "permutation_replicates": 1,
                "confidence_level": 0.95,
                "cluster_unit": "source_image",
                "bootstrap_seed": 1729,
            },
            random_seed=seed,
        )
    finally:
        core._e07_readout_space_rows = original_space
        core._paired_ci = original_paired
    return captured


def _readout(
    layout: RunLayout,
    out: Path,
    models: tuple[str, ...],
    datasets: tuple[str, ...],
    perturbations: tuple[str, ...],
    seeds: tuple[int, ...],
    replicates: int,
) -> None:
    cache = out / "cache"
    pools = {dataset: _pool(layout, dataset, models, seeds, cache) for dataset in datasets}
    counts = {dataset: _counts(dataset, pools[dataset], replicates, cache) for dataset in datasets}
    captured: dict[tuple, dict] = {}
    for seed in seeds:
        for model in models:
            feat = Features(layout, seed, model)
            try:
                for dataset in datasets:
                    for perturbation in perturbations:
                        captured[(seed, model, dataset, perturbation)] = _readout_capture(
                            feat, seed, model, dataset, perturbation
                        )
            finally:
                feat.close()
            print(f"  readout {model} {seed}", flush=True)

    readouts = (
        ("Linear acc.", "frozen_linear", "own_argmax_drop"),
        ("Linear margin", "frozen_linear", "margin_drop"),
        ("Prototype acc.", "frozen_prototype", "own_argmax_drop"),
    )
    conditions = READOUT_CONDITIONS + (READOUT_REFERENCE,)
    cells = [(model, dataset, perturbation) for model in models for dataset in datasets for perturbation in perturbations]
    reps: dict[tuple[str, str, int], list[np.ndarray]] = {}
    object_point: dict[tuple[str, str, int], list[float]] = {}
    for cell_index, (model, dataset, perturbation) in enumerate(cells):
        for seed in seeds:
            entry = captured[(seed, model, dataset, perturbation)]
            for readout, prefix, metric in readouts:
                for condition, _label in conditions:
                    item = entry[(condition, prefix, metric)]
                    values = np.asarray(item["image_values"], dtype=np.float64)
                    ids = np.asarray(item["image_ids"], dtype=str)
                    order = np.argsort(ids)
                    ids, values = ids[order], values[order]
                    current_counts = _cell_counts(pools[dataset], counts[dataset], ids)
                    denominator = current_counts.sum(axis=1)
                    samples = np.divide(
                        current_counts @ values,
                        denominator,
                        out=np.full(current_counts.shape[0], np.nan, dtype=np.float64),
                        where=denominator > 0,
                    )
                    key = (readout, condition, cell_index)
                    reps.setdefault(key, []).append(samples)
                    object_point.setdefault(key, []).append(item["object_mean"])
        for readout, _prefix, _metric in readouts:
            for condition, _label in conditions:
                key = (readout, condition, cell_index)
                reps[key] = _seed_average(reps[key])
                object_point[key] = float(np.mean(object_point[key]))
        if cell_index % 10 == 0:
            print(f"  readout cells {cell_index + 1}/{len(cells)}", flush=True)

    rows = []
    for readout, _prefix, _metric in readouts:
        own_cells = np.asarray([object_point[(readout, "own_class", index)] for index in range(len(cells))])
        for condition, label in conditions:
            values = np.asarray([object_point[(readout, condition, index)] for index in range(len(cells))])
            tensor = np.stack([reps[(readout, condition, index)] for index in range(len(cells))], axis=0)
            bundle = _interval(tensor, replicates, _b7_seed("outer", "S13.3", readout, condition), outer=True)
            if condition == "own_class":
                p = float("nan")
            else:
                p = _sign_flip(
                    own_cells - values,
                    greater=condition != "classifier_weight_aligned_random",
                    replicates=replicates,
                    seed=_b6_seed("s133p", readout, condition),
                )
            rows.append(
                dict(
                    table="S13.3", row=readout, column=label, metric="drop",
                    estimate=float(values.mean()), ci_low=bundle["ci_low"], ci_high=bundle["ci_high"],
                    ci_low_noouter=bundle["ci_low_noouter"], ci_high_noouter=bundle["ci_high_noouter"],
                    p=p, n_cells=len(cells), replicates=replicates, pool_size=float("nan"),
                )
            )
    frame = pd.DataFrame(rows)
    frame["q"] = np.nan
    for readout, group in frame.groupby("row"):
        index = group.index[group.p.notna()]
        if len(index):
            frame.loc[index, "q"] = _bh(frame.loc[index, "p"].astype(float).values)
    frame.to_csv(out / "intervals_readout.csv", index=False)


def run(
    layout: RunLayout,
    out: Path,
    *,
    models: tuple[str, ...] = MODELS,
    seeds: tuple[int, ...] | None = None,
    datasets: tuple[str, ...] = DATASETS,
    perturbations: tuple[str, ...] = PERTURBATIONS,
    max_cells: int = 0,
    tables: tuple[str, ...] = ("rank", "operators", "readout", "regions"),
    replicates: int = 5000,
    permutations: int = 10000,
) -> None:
    """Run the selected interval tables."""

    out.mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)
    seeds = seeds or layout.seeds
    if max_cells:
        # Keep the interface honest: interval tables are defined over all cells.
        raise ValueError("--max-cells is not supported for interval tables; use --tables or a subset of models")
    if "rank" in tables:
        rank_models = tuple(model for model in models if model in MODELS4)
        _rank(layout, out, rank_models or MODELS4, datasets, seeds, replicates, permutations)
    if "operators" in tables:
        _operators(layout, out, models, datasets, perturbations, seeds, replicates, permutations)
    if "readout" in tables:
        _readout(layout, out, models, datasets, perturbations, seeds, replicates)
    if "regions" in tables:
        _regions(layout, out, models, datasets, perturbations, seeds, replicates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default="")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--perturbations", default=",".join(PERTURBATIONS))
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--tables", default="rank,operators,readout,regions")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = {}
    for item in args.runs.split(","):
        seed, name = item.split("=", 1)
        runs[int(seed)] = name
    run(
        RunLayout(args.run_root, runs),
        args.out,
        models=tuple(item for item in args.models.split(",") if item),
        seeds=tuple(int(item) for item in args.seeds.split(",") if item) or None,
        datasets=tuple(item for item in args.datasets.split(",") if item),
        perturbations=tuple(item for item in args.perturbations.split(",") if item),
        max_cells=args.max_cells,
        tables=tuple(item for item in args.tables.split(",") if item),
        replicates=args.replicates,
        permutations=args.permutations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
