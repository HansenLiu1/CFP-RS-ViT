"""Layerwise common-mode decomposition and single-direction removal (S11/Fig. 7b).

Two analyses share the persisted ``layerwise_<model>_<dataset>.npz`` artifacts:

``depth``
    The class-mean origin and the class-specific residual are measured at every
    layer under the legacy object-weighted origin and the M1 class-balanced
    origin (both raw and unit deltas).

``removal``
    A readout is fitted on ``raw`` readout-fit features, then the own, wrong,
    common-mode or random class direction is removed from the confirmation
    features at every layer with the per-sample norm preserved.  Frozen linear
    and prototype readouts are both reported.

The task is CPU-only but the full 45-cell, 14-layer run is substantial; use
``--mode depth`` or ``--layers L01,final`` for a smoke check.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cfra.analysis.common import (
    DATASETS,
    EPS,
    FIT_LIMIT,
    MODELS,
    PERTURBATIONS,
    RunLayout,
    common_mode,
    fit_operator,
    limit_fit,
    unit,
)

LAYERS = ("patch_emb",) + tuple(f"L{index:02d}" for index in range(1, 13)) + ("final",)
MAIN_LAYERS = tuple(f"L{index:02d}" for index in range(1, 13)) + ("final",)
FORMS = ("legacy", "class_balanced", "class_balanced_unit")
DELTA_KINDS = ("unit", "raw")
CONDITIONS = ("own", "wrong", "common", "random")
BUDGETS = ("fit15", "fitall")
READOUTS = ("linear", "prototype")

warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")


def _meta(layout: RunLayout, seed: int, model: str, dataset: str) -> dict:
    npz_path = layout.layerwise_path(seed, model, dataset)
    json_path = npz_path.with_suffix(".json")
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"missing layerwise artifact for {seed} {model} {dataset}: {npz_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def _meta_arrays(meta: dict) -> dict[str, np.ndarray]:
    rows = meta["row_metadata"]
    return {
        "roles": np.asarray([row["role"] for row in rows]),
        "perturbations": np.asarray([row["perturbation"] for row in rows]),
        "labels": np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64),
        "objects": np.asarray([row["object_id"] for row in rows]),
        "excluded": np.asarray([bool(row.get("excluded_from_main", False)) for row in rows]),
    }


def _fit_indices(arrays: dict[str, np.ndarray], perturbation: str, limit: int | None) -> np.ndarray:
    mask = (
        (arrays["roles"] == "direction_fit")
        & (arrays["perturbations"] == perturbation)
        & ~arrays["excluded"]
    )
    if limit is None:
        return np.flatnonzero(mask)
    return np.flatnonzero(limit_fit(mask, arrays["labels"], arrays["objects"], limit))


def _stats(values: np.ndarray, labels: np.ndarray, form: str) -> dict:
    classes = np.unique(labels)
    class_means = np.stack([values[labels == klass].mean(axis=0) for klass in classes.tolist()])
    unit_class_means = unit(class_means)
    mean_d = float(np.mean(np.linalg.norm(values, axis=1)))
    if form == "class_balanced_unit":
        origin = unit_class_means.mean(axis=0)
        axis = unit(origin[None, :])[0]
        cosines = unit_class_means @ axis
        orthogonal = unit_class_means - cosines[:, None] * axis[None, :]
        class_residual = float(np.mean(np.linalg.norm(orthogonal, axis=1)))
        share = float(np.mean(cosines))
        return {
            "n": int(len(labels)),
            "n_classes": int(len(classes)),
            "mean_d": mean_d,
            "norm_origin": float(np.linalg.norm(origin)),
            "classres": class_residual,
            "A_share": share,
            "classres_share": class_residual,
            "classres_over_A": class_residual / max(share, EPS),
        }
    if form == "legacy":
        origin = values.mean(axis=0)
    elif form == "class_balanced":
        origin = unit_class_means.mean(axis=0)
    else:
        raise ValueError(form)
    class_residual = float(np.mean(np.linalg.norm(class_means - origin, axis=1)))
    norm_origin = float(np.linalg.norm(origin))
    return {
        "n": int(len(labels)),
        "n_classes": int(len(classes)),
        "mean_d": mean_d,
        "norm_origin": norm_origin,
        "classres": class_residual,
        "A_share": norm_origin / max(mean_d, EPS),
        "classres_share": class_residual / max(mean_d, EPS),
        "classres_over_A": class_residual / max(norm_origin, EPS),
    }


def _depth(
    layout: RunLayout,
    seed: int,
    model: str,
    datasets: tuple[str, ...],
    perturbations: tuple[str, ...],
    layers: tuple[str, ...],
    max_cells: int,
) -> list[dict]:
    rows = []
    for dataset in datasets:
        meta = _meta(layout, seed, model, dataset)
        arrays = _meta_arrays(meta)
        picks = {perturbation: _fit_indices(arrays, perturbation, FIT_LIMIT) for perturbation in perturbations}
        with np.load(layout.layerwise_path(seed, model, dataset), allow_pickle=False) as payload:
            for layer in layers:
                if f"{layer}__clear" not in payload:
                    continue
                clear = payload[f"{layer}__clear"]
                degraded = payload[f"{layer}__degraded"]
                for perturbation in perturbations:
                    index = picks[perturbation]
                    if len(index) < 4:
                        continue
                    delta = np.asarray(clear[index], dtype=np.float64) - np.asarray(degraded[index], dtype=np.float64)
                    norms = np.linalg.norm(delta, axis=1)
                    keep = norms > EPS
                    delta, labels = delta[keep], arrays["labels"][index][keep]
                    if len(np.unique(labels)) < 3:
                        continue
                    variants = {"raw": delta, "unit": unit(delta)}
                    for kind in DELTA_KINDS:
                        for form in FORMS:
                            rows.append(
                                dict(
                                    seed=seed,
                                    model=model,
                                    dataset=dataset,
                                    perturbation=perturbation,
                                    layer=layer,
                                    support="fit15",
                                    readout="none",
                                    condition="all",
                                    form=form,
                                    deltas=kind,
                                    ** _stats(variants[kind], labels, form),
                                )
                            )
                del clear, degraded
                gc.collect()
                print(f"  depth {seed} {model} {dataset} {layer}", flush=True)
        if max_cells and len(datasets) > 1:
            pass
    return rows


def _remove(features: np.ndarray, labels: np.ndarray, direction_map: dict[int, np.ndarray], alpha: float = 1.0) -> np.ndarray:
    directions = unit(np.stack([direction_map[int(label)] for label in labels.tolist()]))
    projection = np.sum(features * directions, axis=1, keepdims=True)
    transformed = features - alpha * projection * directions
    original = np.linalg.norm(features, axis=1, keepdims=True)
    updated = np.linalg.norm(transformed, axis=1, keepdims=True)
    return transformed * (original / np.clip(updated, EPS, None))


def _nearest_wrong(centroids: np.ndarray, classes: np.ndarray) -> dict[int, int]:
    similarity = centroids @ centroids.T
    np.fill_diagonal(similarity, -np.inf)
    return {
        int(class_id): int(classes[int(np.argmax(similarity[index]))])
        for index, class_id in enumerate(classes.tolist())
    }


def _margin_accuracy(scores: np.ndarray, classes: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    classes = np.asarray(classes)
    position = {int(class_id): index for index, class_id in enumerate(classes.tolist())}
    predicted = classes[np.argmax(scores, axis=1)]
    accuracy = float(np.mean(predicted == labels))
    rows = np.arange(len(labels))
    own = scores[rows, np.asarray([position[int(label)] for label in labels.tolist()])]
    blocked = scores.copy()
    blocked[rows, np.asarray([position[int(label)] for label in labels.tolist()])] = -np.inf
    other = blocked.max(axis=1)
    return accuracy, float(np.mean(own - other))


def _random_directions(
    seed: int,
    model: str,
    dataset: str,
    perturbation: str,
    layer: str,
    classes: np.ndarray,
    dim: int,
) -> dict[int, np.ndarray]:
    key = f"{seed}:{model}:{dataset}:{perturbation}:{layer}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    return {int(class_id): unit(rng.normal(size=(1, dim)))[0] for class_id in classes.tolist()}


def _removal(
    layout: RunLayout,
    seed: int,
    model: str,
    datasets: tuple[str, ...],
    perturbations: tuple[str, ...],
    layers: tuple[str, ...],
    budgets: tuple[str, ...],
    max_cells: int,
) -> list[dict]:
    rows = []
    for dataset in datasets:
        meta = _meta(layout, seed, model, dataset)
        arrays = _meta_arrays(meta)
        picks = {}
        for perturbation in perturbations:
            fit = _fit_indices(arrays, perturbation, None)
            readout_fit = np.flatnonzero(
                (arrays["roles"] == "readout_fit")
                & (arrays["perturbations"] == perturbation)
                & ~arrays["excluded"]
            )
            confirmation = np.flatnonzero(
                (arrays["roles"] == "confirmation")
                & (arrays["perturbations"] == perturbation)
                & ~arrays["excluded"]
            )
            if min(len(fit), len(readout_fit), len(confirmation)) < 3:
                continue
            fits = {}
            mask = np.zeros(len(arrays["roles"]), dtype=bool)
            mask[fit] = True
            for budget in budgets:
                fits[budget] = fit if budget == "fitall" else np.flatnonzero(
                    limit_fit(mask, arrays["labels"], arrays["objects"], FIT_LIMIT)
                )
            picks[perturbation] = (readout_fit, confirmation, fits)

        with np.load(layout.layerwise_path(seed, model, dataset), allow_pickle=False) as payload:
            for layer in layers:
                if f"{layer}__raw" not in payload:
                    continue
                raw = payload[f"{layer}__raw"]
                clear = payload[f"{layer}__clear"]
                degraded = payload[f"{layer}__degraded"]
                for perturbation in perturbations:
                    if perturbation not in picks:
                        continue
                    readout_fit, confirmation, fits = picks[perturbation]
                    fit_raw = np.asarray(raw[readout_fit], dtype=np.float64)
                    center = fit_raw.mean(axis=0)
                    centered_fit = fit_raw - center
                    centered_eval = np.asarray(raw[confirmation], dtype=np.float64) - center
                    eval_labels = arrays["labels"][confirmation]

                    classifier = LogisticRegression(max_iter=1000, random_state=seed)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        classifier.fit(centered_fit, arrays["labels"][readout_fit])
                    classes = np.asarray(classifier.classes_, dtype=np.int64)
                    centroids = np.stack(
                        [centered_fit[arrays["labels"][readout_fit] == class_id].mean(axis=0) for class_id in classes]
                    )
                    centroids = unit(centroids)
                    keep = np.isin(eval_labels, classes)
                    centered_eval, eval_labels = centered_eval[keep], eval_labels[keep]
                    if len(eval_labels) < 3:
                        continue

                    baseline_linear = _margin_accuracy(classifier.decision_function(centered_eval), classes, eval_labels)
                    baseline_prototype = _margin_accuracy(unit(centered_eval) @ centroids.T, classes, eval_labels)
                    wrong = _nearest_wrong(centroids, classes)
                    random_maps = _random_directions(seed, model, dataset, perturbation, layer, classes, centered_eval.shape[1])

                    for budget in budgets:
                        index = fits[budget]
                        values, valid = _fit_values(clear[index], degraded[index])
                        fit_labels = arrays["labels"][index][valid]
                        operator, directions, direction_classes = fit_operator(values, fit_labels, "M1")
                        class_index = {int(class_id): index for index, class_id in enumerate(direction_classes.tolist())}
                        axis = unit(common_mode(operator)[None, :])[0]
                        maps = {
                            "own": {
                                int(class_id): directions[class_index[int(class_id)]]
                                for class_id in classes.tolist()
                                if int(class_id) in class_index
                            },
                            "wrong": {
                                int(class_id): directions[class_index[int(wrong[int(class_id)])]]
                                for class_id in classes.tolist()
                                if int(class_id) in class_index and int(wrong[int(class_id)]) in class_index
                            },
                            "common": {int(class_id): axis for class_id in classes.tolist() if int(class_id) in class_index},
                            "random": random_maps,
                        }
                        for condition in CONDITIONS:
                            direction_map = maps[condition]
                            if not all(int(label) in direction_map for label in eval_labels.tolist()):
                                continue
                            transformed = _remove(centered_eval, eval_labels, direction_map)
                            linear = _margin_accuracy(classifier.decision_function(transformed), classes, eval_labels)
                            prototype = _margin_accuracy(unit(transformed) @ centroids.T, classes, eval_labels)
                            for readout, after, baseline in (
                                ("linear", linear, baseline_linear),
                                ("prototype", prototype, baseline_prototype),
                            ):
                                rows.append(
                                    dict(
                                        seed=seed,
                                        model=model,
                                        dataset=dataset,
                                        perturbation=perturbation,
                                        layer=layer,
                                        support=budget,
                                        readout=readout,
                                        condition=condition,
                                        form="centered_norm_preserved_alpha1",
                                        drop=baseline[0] - after[0],
                                        margin_drop=baseline[1] - after[1],
                                        baseline_accuracy=baseline[0],
                                        baseline_margin=baseline[1],
                                        n_eval=int(len(eval_labels)),
                                        n_readout_fit=int(len(readout_fit)),
                                        n_direction_fit=int(len(index)),
                                        n_direction_classes=int(len(direction_classes)),
                                    )
                                )
                del raw, clear, degraded
                gc.collect()
                print(f"  removal {seed} {model} {dataset} {layer}", flush=True)
    return rows


def _fit_values(clear: np.ndarray, degraded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(clear, dtype=np.float64) - np.asarray(degraded, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    valid = norms > EPS
    return unit(values[valid]), valid


def _seed_summary(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    per_seed = frame.groupby(["seed", *keys])[values].mean().reset_index()
    grouped = per_seed.groupby(keys)[values]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out.columns = [f"{left}_{right}" if right else left for left, right in out.columns.to_flat_index()]
    out = out.rename(columns={f"{metric}_mean": metric for metric in values})
    out = out.rename(columns={f"{metric}_std": f"{metric}_seed_sd" for metric in values})
    out = out.rename(columns={f"{metric}_count": "n_seeds" for metric in values})
    for metric in values[1:]:
        out = out.drop(columns=[f"n_seeds_{metric}"], errors="ignore")
    out = out.rename(columns={f"n_seeds_{values[0]}": "n_seeds"})
    out.insert(0, "seed", "mean3")
    return out


def _depth_summaries(detail: pd.DataFrame, out: Path) -> None:
    # The class-balanced origin is defined on unit class directions; its raw
    # share is not a meaningful quantity, so keep raw only for the legacy form.
    detail = detail[(detail.deltas == "unit") | (detail.form == "legacy")].reset_index(drop=True)
    detail.to_csv(out / "layerwise_depth_per_seed.csv", index=False)
    keys = ["form", "deltas", "layer"]
    summary = _seed_summary(detail, keys, ["A_share", "classres_share", "classres_over_A"])
    summary["layer"] = pd.Categorical(summary["layer"], categories=LAYERS, ordered=True)
    summary.sort_values(["form", "deltas", "layer"]).to_csv(out / "layerwise_depth_3seed.csv", index=False)

    ref_layers = ["patch_emb", "L01", "L06", "L12", "final"]
    rows = []
    for form in FORMS:
        for kind in DELTA_KINDS:
            subset = detail[(detail.form == form) & (detail.deltas == kind)]
            for factor in ("perturbation", "model", "dataset"):
                for level in sorted(subset[factor].unique()):
                    group = subset[subset[factor] == level]
                    for layer in ref_layers:
                        part = group[group.layer == layer]
                        if part.empty:
                            continue
                        for metric in ("A_share", "classres_share", "classres_over_A"):
                            rows.append(
                                dict(
                                    form=form,
                                    deltas=kind,
                                    factor=factor,
                                    level=level,
                                    layer=layer,
                                    metric=metric,
                                    mean=float(part[metric].mean()),
                                    sd=float(part[metric].std()),
                                    n=int(part["seed"].nunique()),
                                )
                            )
    pd.DataFrame(rows).to_csv(out / "layerwise_depth_breakdown.csv", index=False)

    rows = []
    for form in FORMS:
        for kind in DELTA_KINDS:
            subset = detail[(detail.form == form) & (detail.deltas == kind)]
            if subset.empty:
                continue
            mean = subset.groupby(["layer", "seed"])[["A_share", "classres_share", "classres_over_A"]].mean()
            mean = mean.groupby("layer").agg(["mean", "std"])
            for layer in ("L01", "L02", "L04", "L06", "L08", "L10", "L12", "final"):
                if layer not in mean.index:
                    continue
                for metric in ("A_share", "classres_share", "classres_over_A"):
                    rows.append(
                        dict(
                            form=form,
                            deltas=kind,
                            item=f"{metric}@{layer}",
                            layer=layer,
                            value=float(mean.loc[layer, (metric, "mean")]),
                            sd=float(mean.loc[layer, (metric, "std")]),
                        )
                    )
    pd.DataFrame(rows).to_csv(out / "layerwise_depth_headline.csv", index=False)


def _removal_summaries(detail: pd.DataFrame, out: Path) -> None:
    detail.to_csv(out / "layerwise_removal_per_seed.csv", index=False)
    keys = ["support", "readout", "condition", "layer"]
    summary = _seed_summary(detail, keys, ["drop", "margin_drop"])
    summary["layer"] = pd.Categorical(summary["layer"], categories=LAYERS, ordered=True)
    summary.sort_values(["support", "readout", "condition", "layer"]).to_csv(
        out / "layerwise_removal_3seed.csv", index=False
    )

    cells = detail.groupby(
        ["model", "dataset", "perturbation", "layer", "readout", "condition", "support"], as_index=False
    ).agg(drop_mean=("drop", "mean"), drop_sd=("drop", "std"), margin_drop_mean=("margin_drop", "mean"))
    cells.to_csv(out / "layerwise_removal_cellmean.csv", index=False)

    rows = []
    for support in sorted(detail.support.unique()):
        for readout in READOUTS:
            subset = detail[(detail.support == support) & (detail.readout == readout)]
            if subset.empty:
                continue
            wide = subset.pivot_table(
                index=["seed", "model", "dataset", "perturbation", "layer"],
                columns="condition",
                values="drop",
                aggfunc="first",
            ).reset_index()
            for control in ("wrong", "common"):
                if "own" not in wide or control not in wide:
                    continue
                wide[f"own_minus_{control}"] = wide["own"] - wide[control]
                per_seed = wide.groupby(["seed", "layer"])[f"own_minus_{control}"].mean().reset_index()
                grouped = per_seed.groupby("layer")[f"own_minus_{control}"]
                stats = grouped.agg(["mean", "std", "count"]).reset_index()
                for row in stats.itertuples(index=False):
                    rows.append(
                        dict(
                            support=support,
                            readout=readout,
                            contrast=f"own_minus_{control}",
                            layer=row.layer,
                            mean=float(row.mean),
                            sd=float(row.std),
                            n_seeds=int(row.count),
                        )
                    )
    pd.DataFrame(rows).to_csv(out / "layerwise_removal_contrasts.csv", index=False)


def run(
    layout: RunLayout,
    out: Path,
    *,
    models: tuple[str, ...] = MODELS,
    seeds: tuple[int, ...] | None = None,
    datasets: tuple[str, ...] = DATASETS,
    perturbations: tuple[str, ...] = PERTURBATIONS,
    max_cells: int = 0,
    layers: tuple[str, ...] = LAYERS,
    budgets: tuple[str, ...] = BUDGETS,
    mode: str = "both",
) -> None:
    """Run layerwise depth and/or removal analyses."""

    if mode not in {"depth", "removal", "both"}:
        raise ValueError(mode)
    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds or layout.seeds
    if max_cells:
        pairs = [(dataset, perturbation) for dataset in datasets for perturbation in perturbations][:max_cells]
        datasets = tuple(dict.fromkeys(dataset for dataset, _ in pairs))
        perturbations = tuple(dict.fromkeys(perturbation for _, perturbation in pairs))
    depth_rows: list[dict] = []
    removal_rows: list[dict] = []
    for seed in seeds:
        for model in models:
            if mode in {"depth", "both"}:
                depth_rows.extend(
                    _depth(layout, seed, model, datasets, perturbations, layers, max_cells)
                )
            if mode in {"removal", "both"}:
                removal_rows.extend(
                    _removal(layout, seed, model, datasets, perturbations, layers, budgets, max_cells)
                )
    if depth_rows:
        _depth_summaries(pd.DataFrame(depth_rows), out)
    if removal_rows:
        _removal_summaries(pd.DataFrame(removal_rows), out)


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
    parser.add_argument("--layers", default=",".join(LAYERS))
    parser.add_argument("--budgets", default="fit15,fitall")
    parser.add_argument("--mode", choices=("depth", "removal", "both"), default="both")
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
        layers=tuple(item for item in args.layers.split(",") if item),
        budgets=tuple(item for item in args.budgets.split(",") if item),
        mode=args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
