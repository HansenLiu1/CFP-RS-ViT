"""Readout transfer baselines (Table III, S5e, S5i).

For each cell the task compares the pipeline M1 class directions with:

* raw prototype directions (class mean minus global mean);
* the E03 unit-class-mean prototype;
* clear/degraded CAV directions;
* a validation-selected z-score fusion of prototype and CFP scores.

Queries are clean-ROI (confirmation) features, either raw or centred by the
readout-fit mean.  Direction and readout budgets are reported explicitly as
``fit15`` (15 objects per class in sha256 order) and ``all``.
"""

from __future__ import annotations

import argparse
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
    fit_operator,
    limit_fit,
    unit,
)
from cfra.evaluation.metrics import macro_ovr_auc, macro_ovr_auc_from_scores
from cfra.methods.baselines import cav_direction, fit_clear_degraded_cav, fit_raw_prototype_directions
from cfra.representations.deltas import cosine_scores

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _readout_mask(feat: Features, dataset: str, perturbation: str, budget: str) -> np.ndarray:
    mask = feat.mask(dataset, perturbation, "readout_fit")
    if budget == "all":
        return np.flatnonzero(mask)
    if budget != "fit15":
        raise ValueError(budget)
    limited = limit_fit(mask, feat.meta["class_id"], feat.meta["object_id"], FIT_LIMIT)
    return np.flatnonzero(limited)


def _valid_rows(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values).all(axis=1) & (np.linalg.norm(values, axis=1) > EPS)


def _directions_for(feat: Features, dataset: str, perturbation: str, budget: str):
    cell = cell_delta(
        feat,
        dataset,
        perturbation,
        "direction_fit",
        limit=FIT_LIMIT if budget == "fit15" else None,
    )
    _operator, directions, classes = fit_operator(cell["values"], cell["labels"], "M1")
    return np.asarray(directions, dtype=np.float64), np.asarray(classes, dtype=np.int64), len(cell["values"])


def _prototypes(readout_features: np.ndarray, labels: np.ndarray, classes: np.ndarray):
    raw = fit_raw_prototype_directions(readout_features, labels)
    prototype = np.stack([raw[int(class_id)] for class_id in classes])
    e03 = unit(np.stack([readout_features[labels == class_id].mean(axis=0) for class_id in classes]))
    return prototype, e03


def _cavs(
    clear: np.ndarray,
    degraded: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    center: np.ndarray,
):
    raw = []
    centered = []
    for class_id in classes.tolist():
        selected = labels == class_id
        raw.append(cav_direction(fit_clear_degraded_cav(clear[selected], degraded[selected], class_id=int(class_id))))
        centered.append(
            cav_direction(
                fit_clear_degraded_cav(
                    clear[selected] - center,
                    degraded[selected] - center,
                    class_id=int(class_id),
                )
            )
        )
    return np.stack(raw), np.stack(centered)


def _fusion_auc(
    feat: Features,
    dataset: str,
    perturbation: str,
    prototype: np.ndarray,
    prototype_classes: np.ndarray,
    cfp: np.ndarray,
    cfp_classes: np.ndarray,
    query: np.ndarray,
    query_labels: np.ndarray,
):
    validation_index = np.flatnonzero(feat.mask(dataset, perturbation, "validation"))
    validation = np.asarray(feat.array("raw")[validation_index], dtype=np.float64)
    good = _valid_rows(validation)
    validation_index, validation = validation_index[good], validation[good]
    validation_labels = feat.meta["class_id"][validation_index].astype(np.int64)
    shared = np.asarray(sorted(set(prototype_classes.tolist()) & set(cfp_classes.tolist())), dtype=np.int64)
    if len(shared) < 2 or len(validation) == 0:
        return float("nan"), float("nan"), 0
    prototype_position = {int(class_id): index for index, class_id in enumerate(prototype_classes.tolist())}
    cfp_position = {int(class_id): index for index, class_id in enumerate(cfp_classes.tolist())}
    prototype_matrix = np.stack([prototype[prototype_position[int(class_id)]] for class_id in shared])
    cfp_matrix = np.stack([cfp[cfp_position[int(class_id)]] for class_id in shared])
    keep = np.isin(validation_labels, shared)
    prototype_scores = cosine_scores(validation[keep], prototype_matrix)
    cfp_scores = cosine_scores(validation[keep], cfp_matrix)
    validation_labels = validation_labels[keep]
    if len(validation_labels) == 0:
        return float("nan"), float("nan"), 0
    p_mean, p_std = prototype_scores.mean(axis=0), np.where(prototype_scores.std(axis=0) > 1e-8, prototype_scores.std(axis=0), 1.0)
    c_mean, c_std = cfp_scores.mean(axis=0), np.where(cfp_scores.std(axis=0) > 1e-8, cfp_scores.std(axis=0), 1.0)
    aucs = []
    for weight in LAMBDAS:
        fused = (1.0 - weight) * (prototype_scores - p_mean) / p_std + weight * (cfp_scores - c_mean) / c_std
        aucs.append(macro_ovr_auc_from_scores(fused, validation_labels, shared))
    weight = float(LAMBDAS[int(np.nanargmax(aucs))]) if np.isfinite(aucs).any() else 0.0
    keep = np.isin(query_labels, shared)
    z_prototype = (cosine_scores(query[keep], prototype_matrix) - p_mean) / p_std
    z_cfp = (cosine_scores(query[keep], cfp_matrix) - c_mean) / c_std
    fused = (1.0 - weight) * z_prototype + weight * z_cfp
    return float(macro_ovr_auc_from_scores(fused, query_labels[keep], shared)), weight, len(validation_labels)


def _cell(seed: int, model: str, feat: Features, dataset: str, perturbation: str) -> list[dict]:
    labels_all = feat.meta["class_id"].astype(np.int64)
    raw = feat.array("raw")
    clear_all = feat.array("true_clear_degraded")
    degraded_all = feat.array("true_degraded_degraded")

    query_index = np.flatnonzero(feat.mask(dataset, perturbation, "confirmation"))
    query_full = raw[query_index].astype(np.float64)
    good = _valid_rows(query_full)
    query_index, query_full = query_index[good], query_full[good]
    query_labels_full = labels_all[query_index]

    directions = {budget: _directions_for(feat, dataset, perturbation, budget) for budget in ("fit15", "all")}
    common = np.asarray(
        sorted(set(directions["fit15"][1].tolist()) & set(directions["all"][1].tolist())),
        dtype=np.int64,
    )
    if len(common) < 2:
        return []

    def aligned(matrix: np.ndarray, classes: np.ndarray) -> np.ndarray:
        position = {int(class_id): index for index, class_id in enumerate(classes.tolist())}
        return np.stack([matrix[position[int(class_id)]] for class_id in common])

    dirs15 = aligned(directions["fit15"][0], directions["fit15"][1])
    keep_query = np.isin(query_labels_full, common)
    query, query_labels = query_full[keep_query], query_labels_full[keep_query]

    prototypes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cavs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    centers: dict[str, np.ndarray] = {}
    n_fit: dict[str, int] = {}
    for budget in ("fit15", "all"):
        readout_index = _readout_mask(feat, dataset, perturbation, budget)
        labels = labels_all[readout_index]
        keep = np.isin(labels, common)
        readout_index, labels = readout_index[keep], labels[keep]
        readout = raw[readout_index].astype(np.float64)
        n_fit[budget] = len(labels)
        centers[budget] = readout.mean(axis=0)
        prototypes[budget] = _prototypes(readout, labels, common)
        cavs[budget] = _cavs(
            clear_all[readout_index].astype(np.float64),
            degraded_all[readout_index].astype(np.float64),
            labels,
            common,
            centers[budget],
        )

    ident = dict(seed=seed, model=model, dataset=dataset, perturbation=perturbation, n_classes=len(common))
    rows = []
    for budget in ("fit15", "all"):
        for name, matrix in (
            ("prototype", prototypes[budget][0]),
            ("prototype_e03", prototypes[budget][1]),
            ("CAV", cavs[budget][0]),
            ("CAV_centered_fit", cavs[budget][1]),
        ):
            for space, values in (("query_raw", query), ("query_centered", query - centers[budget])):
                rows.append(
                    dict(
                        **ident,
                        form=name,
                        support=budget,
                        space=space,
                        readout=name.split("_")[0],
                        mode="n/a",
                        auc=macro_ovr_auc(values, query_labels, matrix, common),
                        n_fit=n_fit[budget],
                        n_query=len(query_labels),
                        n_fit_dirs=directions[budget][2],
                    )
                )

    for budget in ("all", "fit15"):
        for name, matrix in (("prototype", prototypes[budget][0]), ("CAV", cavs[budget][0])):
            rows.append(
                dict(
                    **ident,
                    form=name,
                    support=budget,
                    space="query_raw",
                    readout=name,
                    mode="n/a",
                    auc=macro_ovr_auc(query, query_labels, matrix, common),
                    n_fit=n_fit[budget],
                    n_query=len(query_labels),
                    n_fit_dirs=directions[budget][2],
                )
            )
        rows.append(
            dict(
                **ident,
                form="CFP",
                support="fit15",
                space="query_raw",
                readout="CFP",
                mode="n/a",
                auc=macro_ovr_auc(query, query_labels, dirs15, common),
                n_fit=n_fit[budget],
                n_query=len(query_labels),
                n_fit_dirs=directions["fit15"][2],
            )
        )
        fused, weight, n_validation = _fusion_auc(
            feat,
            dataset,
            perturbation,
            prototypes[budget][0],
            common,
            dirs15,
            common,
            query,
            query_labels,
        )
        rows.append(
            dict(
                **ident,
                form="fusion",
                support=budget,
                space="query_raw",
                readout="fusion",
                mode="n/a",
                auc=fused,
                n_fit=n_fit[budget],
                n_query=len(query_labels),
                n_fit_dirs=directions["fit15"][2],
                fusion_weight=weight,
                n_validation=n_validation,
            )
        )

    for budget in ("fit15", "all"):
        for space, values in (("query_raw", query), ("query_centered", query - centers[budget])):
            rows.append(
                dict(
                    **ident,
                    form="CFP_transfer",
                    support=budget,
                    space=space,
                    readout="CFP",
                    mode="n/a",
                    auc=macro_ovr_auc(values, query_labels, directions[budget][0], common),
                    n_fit=n_fit[budget],
                    n_query=len(query_labels),
                    n_fit_dirs=directions[budget][2],
                )
            )
    return rows


def _aggregate(frame: pd.DataFrame, keys: list[str], value: str = "auc") -> pd.DataFrame:
    per_seed = frame.groupby(["seed", *keys])[value].mean().reset_index()
    grouped = per_seed.groupby(keys)[value]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": f"{value}_mean", "std": f"{value}_sd", "count": "n_seeds"})
    out.insert(0, "seed", "3seed")
    return out


def run(
    layout: RunLayout,
    out: Path,
    *,
    models: tuple[str, ...] = MODELS,
    seeds: tuple[int, ...] | None = None,
    datasets: tuple[str, ...] = DATASETS,
    perturbations: tuple[str, ...] = PERTURBATIONS,
    max_cells: int = 0,
) -> None:
    """Write per-seed transfer rows, Table III, S5e and S5i summaries."""

    out.mkdir(parents=True, exist_ok=True)
    parts = out / "detail"
    parts.mkdir(exist_ok=True)
    seeds = seeds or layout.seeds
    rows: list[dict] = []
    for seed in seeds:
        for model in models:
            path = parts / f"{seed}_{model}.csv"
            if max_cells == 0 and path.exists():
                rows.extend(pd.read_csv(path).to_dict("records"))
                continue
            feat = Features(layout, seed, model)
            current: list[dict] = []
            try:
                count = 0
                for dataset in datasets:
                    for perturbation in perturbations:
                        current.extend(_cell(seed, model, feat, dataset, perturbation))
                        count += 1
                        print(f"  {seed} {model} {dataset}/{perturbation}", flush=True)
                        if max_cells and count >= max_cells:
                            break
                    if max_cells and count >= max_cells:
                        break
            finally:
                feat.close()
            pd.DataFrame(current).to_csv(path, index=False)
            rows.extend(current)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame.to_csv(out / "readout_transfer_per_seed.csv", index=False)
    _aggregate(frame, ["form", "support", "space"]).to_csv(out / "readout_transfer_summary.csv", index=False)
    _aggregate(frame, ["form", "support", "space", "perturbation"]).to_csv(
        out / "readout_transfer_by_perturbation.csv", index=False
    )
    _aggregate(frame, ["form", "support", "space", "model"]).to_csv(
        out / "readout_transfer_by_model.csv", index=False
    )

    table3 = frame[frame["form"].isin(["prototype", "CAV", "CFP", "fusion"]) & (frame["space"] == "query_raw")]
    _aggregate(table3, ["form", "support"]).to_csv(out / "readout_transfer_table_iii.csv", index=False)

    s5e = frame[frame["form"].isin(["prototype", "prototype_e03", "CAV", "CAV_centered_fit"])]
    _aggregate(s5e, ["form", "support", "space"]).to_csv(out / "readout_transfer_s5e.csv", index=False)

    s5i = frame[frame["form"] == "CFP_transfer"]
    rows_s5i = []
    for budget in ("fit15", "all"):
        subset = s5i[s5i["support"] == budget]
        if subset.empty:
            continue
        wide = subset.pivot_table(
            index=["seed", "model", "dataset", "perturbation"],
            columns="space",
            values="auc",
        ).reset_index()
        if "query_raw" not in wide or "query_centered" not in wide:
            continue
        wide["diff"] = wide["query_centered"] - wide["query_raw"]
        per_seed = wide.groupby("seed")[["query_raw", "query_centered", "diff"]].mean()
        rows_s5i.append(
            dict(
                seed="3seed",
                form="CFP_transfer",
                support=budget,
                space="raw_and_centered",
                readout="CFP",
                auc_raw=float(per_seed["query_raw"].mean()),
                auc_raw_sd=float(per_seed["query_raw"].std(ddof=1)),
                auc_centered=float(per_seed["query_centered"].mean()),
                auc_centered_sd=float(per_seed["query_centered"].std(ddof=1)),
                diff=float(per_seed["diff"].mean()),
                cells=int((wide["diff"] > 0).sum()),
                n_cells=len(wide),
            )
        )
    if rows_s5i:
        pd.DataFrame(rows_s5i).to_csv(out / "readout_transfer_s5i.csv", index=False)

    fusion = frame[frame["form"] == "fusion"]
    if not fusion.empty:
        weights = (
            fusion.groupby(["support", "seed"])["fusion_weight"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        weights.to_csv(out / "readout_transfer_fusion_weights.csv", index=False)


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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
