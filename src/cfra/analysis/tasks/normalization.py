"""Normalization mechanism (Table II/S7/S9, C6 and C7).

Two linked analyses are produced:

* C6: own-class argmax of M0-M6 under ``unit``, ``raw`` and
  ``magnitude_weighted`` delta normalization;
* C7: object-level accuracy as a function of ``r / ||m||``, where ``r`` is the
  query delta norm and ``m`` is the M1 origin of the same fit, with the
  prediction ``argmin_c d_c . unit(m)`` for small ratios.

``magnitude_weighted`` is raw delta divided by one reference norm: the median
valid direction-fit norm of the same cell.  Near-zero rows are removed before
the median is taken.
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
    background_fit,
    cell_delta,
    cell_rows,
    common_mode,
    fit_operator,
    specificity,
)

NORMS = ("unit", "raw", "magnitude_weighted")
OPERATORS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
QUERY_OPERATORS = ("M0", "M1", "M2", "M3")
NBINS = 5


def _cell_inputs(feat: Features, dataset: str, perturbation: str, normalization: str):
    fit_mask = cell_rows(feat, dataset, perturbation, "direction_fit", limit=FIT_LIMIT)
    reference_norm = None
    if normalization == "magnitude_weighted":
        raw = cell_delta(
            feat,
            dataset,
            perturbation,
            "direction_fit",
            limit=FIT_LIMIT,
            normalization="raw",
        )
        reference_norm = float(np.median(raw["norms"]))
    fit = cell_delta(
        feat,
        dataset,
        perturbation,
        "direction_fit",
        limit=FIT_LIMIT,
        normalization=normalization,
        reference_norm=reference_norm,
    )
    evaluation = cell_delta(
        feat,
        dataset,
        perturbation,
        "confirmation",
        normalization=normalization,
        reference_norm=reference_norm,
    )
    background = background_fit(feat, fit_mask, normalization)
    return fit, evaluation, background, reference_norm


def _query_scores(operator, directions: np.ndarray, values: np.ndarray) -> np.ndarray:
    queries = operator.transform(values)
    left = np.linalg.norm(queries, axis=1)
    right = np.linalg.norm(directions, axis=1)
    if np.any(left <= EPS) or np.any(right <= EPS):
        raise ValueError("near-zero query or direction in object-level scoring")
    return (queries / left[:, None]) @ (directions / right[:, None]).T


def _run_cell(
    feat: Features,
    seed: int,
    model: str,
    dataset: str,
    perturbation: str,
    normalization: str,
    operators: tuple[str, ...],
) -> tuple[list[dict], list[dict]]:
    fit, evaluation, background, reference_norm = _cell_inputs(feat, dataset, perturbation, normalization)
    operator_m1, directions_m1, classes_m1 = fit_operator(
        fit["values"], fit["labels"], "M1", background
    )
    origin = common_mode(operator_m1)
    origin_norm = float(np.linalg.norm(origin))
    axis = origin / max(origin_norm, EPS)
    if normalization == "unit":
        ratio_norm = np.ones_like(evaluation["norms"], dtype=np.float64)
    elif normalization == "magnitude_weighted":
        ratio_norm = evaluation["norms"] / reference_norm
    else:
        ratio_norm = evaluation["norms"].astype(np.float64)

    rows: list[dict] = []
    object_rows: list[dict] = []
    for operator_id in operators:
        if operator_id == "M1":
            operator, directions, classes = operator_m1, directions_m1, classes_m1
        else:
            operator, directions, classes = fit_operator(
                fit["values"], fit["labels"], operator_id, background
            )
        metrics = specificity(
            operator,
            directions,
            classes,
            evaluation["values"],
            evaluation["labels"],
            evaluation["images"],
        )
        rows.append(
            dict(
                seed=seed,
                model=model,
                dataset=dataset,
                perturbation=perturbation,
                operator=operator_id,
                normalization=normalization,
                support=f"fit{FIT_LIMIT}",
                own_argmax_rate=metrics["own_argmax_rate"],
                margin=metrics["margin"],
                mean_own_cosine=metrics["mean_own_cosine"],
                macro_ovr_auc=metrics["macro_ovr_auc"],
                n_fit=int(len(fit["values"])),
                n_eval=int(len(evaluation["values"])),
                n_classes=int(len(classes)),
                reference_norm=reference_norm,
                m_norm=origin_norm,
            )
        )
        if operator_id not in QUERY_OPERATORS:
            continue
        scores = _query_scores(operator, directions, evaluation["values"])
        position = {int(class_id): index for index, class_id in enumerate(classes)}
        keep = np.asarray([int(label) in position for label in evaluation["labels"]], dtype=bool)
        truth = np.asarray([position[int(label)] for label in evaluation["labels"][keep]], dtype=np.int64)
        predictions = scores[keep].argmax(axis=1)
        projection = directions @ axis
        predicted_common = int(np.argmin(projection))
        object_rows.append(
            dict(
                dataset=dataset,
                perturbation=perturbation,
                operator=operator_id,
                normalization=normalization,
                ratio=(ratio_norm[keep] / max(origin_norm, EPS)).astype(np.float64),
                correct=predictions == truth,
                predicted_common=predictions == predicted_common,
            )
        )
    return rows, object_rows


def _quantile_bins(ratios: np.ndarray, nbins: int = NBINS) -> np.ndarray:
    order = np.argsort(ratios, kind="stable")
    ranks = np.empty(len(ratios), dtype=np.int64)
    ranks[order] = np.arange(len(ratios))
    return np.minimum((ranks * nbins) // max(len(ratios), 1), nbins - 1)


def _aggregate_objects(
    seed: int,
    model: str,
    normalization: str,
    records: list[dict],
) -> tuple[list[dict], list[dict]]:
    ratio = np.concatenate([record["ratio"] for record in records])
    bins = _quantile_bins(ratio)
    edges = [float(np.quantile(ratio, q)) for q in np.linspace(0, 1, NBINS + 1)]
    offset = 0
    detail = []
    bin_rows = []
    for record in records:
        size = len(record["ratio"])
        record_bins = bins[offset : offset + size]
        offset += size
        for index in range(NBINS):
            mask = record_bins == index
            if not mask.any():
                continue
            detail.append(
                dict(
                    seed=seed,
                    model=model,
                    dataset=record["dataset"],
                    perturbation=record["perturbation"],
                    operator=record["operator"],
                    normalization=normalization,
                    support=f"fit{FIT_LIMIT}",
                    bin=index + 1,
                    n_objects=int(mask.sum()),
                    accuracy=float(record["correct"][mask].mean()),
                    share_predicted_min_ac=float(record["predicted_common"][mask].mean()),
                    ratio_mean=float(record["ratio"][mask].mean()),
                    bin_low=edges[index],
                    bin_high=edges[index + 1],
                )
            )
    for index in range(NBINS):
        mask = bins == index
        bin_rows.append(
            dict(
                seed=seed,
                model=model,
                normalization=normalization,
                support=f"fit{FIT_LIMIT}",
                bin=index + 1,
                n_objects=int(mask.sum()),
                ratio_mean=float(ratio[mask].mean()),
                bin_low=edges[index],
                bin_high=edges[index + 1],
            )
        )
    return detail, bin_rows


def _summarize(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, dropna=False)[values]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out.columns = [f"{left}_{right}" if right else left for left, right in out.columns.to_flat_index()]
    out = out.rename(columns={f"{metric}_mean": metric for metric in values})
    out = out.rename(columns={f"{metric}_std": f"{metric}_seed_sd" for metric in values})
    out = out.rename(columns={f"{metric}_count": "n_seeds" for metric in values})
    for metric in values[1:]:
        out = out.drop(columns=[f"n_seeds_{metric}"], errors="ignore")
    out = out.rename(columns={f"n_seeds_{values[0]}": "n_seeds"})
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
    normalizations: tuple[str, ...] = NORMS,
    operators: tuple[str, ...] = OPERATORS,
) -> None:
    """Run C6/C7 and write per-seed, summary and ratio-bin tables."""

    out.mkdir(parents=True, exist_ok=True)
    parts = out / "detail"
    parts.mkdir(exist_ok=True)
    seeds = seeds or layout.seeds
    metric_rows: list[dict] = []
    object_detail: list[dict] = []
    bin_detail: list[dict] = []

    for seed in seeds:
        for model in models:
            path = parts / f"normalization_{seed}_{model}.csv"
            if max_cells == 0 and path.exists():
                frame = pd.read_csv(path)
                metric_rows.extend(frame.to_dict("records"))
                continue
            feat = Features(layout, seed, model)
            pooled: dict[str, list[dict]] = {name: [] for name in normalizations}
            current: list[dict] = []
            try:
                count = 0
                for dataset in datasets:
                    for perturbation in perturbations:
                        for normalization in normalizations:
                            rows, objects = _run_cell(
                                feat,
                                seed,
                                model,
                                dataset,
                                perturbation,
                                normalization,
                                operators,
                            )
                            current.extend(rows)
                            pooled[normalization].extend(objects)
                        count += 1
                        print(f"  {seed} {model} {dataset}/{perturbation}", flush=True)
                        if max_cells and count >= max_cells:
                            break
                    if max_cells and count >= max_cells:
                        break
            finally:
                feat.close()
            for normalization in normalizations:
                if pooled[normalization]:
                    detail, bins = _aggregate_objects(seed, model, normalization, pooled[normalization])
                    object_detail.extend(detail)
                    bin_detail.extend(bins)
            pd.DataFrame(current).to_csv(path, index=False)
            metric_rows.extend(current)

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        return
    metrics.to_csv(out / "normalization_per_seed.csv", index=False)
    object_frame = pd.DataFrame(object_detail)
    object_frame.to_csv(out / "r_over_m_detail.csv", index=False)
    pd.DataFrame(bin_detail).to_csv(out / "r_over_m_bins.csv", index=False)

    summary = _summarize(metrics, ["operator", "normalization", "support"], ["own_argmax_rate", "margin", "mean_own_cosine"])
    summary.insert(0, "seed", "3seed")
    summary.to_csv(out / "normalization_summary.csv", index=False)

    by_perturbation = _summarize(
        metrics,
        ["operator", "normalization", "perturbation", "support"],
        ["own_argmax_rate"],
    )
    by_perturbation.insert(0, "seed", "3seed")
    by_perturbation.to_csv(out / "normalization_by_perturbation.csv", index=False)

    by_model = _summarize(
        metrics,
        ["operator", "normalization", "model", "support"],
        ["own_argmax_rate"],
    )
    by_model.insert(0, "seed", "3seed")
    by_model.to_csv(out / "normalization_by_model.csv", index=False)

    counts = []
    for normalization in normalizations:
        subset = metrics[metrics["normalization"] == normalization]
        cell = subset.groupby(["seed", "model", "dataset", "perturbation", "operator"])[
            "own_argmax_rate"
        ].mean().unstack()
        if "M0" in cell and "M1" in cell:
            counts.append(
                dict(
                    normalization=normalization,
                    above=int((cell["M1"] > cell["M0"]).sum()),
                    cells=int(len(cell)),
                )
            )
    pd.DataFrame(counts).to_csv(out / "normalization_m1_above_m0.csv", index=False)

    if not object_frame.empty:
        object_summary = _summarize(
            object_frame,
            ["model", "normalization", "operator", "bin", "support"],
            ["accuracy", "share_predicted_min_ac", "ratio_mean"],
        )
        object_summary.insert(0, "seed", "3seed")
        # Seed-level mean and SD: average the 12 cells per seed first.
        seed_level = object_frame.groupby(
            ["seed", "model", "normalization", "operator", "bin", "support"]
        )[["accuracy", "share_predicted_min_ac"]].mean().reset_index()
        seed_stats = _summarize(
            seed_level,
            ["model", "normalization", "operator", "bin", "support"],
            ["accuracy", "share_predicted_min_ac"],
        )
        merged = object_summary.merge(
            seed_stats,
            on=["model", "normalization", "operator", "bin", "support"],
            how="left",
            suffixes=("", "_seedlevel"),
        )
        merged.to_csv(out / "r_over_m_summary.csv", index=False)


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
    parser.add_argument("--normalizations", default=",".join(NORMS))
    parser.add_argument("--operators", default=",".join(OPERATORS))
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
        normalizations=tuple(item for item in args.normalizations.split(",") if item),
        operators=tuple(item for item in args.operators.split(",") if item),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
