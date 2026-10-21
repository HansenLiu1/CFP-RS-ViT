"""Readout-space removal and reflection controls (Table IV, S10, Fig. 7a).

The task calls the public pipeline stage :func:`cfra.pipeline.core.run_e07`
unchanged.  It only builds a per-cell ``FeatureTable`` from the persisted
features and renames direction-fit rows outside the fit15 budget, which is the
same convention used by the paper's E07 analysis:

* readout features are centred by the readout-fit mean;
* removal is ``x' = x - alpha * <x, d> d`` with the per-sample norm preserved;
* the frozen and retrained linear/prototype readouts are both reported;
* ``alpha = 2`` is the reflection control used by S10.

The large nested pipeline diagnostics are flattened to one row per
seed/model/dataset/perturbation/space/condition/alpha.  Bootstrap dictionaries
are retained only when ``--bootstrap-replicates`` is non-zero.
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
    limit_fit,
)
from cfra.pipeline.core import run_e07
from cfra.pipeline.features import FeatureSample, FeatureTable

ROLES = ("direction_fit", "readout_fit", "confirmation")
ARRAYS = ("raw", "true_clear_degraded", "true_degraded_degraded")
STRING_FIELDS = (
    "dataset_id",
    "object_id",
    "source_image_id",
    "class_name",
    "coarse_name",
    "role",
    "perturbation",
)
INT_FIELDS = ("class_id",)
SIMPLE_FIELDS = (
    "dataset_id",
    "model_id",
    "perturbation",
    "condition",
    "control_id",
    "retrained_probe",
    "rebuilt_prototype",
    "alpha",
    "operation",
    "reflection_control",
    "preserve_feature_norm",
    "space",
    "mean_abs_cos_h_w",
    "align_prototype",
    "align_linear_weight",
    "frozen_linear_accuracy",
    "frozen_linear_margin",
    "frozen_prototype_accuracy",
    "frozen_prototype_margin",
    "retrained_linear_accuracy",
    "retrained_linear_margin",
    "retrained_prototype_accuracy",
    "retrained_prototype_margin",
    "frozen_linear_accuracy_drop",
    "frozen_prototype_accuracy_drop",
    "retrained_linear_accuracy_drop",
    "retrained_prototype_accuracy_drop",
    "prototype_priority",
    "prototype_priority_over_linear",
    "retrained_recovery",
    "retrained_prototype_recovery",
    "confirmation_source_image_count",
)
TABLE_CONDITIONS = (
    "own_class",
    "wrong_class",
    "class_mean_axis",
    "projection_matched_random",
    "random_unit",
    "classifier_weight_aligned_random",
    "linear_weight",
)
TABLE_LABELS = {
    "own_class": "Own class",
    "wrong_class": "Wrong class",
    "class_mean_axis": "Common-mode axis",
    "projection_matched_random": "Projection-matched",
    "random_unit": "Random",
    "classifier_weight_aligned_random": "Weight-aligned",
    "linear_weight": "Own readout weight",
}


def _cell_index(feat: Features, dataset: str, perturbation: str) -> np.ndarray:
    dataset_ids = np.asarray(feat.meta["dataset_id"])
    perturbations = np.asarray(feat.meta["perturbation"])
    roles = np.asarray(feat.meta["role"])
    return np.flatnonzero(
        (dataset_ids == dataset)
        & (perturbations == perturbation)
        & np.isin(roles, ROLES)
    )


def build_table(
    feat: Features,
    seed: int,
    model: str,
    dataset: str,
    perturbation: str,
    budget: str,
) -> FeatureTable:
    """Build the minimal FeatureTable consumed by ``run_e07``."""

    index = _cell_index(feat, dataset, perturbation)
    metadata = {name: np.asarray(feat.meta[name])[index] for name in STRING_FIELDS + INT_FIELDS}
    geometry = np.asarray(feat.meta["geometry"], dtype=np.float64)[index]
    arrays = {name: np.asarray(feat.array(name))[index] for name in ARRAYS}
    roles = np.asarray(metadata["role"].astype(str).tolist(), dtype=object)
    if budget == "fit15":
        direction_fit = roles == "direction_fit"
        kept = limit_fit(direction_fit, metadata["class_id"].astype(np.int64), metadata["object_id"].astype(str), FIT_LIMIT)
        roles[direction_fit & ~kept] = "direction_fit_unused"
    elif budget != "fitall":
        raise ValueError(f"unknown budget: {budget}")
    samples = tuple(
        FeatureSample(
            dataset_id=str(metadata["dataset_id"][i]),
            object_id=str(metadata["object_id"][i]),
            source_image_id=str(metadata["source_image_id"][i]),
            class_id=int(metadata["class_id"][i]),
            class_name=str(metadata["class_name"][i]),
            coarse_name=str(metadata["coarse_name"][i]),
            role=str(roles[i]),
            perturbation=str(metadata["perturbation"][i]),
            seed=int(seed),
            geometry=tuple(float(value) for value in geometry[i]),
            excluded_from_main=False,
        )
        for i in range(len(index))
    )
    return FeatureTable(
        model_id=model,
        metadata=samples,
        arrays=arrays,
        representation={"built_by": "cfra.analysis.tasks.readout_removal"},
    )


def _flatten(rows: list[dict], seed: int, model: str, budget: str) -> list[dict]:
    output = []
    for row in rows:
        record = {field: row.get(field) for field in SIMPLE_FIELDS}
        record.update(seed=seed, model=model, budget=budget)
        drops = row.get("removal_drops", {}) or {}
        for name in ("frozen_linear_margin_drop", "frozen_prototype_margin_drop"):
            item = drops.get(name, {}) if isinstance(drops, dict) else {}
            record[f"{name}_estimate"] = item.get("estimate")
            record[f"{name}_ci_low"] = item.get("ci_low")
            record[f"{name}_ci_high"] = item.get("ci_high")
        bootstrap = row.get("bootstrap", {}) or {}
        if isinstance(bootstrap, dict):
            for name, item in bootstrap.items():
                if isinstance(item, dict):
                    record[f"bootstrap_{name}_ci_low"] = item.get("ci_low")
                    record[f"bootstrap_{name}_ci_high"] = item.get("ci_high")
        output.append(record)
    return output


def _run_cell(
    feat: Features,
    seed: int,
    model: str,
    dataset: str,
    perturbation: str,
    budget: str,
    alphas: list[float],
    reflection_alpha: float,
    bootstrap_replicates: int,
) -> list[dict]:
    table = build_table(feat, seed, model, dataset, perturbation, budget)
    statistics_config = {
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_ci": int(bootstrap_replicates) > 0,
        "permutation_replicates": 1,
        "confidence_level": 0.95,
        "cluster_unit": "source_image",
        "bootstrap_seed": 1729,
    }
    rows = run_e07(
        table,
        dataset_ids=[dataset],
        perturbations=[perturbation],
        removal_alphas=list(alphas),
        reflection_alpha=float(reflection_alpha),
        preserve_norm=True,
        epsilon=EPS,
        statistics_config=statistics_config,
        random_seed=seed,
    )
    return _flatten(rows, seed, model, budget)


def _seed_summary(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    per_seed = frame.groupby(["seed", *keys])[metrics].mean().reset_index()
    grouped = per_seed.groupby(keys)[metrics]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out.columns = [f"{left}_{right}" if right else left for left, right in out.columns.to_flat_index()]
    out = out.rename(columns={f"{metric}_mean": metric for metric in metrics})
    out = out.rename(columns={f"{metric}_std": f"{metric}_sd" for metric in metrics})
    out = out.rename(columns={f"{metric}_count": "n_seeds" for metric in metrics})
    for metric in metrics[1:]:
        out = out.drop(columns=[f"n_seeds_{metric}"], errors="ignore")
    out = out.rename(columns={f"n_seeds_{metrics[0]}": "n_seeds"})
    return out


def _table_iv(detail: pd.DataFrame) -> pd.DataFrame:
    selected = detail[
        (detail["space"] == "centered")
        & np.isclose(detail["alpha"], 1.0)
        & detail["condition"].isin(TABLE_CONDITIONS)
    ].copy()
    metrics = {
        "linear_accuracy_drop": "frozen_linear_accuracy_drop",
        "linear_margin_drop": "frozen_linear_margin_drop_estimate",
        "prototype_accuracy_drop": "frozen_prototype_accuracy_drop",
        "prototype_margin_drop": "frozen_prototype_margin_drop_estimate",
    }
    rows = []
    for label, column in metrics.items():
        subset = selected[["seed", "model_id", "dataset_id", "perturbation", "condition", column]].dropna()
        if subset.empty:
            continue
        per_seed = subset.rename(columns={column: "value"}).groupby(
            ["seed", "condition"]
        )["value"].mean().reset_index()
        grouped = per_seed.groupby("condition")["value"]
        out = grouped.agg(["mean", "std", "count"]).reset_index()
        out["metric"] = label
        out["row"] = out["condition"].map(TABLE_LABELS)
        out = out.rename(columns={"mean": "estimate", "std": "seed_sd", "count": "n_seeds"})
        rows.append(out[["metric", "row", "condition", "estimate", "seed_sd", "n_seeds"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _reflection(detail: pd.DataFrame) -> pd.DataFrame:
    selected = detail[detail["operation"] == "reflection_sign_reversal"].copy()
    if selected.empty:
        return selected
    metrics = [
        "frozen_linear_accuracy",
        "frozen_linear_margin",
        "frozen_prototype_accuracy",
        "frozen_prototype_margin",
        "frozen_linear_accuracy_drop",
        "frozen_prototype_accuracy_drop",
    ]
    available = [metric for metric in metrics if metric in selected]
    return _seed_summary(selected, ["space", "condition", "alpha"], available)


def run(
    layout: RunLayout,
    out: Path,
    *,
    models: tuple[str, ...] = MODELS,
    seeds: tuple[int, ...] | None = None,
    datasets: tuple[str, ...] = DATASETS,
    perturbations: tuple[str, ...] = PERTURBATIONS,
    max_cells: int = 0,
    budgets: tuple[str, ...] = ("fit15", "fitall"),
    alphas: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    reflection_alpha: float = 2.0,
    bootstrap_replicates: int = 0,
) -> None:
    """Run E07 for each cell and write the Table IV/S10 summaries."""

    out.mkdir(parents=True, exist_ok=True)
    parts = out / "detail"
    parts.mkdir(exist_ok=True)
    seeds = seeds or layout.seeds
    all_rows: list[dict] = []
    for seed in seeds:
        for model in models:
            for budget in budgets:
                path = parts / f"{seed}_{model}_{budget}.csv"
                if max_cells == 0 and path.exists():
                    all_rows.extend(pd.read_csv(path).to_dict("records"))
                    continue
                feat = Features(layout, seed, model)
                rows: list[dict] = []
                try:
                    count = 0
                    for dataset in datasets:
                        for perturbation in perturbations:
                            rows.extend(
                                _run_cell(
                                    feat,
                                    seed,
                                    model,
                                    dataset,
                                    perturbation,
                                    budget,
                                    list(alphas),
                                    reflection_alpha,
                                    bootstrap_replicates,
                                )
                            )
                            count += 1
                            print(f"  {seed} {model} {budget} {dataset}/{perturbation}", flush=True)
                            if max_cells and count >= max_cells:
                                break
                        if max_cells and count >= max_cells:
                            break
                finally:
                    feat.close()
                pd.DataFrame(rows).to_csv(path, index=False)
                all_rows.extend(rows)

    detail = pd.DataFrame(all_rows)
    if detail.empty:
        return
    detail.to_csv(out / "readout_removal_detail.csv", index=False)

    keys = ["budget", "space", "condition", "alpha"]
    metrics = [
        "frozen_linear_accuracy",
        "frozen_linear_margin",
        "frozen_prototype_accuracy",
        "frozen_prototype_margin",
        "frozen_linear_accuracy_drop",
        "frozen_prototype_accuracy_drop",
        "frozen_linear_margin_drop_estimate",
        "frozen_prototype_margin_drop_estimate",
    ]
    available = [metric for metric in metrics if metric in detail]
    summary = _seed_summary(detail, keys, available)
    summary.to_csv(out / "readout_removal_summary.csv", index=False)

    table = _table_iv(detail)
    if not table.empty:
        table.to_csv(out / "readout_removal_table_iv.csv", index=False)
    reflection = _reflection(detail)
    if not reflection.empty:
        reflection.to_csv(out / "readout_removal_reflection.csv", index=False)
    dose = summary[
        (summary["space"] == "centered")
        & summary["condition"].isin(TABLE_CONDITIONS)
    ].copy()
    if not dose.empty:
        dose.to_csv(out / "readout_removal_dose.csv", index=False)


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
    parser.add_argument("--budgets", default="fit15,fitall")
    parser.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--reflection-alpha", type=float, default=2.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
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
        budgets=tuple(item for item in args.budgets.split(",") if item),
        alphas=tuple(float(item) for item in args.alphas.split(",") if item),
        reflection_alpha=args.reflection_alpha,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
