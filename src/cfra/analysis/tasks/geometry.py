"""Cross-image class-direction geometry (S4b/S4c).

For every cell we fit the pipeline M1 directions on the direction-fit role,
transform the confirmation deltas with the same operator, average them per
class and source image, and score the resulting class-by-class cosine matrix.
The diagonal (same-class) and off-diagonal (different-class) means are reported
together with a class-label permutation null.

This is the public counterpart of the internal cross-image cosine batch.  It is
deliberately called ``geometry`` rather than ``cosine`` because the same task
also documents the direction-matrix geometry used by the subspace analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cfra.analysis.common import (
    DATASETS,
    MODELS,
    PERTURBATIONS,
    Features,
    RunLayout,
    cell_delta,
    fit_operator,
)
from cfra.analysis.matrix import class_unit_mean, same_diff

N_PERM = 200


def _cell(
    seed: int,
    model: str,
    feat: Features,
    dataset: str,
    perturbation: str,
    permutations: int,
) -> dict:
    fit = cell_delta(feat, dataset, perturbation, "direction_fit", limit=15)
    confirmation = cell_delta(feat, dataset, perturbation, "confirmation", limit=None)
    operator, directions, classes = fit_operator(fit["values"], fit["labels"], "M1")
    classes = np.asarray(classes, dtype=np.int64)
    if not np.all(np.isin(confirmation["labels"], classes)):
        raise RuntimeError(f"confirmation class missing from fit for {seed} {model} {dataset}/{perturbation}")

    queries = operator.transform(confirmation["values"])
    query_means = class_unit_mean(queries, confirmation["labels"], classes)
    same, different = same_diff(directions @ query_means.T)

    rng = np.random.default_rng(seed)
    null_same = []
    null_diff = []
    for _ in range(permutations):
        permuted = class_unit_mean(queries, rng.permutation(confirmation["labels"]), classes)
        current_same, current_diff = same_diff(directions @ permuted.T)
        null_same.append(current_same)
        null_diff.append(current_diff)
    null_same = np.asarray(null_same, dtype=np.float64)
    null_diff = np.asarray(null_diff, dtype=np.float64)
    counts = np.asarray([(confirmation["labels"] == class_id).sum() for class_id in classes])
    return dict(
        seed=seed,
        model=model,
        dataset=dataset,
        perturbation=perturbation,
        form="M1",
        support="fit15",
        role="confirmation",
        layer="final",
        n_classes=int(len(classes)),
        n_conf_objects=int(len(confirmation["labels"])),
        mean_class_n=float(counts.mean()),
        min_class_n=int(counts.min()),
        n_fit_rows=int(len(fit["labels"])),
        same=float(same),
        diff=float(different),
        gap=float(same - different),
        null_same=float(null_same.mean()),
        null_same_sd=float(null_same.std()),
        null_same_p95=float(np.quantile(null_same, 0.95)),
        null_diff=float(null_diff.mean()),
        exceeds_p95=bool(same > np.quantile(null_same, 0.95)),
        z_same=float((same - null_same.mean()) / max(null_same.std(), 1.0e-12)),
        n_perm=int(permutations),
    )


def _seed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "same",
        "diff",
        "gap",
        "null_same",
        "null_same_p95",
        "null_diff",
        "z_same",
        "n_classes",
        "n_conf_objects",
        "n_fit_rows",
    ]
    grouped = frame.groupby(["model", "dataset", "perturbation"])
    mean = grouped[metrics].mean().add_suffix("_mean")
    sd = grouped[metrics].std(ddof=1).add_suffix("_seed_sd")
    summary = pd.concat([mean, sd], axis=1).reset_index()
    summary.insert(0, "seed", "3seed")
    summary["n_seeds"] = grouped.size().values
    for column, value in (("form", "M1"), ("support", "fit15"), ("role", "confirmation"), ("layer", "final")):
        summary.insert(1, column, value)
    return summary


def _by_dimension(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    metrics = ["same", "diff", "gap", "null_same", "null_same_p95", "null_diff", "z_same"]
    per_seed = frame.groupby(["seed", dimension])[metrics].mean().reset_index()
    grouped = per_seed.groupby(dimension)[metrics]
    out = grouped.agg(["mean", "std"]).reset_index()
    out.columns = [f"{left}_{right}" if right else left for left, right in out.columns.to_flat_index()]
    out = out.rename(columns={f"{metric}_mean": metric for metric in metrics})
    out = out.rename(columns={f"{metric}_std": f"{metric}_seed_sd" for metric in metrics})
    out["n_seeds"] = grouped.size().values
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
    permutations: int = N_PERM,
) -> None:
    """Write per-seed geometry, the three-seed summary and exceedance counts."""

    out.mkdir(parents=True, exist_ok=True)
    parts = out / "detail"
    parts.mkdir(exist_ok=True)
    seeds = seeds or layout.seeds
    rows: list[dict] = []
    for seed in seeds:
        for model in models:
            path = parts / f"geometry_{seed}_{model}.csv"
            if max_cells == 0 and path.exists():
                rows.extend(pd.read_csv(path).to_dict("records"))
                continue
            feat = Features(layout, seed, model)
            current: list[dict] = []
            try:
                count = 0
                for dataset in datasets:
                    for perturbation in perturbations:
                        current.append(_cell(seed, model, feat, dataset, perturbation, permutations))
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

    frame = pd.DataFrame(rows).sort_values(["seed", "model", "dataset", "perturbation"]).reset_index(drop=True)
    frame.to_csv(out / "geometry_per_seed.csv", index=False)

    summary = _seed_summary(frame)
    summary.to_csv(out / "geometry_3seed.csv", index=False)
    _by_dimension(frame, "model").to_csv(out / "geometry_by_model.csv", index=False)
    _by_dimension(frame, "dataset").to_csv(out / "geometry_by_dataset.csv", index=False)
    _by_dimension(frame, "perturbation").to_csv(out / "geometry_by_perturbation.csv", index=False)

    exceed = frame.groupby("seed").agg(
        n_cells=("exceeds_p95", "size"),
        n_above_p95=("exceeds_p95", "sum"),
    ).reset_index()
    exceed.to_csv(out / "geometry_exceedance_per_seed.csv", index=False)
    pd.DataFrame(
        [
            dict(
                form="M1",
                support="fit15",
                role="confirmation",
                layer="final",
                n_above_p95_mean=float(exceed["n_above_p95"].mean()),
                n_above_p95_seed_sd=float(exceed["n_above_p95"].std()),
                n_cells=int(exceed["n_cells"].iloc[0]),
                per_seed=";".join(f"{int(row.seed)}:{int(row.n_above_p95)}" for row in exceed.itertuples()),
            )
        ]
    ).to_csv(out / "geometry_exceedance_3seed.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--runs", required=True, help="comma-separated seed=run-directory pairs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default="")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--perturbations", default=",".join(PERTURBATIONS))
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=N_PERM)
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
        permutations=args.permutations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
