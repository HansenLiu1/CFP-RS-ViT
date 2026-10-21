"""Export-support statistics: rank, near-zero, interaction and permutation.

This task fills the gap left by the frozen ``artifacts/paper`` tables.  It
provides the CPU-side quantities used by Table I/S4a and several supplement
sections:

``rank``
    Own-class rank distribution (r1-r4) and margin for M0/M1, with the FAIR1M
    coarse-class variant.
``nearzero``
    Near-zero foreground responses under block masking, per role, class and
    dataset.
``interaction``
    Foreground x background interaction ratios and cosines, object- and
    source-image-weighted.
``permutation``
    Label-permutation controls under the Table I definition (refit and
    matrix-column variants).
"""

from __future__ import annotations

import argparse
import hashlib
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
    limit_fit,
    specificity,
    unit_delta,
)

NEARZERO_ROLES = ("direction_fit", "readout_fit", "validation", "confirmation")
PERMUTATION_VARIANTS = ("A_pipeline_refit", "A_fine_refit", "C_matrix_columns")


def rows_from_mask(feat: Features, mask: np.ndarray, normalization: str = "unit") -> dict:
    clear = feat.array("true_clear_degraded")[mask]
    degraded = feat.array("true_degraded_degraded")[mask]
    values, valid, norms = unit_delta(clear, degraded, normalization)
    index = np.flatnonzero(mask)[valid]
    return {
        "values": values,
        "labels": feat.meta["class_id"][index].astype(np.int64),
        "images": feat.meta["source_image_id"][index],
        "objects": feat.meta["object_id"][index],
        "coarse": feat.meta["coarse_name"][index],
        "index": index,
        "norms": norms[valid],
        "row_mask": mask,
    }


def coarse_codes(names: np.ndarray) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(sorted({str(value) for value in names.tolist()}))}
    return np.asarray([mapping[str(value)] for value in names.tolist()], dtype=np.int64)


def coarse_fit_rows(feat: Features, dataset: str, perturbation: str, support: int = FIT_LIMIT) -> dict:
    mask = cell_rows(feat, dataset, perturbation, "direction_fit", limit=None)
    mask = limit_fit(mask, feat.meta["class_id"], feat.meta["object_id"], support)
    return rows_from_mask(feat, mask)


def rank_stats(matrix: np.ndarray) -> dict:
    values = np.asarray(matrix, dtype=np.float64)
    count = values.shape[0]
    own = np.diag(values)
    better = (values > own[:, None]).sum(axis=1)
    row_max = values.max(axis=1)
    off = values.copy()
    np.fill_diagonal(off, np.nan)
    with np.errstate(all="ignore"):
        hardest = np.nanmax(off, axis=1)
    out = {
        "classes": int(count),
        "r1": float(np.mean(own >= row_max)),
        "r4": float(np.mean(better >= 3)),
        "margin": float(np.nanmean(own - hardest)),
    }
    for rank in (1, 2, 3):
        out[f"r{rank}"] = float(np.mean(better <= rank - 1))
    return out


def _rank_cell(feat: Features, dataset: str, perturbation: str) -> list[dict]:
    rows = []
    fit = cell_delta(feat, dataset, perturbation, "direction_fit", limit=FIT_LIMIT)
    evaluation = rows_from_mask(feat, cell_rows(feat, dataset, perturbation, "confirmation"))
    for operator_name in ("M0", "M1"):
        operator, directions, classes = fit_operator(fit["values"], fit["labels"], operator_name)
        matrix = np.asarray(
            specificity(
                operator,
                directions,
                classes,
                evaluation["values"],
                evaluation["labels"],
                evaluation["images"],
            )["score_matrix"],
            dtype=np.float64,
        )
        rows.append(
            dict(
                dataset=dataset,
                perturbation=perturbation,
                operator=operator_name,
                support=FIT_LIMIT,
                role="confirmation",
                normalization="unit",
                coarse="fine",
                **rank_stats(matrix),
            )
        )

    if dataset == "fair1m":
        coarse_fit = coarse_fit_rows(feat, dataset, perturbation)
        coarse_eval = evaluation
        names = np.concatenate([coarse_fit["coarse"], coarse_eval["coarse"]])
        mapping = {name: index for index, name in enumerate(sorted({str(value) for value in names.tolist()}))}
        fit_labels = np.asarray([mapping[str(value)] for value in coarse_fit["coarse"].tolist()], dtype=np.int64)
        eval_labels = np.asarray([mapping[str(value)] for value in coarse_eval["coarse"].tolist()], dtype=np.int64)
        for operator_name in ("M0", "M1"):
            operator, directions, classes = fit_operator(coarse_fit["values"], fit_labels, operator_name)
            matrix = np.asarray(
                specificity(
                    operator,
                    directions,
                    classes,
                    coarse_eval["values"],
                    eval_labels,
                    coarse_eval["images"],
                )["score_matrix"],
                dtype=np.float64,
            )
            rows.append(
                dict(
                    dataset=dataset,
                    perturbation=perturbation,
                    operator=operator_name,
                    support=FIT_LIMIT,
                    role="confirmation",
                    normalization="unit",
                    coarse="coarse",
                    **rank_stats(matrix),
                )
            )
    return rows


def _nearzero(feat: Features, seed: int, model: str) -> tuple[list[dict], list[dict]]:
    rows = []
    class_rows = []
    area = np.expm1(feat.meta["geometry"][:, 0])
    for dataset in DATASETS:
        for role in NEARZERO_ROLES:
            mask = cell_rows(feat, dataset, "patch_mask", role)
            delta = (
                feat.array("true_clear_degraded")[mask].astype(np.float64)
                - feat.array("true_degraded_degraded")[mask].astype(np.float64)
            )
            norms = np.linalg.norm(delta, axis=1)
            near = norms <= EPS
            rows.append(
                dict(
                    seed=seed,
                    model=model,
                    dataset=dataset,
                    perturbation="patch_mask",
                    operator="none",
                    support="all",
                    role=role,
                    normalization="raw",
                    n_total=int(len(norms)),
                    n_near_zero=int(near.sum()),
                    fraction=float(near.mean()) if len(norms) else float("nan"),
                    median_area_near=float(np.median(area[mask][near])) if near.any() else float("nan"),
                    median_area_valid=float(np.median(area[mask][~near])) if (~near).any() else float("nan"),
                    n_images=int(len(set(feat.meta["source_image_id"][mask].tolist()))),
                )
            )
            labels = feat.meta["class_id"][mask]
            names = feat.meta["class_name"][mask]
            for label in sorted(set(labels.tolist())):
                selected = labels == label
                class_rows.append(
                    dict(
                        seed=seed,
                        model=model,
                        dataset=dataset,
                        perturbation="patch_mask",
                        operator="none",
                        support="all",
                        role=role,
                        normalization="raw",
                        class_id=int(label),
                        class_name=str(names[selected][0]),
                        n_total=int(selected.sum()),
                        n_near_zero=int(near[selected].sum()),
                        fraction=float(near[selected].mean()),
                    )
                )
    return rows, class_rows


def _aggregate(values: np.ndarray, images: np.ndarray, aggregation: str) -> np.ndarray:
    if aggregation == "object":
        return values
    return np.asarray(
        [float(np.mean(values[images == image])) for image in dict.fromkeys(images.tolist())],
        dtype=np.float64,
    )


def _interaction(feat: Features, seed: int, model: str) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for perturbation in PERTURBATIONS:
            for role in ("direction_fit", "confirmation"):
                mask = cell_rows(feat, dataset, perturbation, role)
                labels = feat.meta["class_id"][mask]
                images = feat.meta["source_image_id"][mask]
                clear_clean = feat.array("true_clear_clean")[mask].astype(np.float64)
                degraded_clean = feat.array("true_degraded_clean")[mask].astype(np.float64)
                clear_degraded = feat.array("true_clear_degraded")[mask].astype(np.float64)
                degraded_degraded = feat.array("true_degraded_degraded")[mask].astype(np.float64)
                d_bc = clear_clean - degraded_clean
                d_bd = clear_degraded - degraded_degraded
                interaction = d_bc - d_bd
                good = (np.linalg.norm(d_bd, axis=1) > EPS) & (np.linalg.norm(d_bc, axis=1) > EPS)
                d_bc, d_bd, interaction = d_bc[good], d_bd[good], interaction[good]
                images, labels = images[good], labels[good]
                n_bc = np.linalg.norm(d_bc, axis=1)
                n_bd = np.linalg.norm(d_bd, axis=1)
                n_inter = np.linalg.norm(interaction, axis=1)
                cosine = np.sum(d_bc * d_bd, axis=1) / np.maximum(n_bc * n_bd, EPS)

                class_ratios = []
                class_cosines = []
                for label in sorted(set(labels.tolist())):
                    selected = labels == label
                    mean_bc = d_bc[selected].mean(axis=0)
                    mean_bd = d_bd[selected].mean(axis=0)
                    class_ratios.append(
                        float(np.linalg.norm(mean_bc) / max(np.linalg.norm(mean_bd), EPS))
                    )
                    class_cosines.append(
                        float(mean_bc @ mean_bd / max(np.linalg.norm(mean_bc) * np.linalg.norm(mean_bd), EPS))
                    )

                for aggregation in ("object_then_image", "object"):
                    units_bc = _aggregate(n_bc, images, aggregation)
                    units_bd = _aggregate(n_bd, images, aggregation)
                    units_inter = _aggregate(n_inter, images, aggregation)
                    units_cos = _aggregate(cosine, images, aggregation)
                    rows.append(
                        dict(
                            seed=seed,
                            model=model,
                            dataset=dataset,
                            perturbation=perturbation,
                            operator="none",
                            support="all",
                            role=role,
                            normalization="raw",
                            aggregation=aggregation,
                            n_objects=int(good.sum()),
                            n_units=int(len(units_bc)),
                            ratio_inter_bd=float(units_inter.mean() / max(units_bd.mean(), EPS)),
                            ratio_bc_bd=float(units_bc.mean() / max(units_bd.mean(), EPS)),
                            cos_object=float(units_cos.mean()),
                            ratio_bc_bd_classmean=float(np.mean(class_ratios)),
                            cos_classmean=float(np.mean(class_cosines)),
                        )
                    )
    return rows


def _permutation_seed(seed: int, model: str, dataset: str, perturbation: str) -> int:
    digest = hashlib.sha256(f"label-permutation:{seed}:{model}:{dataset}:{perturbation}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _permutation(feat: Features, seed: int, model: str) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        evaluation = rows_from_mask(feat, cell_rows(feat, dataset, "blur", "confirmation"))
        fit = cell_delta(feat, dataset, "blur", "direction_fit", limit=FIT_LIMIT)
        if dataset == "fair1m":
            coarse_fit = coarse_fit_rows(feat, dataset, "blur")
            names = np.concatenate([coarse_fit["coarse"], evaluation["coarse"]])
            mapping = {name: index for index, name in enumerate(sorted({str(value) for value in names.tolist()}))}
            fit_labels = np.asarray([mapping[str(value)] for value in coarse_fit["coarse"].tolist()], dtype=np.int64)
            eval_labels = np.asarray([mapping[str(value)] for value in evaluation["coarse"].tolist()], dtype=np.int64)
            fit_values = coarse_fit["values"]
        else:
            fit_labels = fit["labels"]
            eval_labels = evaluation["labels"]
            fit_values = fit["values"]

        rng = np.random.default_rng(_permutation_seed(seed, model, dataset, "blur"))
        permuted = rng.permutation(fit_labels)
        operator, directions, classes = fit_operator(fit_values, permuted, "M1")
        matrix = np.asarray(
            specificity(
                operator,
                directions,
                classes,
                evaluation["values"],
                eval_labels,
                evaluation["images"],
            )["score_matrix"],
            dtype=np.float64,
        )
        stats = rank_stats(matrix)
        rows.append(
            dict(
                seed=seed,
                model=model,
                dataset=dataset,
                perturbation="blur",
                operator="M1",
                support=FIT_LIMIT,
                role="confirmation",
                normalization="unit",
                variant="A_pipeline_refit",
                classes=stats["classes"],
                r1=stats["r1"],
                r2=stats["r2"],
                r4=stats["r4"],
            )
        )

        if dataset == "fair1m":
            rng = np.random.default_rng(_permutation_seed(seed, model, dataset, "blur"))
            permuted_fine = rng.permutation(fit["labels"])
            operator, directions, classes = fit_operator(fit["values"], permuted_fine, "M1")
            matrix_fine = np.asarray(
                specificity(
                    operator,
                    directions,
                    classes,
                    evaluation["values"],
                    evaluation["labels"],
                    evaluation["images"],
                )["score_matrix"],
                dtype=np.float64,
            )
            fine = rank_stats(matrix_fine)
        else:
            fine = dict(r1=np.nan, r2=np.nan, r4=np.nan, classes=np.nan)
        rows.append(
            dict(
                seed=seed,
                model=model,
                dataset=dataset,
                perturbation="blur",
                operator="M1",
                support=FIT_LIMIT,
                role="confirmation",
                normalization="unit",
                variant="A_fine_refit",
                classes=fine["classes"],
                r1=fine["r1"],
                r2=fine["r2"],
                r4=fine["r4"],
            )
        )

        operator, directions, classes = fit_operator(fit_values, fit_labels, "M1")
        true_matrix = np.asarray(
            specificity(
                operator,
                directions,
                classes,
                evaluation["values"],
                eval_labels,
                evaluation["images"],
            )["score_matrix"],
            dtype=np.float64,
        )
        rng = np.random.default_rng(seed)
        shuffle = rng.permutation(true_matrix.shape[0])
        stats = rank_stats(true_matrix[:, shuffle])
        rows.append(
            dict(
                seed=seed,
                model=model,
                dataset=dataset,
                perturbation="blur",
                operator="M1",
                support=FIT_LIMIT,
                role="confirmation",
                normalization="unit",
                variant="C_matrix_columns",
                classes=stats["classes"],
                r1=stats["r1"],
                r2=stats["r2"],
                r4=stats["r4"],
            )
        )
    return rows


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
    analyses: tuple[str, ...] = ("rank", "nearzero", "interaction", "permutation"),
) -> None:
    """Run the selected export-support analyses."""

    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds or layout.seeds
    for seed in seeds:
        for model in models:
            feat = Features(layout, seed, model)
            try:
                if "rank" in analyses:
                    rank_path = out / f"rank_{seed}_{model}.csv"
                    rows = []
                    count = 0
                    for dataset in datasets:
                        for perturbation in perturbations:
                            rows.extend(_rank_cell(feat, dataset, perturbation))
                            count += 1
                            if max_cells and count >= max_cells:
                                break
                        if max_cells and count >= max_cells:
                            break
                    frame = pd.DataFrame(rows)
                    frame.insert(0, "seed", seed)
                    frame.insert(1, "model", model)
                    frame.to_csv(rank_path, index=False)
                if "nearzero" in analyses:
                    rows, class_rows = _nearzero(feat, seed, model)
                    pd.DataFrame(rows).to_csv(out / f"nearzero_{seed}_{model}.csv", index=False)
                    pd.DataFrame(class_rows).to_csv(out / f"nearzero_by_class_{seed}_{model}.csv", index=False)
                if "interaction" in analyses:
                    pd.DataFrame(_interaction(feat, seed, model)).to_csv(
                        out / f"interaction_{seed}_{model}.csv", index=False
                    )
                if "permutation" in analyses:
                    pd.DataFrame(_permutation(feat, seed, model)).to_csv(
                        out / f"permutation_{seed}_{model}.csv", index=False
                    )
            finally:
                feat.close()
            print(f"  {seed} {model}", flush=True)

    def combine(prefix: str, keys: list[str], values: list[str], name: str) -> None:
        files = sorted(out.glob(f"{prefix}_[0-9]*_*.csv"))
        if prefix == "nearzero_by_class":
            files = sorted(out.glob("nearzero_by_class_[0-9]*_*.csv"))
        if not files:
            return
        frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        frame.to_csv(out / f"{name}_per_seed.csv", index=False)
        summary = _seed_summary(frame, keys, values)
        summary.to_csv(out / f"{name}_3seed.csv", index=False)

    if "rank" in analyses:
        combine(
            "rank",
            ["model", "dataset", "perturbation", "operator", "coarse"],
            ["r1", "r2", "r3", "r4", "margin"],
            "rank",
        )
    if "nearzero" in analyses:
        combine(
            "nearzero",
            ["dataset", "role"],
            ["fraction", "median_area_near", "median_area_valid"],
            "nearzero",
        )
        combine(
            "nearzero_by_class",
            ["dataset", "role", "class_id", "class_name"],
            ["fraction"],
            "nearzero_by_class",
        )
    if "interaction" in analyses:
        combine(
            "interaction",
            ["role", "aggregation", "perturbation"],
            ["ratio_inter_bd", "ratio_bc_bd", "cos_object", "ratio_bc_bd_classmean", "cos_classmean"],
            "interaction",
        )
    if "permutation" in analyses:
        combine(
            "permutation",
            ["dataset", "variant"],
            ["r1", "r2", "r4"],
            "permutation",
        )


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
    parser.add_argument("--analyses", default="rank,nearzero,interaction,permutation")
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
        analyses=tuple(item for item in args.analyses.split(",") if item),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
