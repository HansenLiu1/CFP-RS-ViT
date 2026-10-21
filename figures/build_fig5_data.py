"""Recompute the Fig. 5 layerwise specificity table with the primary protocol.

The persisted E06 export in ``12_layerwise_specificity.csv`` was computed on
the full direction-fit cache.  Fig. 5 is anchored to the fit15 specificity
analysis, so this script re-fits M1 on the same layerwise artifacts with the
same deterministic 15-per-class budget used by ``core._limit_fit_mask``.

Protocol per (seed, model, dataset, perturbation):
  * direction-fit: up to 15 objects per class, hash order, not excluded;
  * M1: A_c = mean_i unit(delta_i); m = mean_c unit(A_c);
        d_c = unit(A_c - m); query = unit(unit(delta_q) - m);
  * confirmation: source-image-disjoint; per query class, scores are averaged
    within each source image and the image means are averaged with equal
    weight (same estimator as the main specificity export);
  * metric: own-class argmax rate (absolute; chance = 1/K).

Plotting tables are absolute rates: within each seed the model (x perturbation)
cells are averaged, then the three per-seed values give the mean and sample SD.

Outputs
  data/fig5_layerwise_raw.csv   long table, 3 seeds
  data/fig5_dataset.csv         absolute curves per dataset (5 models x 4 perturbations)
  data/fig5_fair1m.csv          absolute FAIR1M curves per perturbation (5 models)
  data/fig5_final_anchor.csv    final-layer values for checking against S4a

Usage
  python build_fig5_data.py                   re-fit from the layerwise caches
  python build_fig5_data.py --aggregate-only  rebuild tables from the raw CSV
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cfra.directions import fit, unit

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
MODELS = ["remoteclip", "clip", "siglip2", "dinov2", "mae"]
DATASETS = ["dota", "dior", "fair1m"]
PERTURBATIONS = ["blur", "brightness", "gaussian_noise", "patch_mask"]
LAYERS = [f"L{i:02d}" for i in range(1, 13)] + ["final"]
FIT_LIMIT = 15
EPS = 1e-8


def limit_fit(indices: np.ndarray, labels: np.ndarray, object_ids: np.ndarray) -> np.ndarray:
    picked: list[int] = []
    for class_id in sorted(set(labels[indices].tolist())):
        subset = indices[labels[indices] == class_id]
        ordered = sorted(
            subset.tolist(),
            key=lambda index: (
                hashlib.sha256(str(object_ids[index]).encode("utf-8")).hexdigest(),
                str(object_ids[index]),
            ),
        )
        picked.extend(ordered[:FIT_LIMIT])
    return np.asarray(sorted(picked), dtype=np.int64)


def own_argmax(
    fit_delta: np.ndarray,
    fit_labels: np.ndarray,
    query_delta: np.ndarray,
    query_labels: np.ndarray,
    query_images: np.ndarray,
) -> tuple[float, int, int]:
    # The 15-per-class cap is applied before invalid (zero-change) deltas are
    # dropped, as in core._limit_fit_mask followed by the validity filter.
    fit_unit = unit(fit_delta)
    fit_valid = np.linalg.norm(fit_delta, axis=1) > EPS
    fit_unit, fit_labels = fit_unit[fit_valid], fit_labels[fit_valid]

    query_unit = unit(query_delta)
    query_valid = np.linalg.norm(query_delta, axis=1) > EPS
    query_unit, query_labels, query_images = (
        query_unit[query_valid],
        query_labels[query_valid],
        query_images[query_valid],
    )

    if len(set(fit_labels.tolist()) & set(query_labels.tolist())) < 2:
        return float("nan"), len(fit_labels), len(query_labels)

    operator, directions, classes = fit(fit_unit, fit_labels, operator="M1")
    keep = np.isin(query_labels, classes)
    query_unit, query_labels, query_images = query_unit[keep], query_labels[keep], query_images[keep]
    if len(query_unit) == 0:
        return float("nan"), len(fit_labels), len(query_labels)
    scores = operator.transform(query_unit) @ directions.T

    # Match evaluation.metrics._specificity_summary: for each query class,
    # average the scores within each source image, then average the image
    # means with equal weight.  The direction counts as a hit when its own
    # column is the row maximum (ties count).
    matrix = np.full((len(classes), len(classes)), np.nan, dtype=np.float64)
    for column, class_id in enumerate(classes.tolist()):
        selected = query_labels == class_id
        if not selected.any():
            continue
        class_scores = scores[selected]
        class_images = query_images[selected]
        image_means = [
            class_scores[class_images == image].mean(axis=0)
            for image in dict.fromkeys(class_images.tolist())
        ]
        matrix[:, column] = np.mean(image_means, axis=0)
    if not np.isfinite(matrix).all():
        return float("nan"), len(fit_labels), len(query_labels)
    hits = np.diag(matrix) >= matrix.max(axis=1)
    return float(np.mean(hits)), len(fit_labels), len(query_labels)


def main(run_root: Path, runs: dict[int, str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for seed, run in runs.items():
        feature_dir = run_root / run / "intermediates" / "features"
        for model in MODELS:
            for dataset in DATASETS:
                metadata = json.loads(
                    (feature_dir / f"layerwise_{model}_{dataset}.json").read_text(encoding="utf-8")
                )["row_metadata"]
                role = np.asarray([row["role"] for row in metadata])
                perturbation = np.asarray([row["perturbation"] for row in metadata])
                labels = np.asarray([int(row["class_id"]) for row in metadata], dtype=np.int64)
                object_ids = np.asarray([row["object_id"] for row in metadata])
                images = np.asarray([row["source_image_id"] for row in metadata])
                excluded = np.asarray(
                    [bool(row.get("excluded_from_main", False)) for row in metadata]
                )

                fit_picks = {}
                query_picks = {}
                for pert in PERTURBATIONS:
                    fit = np.flatnonzero(
                        (role == "direction_fit") & (perturbation == pert) & ~excluded
                    )
                    query = np.flatnonzero(
                        (role == "confirmation") & (perturbation == pert) & ~excluded
                    )
                    if len(fit) == 0 or len(query) == 0:
                        continue
                    fit_picks[pert] = limit_fit(fit, labels, object_ids)
                    query_picks[pert] = query

                if not fit_picks:
                    continue

                with np.load(
                    feature_dir / f"layerwise_{model}_{dataset}.npz",
                    allow_pickle=False,
                    mmap_mode="r",
                ) as payload:
                    for layer in LAYERS:
                        clear = np.asarray(payload[f"{layer}__clear"], dtype=np.float64)
                        degraded = np.asarray(payload[f"{layer}__degraded"], dtype=np.float64)
                        for pert in PERTURBATIONS:
                            if pert not in fit_picks:
                                continue
                            fit = fit_picks[pert]
                            query = query_picks[pert]
                            fit_delta = clear[fit] - degraded[fit]
                            query_delta = clear[query] - degraded[query]
                            value, n_fit, n_query = own_argmax(
                                fit_delta,
                                labels[fit],
                                query_delta,
                                labels[query],
                                images[query],
                            )
                            rows.append(
                                {
                                    "seed": seed,
                                    "model": model,
                                    "dataset": dataset,
                                    "perturbation": pert,
                                    "layer": layer,
                                    "own_argmax": value,
                                    "n_fit": n_fit,
                                    "n_query": n_query,
                                }
                            )
                print(f"done {seed} {model}/{dataset}", flush=True)

    raw_path = OUT / "fig5_layerwise_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {raw_path} rows={len(rows)}")


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} rows={len(rows)}")


def seed_level_curves(rows: list[dict], group_field: str, keep) -> list[dict]:
    """Absolute curves: per seed, average the cells in each group; then report
    the mean and sample SD of the three per-seed values."""

    cells: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if not keep(row):
            continue
        value = float(row["own_argmax"])
        if not np.isfinite(value):
            continue
        cells[(int(row["seed"]), row[group_field], row["layer"])].append(value)

    seeds = sorted({key[0] for key in cells})
    groups = sorted({key[1] for key in cells}, key=lambda g: (DATASETS + PERTURBATIONS).index(g))
    out = []
    for group in groups:
        for layer in LAYERS:
            per_seed = [cells[(seed, group, layer)] for seed in seeds if (seed, group, layer) in cells]
            sizes = {len(values) for values in per_seed}
            if len(per_seed) != len(seeds) or len(sizes) != 1:
                raise ValueError(f"unbalanced cells for {group}/{layer}: {sizes}")
            means = [float(np.mean(values)) for values in per_seed]
            out.append(
                {
                    group_field: group,
                    "layer": layer,
                    "mean": float(np.mean(means)),
                    "sd": float(np.std(means, ddof=1)),
                    "n_seeds": len(means),
                    "cells_per_seed": sizes.pop(),
                }
            )
    return out


def aggregate(rows: list[dict]) -> None:
    rows = [row for row in rows if row["layer"] in LAYERS]

    anchors = [
        {
            "seed": row["seed"],
            "model": row["model"],
            "dataset": row["dataset"],
            "perturbation": row["perturbation"],
            "final": row["own_argmax"],
        }
        for row in rows
        if row["layer"] == "final"
    ]
    write_rows(OUT / "fig5_final_anchor.csv", list(anchors[0]), anchors)

    fields = ["mean", "sd", "n_seeds", "cells_per_seed"]
    dataset_out = seed_level_curves(rows, "dataset", lambda row: True)
    write_rows(OUT / "fig5_dataset.csv", ["dataset", "layer", *fields], dataset_out)

    fair_out = seed_level_curves(rows, "perturbation", lambda row: row["dataset"] == "fair1m")
    write_rows(OUT / "fig5_fair1m.csv", ["perturbation", "layer", *fields], fair_out)

    print("\nabsolute own-class argmax rate (3-seed mean):")
    for row in dataset_out:
        if row["layer"] in {"L01", "L06", "L09", "L12", "final"}:
            print(f"  {row['dataset']:7s} {row['layer']:5s} {row['mean']:.3f} +/- {row['sd']:.3f}")


def read_raw() -> list[dict]:
    path = OUT / "fig5_layerwise_raw.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run without --aggregate-only first")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--runs",
        required=True,
        help="comma-separated seed=run-directory pairs",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="skip the layerwise re-fit and rebuild the plotting tables from fig5_layerwise_raw.csv",
    )
    args = parser.parse_args()
    runs = {}
    for item in args.runs.split(","):
        seed, name = item.split("=", 1)
        runs[int(seed)] = name
    if not args.aggregate_only:
        main(args.run_root, runs)
    aggregate(read_raw())
