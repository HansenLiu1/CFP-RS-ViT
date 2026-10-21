"""Text anchoring of M1 class directions (S12b).

For each seed/model/dataset/perturbation, the task compares the M1 (and M0
control) class directions with class-name text prototypes in the model's shared
image-text space.  Four prompt templates and their unit mean ensemble are
reported, together with a class-label permutation null shared by the seeds.

As with the text baseline, this task needs ``.[models]``, local model weights and
a valid ``config/local.toml``; the CPU-only CI job does not run it.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from cfra.analysis.common import DATASETS, PERTURBATIONS, Features, RunLayout, cell_delta, fit_operator, unit
from cfra.analysis.tasks.text_baseline import (
    SHARED_SPACE,
    TEXT_MODELS,
    _clip_projection,
    _runtime,
)
from cfra.configuration import load_local_config
from cfra.data.index import load_class_registry
from cfra.models import resolve_model_assets
from cfra.models.text import DEFAULT_TEXT_TEMPLATES, load_shared_text_encoder
from cfra.offline import _text_class_names

TEMPLATE_LABELS = ("t1", "t2", "t3", "t4")


def _prototypes(encoder, registry, dataset: str) -> dict:
    names = _text_class_names(registry, dataset)
    class_ids = sorted(int(key.split(":")[1]) for key in names)
    ordered = [names[f"{dataset}:{class_id}"] for class_id in class_ids]
    matrices = {}
    for label, template in zip(TEMPLATE_LABELS, DEFAULT_TEXT_TEMPLATES):
        prompts = [template.format(class_name=name) for name in ordered]
        encoded = encoder.encode(prompts, normalize=True, output_device="cpu")
        values = encoded.values.detach().cpu().numpy() if hasattr(encoded.values, "detach") else encoded.values
        matrices[label] = np.asarray(values, dtype=np.float64)
    matrices["ensemble"] = unit(np.mean([matrices[label] for label in TEMPLATE_LABELS], axis=0))
    return {
        "class_ids": class_ids,
        "row": {class_id: index for index, class_id in enumerate(class_ids)},
        "matrices": matrices,
    }


def _metrics(scores: np.ndarray) -> dict:
    count = scores.shape[0]
    own = np.diag(scores)
    off = scores.copy()
    np.fill_diagonal(off, -np.inf)
    wrong_max = off.max(axis=1)
    ranks = 1 + (scores > own[:, None]).sum(axis=1)
    return {
        "classes": count,
        "chance": 1.0 / count,
        "own_top1": float(np.mean(own >= wrong_max)),
        "own_top3": float(np.mean(ranks <= 3)),
        "margin": float(np.mean(own - wrong_max)),
        "mean_own": float(np.mean(own)),
        "mean_wrong": float(np.mean((scores.sum(axis=1) - own) / max(1, count - 1))),
    }


def _permutation(score_list: list[np.ndarray], observed: float, draws: int, seed: int) -> dict:
    count = score_list[0].shape[0]
    index = np.arange(count)
    rng = np.random.default_rng(seed)
    permutations = np.argsort(rng.random((draws, count)), axis=1)
    null = np.zeros(draws)
    for scores in score_list:
        row_max = scores.max(axis=1)
        own = scores[index[None, :], permutations]
        null += np.mean(own >= row_max[None, :], axis=1)
    null /= len(score_list)
    p = (1.0 + float(np.sum(null >= observed))) / (draws + 1.0)
    return {"p_value": p, "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95))}


def _bh(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(p), dtype=np.float64)
    out[order] = np.minimum(ranked, 1.0)
    return out.tolist()


def run(
    layout: RunLayout,
    out: Path,
    *,
    project_root: Path,
    models: tuple[str, ...] = TEXT_MODELS,
    seeds: tuple[int, ...],
    datasets: tuple[str, ...] = DATASETS,
    perturbations: tuple[str, ...] = PERTURBATIONS,
    permutations: int = 10000,
    clip_projection: Path | None = None,
    max_cells: int = 0,
) -> None:
    """Write per-seed anchoring cells, the three-seed summary and permutation nulls."""

    out.mkdir(parents=True, exist_ok=True)
    project_root = project_root.expanduser().resolve()
    registry = load_class_registry(project_root / "registry")
    local = load_local_config(project_root / "config" / "local.toml")
    cells: list[dict] = []
    permutation_rows: list[dict] = []
    for model in models:
        if model not in SHARED_SPACE:
            raise ValueError(f"text anchoring supports {TEXT_MODELS}, got {model!r}")
        asset_model = model
        source_model = "clip" if model == "clip_shared" else model
        assets = resolve_model_assets(local, [asset_model], registry_root=project_root / "registry")
        encoder = load_shared_text_encoder(
            asset_model,
            assets[asset_model],
            registry_root=project_root / "registry",
            runtime=_runtime(layout, seeds[0], asset_model),
            device="auto",
        )
        projection = _clip_projection(project_root, layout, seeds[0], clip_projection) if model == "clip_shared" else None
        prototypes = {dataset: _prototypes(encoder, registry, dataset) for dataset in datasets}
        score_cache: dict[tuple, list[np.ndarray]] = {}
        try:
            for seed in seeds:
                feat = Features(layout, seed, source_model)
                try:
                    pairs = [(dataset, perturbation) for dataset in datasets for perturbation in perturbations]
                    if max_cells:
                        pairs = pairs[:max_cells]
                    for dataset, perturbation in pairs:
                        delta = cell_delta(feat, dataset, perturbation, "direction_fit", limit=15)
                        if len(np.unique(delta["labels"])) < 3:
                            continue
                        for operator_name in ("M1", "M0"):
                            _operator, directions, classes = fit_operator(
                                delta["values"], delta["labels"], operator_name
                            )
                            directions = unit(np.asarray(directions, dtype=np.float64))
                            if projection is not None:
                                directions = unit(directions @ projection.T)
                            rows = [prototypes[dataset]["row"][int(class_id)] for class_id in classes]
                            for label in (*TEMPLATE_LABELS, "ensemble"):
                                scores = directions @ prototypes[dataset]["matrices"][label][rows].T
                                metrics = _metrics(scores)
                                cells.append(
                                    dict(
                                        seed=seed,
                                        model=model,
                                        dataset=dataset,
                                        perturbation=perturbation,
                                        operator=operator_name,
                                        template=label,
                                        classes=metrics["classes"],
                                        chance=metrics["chance"],
                                        own_top1=metrics["own_top1"],
                                        own_top3=metrics["own_top3"],
                                        margin=metrics["margin"],
                                        mean_own=metrics["mean_own"],
                                        mean_wrong=metrics["mean_wrong"],
                                        feature_dim=int(directions.shape[1]),
                                        shared_space=SHARED_SPACE[model],
                                    )
                                )
                                if label == "ensemble":
                                    score_cache.setdefault((dataset, perturbation, operator_name), []).append(scores)
                finally:
                    feat.close()
                print(f"  text-anchoring {model} {seed}", flush=True)
        finally:
            encoder.close()

        for (dataset, perturbation, operator_name), scores in score_cache.items():
            observed_rows = [
                row
                for row in cells
                if row["model"] == model
                and row["dataset"] == dataset
                and row["perturbation"] == perturbation
                and row["operator"] == operator_name
                and row["template"] == "ensemble"
            ]
            observed = float(np.mean([row["own_top1"] for row in observed_rows]))
            result = _permutation(
                scores, observed, permutations,
                int.from_bytes(
                    hashlib.sha256(
                        f"text-anchor:{model}:{dataset}:{perturbation}:{operator_name}".encode()
                    ).digest()[:4],
                    "big",
                ),
            )
            permutation_rows.append(
                dict(
                    model=model,
                    dataset=dataset,
                    perturbation=perturbation,
                    operator=operator_name,
                    classes=scores[0].shape[0],
                    chance=1.0 / scores[0].shape[0],
                    observed_own_top1=observed,
                    **result,
                )
            )

    frame = pd.DataFrame(cells)
    frame.to_csv(out / "text_anchoring_cells.csv", index=False)
    summary = frame.groupby(
        ["model", "dataset", "perturbation", "operator", "template"], as_index=False
    ).agg(
        own_top1_mean=("own_top1", "mean"),
        own_top1_sd=("own_top1", "std"),
        own_top3_mean=("own_top3", "mean"),
        margin_mean=("margin", "mean"),
        margin_sd=("margin", "std"),
        mean_own_mean=("mean_own", "mean"),
        mean_wrong_mean=("mean_wrong", "mean"),
        classes=("classes", "first"),
        chance=("chance", "first"),
        n_seeds=("seed", "nunique"),
    )
    summary.to_csv(out / "text_anchoring_summary.csv", index=False)

    for operator_name in ("M0", "M1"):
        indices = [index for index, row in enumerate(permutation_rows) if row["operator"] == operator_name]
        if indices:
            q_values = _bh([permutation_rows[index]["p_value"] for index in indices])
            for index, q_value in zip(indices, q_values):
                permutation_rows[index]["q_value"] = q_value
    pd.DataFrame(permutation_rows).to_csv(out / "text_anchoring_permutation.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--models", default=",".join(TEXT_MODELS))
    parser.add_argument("--seeds", default="")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--perturbations", default=",".join(PERTURBATIONS))
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--clip-projection", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = {}
    for item in args.runs.split(","):
        seed, name = item.split("=", 1)
        runs[int(seed)] = name
    layout = RunLayout(args.run_root, runs)
    run(
        layout,
        args.out,
        project_root=args.project_root,
        models=tuple(item for item in args.models.split(",") if item),
        seeds=tuple(int(item) for item in args.seeds.split(",") if item) or layout.seeds,
        datasets=tuple(item for item in args.datasets.split(",") if item),
        perturbations=tuple(item for item in args.perturbations.split(",") if item),
        permutations=args.permutations,
        clip_projection=args.clip_projection,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
