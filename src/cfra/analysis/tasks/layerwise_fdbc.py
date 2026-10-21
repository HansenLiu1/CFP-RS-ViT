"""Layerwise foreground x background interaction from FdBc artifacts (S8b).

The persisted layerwise artifact contains FcBc, FcBd and FdBd.  The fourth
factorial cell FdBc must be extracted once with the frozen encoder (GPU).  This
task consumes those extracted arrays and reports, per layer and perturbation:

    d_Bc = h(FcBc) - h(FdBc)
    d_Bd = h(FcBd) - h(FdBd)
    I_FB = d_Bc - d_Bd

The scale-free summaries are the mean cosine, absolute cosine, sine and unit
disagreement between d_Bc and d_Bd.  Extraction is intentionally a separate GPU
step; the analysis itself is CPU-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfra.analysis.common import DATASETS, MODELS, RunLayout

LAYERS = ("patch_emb",) + tuple(f"L{index:02d}" for index in range(1, 13)) + ("final",)
FDBC_PERTURBATIONS = ("blur", "gaussian_noise", "patch_mask")
METRICS = ("cos", "abs_cos", "sin", "disagreement")
EPS = 1.0e-8


def _unit(values: np.ndarray) -> np.ndarray:
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), EPS, None)


def _stats(d_bc: np.ndarray, d_bd: np.ndarray) -> dict:
    d_bc = np.asarray(d_bc, dtype=np.float64)
    d_bd = np.asarray(d_bd, dtype=np.float64)
    u_bc = _unit(d_bc)
    u_bd = _unit(d_bd)
    cosine = np.sum(u_bc * u_bd, axis=1)
    return {
        "n": int(len(cosine)),
        "cos": float(np.mean(cosine)),
        "abs_cos": float(np.mean(np.abs(cosine))),
        "sin": float(np.mean(np.sqrt(np.clip(1.0 - cosine**2, 0.0, 1.0)))),
        "disagreement": float(np.mean(np.linalg.norm(u_bc - u_bd, axis=1))),
    }


def _fdbc_map(value: str | None, layout: RunLayout, seeds: tuple[int, ...]) -> dict[int, Path]:
    if not value:
        # Default layout: sibling of the run root, one directory per seed.
        return {seed: layout.run_root.parent / f"features_fdbc_{seed}" for seed in seeds}
    result = {}
    for item in value.split(","):
        seed, path = item.split("=", 1)
        result[int(seed)] = Path(path)
    return result


def _cell_rows(
    layout: RunLayout,
    seed: int,
    model: str,
    dataset: str,
    fdbc_dir: Path,
    perturbations: tuple[str, ...],
    layers: tuple[str, ...],
) -> list[dict]:
    artifact = layout.layerwise_path(seed, model, dataset)
    fdbc_path = fdbc_dir / f"layerwise_fdbc_{model}_{dataset}.npz"
    if not artifact.is_file() or not fdbc_path.is_file():
        return []
    metadata = json.loads((fdbc_dir / f"layerwise_fdbc_{model}_{dataset}.json").read_text(encoding="utf-8"))
    rows = []
    with np.load(artifact, allow_pickle=False) as artifact_payload, np.load(fdbc_path, allow_pickle=False) as fdbc_payload:
        row_index = np.asarray(fdbc_payload["row_index"], dtype=np.int64)
        kept_perturbations = np.asarray(fdbc_payload["perturbation"])
        keep = [perturbation for perturbation in perturbations if perturbation in set(kept_perturbations.tolist())]
        for layer in layers:
            if f"{layer}__fdbc" not in fdbc_payload.files:
                continue
            fdbc = np.asarray(fdbc_payload[f"{layer}__fdbc"], dtype=np.float64)
            if fdbc.shape[0] != len(row_index):
                raise ValueError(f"{fdbc_path.name}: {layer} rows do not match row_index")
            raw = artifact_payload[f"{layer}__raw"][row_index].astype(np.float64)
            clear = artifact_payload[f"{layer}__clear"][row_index].astype(np.float64)
            degraded = artifact_payload[f"{layer}__degraded"][row_index].astype(np.float64)
            d_bc = raw - fdbc
            d_bd = clear - degraded
            for perturbation in keep + ["all"]:
                mask = np.ones(len(row_index), dtype=bool) if perturbation == "all" else (kept_perturbations == perturbation)
                if mask.sum() < 2:
                    continue
                rows.append(
                    dict(
                        model=model,
                        dataset=dataset,
                        layer=layer,
                        perturbation=perturbation,
                        ** _stats(d_bc[mask], d_bd[mask]),
                    )
                )
    if not rows:
        return []
    print(f"  fdbc {seed} {model}/{dataset}: {len(rows)} rows (n={metadata.get('n_kept')})", flush=True)
    return rows


def _three_seed(detail: dict[int, pd.DataFrame]) -> pd.DataFrame:
    keys = ["model", "dataset", "layer", "perturbation"]
    frames = [frame[keys + ["n", *METRICS]].assign(seed=seed) for seed, frame in detail.items()]
    stacked = pd.concat(frames, ignore_index=True)
    grouped = stacked.groupby(keys, sort=False)
    out = grouped[list(METRICS)].agg(["mean", "std"]).reset_index()
    out.columns = [left if right == "" else f"{left}_{right}" for left, right in out.columns.to_flat_index()]
    out = out.rename(columns={f"{metric}_mean": metric for metric in METRICS})
    out = out.rename(columns={f"{metric}_std": f"{metric}_sd" for metric in METRICS})
    counts = grouped["n"].agg(["count", "sum"]).reset_index().rename(columns={"count": "n_seeds", "sum": "n_objects"})
    return out.merge(counts, on=keys, how="left")


def _trajectory(detail: dict[int, pd.DataFrame], layers: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = []
    for seed, frame in detail.items():
        subset = frame[frame.perturbation == "all"]
        aggregated = subset.groupby("layer")[list(METRICS)].mean().reindex([layer for layer in layers if layer in subset.layer.values])
        aggregated["seed"] = seed
        per_seed.append(aggregated.reset_index())
    stacked = pd.concat(per_seed, ignore_index=True)
    summary = stacked.groupby("layer")[list(METRICS)].agg(["mean", "std"]).reset_index()
    summary.columns = [left if right == "" else f"{left}_{right}" for left, right in summary.columns.to_flat_index()]
    summary = summary.rename(columns={f"{metric}_mean": metric for metric in METRICS})
    summary = summary.rename(columns={f"{metric}_std": f"{metric}_sd" for metric in METRICS})
    wide = stacked.pivot(index="layer", columns="seed", values=list(METRICS))
    wide.columns = [f"{metric}_seed{seed}" for metric, seed in wide.columns]
    merged = summary.merge(wide.reset_index(), on="layer", how="left")
    merged["layer"] = pd.Categorical(merged["layer"], categories=[layer for layer in layers if layer in merged.layer.values], ordered=True)
    return merged.sort_values("layer").reset_index(drop=True), stacked


def run(
    layout: RunLayout,
    out: Path,
    *,
    models: tuple[str, ...] = MODELS,
    seeds: tuple[int, ...] | None = None,
    datasets: tuple[str, ...] = DATASETS,
    perturbations: tuple[str, ...] = FDBC_PERTURBATIONS,
    layers: tuple[str, ...] = LAYERS,
    fdbc: str | None = None,
) -> None:
    """Read FdBc artifacts and write per-seed, three-seed and trajectory tables."""

    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds or layout.seeds
    fdbc_dirs = _fdbc_map(fdbc, layout, seeds)
    detail: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        rows = []
        for model in models:
            for dataset in datasets:
                rows.extend(
                    _cell_rows(layout, seed, model, dataset, fdbc_dirs[seed], perturbations, layers)
                )
        if not rows:
            print(f"  fdbc seed {seed}: no rows", flush=True)
            continue
        frame = pd.DataFrame(rows)
        frame.insert(0, "seed", seed)
        frame.to_csv(out / f"layerwise_fdbc_per_seed_{seed}.csv", index=False)
        detail[seed] = frame
    if not detail:
        return
    combined = pd.concat(detail.values(), ignore_index=True)
    combined.to_csv(out / "layerwise_fdbc_detail.csv", index=False)
    _three_seed(detail).to_csv(out / "layerwise_fdbc_3seed.csv", index=False)
    trajectory, per_seed = _trajectory(detail, layers)
    trajectory.to_csv(out / "layerwise_fdbc_trajectory_3seed.csv", index=False)
    per_seed.to_csv(out / "layerwise_fdbc_trajectory_per_seed.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default="")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--perturbations", default=",".join(FDBC_PERTURBATIONS))
    parser.add_argument("--layers", default=",".join(LAYERS))
    parser.add_argument("--fdbc", default=None, help="comma-separated seed=directory FdBc overrides")
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
        models=tuple(item for item in args.models.split(",") if item),
        seeds=tuple(int(item) for item in args.seeds.split(",") if item) or None,
        datasets=tuple(item for item in args.datasets.split(",") if item),
        perturbations=tuple(item for item in args.perturbations.split(",") if item),
        layers=tuple(item for item in args.layers.split(",") if item),
        fdbc=args.fdbc,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
