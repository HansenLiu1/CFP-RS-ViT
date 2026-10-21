"""Subspace-alignment analysis (paper S4c and main Sec. 6.4).

The readout-space removal experiment removes the top principal directions of
the M1 class-direction matrix and then refits the linear readout.  Recovery is
almost complete.  This task quantifies how strongly that removed subspace
aligns with the readout weights, the between-class scatter and the total
feature variance, and compares it with random and feature-PC subspaces.

All directions are fitted through :func:`cfra.directions.fit`; the readout
helpers live in :mod:`cfra.analysis.readouts`.
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from cfra.analysis.common import (
    DATASETS,
    FIT_LIMIT,
    MODELS,
    PERTURBATIONS,
    Features,
    RunLayout,
    cell_delta,
    fit_operator,
    unit,
)
from cfra.analysis.readouts import LinearReadout, PrototypeReadout

KS = (1, 3, 8, 16)
N_RANDOM = 5
METRICS = ("w_energy_raw", "w_energy_cen", "scatter", "total_var")
KEYS = ["model", "dataset", "perturbation", "k", "dim"]
DERIVED = [f"{metric}_{suffix}" for metric in METRICS for suffix in ("top", "rand", "bottom", "ratio")]


def project_out(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project rows onto the orthogonal complement of ``basis``, preserving norm."""

    values = np.asarray(x, dtype=np.float64)
    if basis.shape[0] == 0:
        return values.copy()
    modified = values - (values @ basis.T) @ basis
    n0 = np.linalg.norm(values, axis=-1, keepdims=True)
    n1 = np.linalg.norm(modified, axis=-1, keepdims=True)
    return modified * np.where(n1 > 1e-12, n0 / np.maximum(n1, 1e-12), 1.0)


def random_basis(dim: int, k: int, key: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(key.encode("utf-8")))
    q, _ = np.linalg.qr(rng.standard_normal((k, dim)).T)
    return q.T


def accuracy(readout, X: np.ndarray, y: np.ndarray) -> float:
    classes = np.asarray(readout.classes)
    return float(np.mean(classes[np.argmax(readout.scores(X), axis=1)] == y))


def frac(A: np.ndarray, B: np.ndarray) -> float:
    """``||A B^T||_F^2 / ||A||_F^2`` for orthonormal rows of ``B``."""

    denominator = float(np.sum(A * A))
    if denominator <= 0.0:
        return float("nan")
    projected = A @ B.T
    return float(np.sum(projected * projected) / denominator)


def cell_frame(feat: Features, dataset: str, perturbation: str):
    """Fit M1 and return centered readout/query features for one cell."""

    cell = cell_delta(feat, dataset, perturbation, "direction_fit", limit=FIT_LIMIT)
    operator, directions, classes = fit_operator(cell["values"], cell["labels"], "M1")
    common_axis = unit(operator.mean[None, :])[0]

    readout_index = np.flatnonzero(feat.mask(dataset, perturbation, "readout_fit"))
    query_index = np.flatnonzero(feat.mask(dataset, perturbation, "confirmation"))
    raw = feat.array("raw")
    labels = feat.meta["class_id"].astype(np.int64)

    fit_raw, fit_labels = raw[readout_index].astype(np.float64), labels[readout_index]
    keep = np.isin(fit_labels, classes)
    fit_raw, fit_labels = fit_raw[keep], fit_labels[keep]

    query_raw, query_labels = raw[query_index].astype(np.float64), labels[query_index]
    keep = np.isin(query_labels, classes)
    query_raw, query_labels = query_raw[keep], query_labels[keep]

    mean = fit_raw.mean(axis=0)
    return directions, classes, common_axis, fit_raw - mean, fit_labels, query_raw - mean, query_labels


def run_cell(
    seed: int,
    model: str,
    feat: Features,
    dataset: str,
    perturbation: str,
) -> tuple[list[dict], dict | None]:
    directions, classes, common_axis, Xf, yf, Xq, yq = cell_frame(feat, dataset, perturbation)
    if len(np.unique(yf)) < 3 or len(yq) == 0:
        return [], None

    dim = Xq.shape[1]
    n_classes = directions.shape[0]
    kmax = min(max(KS), n_classes - 2, dim - 1)
    _, singular, vt = np.linalg.svd(directions, full_matrices=False)
    rank_d = int(np.sum(singular > 1e-6 * singular[0]))

    linear = LinearReadout(Xf, yf)
    prototype = PrototypeReadout(Xf, yf, centered=False)
    W = np.asarray(linear.clf.coef_, dtype=np.float64)
    Wc = W - W.mean(axis=0, keepdims=True)
    Xc = Xf - Xf.mean(axis=0, keepdims=True)
    means = [Xf[yf == class_id].mean(axis=0) for class_id in classes if np.any(yf == class_id)]
    M = np.stack(means)
    M = M - M.mean(axis=0, keepdims=True)

    ident = dict(
        seed=seed,
        model=model,
        dataset=dataset,
        perturbation=perturbation,
        n_classes=n_classes,
        n_fit=int(len(yf)),
        n_query=int(len(yq)),
        dim=dim,
        rank_d=rank_d,
        support="readout_all",
        space="readout_centered",
        direction_fit="M1_fit15",
    )

    bases: list[tuple[str, int, int, np.ndarray]] = []
    for k in KS:
        if k > kmax:
            continue
        bases.append(("top", k, 0, vt[:k]))
        bases.append(("bottom", k, 0, vt[-k:]))
        for draw in range(N_RANDOM):
            key = f"b8:{seed}:{model}:{dataset}:{perturbation}:{k}:{draw}"
            bases.append(("random", k, draw, random_basis(dim, k, key)))
    k_full = min(rank_d, dim)
    bases.append(("full_D", k_full, 0, vt[:k_full]))

    rows = []
    for subspace, k, draw, basis in bases:
        row = dict(
            **ident,
            subspace=subspace,
            k=int(k),
            draw=int(draw),
            isotropic=float(k) / float(dim),
        )
        for name, matrix in (
            ("w_energy_raw", W),
            ("w_energy_cen", Wc),
            ("scatter", M),
            ("total_var", Xc),
        ):
            row[name] = frac(matrix, basis)
        rows.append(row)

    validation = dict(
        **ident,
        baseline_lin=accuracy(linear, Xq, yq),
        baseline_pro=accuracy(prototype, Xq, yq),
        acc_lin_e=accuracy(linear, project_out(Xq, common_axis[None, :]), yq),
        acc_pro_e=accuracy(prototype, project_out(Xq, common_axis[None, :]), yq),
    )
    k_validation = 8 if 8 <= kmax else max(k for k in KS if k <= kmax)
    basis8 = vt[:k_validation]
    Xq8, Xf8 = project_out(Xq, basis8), project_out(Xf, basis8)
    validation["k_val"] = int(k_validation)
    validation["acc_lin_frozen"] = accuracy(linear, Xq8, yq)
    validation["acc_pro_frozen"] = accuracy(prototype, Xq8, yq)
    validation["acc_lin_refit"] = accuracy(LinearReadout(Xf8, yf), Xq8, yq)
    validation["acc_pro_refit"] = accuracy(PrototypeReadout(Xf8, yf, centered=False), Xq8, yq)
    validation["drop_lin_frozen"] = validation["baseline_lin"] - validation["acc_lin_frozen"]
    validation["drop_lin_refit"] = validation["baseline_lin"] - validation["acc_lin_refit"]
    validation["drop_pro_frozen"] = validation["baseline_pro"] - validation["acc_pro_frozen"]
    validation["drop_pro_refit"] = validation["baseline_pro"] - validation["acc_pro_refit"]
    return rows, validation


def pc_cell(
    seed: int,
    model: str,
    feat: Features,
    dataset: str,
    perturbation: str,
) -> list[dict]:
    """Top feature-PC control for the same readout-fit features."""

    cell = cell_delta(feat, dataset, perturbation, "direction_fit", limit=FIT_LIMIT)
    _operator, _directions, classes = fit_operator(cell["values"], cell["labels"], "M1")
    readout_index = np.flatnonzero(feat.mask(dataset, perturbation, "readout_fit"))
    labels = feat.meta["class_id"].astype(np.int64)
    Xf, yf = feat.array("raw")[readout_index].astype(np.float64), labels[readout_index]
    keep = np.isin(yf, classes)
    Xf, yf = Xf[keep], yf[keep]
    if len(np.unique(yf)) < 3:
        return []

    Xc = Xf - Xf.mean(axis=0, keepdims=True)
    linear = LinearReadout(Xf, yf)
    W = np.asarray(linear.clf.coef_, dtype=np.float64)
    M = np.stack([Xf[yf == class_id].mean(axis=0) for class_id in classes if np.any(yf == class_id)])
    M = M - M.mean(axis=0, keepdims=True)
    vt = np.linalg.svd(Xc, full_matrices=False)[2]

    rows = []
    for k in KS:
        if k >= Xc.shape[0] or k > Xc.shape[1]:
            continue
        basis = vt[:k]
        row = dict(
            seed=seed,
            model=model,
            dataset=dataset,
            perturbation=perturbation,
            k=k,
            dim=Xc.shape[1],
            isotropic=k / Xc.shape[1],
        )
        for name, matrix in (("w_energy_raw", W), ("scatter", M), ("total_var", Xc)):
            row[name] = frac(matrix, basis)
        rows.append(row)
    return rows


def _per_seed_wide(long: pd.DataFrame) -> pd.DataFrame:
    rand = long[long.subspace == "random"].groupby(["seed", *KEYS])[list(METRICS)].std(ddof=0)
    rand.columns = [f"{metric}_rand_drawsd" for metric in METRICS]
    mean = long[long.subspace.isin(("top", "random", "bottom"))].groupby(
        ["seed", *KEYS, "subspace"]
    )[list(METRICS)].mean()
    wide = mean.unstack("subspace")
    out = pd.DataFrame(index=wide.index)
    for metric in METRICS:
        for source, name in (("top", "top"), ("random", "rand"), ("bottom", "bottom")):
            if (metric, source) in wide.columns:
                out[f"{metric}_{name}"] = wide[(metric, source)]
        if f"{metric}_top" in out and f"{metric}_rand" in out:
            out[f"{metric}_ratio"] = out[f"{metric}_top"] / out[f"{metric}_rand"]
        iso = out.index.get_level_values("k").to_numpy() / out.index.get_level_values("dim").to_numpy()
        out["iso"] = iso
        out[f"{metric}_xiso"] = out[f"{metric}_top"] / iso
    return out.join(rand).reset_index()


def _aggregate(frame: pd.DataFrame, keys: list[str], cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or [column for column in DERIVED + ["iso"] if column in frame]
    inner = frame.groupby(["seed", *keys])[cols].mean()
    grouped = inner.groupby(keys)[cols]
    out = grouped.agg(["mean", "std"])
    out.columns = [f"{a}_{'mean' if b == 'mean' else 'sd'}" for a, b in out.columns]
    out["n_seeds"] = grouped.size()
    return out.reset_index()


def summarize(out: Path) -> None:
    """Aggregate the per-seed alignment, PC control and validation tables."""

    long = pd.read_csv(out / "alignment_per_seed.csv")
    main = long[long.subspace != "full_D"].copy()
    wide = _per_seed_wide(main)

    cell = _aggregate(wide, KEYS)
    cell.insert(0, "seed", "3seed")
    cell.to_csv(out / "alignment_3seed.csv", index=False)

    pooled = _aggregate(wide, ["k"])
    pooled.insert(0, "scope", "pooled")
    pooled.to_csv(out / "alignment_pooled.csv", index=False)
    for dimension in ("perturbation", "model", "dataset"):
        table = _aggregate(wide, ["k", dimension])
        table.insert(0, "scope", dimension)
        table.to_csv(out / f"alignment_by_{dimension}.csv", index=False)

    full = long[long.subspace == "full_D"].copy()
    full = full.groupby(["seed", "model", "dataset", "perturbation", "k", "dim"])[list(METRICS)].mean().reset_index()
    full["iso"] = full["k"] / full["dim"]
    for metric in METRICS:
        full[f"{metric}_xiso"] = full[metric] / full["iso"]
    frames = []
    columns = list(METRICS) + ["iso"] + [f"{metric}_xiso" for metric in METRICS]
    table = _aggregate(full, ["k"], cols=columns)
    table.insert(0, "scope", "pooled")
    frames.append(table)
    table = _aggregate(full, ["k", "model"], cols=columns)
    table.insert(0, "scope", "model")
    frames.append(table)
    pd.concat(frames, ignore_index=True).to_csv(out / "alignment_fullD.csv", index=False)

    validation = pd.read_csv(out / "alignment_validation.csv")
    drops = ["drop_lin_frozen", "drop_lin_refit", "drop_pro_frozen", "drop_pro_refit"]
    rows = []
    for scope, frame, keys in (
        ("pooled", validation, []),
        ("model", validation, ["model"]),
        ("perturbation", validation, ["perturbation"]),
        ("dataset", validation, ["dataset"]),
        ("seed", validation, ["seed"]),
    ):
        groups = frame.groupby(keys) if keys else [("all", frame)]
        for level, group in groups:
            per_seed = group.groupby("seed")[drops + ["baseline_lin", "baseline_pro"]].mean()
            for metric in drops + ["baseline_lin", "baseline_pro"]:
                rows.append(
                    dict(
                        scope=scope,
                        level=str(level),
                        metric=metric,
                        n_cells=len(group),
                        mean=float(per_seed[metric].mean()),
                        sd=float(np.std(per_seed[metric], ddof=1)),
                    )
                )
    pd.DataFrame(rows).to_csv(out / "alignment_validation_summary.csv", index=False)

    pc_path = out / "pc_control_per_seed.csv"
    if pc_path.exists():
        pc = pd.read_csv(pc_path)
        cols = ["w_energy_raw", "scatter", "total_var", "isotropic"]
        frames = []
        for keys in (["k"], ["k", "model"]):
            table = _aggregate(pc, keys, cols=cols)
            table.insert(0, "scope", "pooled" if len(keys) == 1 else keys[-1])
            frames.append(table)
        pd.concat(frames, ignore_index=True).to_csv(out / "pc_control_pooled.csv", index=False)


def run(
    layout: RunLayout,
    out: Path,
    *,
    models: tuple[str, ...] = MODELS,
    seeds: tuple[int, ...] | None = None,
    datasets: tuple[str, ...] = DATASETS,
    perturbations: tuple[str, ...] = PERTURBATIONS,
    max_cells: int = 0,
    summarize_output: bool = True,
) -> None:
    """Run the alignment and feature-PC control, then write summaries."""

    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds or layout.seeds
    detail = out / "detail"
    pc_detail = out / "pc_detail"
    detail.mkdir(exist_ok=True)
    pc_detail.mkdir(exist_ok=True)

    alignment_rows: list[dict] = []
    validation_rows: list[dict] = []
    pc_rows: list[dict] = []

    for seed in seeds:
        for model in models:
            alignment_path = detail / f"{seed}_{model}.csv"
            validation_path = detail / f"{seed}_{model}_validation.csv"
            pc_path = pc_detail / f"{seed}_{model}.csv"
            if alignment_path.exists() and validation_path.exists() and pc_path.exists():
                alignment_rows.extend(pd.read_csv(alignment_path).to_dict("records"))
                validation_rows.extend(pd.read_csv(validation_path).to_dict("records"))
                pc_rows.extend(pd.read_csv(pc_path).to_dict("records"))
                continue

            feat = Features(layout, seed, model)
            cell_alignment: list[dict] = []
            cell_validation: list[dict] = []
            cell_pc: list[dict] = []
            try:
                count = 0
                for dataset in datasets:
                    for perturbation in perturbations:
                        rows, validation = run_cell(seed, model, feat, dataset, perturbation)
                        cell_alignment.extend(rows)
                        if validation is not None:
                            cell_validation.append(validation)
                        cell_pc.extend(pc_cell(seed, model, feat, dataset, perturbation))
                        count += 1
                        print(f"  {seed} {model} {dataset}/{perturbation} rows={len(rows)}", flush=True)
                        if max_cells and count >= max_cells:
                            break
                    if max_cells and count >= max_cells:
                        break
            finally:
                feat.close()
            pd.DataFrame(cell_alignment).to_csv(alignment_path, index=False)
            pd.DataFrame(cell_validation).to_csv(validation_path, index=False)
            pd.DataFrame(cell_pc).to_csv(pc_path, index=False)
            alignment_rows.extend(cell_alignment)
            validation_rows.extend(cell_validation)
            pc_rows.extend(cell_pc)

    pd.DataFrame(alignment_rows).to_csv(out / "alignment_per_seed.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(out / "alignment_validation.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(out / "pc_control_per_seed.csv", index=False)
    if summarize_output:
        summarize(out)


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
        max_cells=args.max_cells,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
