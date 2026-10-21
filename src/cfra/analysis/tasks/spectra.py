"""Final-layer class-direction spectra and removal geometry (S4b/S4c-1).

This task computes the spectra of the clean-ROI prototypes and of the M0-M6
class-direction matrices, the M1 common-mode decomposition, the top-k removal
curves, and the class-matched 16-class control.  Class directions are always
fitted through :func:`cfra.analysis.common.fit_operator`, so there is only one
direction implementation in the public repository.

The output is intentionally close to the archived batch tables:

``isotropic_null.csv``
    Deterministic isotropic nulls for the three class counts and the two
    encoder dimensions.
``spectra_per_seed.csv`` / ``spectra_3seed.csv``
    Spectrum summaries for clean prototypes and M0-M6 plus the retired
    ``w_form`` reference.
``residual_spectra_*``
    Spectra of the orthogonal common-mode residual and its unit version.
``decomposition_*``
    Common-mode projections and top-3 class diagnostics.
``removal_curves_*``
    Top-k removal curves in the final-layer direction space.
``matched16_*``
    DIOR-R/FAIR1M class-matched draws against the 16 DOTA classes.
``criteria_summary.csv``
    Counts for the two-cluster criteria used in the supplement.
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
    spectrum,
    unit,
)
from cfra.analysis.matrix import paired_cos, project_out, stable_seed

FORMS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "w_form")
REMOVAL_FORMS = ("M1", "M1:O", "M0", "M2", "M3", "w_form")
NULL_CLASSES = (16, 20, 37)
NULL_DIMS = (512, 768)
CRIT_COS = 0.5
CRIT_PAIR = 0.2


def _key(seed: int, model: str, dataset: str, perturbation: str, form: str) -> tuple:
    return seed, model, dataset, perturbation, form


def isotropic_null(n_classes: int, dim: int, reps: int = 200, seed: int = 0) -> dict[str, float]:
    """Isotropic null for the uncentred and centred direction spectra."""

    rng = np.random.default_rng(seed)
    plain = [spectrum(unit(rng.standard_normal((n_classes, dim)))) for _ in range(reps)]
    centered = []
    for _ in range(reps):
        rows = unit(rng.standard_normal((n_classes, dim)))
        centered.append(spectrum(unit(rows - rows.mean(axis=0))))
    p = {name: float(np.mean([row[name] for row in plain])) for name in plain[0]}
    q = {name: float(np.mean([row[name] for row in centered])) for name in centered[0]}
    return {
        "iso_top1": p["top1_share"],
        "iso_er": p["effective_rank"],
        "iso_n50": p["n50"],
        "iso_n80": p["n80"],
        "isoc_top1": q["top1_share"],
        "isoc_er": q["effective_rank"],
        "isoc_n50": q["n50"],
        "isoc_n80": q["n80"],
    }


def isotropic_centered_removed(n_classes: int, dim: int, k: int = 1, reps: int = 200, seed: int = 0) -> float:
    """Centred isotropic null after removing its top-k directions."""

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        rows = unit(rng.standard_normal((n_classes, dim)))
        rows = unit(rows - rows.mean(axis=0))
        _, _, vt = np.linalg.svd(rows, full_matrices=False)
        values.append(spectrum(unit(rows - (rows @ vt[:k].T) @ vt[:k]))["top1_share"])
    return float(np.mean(values))


def build_nulls(reps: int) -> tuple[dict[tuple[int, int], dict[str, float]], pd.DataFrame]:
    """Build the deterministic isotropic tables used by every cell."""

    nulls: dict[tuple[int, int], dict[str, float]] = {}
    rows = []
    for n_classes in NULL_CLASSES:
        for dim in NULL_DIMS:
            values = isotropic_null(n_classes, dim, reps=reps, seed=0)
            values["isoc_rem1_top1"] = isotropic_centered_removed(n_classes, dim, k=1, reps=reps, seed=0)
            nulls[(n_classes, dim)] = values
            rows.append(dict(values, n_classes=n_classes, dim=dim, reps=reps, null_seed=0))
    return nulls, pd.DataFrame(rows)


def _cell_matrices(
    feat: Features, dataset: str, perturbation: str
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit M0-M6 on one cell and add the retired w-form reference."""

    fit = cell_delta(feat, dataset, perturbation, "direction_fit", limit=FIT_LIMIT)
    if len(np.unique(fit["labels"])) < 3:
        raise ValueError("cell has fewer than three classes")
    fit_mask = cell_rows(feat, dataset, perturbation, "direction_fit", limit=FIT_LIMIT)
    background = background_fit(feat, fit_mask, normalization="unit")
    values, labels = fit["values"], fit["labels"]
    matrices: dict[str, np.ndarray] = {}
    classes = None
    operator_m1 = None
    for form in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
        operator, directions, current_classes = fit_operator(
            values,
            labels,
            form,
            background if form == "M6" else None,
        )
        matrices[form] = np.asarray(directions, dtype=np.float64)
        if classes is None:
            classes = np.asarray(current_classes, dtype=np.int64)
        elif not np.array_equal(classes, current_classes):
            raise RuntimeError(f"class order changed between operators for {dataset}/{perturbation}")
        if form == "M1":
            operator_m1 = operator
    if operator_m1 is None:
        raise AssertionError("M1 was not fitted")
    origin = common_mode(operator_m1)
    matrices["w_form"] = unit(matrices["M0"] - origin)
    return matrices, np.asarray(classes, dtype=np.int64), origin, values, labels, fit["norms"]


def _class_moments(
    values: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    raw_norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    norms = []
    magnitudes = []
    raw_norms = np.asarray(raw_norms, dtype=np.float64)
    for class_id in classes.tolist():
        selected = labels == class_id
        if not selected.any():
            raise ValueError("class missing from fit rows")
        class_mean = values[selected].mean(axis=0)
        norms.append(float(np.linalg.norm(class_mean)))
        magnitudes.append(float(raw_norms[selected].mean()))
    return np.asarray(norms), np.asarray(magnitudes)


def _spectrum_row(
    *,
    seed: int,
    model: str,
    dataset: str,
    perturbation: str,
    form: str,
    matrix: np.ndarray,
    null: dict[str, float],
    support: str = "fit15",
) -> dict:
    values = spectrum(matrix)
    return dict(
        seed=seed,
        model=model,
        dataset=dataset,
        perturbation=perturbation,
        form=form,
        support=support,
        layer="final",
        normalization="unit",
        n_classes=int(matrix.shape[0]),
        dim=int(matrix.shape[1]),
        **values,
        **null,
        ratio_er=values["effective_rank"] / null["isoc_er"],
        ratio_top1=values["top1_share"] / null["isoc_top1"],
    )


def _decomposition_row(
    *,
    seed: int,
    model: str,
    dataset: str,
    perturbation: str,
    form: str,
    matrix: np.ndarray,
    origin: np.ndarray,
    classes: np.ndarray,
    magnitudes: np.ndarray,
    class_norms: np.ndarray,
) -> dict:
    axis = unit(origin[None, :])[0]
    a = matrix @ axis
    residual = matrix - np.outer(a, axis)
    residual_unit = unit(residual)
    v1 = np.linalg.svd(matrix, full_matrices=False)[2][0]
    residual_v1 = np.linalg.svd(residual, full_matrices=False)[2][0]
    residual_unit_v1 = np.linalg.svd(residual_unit, full_matrices=False)[2][0]
    top3 = np.argsort(-magnitudes, kind="stable")[:3]
    residual_spectrum = spectrum(residual)
    residual_unit_spectrum = spectrum(residual_unit)
    return dict(
        seed=seed,
        model=model,
        dataset=dataset,
        perturbation=perturbation,
        form=form,
        support="fit15",
        layer="final",
        normalization="unit",
        n_classes=int(matrix.shape[0]),
        dim=int(matrix.shape[1]),
        m_norm=float(np.linalg.norm(origin)),
        mean_A_norm=float(class_norms.mean()),
        mean_a=float(a.mean()),
        frac_a_neg=float(np.mean(a < 0)),
        cos_v1_e=float(abs(v1 @ axis)),
        top3_classes="|".join(str(int(classes[index])) for index in top3),
        top3_magnitude=float(magnitudes[top3].mean()),
        top3_pair_cos=paired_cos(matrix[top3]),
        top3_e_cos=float(np.mean(matrix[top3] @ axis)),
        o_mean_norm=float(np.linalg.norm(residual, axis=1).mean()),
        o_min_norm=float(np.linalg.norm(residual, axis=1).min()),
        o_frac_neg=float(np.mean((residual @ axis) < 0)),
        o_cos_v1_e=float(abs(residual_v1 @ axis)),
        ou_cos_v1_e=float(abs(residual_unit_v1 @ axis)),
        o_top3_pair_cos=paired_cos(residual_unit[top3]),
        o_top3_e_cos=float(np.mean(residual_unit[top3] @ axis)),
        o_top1_share=residual_spectrum["top1_share"],
        o_effective_rank=residual_spectrum["effective_rank"],
        o_n50=residual_spectrum["n50"],
        o_n80=residual_spectrum["n80"],
        ou_top1_share=residual_unit_spectrum["top1_share"],
        ou_effective_rank=residual_unit_spectrum["effective_rank"],
        ou_n50=residual_unit_spectrum["n50"],
        ou_n80=residual_unit_spectrum["n80"],
    )


def _residual_rows(
    *,
    seed: int,
    model: str,
    dataset: str,
    perturbation: str,
    form: str,
    matrix: np.ndarray,
    origin: np.ndarray,
    null: dict[str, float],
) -> list[dict]:
    axis = unit(origin[None, :])[0]
    a = matrix @ axis
    residual = matrix - np.outer(a, axis)
    residual_unit = unit(residual)
    rows = []
    for label, values in (("O", residual), ("Ounit", residual_unit)):
        metrics = spectrum(values)
        rows.append(
            dict(
                seed=seed,
                model=model,
                dataset=dataset,
                perturbation=perturbation,
                form=f"{form}:{label}",
                space="orthogonal" if label == "O" else "orthogonal_unit",
                support="fit15",
                layer="final",
                normalization="unit",
                n_classes=int(values.shape[0]),
                dim=int(values.shape[1]),
                **metrics,
                **null,
                ratio_er=metrics["effective_rank"] / null["isoc_er"],
                ratio_top1=metrics["top1_share"] / null["isoc_top1"],
            )
        )
    return rows


def _removal_matrix(matrices: dict[str, np.ndarray], origin: np.ndarray, form: str) -> np.ndarray:
    if form == "w_form":
        return matrices["w_form"]
    if form == "M1:O":
        axis = unit(origin[None, :])[0]
        values = matrices["M1"]
        return values - np.outer(values @ axis, axis)
    return matrices[form]


def _removal_rows(
    *,
    seed: int,
    model: str,
    dataset: str,
    perturbation: str,
    form: str,
    matrix: np.ndarray,
) -> list[dict]:
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    rows = []
    for removed in range(0, matrix.shape[0] - 1):
        basis = vt[:removed]
        values = unit(project_out(matrix, basis, )) if removed else matrix.copy()
        metrics = spectrum(values)
        rows.append(
            dict(
                seed=seed,
                model=model,
                dataset=dataset,
                perturbation=perturbation,
                form=form,
                support="fit15",
                layer="final",
                normalization="unit",
                n_classes=int(matrix.shape[0]),
                removed=int(removed),
                **metrics,
            )
        )
    return rows


def _matched16_rows(
    *,
    seed: int,
    model: str,
    dataset: str,
    perturbation: str,
    values: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    draws: int,
) -> list[dict]:
    if draws <= 0:
        return []
    if dataset == "dota" and len(classes) == 16:
        selections = [(0, np.arange(len(classes)))]
        support = "fit15_matched16_full"
    else:
        if len(classes) < 16:
            return []
        selections = []
        for draw in range(draws):
            rng = np.random.default_rng(stable_seed(seed, model, dataset, draw, tag="matched16"))
            selections.append((draw, np.sort(rng.choice(len(classes), 16, replace=False))))
        support = "fit15_matched16_draw40"

    rows = []
    for draw, chosen in selections:
        selected_classes = classes[chosen]
        keep = np.isin(labels, selected_classes)
        if len(np.unique(labels[keep])) != 16:
            continue
        operator, directions, current_classes = fit_operator(values[keep], labels[keep], "M1")
        if len(current_classes) != 16:
            continue
        _, _, vt = np.linalg.svd(directions, full_matrices=False)
        for removed in range(0, 15):
            basis = vt[:removed]
            result = unit(project_out(directions, basis)) if removed else directions.copy()
            metrics = spectrum(result)
            rows.append(
                dict(
                    seed=seed,
                    model=model,
                    dataset=dataset,
                    perturbation=perturbation,
                    form="M1",
                    support=support,
                    layer="final",
                    normalization="unit",
                    draw=int(draw),
                    n_classes=16,
                    removed=int(removed),
                    m_norm=float(np.linalg.norm(common_mode(operator))),
                    **metrics,
                )
            )
    return rows


def _criteria_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    values = frame.copy()
    values["c1c2"] = (values["c1"] > 0) & (values["c2"] > 0)
    means = values.groupby(["model", "dataset", "perturbation", "form"])[
        ["cos_v1_e", "top3_o_pair_cos", "rem1_ratio"]
    ].mean().reset_index()
    seed_mean = values.groupby(["model", "dataset", "perturbation", "form"])[["c1", "c2"]].mean().reset_index()
    seed_mean = seed_mean.merge(means, on=["model", "dataset", "perturbation", "form"])
    seed_mean["c1"] = seed_mean["cos_v1_e"] > CRIT_COS
    seed_mean["c2"] = seed_mean["top3_o_pair_cos"] > CRIT_PAIR
    seed_mean["c1c2"] = seed_mean["c1"] & seed_mean["c2"]
    all_seeds = values.groupby(["model", "dataset", "perturbation", "form"])[["c1", "c2", "c1c2"]].min().reset_index()
    all_seeds = all_seeds.merge(means, on=["model", "dataset", "perturbation", "form"])

    rows = []
    for mode, table in (("per_seed_cell", values), ("seed_mean", seed_mean), ("all_three_seeds", all_seeds)):
        for (perturbation, form), group in table.groupby(["perturbation", "form"], sort=False):
            rows.append(
                dict(
                    mode=mode,
                    perturbation=perturbation,
                    form=form,
                    n_cells=int(len(group)),
                    n_c1=int(group["c1"].sum()),
                    n_c2=int(group["c2"].sum()),
                    n_c1_and_c2=int(group["c1c2"].sum()),
                    frac_c1=float(group["c1"].mean()),
                    frac_c2=float(group["c2"].mean()),
                    frac_c1_and_c2=float(group["c1c2"].mean()),
                    mean_cos_v1_e=float(group["cos_v1_e"].mean()),
                    mean_top3_o_pair=float(group["top3_o_pair_cos"].mean()),
                    mean_rem1_ratio=float(group["rem1_ratio"].mean()),
                    median_rem1_ratio=float(group["rem1_ratio"].median()),
                    n_rem1_ratio_gt1=int((group["rem1_ratio"] > 1).sum()),
                )
            )
    return pd.DataFrame(rows)


def _aggregate(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    per_seed = frame.groupby(["seed", *keys])[metrics].mean().reset_index()
    grouped = per_seed.groupby(keys)[metrics]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out.columns = [f"{left}_{right}" if right else left for left, right in out.columns.to_flat_index()]
    out = out.rename(columns={f"{metric}_mean": metric for metric in metrics})
    out = out.rename(columns={f"{metric}_std": f"{metric}_sd" for metric in metrics})
    out = out.rename(columns={f"{metric}_count": f"n_seeds_{metric}" for metric in metrics})
    first_count = f"n_seeds_{metrics[0]}"
    if first_count in out:
        out = out.rename(columns={first_count: "n_seeds"})
        out = out.drop(columns=[f"n_seeds_{metric}" for metric in metrics[1:] if f"n_seeds_{metric}" in out])
    return out


def _read_parts(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _combine(parts: Path, stem: str, keys: list[str], metrics: list[str], out: Path) -> None:
    files = sorted(parts.glob(f"{stem}_*.csv"))
    frame = _read_parts(files)
    if frame.empty:
        return
    frame.to_csv(out / f"{stem}_per_seed.csv", index=False)
    summary = _aggregate(frame, keys, metrics)
    if not summary.empty:
        summary.to_csv(out / f"{stem}_3seed.csv", index=False)


def run(
    layout: RunLayout,
    out: Path,
    *,
    models: tuple[str, ...] = MODELS,
    seeds: tuple[int, ...] | None = None,
    datasets: tuple[str, ...] = DATASETS,
    perturbations: tuple[str, ...] = PERTURBATIONS,
    max_cells: int = 0,
    null_reps: int = 200,
    matched16_draws: int = 40,
) -> None:
    """Compute spectra, common-mode decomposition and removal curves."""

    out.mkdir(parents=True, exist_ok=True)
    parts = out / "detail"
    parts.mkdir(exist_ok=True)
    seeds = seeds or layout.seeds
    nulls, null_frame = build_nulls(null_reps)
    null_frame.to_csv(out / "isotropic_null.csv", index=False)

    expected = ("spectra", "residual", "decomposition", "criteria", "removal", "matched16")
    for seed in seeds:
        for model in models:
            paths = {name: parts / f"{name}_{seed}_{model}.csv" for name in expected}
            if max_cells == 0 and all(path.exists() for path in paths.values()):
                continue
            feat = Features(layout, seed, model)
            spectra_rows: list[dict] = []
            residual_rows: list[dict] = []
            decomposition_rows: list[dict] = []
            criteria_rows: list[dict] = []
            removal_rows: list[dict] = []
            matched_rows: list[dict] = []
            try:
                cell_count = 0
                for dataset in datasets:
                    prototype_mask = cell_rows(feat, dataset, "blur", "direction_fit", limit=FIT_LIMIT)
                    raw = np.asarray(feat.array("raw")[prototype_mask], dtype=np.float64)
                    prototype_labels = feat.meta["class_id"][prototype_mask].astype(np.int64)
                    valid = np.linalg.norm(raw, axis=1) > EPS
                    raw, prototype_labels = raw[valid], prototype_labels[valid]
                    prototype_rows = unit(raw)
                    prototype_classes = np.asarray(sorted(set(prototype_labels.tolist())), dtype=np.int64)
                    prototype = np.stack(
                        [unit(prototype_rows[prototype_labels == class_id].mean(axis=0)) for class_id in prototype_classes]
                    )
                    prototype_null = nulls[(len(prototype_classes), prototype.shape[1])]
                    spectra_rows.append(
                        _spectrum_row(
                            seed=seed,
                            model=model,
                            dataset=dataset,
                            perturbation="-",
                            form="clean_proto",
                            matrix=prototype,
                            null=prototype_null,
                        )
                    )
                    for perturbation in perturbations:
                        matrices, classes, origin, values, labels, raw_norms = _cell_matrices(feat, dataset, perturbation)
                        null = nulls[(len(classes), matrices["M1"].shape[1])]
                        class_norms, magnitudes = _class_moments(values, labels, classes, raw_norms)
                        for form in FORMS:
                            matrix = matrices[form]
                            spectra_rows.append(
                                _spectrum_row(
                                    seed=seed,
                                    model=model,
                                    dataset=dataset,
                                    perturbation=perturbation,
                                    form=form,
                                    matrix=matrix,
                                    null=null,
                                )
                            )
                            decomposition_rows.append(
                                _decomposition_row(
                                    seed=seed,
                                    model=model,
                                    dataset=dataset,
                                    perturbation=perturbation,
                                    form=form,
                                    matrix=matrix,
                                    origin=origin,
                                    classes=classes,
                                    magnitudes=magnitudes,
                                    class_norms=class_norms,
                                )
                            )
                            residual_rows.extend(
                                _residual_rows(
                                    seed=seed,
                                    model=model,
                                    dataset=dataset,
                                    perturbation=perturbation,
                                    form=form,
                                    matrix=matrix,
                                    origin=origin,
                                    null=null,
                                )
                            )
                            if form in {"M0", "M1", "M2", "M3", "M4", "M5", "M6", "w_form"}:
                                axis = unit(origin[None, :])[0]
                                residual = matrix - np.outer(matrix @ axis, axis)
                                residual_unit = unit(residual)
                                top3 = np.argsort(-magnitudes, kind="stable")[:3]
                                v1 = np.linalg.svd(matrix, full_matrices=False)[2][0]
                                criteria_rows.append(
                                    dict(
                                        seed=seed,
                                        model=model,
                                        dataset=dataset,
                                        perturbation=perturbation,
                                        form=form,
                                        support="fit15",
                                        layer="final",
                                        normalization="unit",
                                        cos_v1_e=float(abs(v1 @ axis)),
                                        top3_o_pair_cos=paired_cos(residual_unit[top3]),
                                        c1=float(abs(v1 @ axis) > CRIT_COS),
                                        c2=float(paired_cos(residual_unit[top3]) > CRIT_PAIR),
                                        rem1_top1=spectrum(unit(project_out(matrix, np.linalg.svd(matrix, full_matrices=False)[2][:1])))[
                                            "top1_share"
                                        ],
                                        isoc_rem1_top1=null["isoc_rem1_top1"],
                                    )
                                )
                                criteria_rows[-1]["rem1_ratio"] = (
                                    criteria_rows[-1]["rem1_top1"] / null["isoc_rem1_top1"]
                                )
                        for removal_form in REMOVAL_FORMS:
                            removal_matrix = _removal_matrix(matrices, origin, removal_form)
                            removal_rows.extend(
                                _removal_rows(
                                    seed=seed,
                                    model=model,
                                    dataset=dataset,
                                    perturbation=perturbation,
                                    form=removal_form,
                                    matrix=removal_matrix,
                                )
                            )
                        matched_rows.extend(
                            _matched16_rows(
                                seed=seed,
                                model=model,
                                dataset=dataset,
                                perturbation=perturbation,
                                values=values,
                                labels=labels,
                                classes=classes,
                                draws=matched16_draws,
                            )
                        )
                        cell_count += 1
                        print(f"  {seed} {model} {dataset}/{perturbation}", flush=True)
                        if max_cells and cell_count >= max_cells:
                            break
                    if max_cells and cell_count >= max_cells:
                        break
            finally:
                feat.close()
            pd.DataFrame(spectra_rows).to_csv(paths["spectra"], index=False)
            pd.DataFrame(residual_rows).to_csv(paths["residual"], index=False)
            pd.DataFrame(decomposition_rows).to_csv(paths["decomposition"], index=False)
            pd.DataFrame(criteria_rows).to_csv(paths["criteria"], index=False)
            pd.DataFrame(removal_rows).to_csv(paths["removal"], index=False)
            pd.DataFrame(matched_rows).to_csv(paths["matched16"], index=False)

    keys = ["model", "dataset", "perturbation", "form", "support", "layer", "normalization"]
    _combine(
        parts,
        "spectra",
        keys,
        ["n_classes", "dim", "top1_share", "effective_rank", "n50", "n80", "ratio_er", "ratio_top1"],
        out,
    )
    _combine(
        parts,
        "residual",
        keys + ["space"],
        ["top1_share", "effective_rank", "n50", "n80", "ratio_er", "ratio_top1"],
        out,
    )
    _combine(
        parts,
        "decomposition",
        keys,
        [
            "m_norm",
            "mean_A_norm",
            "mean_a",
            "frac_a_neg",
            "cos_v1_e",
            "top3_pair_cos",
            "top3_e_cos",
            "o_mean_norm",
            "o_cos_v1_e",
            "ou_cos_v1_e",
            "o_top3_pair_cos",
            "o_top3_e_cos",
            "o_top1_share",
            "o_effective_rank",
            "ou_top1_share",
            "ou_effective_rank",
        ],
        out,
    )
    _combine(
        parts,
        "removal",
        keys + ["removed"],
        ["n_classes", "top1_share", "effective_rank", "n50", "n80"],
        out,
    )

    criteria = _read_parts(sorted(parts.glob("criteria_*.csv")))
    if not criteria.empty:
        criteria.to_csv(out / "criteria_per_seed.csv", index=False)
        _criteria_summary(criteria).to_csv(out / "criteria_summary.csv", index=False)

    matched = _read_parts(sorted(parts.glob("matched16_*.csv")))
    if not matched.empty:
        matched.to_csv(out / "matched16_per_seed.csv", index=False)
        per_draw = matched.groupby(
            ["seed", "model", "dataset", "perturbation", "form", "removed"], as_index=False
        )[["top1_share", "effective_rank", "n50", "n80", "m_norm"]].mean()
        per_draw["support"] = np.where(
            per_draw["dataset"] == "dota", "fit15_matched16_full", "fit15_matched16_draw40"
        )
        per_draw["layer"] = "final"
        per_draw["normalization"] = "unit"
        summary = _aggregate(
            per_draw,
            ["model", "dataset", "perturbation", "form", "support", "layer", "normalization", "removed"],
            ["top1_share", "effective_rank", "n50", "n80", "m_norm"],
        )
        summary.to_csv(out / "matched16_3seed.csv", index=False)


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
    parser.add_argument("--null-reps", type=int, default=200)
    parser.add_argument("--matched16-draws", type=int, default=40)
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
        null_reps=args.null_reps,
        matched16_draws=args.matched16_draws,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
