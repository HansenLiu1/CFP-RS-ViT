"""Build compact plotting tables for Fig. 3 and Fig. 4.

Source of truth: the paper export produced by
``scripts/51_export_paper_data.py``.  Pass its root with ``--export`` or set
``CFP_PAPER_EXPORT``.  The export is external and large; the two plotting
scripts only read the small CSV files written here.

Fig. 3:
  own-class argmax rate, M0 (raw) and M1 (CMC), fit15, unit normalization,
  true OBB, confirmation role, 3 seeds, 5 encoders x 3 datasets x 4
  perturbations. Cell values are means over datasets and seeds.

Fig. 4a:
  same metric and fit, control = true / shifted / random geometry matched /
  label permutation. M1 only. Values are means over models and datasets
  within each seed, then mean and sample SD over the three seeds.

Fig. 4b:
  M1 cross-perturbation transfer matrix from the final data ledger (S5a),
  macro OVR AUC, source-operator space, 4 x 4.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

MODELS = ["remoteclip", "clip", "siglip2", "dinov2", "mae"]
PERTURBATIONS = ["blur", "brightness", "gaussian_noise", "patch_mask"]
SEEDS = {1729, 2718, 3141}

M0 = "raw"
M1 = "euclidean_mean_subtraction"

# Table 3, A set (fit15, 3 seeds, true OBB, unit): used as a consistency guard.
EXPECTED_M1 = {
    "blur": 0.703,
    "brightness": 0.743,
    "gaussian_noise": 0.660,
    "patch_mask": 0.704,
}
EXPECTED_M0 = {
    "blur": 0.586,
    "brightness": 0.589,
    "gaussian_noise": 0.552,
    "patch_mask": 0.589,
}

# S5a in T00_Tables_Data.md. Rows = source, columns = target.
TRANSFER = {
    "blur": {
        "blur": 0.874,
        "brightness": 0.613,
        "gaussian_noise": 0.625,
        "patch_mask": 0.663,
    },
    "brightness": {
        "blur": 0.611,
        "brightness": 0.865,
        "gaussian_noise": 0.608,
        "patch_mask": 0.582,
    },
    "gaussian_noise": {
        "blur": 0.639,
        "brightness": 0.605,
        "gaussian_noise": 0.830,
        "patch_mask": 0.623,
    },
    "patch_mask": {
        "blur": 0.665,
        "brightness": 0.586,
        "gaussian_noise": 0.623,
        "patch_mask": 0.796,
    },
}


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_specificity(path: Path) -> csv.DictReader:
    handle = path.open("r", encoding="utf-8", newline="")
    return handle, csv.DictReader(handle)


def base_row(row: dict[str, str]) -> bool:
    if row.get("metric") != "own_argmax_rate":
        return False
    if row.get("normalization") != "unit":
        return False
    if row.get("operator") not in {M0, M1}:
        return False
    if row.get("dataset") not in {"dota", "dior", "fair1m"}:
        return False
    if row.get("perturbation") not in PERTURBATIONS:
        return False
    if row.get("model") not in MODELS:
        return False
    if int(as_float(row.get("split_seed"))) not in SEEDS:
        return False
    return True


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} rows={len(rows)}")


def build_fig3(handle, reader: csv.DictReader) -> None:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in reader:
        if not base_row(row):
            continue
        if row.get("control") != "true_obb":
            continue
        if row.get("stage") != "specificity_recomputed":
            continue
        if abs(as_float(row.get("support_size")) - 15.0) > 1e-9:
            continue
        values[(row["model"], row["perturbation"], row["operator"])].append(
            as_float(row["estimate"])
        )

    rows: list[dict] = []
    for model in MODELS:
        for perturbation in PERTURBATIONS:
            m0 = fmean(values[(model, perturbation, M0)])
            m1 = fmean(values[(model, perturbation, M1)])
            rows.append(
                {
                    "model": model,
                    "perturbation": perturbation,
                    "m0": f"{m0:.6f}",
                    "m1": f"{m1:.6f}",
                    "delta": f"{m1 - m0:.6f}",
                }
            )

    # Guard against a silent export/filter change.
    for perturbation in PERTURBATIONS:
        m0 = fmean(float(r["m0"]) for r in rows if r["perturbation"] == perturbation)
        m1 = fmean(float(r["m1"]) for r in rows if r["perturbation"] == perturbation)
        assert abs(m0 - EXPECTED_M0[perturbation]) < 0.002, (perturbation, m0)
        assert abs(m1 - EXPECTED_M1[perturbation]) < 0.002, (perturbation, m1)

    write_csv(DATA / "fig3_specificity.csv", ["model", "perturbation", "m0", "m1", "delta"], rows)


def build_fig4a(handle, reader: csv.DictReader) -> None:
    controls = {
        "true_obb",
        "shifted_obb",
        "random_geometry_matched_obb",
        "label_permutation",
    }
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in reader:
        if not base_row(row):
            continue
        if row.get("operator") != M1:
            continue
        if row.get("stage") != "E04-4":
            continue
        if row.get("control") not in controls:
            continue
        if abs(as_float(row.get("support_size")) - 15.0) > 1e-9:
            continue
        seed = int(as_float(row["split_seed"]))
        values[(row["control"], row["perturbation"], seed)].append(as_float(row["estimate"]))

    # Average the 5 encoders x 3 datasets within each seed first, then report
    # the mean and sample SD of the three per-seed values.
    rows = []
    for control in [
        "true_obb",
        "shifted_obb",
        "random_geometry_matched_obb",
        "label_permutation",
    ]:
        for perturbation in PERTURBATIONS:
            seed_means = [
                fmean(values[(control, perturbation, seed)])
                for seed in sorted(SEEDS)
                if values[(control, perturbation, seed)]
            ]
            rows.append(
                {
                    "control": control,
                    "perturbation": perturbation,
                    "estimate": f"{fmean(seed_means):.6f}",
                    "sd": f"{stdev(seed_means):.6f}" if len(seed_means) > 1 else "0.000000",
                    "n_seeds": len(seed_means),
                }
            )
    write_csv(
        DATA / "fig4_geometry.csv",
        ["control", "perturbation", "estimate", "sd", "n_seeds"],
        rows,
    )


def build_fig4b() -> None:
    rows = []
    for source in PERTURBATIONS:
        for target in PERTURBATIONS:
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "auc": f"{TRANSFER[source][target]:.6f}",
                    "kind": "diagonal" if source == target else "cross",
                }
            )
    write_csv(
        DATA / "fig4_transfer.csv",
        ["source", "target", "auc", "kind"],
        rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        type=Path,
        default=os.environ.get("CFP_PAPER_EXPORT"),
        help="paper export root containing 05_specificity_full.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.export is None:
        raise SystemExit("--export or CFP_PAPER_EXPORT is required")
    spec_csv = Path(args.export).expanduser().resolve() / "05_specificity_full.csv"
    if not spec_csv.exists():
        raise FileNotFoundError(f"{spec_csv} not found")

    handle, reader = read_specificity(spec_csv)
    try:
        build_fig3(handle, reader)
    finally:
        handle.close()

    handle, reader = read_specificity(spec_csv)
    try:
        build_fig4a(handle, reader)
    finally:
        handle.close()

    build_fig4b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
