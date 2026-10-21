"""Export only the E04-5 perturbation-strength sweep, without any recompute.

Reads the persisted ``intermediates/stage_results/E04-5_<model>.json`` shards of
the configured runs, reconstructs the nominal-vs-variant paired contrasts with
the same estimator used by the full paper export, and writes a self-contained
pair of CSVs.  It never reruns a stage (no E04/E06 recompute, no missing-seed
grid) and never writes inside a run directory.

Example
-------
python scripts/53_export_strength_sweep.py \
    --run 1729=/data/runs/cfp-1729 \
    --run 2718=/data/runs/cfp-2718 \
    --run 3141=/data/runs/cfp-3141 \
    --out /data/exports/cfp-strength
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cfra.paper_export import FIELDS, PaperExport, read_json, write_csv
from cfra.paper_export_contrasts import apply_contrasts
from cfra.paper_export_grid import REGISTERED_SEEDS

logger = logging.getLogger("cfra")




def _load_models(root: Path) -> dict:
    path = root / "manifest" / "models.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["model_id"]: row for row in csv.DictReader(handle)}


def _benjamini_hochberg(rows: list[dict]) -> None:
    """BH within explicit hypothesis families; missing p values are excluded."""

    families: dict[object, list[dict]] = {}
    for row in rows:
        p_value = row.get("p_value")
        if not (isinstance(p_value, (int, float)) and math.isfinite(p_value)):
            continue
        key = row.get("family") or (row.get("stage"), row.get("metric"))
        families.setdefault(key, []).append(row)
    for family_rows in families.values():
        family_rows.sort(key=lambda item: item["p_value"])
        count = len(family_rows)
        q_value = 1.0
        for index in range(count - 1, -1, -1):
            q_value = min(q_value, family_rows[index]["p_value"] * count / (index + 1))
            family_rows[index]["FDR_q"] = q_value


def _cell_sort_key(row: dict) -> tuple:
    return (
        str(row.get("dataset") or ""),
        str(row.get("model") or ""),
        str(row.get("perturbation") or ""),
        str(row.get("operator") or ""),
        json.dumps(row.get("perturbation_strength"), sort_keys=True, default=str),
        str(row.get("metric") or ""),
    )


def _paired_sort_key(row: dict) -> tuple:
    return (
        str(row.get("dataset") or ""),
        str(row.get("model") or ""),
        str(row.get("perturbation") or ""),
        str(row.get("operator") or ""),
        str(row.get("state_b") or ""),
        str(row.get("metric") or ""),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SEED=DIRECTORY")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default="remoteclip,clip,siglip2,dinov2,mae")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--permutation-replicates", type=int, default=10000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = [item for item in args.models.split(",") if item]
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    output = args.out.expanduser().resolve()
    export = PaperExport(
        [],
        output,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_replicates=args.permutation_replicates,
        recompute_layers=False,
        recompute_final=False,
    )
    run_count = 0
    try:
        for item in args.run:
            _seed, run_dir = item.split("=", 1)
            root = Path(run_dir).expanduser().resolve(strict=True)
            export.config = read_json(root / "manifest" / "run.json")["resolved_config"]
            export.active_run = root.name
            export.models = _load_models(root)
            seed = export.config["split"]["seed"]
            for model_id in models:
                stage_path = root / "intermediates/stage_results" / f"E04-5_{model_id}.json"
                if not stage_path.is_file():
                    export.log(f"  missing E04-5 artifact: {stage_path}")
                    continue
                rows = read_json(stage_path).get("models", {}).get(model_id)
                if not rows:
                    export.log(f"  no E04-5 rows for {model_id} in {stage_path.name}")
                    continue
                for item in rows:
                    export.ingest(root, "E04-5", item, stage_path, "strength_sweep")
                apply_contrasts(export, "E04-5", stage_path, rows)
                export.log(f"  {root.name}/{model_id}: seed={seed} rows={len(rows)}")
            run_count += 1
    finally:
        export.units.close()

    table = export.tables["strength_sweep"]
    _benjamini_hochberg(table)
    cells = [row for row in table if not str(row.get("metric", "")).startswith("delta_")]
    paired = [row for row in table if str(row.get("metric", "")).startswith("delta_")]
    cells.sort(key=_cell_sort_key)
    paired.sort(key=_paired_sort_key)
    write_csv(output / "e04_5_cells.csv", cells, FIELDS)
    write_csv(output / "e04_5_paired.csv", paired, FIELDS)
    coverage = {
        "runs": run_count,
        "models": models,
        "registered_seeds": list(REGISTERED_SEEDS),
        "cells_rows": len(cells),
        "paired_rows": len(paired),
        "observed_cells": sum(row.get("status") != "missing" for row in cells),
        "observed_paired": sum(row.get("status") != "missing" for row in paired),
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
    }
    (output / "e04_5_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    export.log(f"wrote {output/'e04_5_cells.csv'} ({len(cells)} rows)")
    export.log(f"wrote {output/'e04_5_paired.csv'} ({len(paired)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
