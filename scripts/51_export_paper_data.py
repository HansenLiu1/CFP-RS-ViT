"""Export the three-seed paper aggregation from completed runs.

The exporter pools the supplied single-seed runs into one metric-level master
and aggregates each identifying cell into ``tables_3seed/seed_summary.csv``.
The declared seed is checked against ``<run>/manifest/run.json`` before any
work starts.  The export is resumable through ``OUTPUT_DIR/_checkpoint``.

Example
-------
python scripts/51_export_paper_data.py \
    --run 1729=/data/runs/cfp-1729 \
    --run 2718=/data/runs/cfp-2718 \
    --run 3141=/data/runs/cfp-3141 \
    --out /data/exports/cfp-paper
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cfra.paper_export import PaperExport

REGISTERED_SEEDS = (1729, 2718, 3141)


def _declared_seed(run_dir: Path) -> int:
    manifest = run_dir / "manifest" / "run.json"
    if not manifest.is_file():
        raise SystemExit(f"run manifest missing (not a completed run): {manifest}")
    config = json.loads(manifest.read_text(encoding="utf-8")).get("resolved_config", {})
    try:
        return int(config.get("split", {}).get("seed"))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"run manifest has no integer split seed: {manifest}") from exc


def validate_runs(runs: list[tuple[int, Path]], *, allow_partial: bool) -> list[str]:
    run_dirs: list[str] = []
    seen: set[int] = set()
    for seed, run_dir in runs:
        path = Path(run_dir).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"run directory does not exist: {path}")
        declared = _declared_seed(path)
        if declared != seed:
            raise SystemExit(
                f"declared seed {seed} does not match {path}/manifest/run.json split seed {declared}"
            )
        if seed in seen:
            raise SystemExit(f"duplicate split seed: {seed}")
        seen.add(seed)
        run_dirs.append(str(path))
    missing = sorted(set(REGISTERED_SEEDS) - seen)
    if missing and not allow_partial:
        raise SystemExit(
            f"missing registered seeds {missing}; the three-seed aggregate needs "
            f"{list(REGISTERED_SEEDS)} (use --allow-partial for an exploratory export)"
        )
    return run_dirs


def report_aggregation(output_dir: Path, expected_seeds: int) -> None:
    summary_path = output_dir / "tables_3seed" / "seed_summary.csv"
    if not summary_path.is_file():
        print(f"  seed summary not found at {summary_path}", flush=True)
        return
    total = full = 0
    per_table: dict[str, list[int]] = {}
    with summary_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            n_seeds = int(float(row["n_seeds"])) if row.get("n_seeds") not in (None, "") else 0
            if n_seeds >= expected_seeds:
                full += 1
            per_table.setdefault(row.get("table", "?"), []).append(n_seeds)
    print(f"  seed_summary rows={total} cells={full}", flush=True)
    for table in sorted(per_table):
        counts = per_table[table]
        print(f"    {table}: {len(counts)} cell(s), max n_seeds={max(counts)}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SEED=DIRECTORY",
        help="repeat for each split seed",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-stages", default="E04-5")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--permutation-replicates", type=int, default=10000)
    parser.add_argument("--skip-layer-recompute", action="store_true")
    parser.add_argument("--skip-final-recompute", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--clear-checkpoint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs: list[tuple[int, Path]] = []
    for item in args.run:
        seed, directory = item.split("=", 1)
        runs.append((int(seed), Path(directory)))

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if os.environ.get("CFRA_BOOTSTRAP_WORKERS") is None:
        os.environ["CFRA_BOOTSTRAP_WORKERS"] = str(max(1, min((os.cpu_count() or 4) - 1, 18)))

    run_dirs = validate_runs(runs, allow_partial=args.allow_partial)
    exclude_stages = tuple(item for item in args.exclude_stages.split(",") if item)
    print("Paper export configuration:", flush=True)
    for seed, run_dir in runs:
        print(f"  seed={seed} run_dir={run_dir}", flush=True)
    print(f"  output_dir={args.out}", flush=True)
    print(f"  exclude_stages={exclude_stages}", flush=True)
    print(
        f"  bootstrap_replicates={args.bootstrap_replicates} "
        f"permutation_replicates={args.permutation_replicates}",
        flush=True,
    )

    started = time.time()
    acceptance = PaperExport(
        run_dirs,
        str(args.out),
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_replicates=args.permutation_replicates,
        recompute_layers=not args.skip_layer_recompute,
        recompute_final=not args.skip_final_recompute,
        exclude_stages=exclude_stages,
        clear_checkpoint=args.clear_checkpoint,
    ).run()
    report_aggregation(args.out, len(runs))
    print(
        f"Paper export completed in {time.time() - started:.1f}s; "
        f"acceptance={acceptance['status']}",
        flush=True,
    )
    if args.require_complete and not acceptance["submission_ready"]:
        raise SystemExit(
            "CSV export finished, but requested data are incomplete. "
            "See acceptance.json (not a model-run failure)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
