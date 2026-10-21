"""OFAT perturbation-strength sweep (E04-5) driver for completed offline runs.

This script never rebuilds a run and never mutates the baseline artifacts it
depends on.  For every configured run it:

* loads the run's frozen ``resolved_config``, manifests and E01 selection;
* loads the run's persisted nominal ``features_<model>.npz`` (read only) to keep
  the nominal reference cell;
* extracts only the foreground-pair features of each sweep variant (one
  perturbation, headline operators M0/M1/M3, unit normalization, no spatial
  controls), caching each variant below
  ``intermediates/features/strength_sweep/<model>/``;
* writes ``intermediates/stage_results/E04-5_<model>.json`` in the same sharded
  layout ``scripts/51_export_paper_data.py`` already reads.

Use ``--dry-run`` to print the resolved grid and extraction plan without
loading a model.  Variant feature files are reused across runs unless
``--force`` is set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cfra.models import load_adapter
from cfra.offline import _rebuild_context
from cfra.pipeline.core import _json_safe
from cfra.pipeline.stages import load_current_manifests, load_e01_selected
from cfra.pipeline.strength_sweep import (
    SWEEP_METHODS,
    SWEEP_NORMALIZATIONS,
    StrengthSweepError,
    extract_pair_table,
    load_nominal_pair_subset,
    load_pair_table,
    nominal_cell,
    resolve_sweep_cells,
    run_sweep_cell,
    save_pair_table,
)
from cfra.resume import write_stage_result
from cfra.runs import write_manifest_json

logger = logging.getLogger("cfra")

@dataclass(frozen=True)
class SweepOptions:
    models: tuple[str, ...]
    levels: tuple[str, ...]
    dry_run: bool = False
    force: bool = False
    bootstrap_replicates: int | None = None
    cpu_workers: int | None = None


def _load_run_config(root: Path) -> dict:
    run_json = root / "manifest" / "run.json"
    if not run_json.is_file():
        raise StrengthSweepError(f"missing run manifest: {run_json}")
    return json.loads(run_json.read_text(encoding="utf-8"))["resolved_config"]


def _statistics_config(config: dict, options: SweepOptions) -> dict:
    stats = dict(config.get("statistics", {}))
    if options.bootstrap_replicates is not None:
        stats["bootstrap_replicates"] = int(options.bootstrap_replicates)
    return stats


def _pair_path(root: Path, model_id: str, variant_id: str, perturbation: str) -> Path:
    return root / "intermediates/features/strength_sweep" / model_id / f"{variant_id}__{perturbation}.npz"


def run_model_sweep(
    *,
    root: Path,
    config: dict,
    manifests: dict,
    selected: dict,
    context: dict,
    model_id: str,
    cells: list,
    options: SweepOptions,
) -> list[dict]:
    datasets = list(config["scope"]["datasets"])
    epsilon = float(config["protocol"]["near_zero_norm_epsilon"])
    fit_size = int(config["protocol"]["primary_fit_size"])
    statistics_config = _statistics_config(config, options)
    seed = int(config["split"]["seed"])
    asset = context["model_assets"][model_id]
    runtime = context["runtime"]
    rows: list[dict] = []

    perturbations = sorted({cell.perturbation for cell in cells})
    for perturbation in perturbations:
        nominal_table = load_nominal_pair_subset(root, model_id, perturbation)
        rows.extend(
            run_sweep_cell(
                nominal_table,
                nominal_cell(config, perturbation),
                dataset_ids=datasets,
                epsilon=epsilon,
                fit_size=fit_size,
                statistics_config=statistics_config,
                random_seed=seed,
            )
        )
        del nominal_table

    pending = [
        cell
        for cell in cells
        if options.force or not _pair_path(root, model_id, cell.variant_id, cell.perturbation).is_file()
    ]
    if pending:
        logger.info(f"  [{model_id}] 提取 {len(pending)}/{len(cells)} 个变体特征")
        with load_adapter(
            model_id,
            asset,
            runtime=runtime,
            registry_root=PROJECT_ROOT / "registry",
            device="auto",
        ) as adapter:
            for cell in pending:
                started = time.time()
                table = extract_pair_table(
                    project_root=PROJECT_ROOT,
                    run_root=root,
                    config=config,
                    manifests=manifests,
                    selected_by_dataset=selected,
                    model_id=model_id,
                    asset=asset,
                    runtime=runtime,
                    perturbation=cell.perturbation,
                    strength_variant=cell.strength_variant,
                    variant_id=cell.variant_id,
                    perturbation_overrides=cell.overrides,
                    cpu_workers=options.cpu_workers,
                    adapter=adapter,
                )
                save_pair_table(root, table)
                logger.info(f"    [{model_id}] {cell.variant_id} 提取完成 (耗时 {time.time() - started:.1f}s)")
                del table
    else:
        logger.info(f"  [{model_id}] 变体特征已全部存在，跳过提取")

    for cell in cells:
        table = load_pair_table(root, model_id, cell.variant_id, cell.perturbation)
        rows.extend(
            run_sweep_cell(
                table,
                cell,
                dataset_ids=datasets,
                epsilon=epsilon,
                fit_size=fit_size,
                statistics_config=statistics_config,
                random_seed=seed,
            )
        )
        del table
    return rows


def dry_run_report(run_dirs: list[Path], options: SweepOptions) -> None:
    for root in run_dirs:
        config = _load_run_config(root)
        cells = resolve_sweep_cells(config, list(options.levels))
        perturbations = sorted({cell.perturbation for cell in cells})
        logger.info(
            f"[DRY] {root.name}: seed={config['split']['seed']} "
            f"datasets={config['scope']['datasets']} cells={len(cells)} "
            f"perturbations={perturbations}"
        )
        for cell in cells:
            logger.info(
                f"    {cell.sweep_level} {cell.variant_id} pert={cell.perturbation} "
                f"strength={cell.strength_value} overrides={cell.overrides}"
            )
        logger.info(
            f"  -> {len(options.models)} models x {len(cells)} variant extractions "
            f"(+ {len(perturbations)} nominal cells/model from persisted table)"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SEED=DIRECTORY")
    parser.add_argument("--models", default="remoteclip,clip,siglip2,dinov2,mae")
    parser.add_argument("--levels", default="P1,P2")
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--cpu-workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = SweepOptions(
        models=tuple(item for item in args.models.split(",") if item),
        levels=tuple(item for item in args.levels.split(",") if item),
        dry_run=args.dry_run,
        force=args.force,
        bootstrap_replicates=args.bootstrap_replicates,
        cpu_workers=args.cpu_workers,
    )
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_dirs = []
    for item in args.run:
        _seed, directory = item.split("=", 1)
        run_dirs.append(Path(directory).expanduser().resolve())
    missing = [str(path) for path in run_dirs if not path.is_dir()]
    if missing:
        raise SystemExit(f"run directories missing: {missing}")

    logger.info("=" * 72)
    logger.info("CFP perturbation-strength sweep (E04-5, OFAT pair-only)")
    logger.info("runs=%s models=%s levels=%s", [path.name for path in run_dirs], options.models, options.levels)
    logger.info(
        "dry_run=%s force=%s bootstrap_replicates=%s",
        options.dry_run,
        options.force,
        options.bootstrap_replicates,
    )
    logger.info("=" * 72)

    if options.dry_run:
        dry_run_report(run_dirs, options)
        logger.info("dry run: no model was loaded and no artifact was written.")
        return 0

    for root in run_dirs:
        config = _load_run_config(root)
        cells = resolve_sweep_cells(config, list(options.levels))
        logger.info("Run %s: seed=%s cells=%s", root.name, config["split"]["seed"], len(cells))
        manifests = load_current_manifests(root, config)
        selected = load_e01_selected(root, config)
        context = _rebuild_context(PROJECT_ROOT, config)
        for model_id in options.models:
            out_path = root / "intermediates/stage_results" / f"E04-5_{model_id}.json"
            if out_path.is_file() and not options.force:
                logger.info("  [%s] existing E04-5 artifact; skipping", model_id)
                continue
            started = time.time()
            logger.info("  [%s] start", model_id)
            rows = run_model_sweep(
                root=root,
                config=config,
                manifests=manifests,
                selected=selected,
                context=context,
                model_id=model_id,
                cells=cells,
                options=options,
            )
            write_stage_result(root, f"E04-5_{model_id}", {"stage": "E04-5", "models": {model_id: _json_safe(rows)}})
            write_manifest_json(
                root,
                f"manifest/E04-5_{model_id}.json",
                {
                    "stage": "E04-5",
                    "model_id": model_id,
                    "protocol": "strength_ofat_pair_only",
                    "levels": list(options.levels),
                    "methods": list(SWEEP_METHODS),
                    "normalization": list(SWEEP_NORMALIZATIONS),
                    "controls": "disabled_except_true_obb",
                    "variants": [cell.to_json() for cell in cells],
                    "row_count": len(rows),
                    "status": "completed",
                },
            )
            logger.info("  [%s] done in %.1fs: %s rows", model_id, time.time() - started, len(rows))
        logger.info("Run %s completed", root.name)

    logger.info("Strength sweep completed.")
    return 0


if __name__ == "__main__":
    if os.environ.get("CFRA_BOOTSTRAP_WORKERS") is None:
        os.environ["CFRA_BOOTSTRAP_WORKERS"] = str(max(1, min((os.cpu_count() or 4) - 1, 18)))
    raise SystemExit(main())
