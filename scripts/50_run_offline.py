"""Run the CFP pipeline in one or more single-seed run directories.

This is the paper-data entry point.  It executes the paired-input audit,
feature extraction and the requested E03--E08 stages.  Each seed is written to
its own run directory; the analysis layer averages the seeds afterwards.

Example
-------
python scripts/50_run_offline.py \
    --run 1729=/data/runs/cfp-1729 \
    --config configs/offline.toml

The script does not create the run directory.  Create it first, then run the
command again to resume from the last completed stage.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cfra.configuration import config_sha256, validate_registry_contract
from cfra.logging_utils import setup_logger
from cfra.offline import (
    OfflineRunnerError,
    build_offline_config,
    open_offline_run,
    parameters_signature,
    run_full_offline,
)

DEFAULT_EXPERIMENTS = [
    "E00",
    "E01",
    "E02",
    "E03",
    "E04",
    "E04-1",
    "E04-2",
    "E04-3",
    "E04-4",
    "E05",
    "E06",
    "E07",
    "E08",
]


def parse_runs(values: list[str]) -> list[tuple[int, Path]]:
    runs: list[tuple[int, Path]] = []
    for item in values:
        try:
            seed_text, directory = item.split("=", 1)
            runs.append((int(seed_text), Path(directory).expanduser().resolve()))
        except ValueError as exc:
            raise SystemExit(f"invalid --run {item!r}; expected SEED=DIRECTORY") from exc
    return runs


def run_one(
    run_dir: Path,
    seed: int,
    *,
    config_path: Path,
    datasets: list[str],
    models: list[str],
    perturbations: list[str],
    experiments: list[str],
    run_e04_5: bool,
) -> None:
    if not run_dir.exists():
        raise SystemExit(f"run directory does not exist: {run_dir}")

    config = build_offline_config(
        config_path,
        datasets=datasets,
        seed=seed,
        models=models,
        perturbations=perturbations,
        experiment_ids=experiments,
        run_e04_5=run_e04_5,
    )
    validate_registry_contract(config, PROJECT_ROOT / "registry")

    signature = parameters_signature(
        datasets=datasets,
        seed=seed,
        models=models,
        perturbations=perturbations,
        experiment_ids=experiments,
        run_e04_5=run_e04_5,
        config_hash=config_sha256(config),
    )

    logger = setup_logger(run_root=run_dir, name="cfra")
    logger.info("=" * 72)
    logger.info("CFRA offline runner | run_dir=%s", run_dir)
    logger.info(
        "datasets=%s seed=%s models=%s perturbations=%s",
        datasets,
        seed,
        models,
        perturbations,
    )
    logger.info(
        "experiments=%s run_e04_5=%s config=%s",
        experiments,
        run_e04_5,
        config_path.name,
    )
    logger.info("parameter_signature=%s...", signature[:16])
    logger.info("=" * 72)

    run_root, mode, completed = open_offline_run(run_dir, config, signature)
    logger.info("[offline] mode=%s completed_stages=%s", mode, len(completed))
    if completed:
        logger.info("[offline] resuming after: %s", ", ".join(completed[:12]))
    run_full_offline(
        run_root=run_root,
        project_root=PROJECT_ROOT,
        config=config,
        config_path=config_path,
        completed=completed,
        run_e04_5=run_e04_5,
        export_only_handoff=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SEED=DIRECTORY",
        help="repeat for each split seed",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "offline.toml")
    parser.add_argument("--datasets", default="dota,dior,fair1m")
    parser.add_argument("--models", default="remoteclip,clip,siglip2,dinov2,mae")
    parser.add_argument("--perturbations", default="blur,patch_mask,gaussian_noise,brightness")
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument("--run-e04-5", action="store_true")
    parser.add_argument("--reuse-e00", action="store_true")
    parser.add_argument("--allow-feature-reset", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reuse_e00:
        os.environ["CFRA_REUSE_E00"] = "1"
    if args.allow_feature_reset:
        os.environ["CFRA_ALLOW_FEATURE_RESET"] = "1"
    if os.environ.get("CFRA_BOOTSTRAP_WORKERS") is None:
        os.environ["CFRA_BOOTSTRAP_WORKERS"] = str(max(1, min((os.cpu_count() or 4) - 1, 18)))
    if os.environ.get("CFRA_CELL_WORKERS") is None:
        os.environ["CFRA_CELL_WORKERS"] = "1"

    for seed, run_dir in parse_runs(args.run):
        try:
            run_one(
                run_dir,
                seed,
                config_path=args.config,
                datasets=[item for item in args.datasets.split(",") if item],
                models=[item for item in args.models.split(",") if item],
                perturbations=[item for item in args.perturbations.split(",") if item],
                experiments=[item for item in args.experiments.split(",") if item],
                run_e04_5=args.run_e04_5,
            )
        except OfflineRunnerError as exc:
            raise SystemExit(f"offline run failed for {run_dir}: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
