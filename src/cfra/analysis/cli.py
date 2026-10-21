"""Command-line entry point for the public paper analyses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURE_SCRIPTS = (
    "fig3_specificity.py",
    "fig4_geometry_transfer.py",
    "fig5_layerwise.py",
    "fig6_common_alignment.py",
    "fig7_readout.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts(artifacts: Path) -> int:
    """Check the frozen artifact manifest and hashes."""

    manifest = artifacts / "manifest.csv"
    if not manifest.is_file():
        print(f"missing manifest: {manifest}", file=sys.stderr)
        return 2
    failures: list[str] = []
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        relative = row.get("path", "").strip()
        expected = row.get("sha256", "").strip().lower()
        if not relative:
            continue
        path = artifacts / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        actual = _sha256(path)
        if expected and actual != expected:
            failures.append(f"hash mismatch {relative}: {actual} != {expected}")
    if failures:
        print("artifact validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"validated {len(rows)} artifact(s) under {artifacts}")
    return 0


def make_figures() -> int:
    """Regenerate Figures 3--7 from the committed plotting tables."""

    figures = REPO_ROOT / "figures"
    for script in FIGURE_SCRIPTS:
        path = figures / script
        if not path.is_file():
            print(f"missing figure script: {path}", file=sys.stderr)
            return 2
        print(f"running {path.relative_to(REPO_ROOT)}")
        result = subprocess.run([sys.executable, str(path)], cwd=REPO_ROOT)
        if result.returncode != 0:
            return result.returncode
    return 0


def _parse_runs(value: str) -> dict[int, str]:
    runs: dict[int, str] = {}
    for item in value.split(","):
        seed, name = item.split("=", 1)
        runs[int(seed)] = name
    return runs


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--runs", required=True, help="comma-separated seed=run-directory pairs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default="remoteclip,clip,siglip2,dinov2,mae")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--datasets", default="dota,dior,fair1m")
    parser.add_argument("--perturbations", default="blur,brightness,gaussian_noise,patch_mask")
    parser.add_argument("--max-cells", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfra-analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="check frozen paper artifacts")
    validate.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "paper")

    sub.add_parser("figures", help="regenerate Figures 3--7 from committed CSVs")

    alignment = sub.add_parser("subspace-alignment", help="S4c subspace alignment and PC control")
    _add_run_arguments(alignment)

    spectra = sub.add_parser("spectra", help="final-layer spectra, common-mode decomposition and removal curves")
    _add_run_arguments(spectra)
    spectra.add_argument("--null-reps", type=int, default=200)
    spectra.add_argument("--matched16-draws", type=int, default=40)

    geometry = sub.add_parser("geometry", help="cross-image class-direction cosines and permutation null")
    _add_run_arguments(geometry)
    geometry.add_argument("--permutations", type=int, default=200)

    removal = sub.add_parser("readout-removal", help="E07 readout removal, reflection and Table IV")
    _add_run_arguments(removal)
    removal.add_argument("--budgets", default="fit15,fitall")
    removal.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    removal.add_argument("--reflection-alpha", type=float, default=2.0)
    removal.add_argument("--bootstrap-replicates", type=int, default=0)

    transfer = sub.add_parser("readout-transfer", help="prototype/CAV/CFP transfer and fusion baselines")
    _add_run_arguments(transfer)

    normalization = sub.add_parser("normalization", help="M0-M6 normalization mechanism and r/|m| bins")
    _add_run_arguments(normalization)
    normalization.add_argument("--normalizations", default="unit,raw,magnitude_weighted")
    normalization.add_argument("--operators", default="M0,M1,M2,M3,M4,M5,M6")

    stats = sub.add_parser("export-stats", help="rank, near-zero, interaction and permutation tables")
    _add_run_arguments(stats)
    stats.add_argument("--analyses", default="rank,nearzero,interaction,permutation")

    layerwise = sub.add_parser("layerwise", help="layerwise depth decomposition and single-direction removal")
    _add_run_arguments(layerwise)
    layerwise.add_argument("--layers", default="patch_emb,L01,L02,L03,L04,L05,L06,L07,L08,L09,L10,L11,L12,final")
    layerwise.add_argument("--budgets", default="fit15,fitall")
    layerwise.add_argument("--mode", choices=("depth", "removal", "both"), default="both")

    intervals = sub.add_parser("intervals", help="S13 source-image cluster intervals")
    _add_run_arguments(intervals)
    intervals.add_argument("--tables", default="rank,operators,readout,regions")
    intervals.add_argument("--replicates", type=int, default=5000)
    intervals.add_argument("--permutations", type=int, default=10000)

    fdbc = sub.add_parser("layerwise-fdbc", help="foreground x background interaction from FdBc arrays")
    _add_run_arguments(fdbc)
    fdbc.add_argument("--layers", default="patch_emb,L01,L02,L03,L04,L05,L06,L07,L08,L09,L10,L11,L12,final")
    fdbc.add_argument("--fdbc", default=None, help="comma-separated seed=directory FdBc overrides")

    text_baseline = sub.add_parser("text-baseline", help="S12 shared-space text prototype baseline")
    _add_run_arguments(text_baseline)
    text_baseline.add_argument("--project-root", type=Path, required=True)
    text_baseline.add_argument("--templates", action="store_true")
    text_baseline.add_argument("--clip-projection", type=Path, default=None)
    text_baseline.add_argument("--k", type=int, default=10)

    text_anchoring = sub.add_parser("text-anchoring", help="S12b text anchoring and permutation null")
    _add_run_arguments(text_anchoring)
    text_anchoring.add_argument("--project-root", type=Path, required=True)
    text_anchoring.add_argument("--permutations", type=int, default=10000)
    text_anchoring.add_argument("--clip-projection", type=Path, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        return validate_artifacts(args.artifacts)
    if args.command == "figures":
        return make_figures()
    if args.command == "subspace-alignment":
        from cfra.analysis.common import RunLayout
        from cfra.analysis.tasks import subspace_alignment

        layout = RunLayout(args.run_root, _parse_runs(args.runs))
        subspace_alignment.run(
            layout,
            args.out,
            models=tuple(item for item in args.models.split(",") if item),
            seeds=tuple(int(item) for item in args.seeds.split(",") if item) or None,
            datasets=tuple(item for item in args.datasets.split(",") if item),
            perturbations=tuple(item for item in args.perturbations.split(",") if item),
            max_cells=args.max_cells,
        )
        return 0
    if args.command in {"spectra", "geometry"}:
        from cfra.analysis.common import RunLayout
        from cfra.analysis.tasks import geometry, spectra

        layout = RunLayout(args.run_root, _parse_runs(args.runs))
        options = dict(
            models=tuple(item for item in args.models.split(",") if item),
            seeds=tuple(int(item) for item in args.seeds.split(",") if item) or None,
            datasets=tuple(item for item in args.datasets.split(",") if item),
            perturbations=tuple(item for item in args.perturbations.split(",") if item),
            max_cells=args.max_cells,
        )
        if args.command == "spectra":
            spectra.run(
                layout,
                args.out,
                null_reps=args.null_reps,
                matched16_draws=args.matched16_draws,
                **options,
            )
        else:
            geometry.run(layout, args.out, permutations=args.permutations, **options)
        return 0
    if args.command == "readout-removal":
        from cfra.analysis.common import RunLayout
        from cfra.analysis.tasks import readout_removal

        layout = RunLayout(args.run_root, _parse_runs(args.runs))
        readout_removal.run(
            layout,
            args.out,
            models=tuple(item for item in args.models.split(",") if item),
            seeds=tuple(int(item) for item in args.seeds.split(",") if item) or None,
            datasets=tuple(item for item in args.datasets.split(",") if item),
            perturbations=tuple(item for item in args.perturbations.split(",") if item),
            max_cells=args.max_cells,
            budgets=tuple(item for item in args.budgets.split(",") if item),
            alphas=tuple(float(item) for item in args.alphas.split(",") if item),
            reflection_alpha=args.reflection_alpha,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        return 0
    if args.command == "readout-transfer":
        from cfra.analysis.common import RunLayout
        from cfra.analysis.tasks import readout_transfer

        layout = RunLayout(args.run_root, _parse_runs(args.runs))
        readout_transfer.run(
            layout,
            args.out,
            models=tuple(item for item in args.models.split(",") if item),
            seeds=tuple(int(item) for item in args.seeds.split(",") if item) or None,
            datasets=tuple(item for item in args.datasets.split(",") if item),
            perturbations=tuple(item for item in args.perturbations.split(",") if item),
            max_cells=args.max_cells,
        )
        return 0
    if args.command == "normalization":
        from cfra.analysis.common import RunLayout
        from cfra.analysis.tasks import normalization

        layout = RunLayout(args.run_root, _parse_runs(args.runs))
        normalization.run(
            layout,
            args.out,
            models=tuple(item for item in args.models.split(",") if item),
            seeds=tuple(int(item) for item in args.seeds.split(",") if item) or None,
            datasets=tuple(item for item in args.datasets.split(",") if item),
            perturbations=tuple(item for item in args.perturbations.split(",") if item),
            max_cells=args.max_cells,
            normalizations=tuple(item for item in args.normalizations.split(",") if item),
            operators=tuple(item for item in args.operators.split(",") if item),
        )
        return 0
    if args.command == "export-stats":
        from cfra.analysis.common import RunLayout
        from cfra.analysis.tasks import export_stats

        layout = RunLayout(args.run_root, _parse_runs(args.runs))
        export_stats.run(
            layout,
            args.out,
            models=tuple(item for item in args.models.split(",") if item),
            seeds=tuple(int(item) for item in args.seeds.split(",") if item) or None,
            datasets=tuple(item for item in args.datasets.split(",") if item),
            perturbations=tuple(item for item in args.perturbations.split(",") if item),
            max_cells=args.max_cells,
            analyses=tuple(item for item in args.analyses.split(",") if item),
        )
        return 0
    if args.command == "layerwise":
        from cfra.analysis.common import RunLayout
        from cfra.analysis.tasks import layerwise

        layout = RunLayout(args.run_root, _parse_runs(args.runs))
        layerwise.run(
            layout,
            args.out,
            models=tuple(item for item in args.models.split(",") if item),
            seeds=tuple(int(item) for item in args.seeds.split(",") if item) or None,
            datasets=tuple(item for item in args.datasets.split(",") if item),
            perturbations=tuple(item for item in args.perturbations.split(",") if item),
            max_cells=args.max_cells,
            layers=tuple(item for item in args.layers.split(",") if item),
            budgets=tuple(item for item in args.budgets.split(",") if item),
            mode=args.mode,
        )
        return 0
    if args.command == "intervals":
        from cfra.analysis.common import RunLayout
        from cfra.analysis.tasks import intervals

        layout = RunLayout(args.run_root, _parse_runs(args.runs))
        intervals.run(
            layout,
            args.out,
            models=tuple(item for item in args.models.split(",") if item),
            seeds=tuple(int(item) for item in args.seeds.split(",") if item) or None,
            datasets=tuple(item for item in args.datasets.split(",") if item),
            perturbations=tuple(item for item in args.perturbations.split(",") if item),
            max_cells=args.max_cells,
            tables=tuple(item for item in args.tables.split(",") if item),
            replicates=args.replicates,
            permutations=args.permutations,
        )
        return 0
    if args.command == "layerwise-fdbc":
        from cfra.analysis.common import RunLayout
        from cfra.analysis.tasks import layerwise_fdbc

        layout = RunLayout(args.run_root, _parse_runs(args.runs))
        layerwise_fdbc.run(
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
    if args.command in {"text-baseline", "text-anchoring"}:
        from cfra.analysis.common import RunLayout
        from cfra.analysis.tasks import text_anchoring, text_baseline

        layout = RunLayout(args.run_root, _parse_runs(args.runs))
        common = dict(
            models=tuple(item for item in args.models.split(",") if item),
            seeds=tuple(int(item) for item in args.seeds.split(",") if item) or layout.seeds,
            datasets=tuple(item for item in args.datasets.split(",") if item),
            perturbations=tuple(item for item in args.perturbations.split(",") if item),
            project_root=args.project_root,
            clip_projection=args.clip_projection,
        )
        if args.command == "text-baseline":
            text_baseline.run(
                layout, args.out, templates=args.templates, k=args.k, max_cells=args.max_cells, **common
            )
        else:
            text_anchoring.run(
                layout, args.out, permutations=args.permutations, max_cells=args.max_cells, **common
            )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
