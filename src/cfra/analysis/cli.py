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
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
