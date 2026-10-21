# Reproducibility guide

The repository exposes three levels of reproduction.  They are deliberately
separate so that a reader does not need a 50--100 GB feature cache to check a
reported number.

## Tier 1: paper audit (CPU, no downloads)

Inputs:

- `artifacts/paper/` -- frozen summary CSVs and `manifest.csv`
- `figures/data/` -- compact plotting tables (a few dozen KB)

Commands:

```bash
python -m cfra.analysis validate --artifacts artifacts/paper
python -m cfra.analysis figures
```

The validation command checks every artifact in `manifest.csv` and its
SHA-256; `docs/paper_map.md` gives the manuscript mapping.  The figure command
regenerates Figures 3--7 as PDF and PNG.  Neither command reads raw imagery or
model checkpoints.

## Tier 2: analysis rerun (CPU, feature artifacts)

Inputs: a completed pipeline run with:

```text
<run>/intermediates/features/features_<model>.npz
<run>/intermediates/features/layerwise_<model>_<dataset>.npz
```

Commands (example):

```bash
python -m cfra.analysis subspace-alignment \
    --run-root /runs \
    --runs 1729=20260831_run1,2718=20260913_run1,3141=20260920_run1 \
    --out out/subspace
```

The exact task list and paper mapping are in `docs/paper_map.md`.  No source
images are needed; all analyses operate on the saved feature tables.  The
subspace task is the first ported analysis entry point; the remaining tasks
use the same `--run-root/--runs/--out` interface as they are added.

## Tier 3: full pipeline (GPU, raw data and checkpoints)

Inputs: the datasets described in `docs/data.md`, model checkpoints, and
`config/local.toml`.

Commands:

```bash
python scripts/40_preflight_check.py
python scripts/50_run_offline.py --help
python scripts/51_export_paper_data.py --help
```

The paper used three split seeds (`1729`, `2718`, `3141`) and the run directory
names recorded in `artifacts/paper/provenance.json`.  Each seed is run as its
own single-seed run; the analysis layer then averages the three seeds with
equal weight.

## Environment

The archived runs used:

- Python 3.13.3
- numpy 2.2.6
- scikit-learn 1.6.1
- scipy 1.17.1
- pandas 3.0.2
- torch 2.6.0+cu124
- transformers 4.57.3
- open_clip_torch 3.3.0

`pyproject.toml` gives compatible ranges.  For exact last-digit reproduction,
pin the versions above and set the BLAS thread count to 3.

## What is not guaranteed

- Bitwise identity across CPU/GPU vendors.
- Identical SVG rendering for Figures 1--2 across font installations.
- Redistribution of datasets, checkpoints or feature caches.

The archived summary values, not bitwise feature arrays, are the release
contract.
