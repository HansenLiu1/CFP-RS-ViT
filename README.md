# CFP-RS-ViT

Code for **Controlled Foreground Perturbation for Diagnosing Layerwise Semantic
Structure in Vision Transformers on Remote-Sensing Imagery**.

The repository is being prepared as the public companion to the manuscript.  It
contains the controlled foreground perturbation (CFP) pipeline, the analysis
code that produces the paper tables and figures, and a compact set of frozen
summary tables that can be checked without downloading feature caches.

## What is included

- `src/cfra/`: data indexing, OBB crop and perturbation operators, frozen model
  adapters, representation deltas, direction fitting, metrics and statistics.
- `src/cfra/analysis/`: paper-level analyses.  The first public task is
  `subspace-alignment`; the remaining tasks share the same CLI and are listed
  with their status in `docs/paper_map.md`.
- `scripts/`: pipeline entry points for feature extraction and paper export.
- `figures/`: scripts and compact CSV inputs for Figures 3--7.
- `artifacts/paper/`: frozen summary tables and a manifest mapping them to the
  manuscript.
- `tests/`: protocol, numerical and release regression tests.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[figures,dev]"
```

The model extras are separate because they download PyTorch and pretrained
encoders:

```bash
pip install -e ".[models]"
```

## Data and model paths

Copy `config/local.example.toml` to `config/local.toml` and fill in the dataset
and checkpoint roots.  The file is ignored by git.  Raw imagery, annotations
and model weights are not redistributed.

The pipeline expects one directory per official split:

```text
<dataset-root>/
  Train/
    Images/
    Labels/
  Val/
    Images/
    Labels/
  Test/
    Images/
    Labels/
```

DOTA v1.5 labels use the official DOTA text format.  DIOR-R and FAIR1M labels
use one line per object:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

with coordinates normalized to `[0, 1]`.  See `docs/data.md` for the exact
class registry and the annotation preparation notes.

## Reproduce the paper tables and figures

The compact paper artifacts do not require a GPU:

```bash
python -m cfra.analysis validate --artifacts artifacts/paper
python -m cfra.analysis figures
```

The first command checks the frozen artifacts and their SHA-256 values; the
manuscript mapping is in `docs/paper_map.md`.  The second regenerates
Figures 3--7 from the committed plotting tables.

## Run the pipeline

After configuring `config/local.toml`:

```bash
python scripts/40_preflight_check.py
python scripts/50_run_offline.py --help
```

`50_run_offline.py` is the paper-data entry point.  It runs the paired-input
audit, feature extraction and E03--E08 stages for one split seed.  The three
paper seeds are run in separate run directories and exported with
`51_export_paper_data.py`.

The public version of the scripts takes paths from the command line or from
`config/local.toml`; no machine-specific absolute path is stored in the
repository.

## Reproduction tiers

1. **Paper audit (CPU only):** use `artifacts/paper/` and the committed figure
   tables to verify every reported number and regenerate Figures 3--7.
2. **Analysis rerun:** use feature artifacts from a completed run to recompute
   spectra, geometry, readout and layerwise analyses.
3. **Full pipeline:** run E00--E08 from raw data and frozen checkpoints, then
   export the paper tables.  This requires the datasets and model weights.

Known numerical differences (BLAS threading, fp16 text encoding and effective
support under block masking) are documented in `docs/known_differences.md`.

## Citation

Citation metadata will be updated when the manuscript is accepted.  The
repository is released under the MIT License; see `LICENSE`.

## Contact

Hanxu Liu, Chengdu University of Technology.
