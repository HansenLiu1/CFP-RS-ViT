# CFP-RS-ViT

Code for **Controlled Foreground Perturbation for Diagnosing Layerwise Semantic
Structure in Vision Transformers on Remote-Sensing Imagery**.

The repository contains the CFP pipeline, the public paper-level analyses, the
scripts that regenerate Figures 3--7, and a compact set of frozen summary tables
for auditing the reported numbers without downloading the full feature caches.

## Status

Implemented and tested:

- `src/cfra/`: data indexing, OBB crops and perturbations, frozen encoder
  adapters, direction fitting, evaluation and statistics.
- `src/cfra/analysis/`: shared analysis primitives, readouts, and the
  `cfra-analysis` command line.
- `src/cfra/analysis/tasks/`: `spectra`, `geometry`, `subspace-alignment`,
  `readout-removal`, `readout-transfer`, `normalization`, `export-stats`,
  `layerwise`, `intervals`, `layerwise-fdbc`, `text-baseline` and
  `text-anchoring`.
- `scripts/40,50,51,52,53`: preflight, main run, paper export, E04-5 full
  sweep, and E04-5 export-only.
- `figures/`: Figures 3--7 and their compact plotting tables.
- `artifacts/paper/`: 45 frozen summary tables plus `manifest.csv`.

All paper-level analysis tasks now have public entry points.  The CPU tasks are
validated against the archived seed-1729 tables; the text tasks and the
layerwise FdBc interaction require the optional model assets and, for FdBc, a
separate GPU extraction.  See [`docs/analyses.md`](docs/analyses.md) for the
task-by-task commands, inputs and outputs.  The frozen `artifacts/paper/`
tables still allow the main tables and supplement values to be checked without
feature caches.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[figures,dev]"
```

For the encoder pipeline, install the model extras as well:

```bash
pip install -e ".[models,figures,dev]"
```

This installs PyTorch, transformers, open_clip and safetensors.  The analysis
and figure paths only need the base and `figures` dependencies.

## Repository layout

```text
CFP-RS-ViT/
├─ config/
│  └─ local.example.toml        # copy to local.toml; never committed
├─ configs/                     # versioned experiment configurations
├─ registry/                    # dataset, model, experiment and class registries
├─ data/                        # data-layout and manifest documentation
├─ src/cfra/                    # pipeline package
│  └─ analysis/                 # paper analyses and CLI
├─ scripts/
│  ├─ 40_preflight_check.py
│  ├─ 50_run_offline.py
│  ├─ 51_export_paper_data.py
│  ├─ 52_run_strength_sweep.py
│  └─ 53_export_strength_sweep.py
├─ figures/                     # Fig 3--7 scripts, data/ and output/
├─ artifacts/paper/             # frozen tables and manifest.csv
├─ tests/
└─ docs/
```

The public repo contains no raw imagery, annotations, model weights, feature
caches, or run directories.

## Data loading

### 1. Configure local paths

Copy the example and fill in the dataset and checkpoint roots:

```bash
cp config/local.example.toml config/local.toml
```

`config/local.toml` is ignored by git.  It contains:

```toml
[paths.datasets]
dota = "E:/data/DOTA/v1.5"
dior = "E:/data/DIOR-R"
fair1m = "E:/data/FAIR1M"

[paths.models]
remoteclip = "D:/models/RemoteCLIP-ViT-B-32.pt"
clip = "D:/models/clip-vit-base-patch16"
siglip2 = "D:/models/siglip2-base-patch16-224"
dinov2 = "D:/models/dinov2-base"
mae = "D:/models/vit-mae-base"
```

### 2. Dataset directory layout

The loader scans one directory per split:

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

The directory names `Images` and `Labels` are case-sensitive on Linux.

**DOTA v1.5** uses the official text labels:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

The two header lines (`imagesource:` and `gsd:`) are ignored.

**DIOR-R and FAIR1M** use one object per line in normalized form:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Coordinates are multiplied by the image width and height.  Class ids follow
`registry/dataset_classes.json`.

> **Annotation preparation.**  The current release expects the normalized
> DIOR-R/FAIR1M layout above.  The official releases distribute annotations in
> their own XML formats.  The paper's archived runs used a fixed normalized
> snapshot; a converter from the official XML files is a separate release task
> and is not yet included.  See [`docs/data.md`](docs/data.md) for details and
> record your converter and its hash in the run manifest if you create one.

### 3. What the loader does

`cfra.data.index` scans the split directories, hashes each image and label,
parses OBB geometry, applies the exclusion rules, and writes a manifest below
the run directory.  The split is source-image-disjoint and assigns each image
to one of four roles:

```text
direction_fit, readout_fit, validation, confirmation
```

The three paper seeds are `1729`, `2718`, and `3141`.  The `fit15` analyses use
at most 15 direction-fit objects per class in `sha256(object_id)` order.

### 4. Preflight

After `config/local.toml` is in place:

```bash
python scripts/40_preflight_check.py
```

It checks dependencies, dataset paths, checkpoint identity, hardware, and a
single-sample forward pass.  It does not create a run directory.

## Script order

The paper path is:

```text
40 preflight
  -> 50 run one directory per seed
  -> 51 export master + three-seed tables
  -> 52 E04-5 full sweep (optional, GPU)
  -> 53 E04-5 export-only (optional, CPU)
  -> cfra-analysis <task> (paper analyses)
  -> cfra-analysis figures
  -> cfra-analysis validate
```

`50_run_offline.py` is the paper-data entry point.  It runs the paired-input
audit, feature extraction, and the requested E03--E08 stages.  It does **not**
create the run directory; create it first.

### Quick smoke run

```bash
mkdir -p runs/smoke-1729

python scripts/50_run_offline.py \
  --run 1729=runs/smoke-1729 \
  --config configs/smoke_offline.toml \
  --datasets dota \
  --models clip \
  --perturbations blur \
  --experiments E00,E01,E02,E03
```

This is an engineering smoke test, not paper evidence.

### Main three-seed run

Create the three run directories first:

```bash
mkdir -p runs/cfp-1729 runs/cfp-2718 runs/cfp-3141
```

Then run all three seeds in one command (sequentially) or on separate machines:

```bash
python scripts/50_run_offline.py \
  --run 1729=runs/cfp-1729 \
  --run 2718=runs/cfp-2718 \
  --run 3141=runs/cfp-3141 \
  --config configs/offline.toml
```

The default experiment list covers `E00` through `E08` but excludes `E04-5`;
the strength sweep is handled separately by `52`/`53`.  Each run takes its own
feature cache; do not point two seeds at the same directory.

If a run is interrupted after a feature-affecting setting changed, rerun the
same command with `--allow-feature-reset`.  To reuse the seed-independent E00
source manifest, pass `--reuse-e00`.

### Paper export

`51_export_paper_data.py` checks the seed recorded in each run manifest and
writes the metric-level master and the three-seed tables:

```bash
python scripts/51_export_paper_data.py \
  --run 1729=runs/cfp-1729 \
  --run 2718=runs/cfp-2718 \
  --run 3141=runs/cfp-3141 \
  --out exports/cfp-paper
```

`E04-5` is excluded by default (`--exclude-stages E04-5`) because its export has
a separate path.  `51` requires all three registered seeds unless
`--allow-partial` is passed.  Use `--clear-checkpoint` to discard the export
checkpoint and recompute from scratch.

### E04-5: full sweep vs export-only

The paper's strength analysis has 18 levels: four nominal cells already present
in the main run, six scalar low/high cells (`P1`), and eight patch-mask
budget/side cells (`P2`).  The full sweep extracts the 14 non-nominal cells per
model and seed.  It reads the nominal `features_<model>.npz` tables written by
`50`, so run `50` first.

**Full sweep (GPU; writes into the run directories):**

```bash
python scripts/52_run_strength_sweep.py \
  --run 1729=runs/cfp-1729 \
  --run 2718=runs/cfp-2718 \
  --run 3141=runs/cfp-3141 \
  --models remoteclip,clip,siglip2,dinov2,mae \
  --levels P1,P2
```

Useful options:

| Option | Meaning |
|---|---|
| `--levels P1` | scalar low/high cells only; skip the 3x3 patch grid |
| `--levels P2` | patch budget/side grid only |
| `--models clip` | subset of encoders |
| `--bootstrap-replicates 50` | cheap smoke statistics |
| `--cpu-workers N` | extraction worker count |
| `--dry-run` | print the resolved grid and plan; load no model |
| `--force` | re-extract existing variant feature files |

**Export-only (CPU; no feature extraction):**

```bash
python scripts/53_export_strength_sweep.py \
  --run 1729=runs/cfp-1729 \
  --run 2718=runs/cfp-2718 \
  --run 3141=runs/cfp-3141 \
  --out exports/cfp-e04-5
```

`53` reads the `E04-5_<model>.json` stage shards produced by `52` and writes:

```text
e04_5_cells.csv
e04_5_paired.csv
e04_5_coverage.json
```

It never extracts features and never writes inside a run directory.  If a
shard is missing, `53` records the cell as missing; it does not silently
fabricate it.

**Quick E04-5 subset:** for an engineering check, run one model and one level:

```bash
python scripts/52_run_strength_sweep.py \
  --run 1729=runs/cfp-1729 \
  --models clip \
  --levels P1 \
  --bootstrap-replicates 50

python scripts/53_export_strength_sweep.py \
  --run 1729=runs/cfp-1729 \
  --models clip \
  --out exports/cfp-e04-5-clip-p1
```

This is a subset, not a reproduction of the paper's full 18-level grid.

### Paper analyses

All tasks use the same run-directory interface:

```bash
python -m cfra.analysis <task> \
  --run-root runs \
  --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/<task>
```

The available tasks are:

| Task | Paper items | Main outputs |
|---|---|---|
| `spectra` | S4b, S4c-1 | spectra, common-mode decomposition, removal curves, matched-16 |
| `geometry` | S4b cross-image cosine | class-direction cosine gaps and permutation null |
| `subspace-alignment` | S4c | alignment, PC control, validation tables |
| `readout-removal` | Table IV, S10, Fig. 7a | E07 removal, reflection and dose tables |
| `readout-transfer` | Table III, S5e, S5i | prototype/CAV/CFP/fusion transfer |
| `normalization` | Table II, S7, S9 | M0-M6 normalization and `r/|m|` bins |
| `export-stats` | Table I, S4a/S4d, S8a | rank, near-zero, interaction, permutation |
| `layerwise` | S11, Fig. 7b | depth decomposition and layerwise removal |
| `intervals` | S13.1-S13.4 | shared-source-image cluster intervals |
| `layerwise-fdbc` | S8b | layerwise foreground x background interaction |
| `text-baseline` | S12 | per-template and ensemble text retrieval |
| `text-anchoring` | S12b | text-direction alignment and permutation null |

Most tasks also accept `--models`, `--seeds`, `--datasets`,
`--perturbations` and `--max-cells` for a smoke run.  `intervals` has
`--tables rank,operators,readout,regions`, `--replicates` and `--permutations`.
The text tasks additionally need `--project-root` pointing at a checkout with
`config/local.toml`, `registry/` and the model weights; install
`.[models]` first.  `layerwise-fdbc` consumes pre-extracted FdBc arrays via
`--fdbc seed=directory,...`.

See [`docs/analyses.md`](docs/analyses.md) for the complete task map, output
files and asset requirements.

### Figures

Figures 3--7 are generated from the committed plotting tables:

```bash
python -m cfra.analysis figures
```

Outputs are written to:

```text
figures/output/Figure 3.pdf
figures/output/Figure 3.png
...
figures/output/Figure 7.pdf
figures/output/Figure 7.png
```

The figure scripts only read `figures/data/`.  They do not need raw imagery,
checkpoints, or a GPU.  Figure 1/2 are conceptual assets and are not generated
by this command.

### Validate the frozen paper tables

```bash
python -m cfra.analysis validate --artifacts artifacts/paper
```

This checks every file in `artifacts/paper/manifest.csv` and its SHA-256.
`docs/paper_map.md` gives the manuscript mapping.  The command does not
recompute values; it verifies that the frozen audit tables are the ones shipped
with the release.

## Result and cache locations

### Run directory

`50` never creates the run directory.  Once created, it writes:

```text
<run>/
├─ manifest/
│  ├─ run.json
│  ├─ e00.json / e01.json
│  ├─ features_<model>.json
│  └─ E04-5_<model>.json        # after 52
├─ intermediates/
│  ├─ features/
│  │  ├─ features_<model>.npz
│  │  ├─ layerwise_<model>_<dataset>.npz
│  │  └─ strength_sweep/<model>/<variant>__<perturbation>.npz   # after 52
│  ├─ e02_pixel_protocol/
│  └─ stage_results/
│     ├─ E03.json ... E08.json
│     └─ E04-5_<model>.json      # after 52
├─ metrics/
├─ statistics/
├─ tables/
├─ figures/
├─ logs/
├─ checkpoint.json
└─ STATUS
```

Feature caches are the large objects (roughly 50--100 GB per three-dataset,
five-model run).  They are never committed to git.

### Export and analysis outputs

| Output | Default path | Written by |
|---|---|---|
| Master metric table | `<export>/00_master_results.csv` | `51` |
| Three-seed summary | `<export>/tables_3seed/seed_summary.csv` | `51` |
| Export acceptance | `<export>/acceptance.json` | `51` |
| Export coverage | `<export>/coverage.json` | `51` |
| E04-5 cells | `<e04-out>/e04_5_cells.csv` | `53` |
| E04-5 paired | `<e04-out>/e04_5_paired.csv` | `53` |
| Analysis task outputs | `<analysis-out>/` (see `docs/analyses.md`) | `cfra-analysis <task>` |
| Figures | `figures/output/` | `cfra-analysis figures` |
| Frozen paper tables | `artifacts/paper/` | release artifact pack |

### How to change locations

- **Run directory / feature cache:** pass a different `--run SEED=DIR` to `50`,
  `51`, `52`, and `53`.  The directory must already exist for `50`.
- **Paper export:** pass a different `--out` to `51`.
- **E04-5 export:** pass a different `--out` to `53`.
- **Analysis output:** pass a different `--out` to the analysis task.
- **Figure output:** edit the `OUT = HERE / "output"` line at the top of each
  `figures/fig*.py` script.
- **Frozen paper tables:** do not edit them by hand.  Regenerate the relevant
  table from the analysis layer and update `artifacts/paper/manifest.csv` with
  its SHA-256.

### How to force recomputation

- **Main stages:** delete the specific `intermediates/stage_results/<stage>.json`
  (and its derived metrics) and rerun `50`; artifacts on disk are reconciled
  against the checkpoint.
- **Feature caches after a feature-affecting change:** rerun `50` with
  `--allow-feature-reset`.
- **Reuse the seed-independent source manifest:** `50 --reuse-e00`.
- **Export:** `51 --clear-checkpoint`.
- **E04-5 extraction:** `52 --force`.
- **E04-5 export:** `53` is cheap; delete its output CSVs and rerun.
- **Figures:** `python -m cfra.analysis figures` overwrites `figures/output/`.
- Do not hand-edit `checkpoint.json`.

## Configuration and environment

Authoritative configs:

| File | Role |
|---|---|
| `configs/base.toml` | protocol, sampling, normalization and statistics defaults |
| `configs/offline.toml` | paper-data path used by `50` |
| `configs/main.toml` | formal four-group reference configuration |
| `configs/smoke_offline.toml` | DOTA/clip/blur engineering smoke |
| `configs/development_core.toml` | small development run |

Machine-local paths belong in `config/local.toml`; never add absolute paths to
a versioned config.

Useful environment variables:

| Variable | Effect |
|---|---|
| `CFRA_BOOTSTRAP_WORKERS` | CPU workers for clustered bootstrap |
| `CFRA_CELL_WORKERS` | cell-level parallel workers |
| `CFRA_REUSE_E00` | reuse a seed-independent E00 manifest |
| `CFRA_ALLOW_FEATURE_RESET` | allow feature caches to be deleted on reset |
| `CFP_PAPER_EXPORT` | default root for `figures/build_data.py` |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | BLAS thread control; set to `3` to match archived values |

CLI overrides:

```text
50: --datasets --models --perturbations --experiments
    --run-e04-5   # only if E04-5 is in --experiments; prefer 52 for the paper sweep
51: --exclude-stages --bootstrap-replicates --permutation-replicates
    --skip-layer-recompute --skip-final-recompute --allow-partial
    --require-complete --clear-checkpoint
52: --models --levels --bootstrap-replicates --cpu-workers --dry-run --force
53: --models --bootstrap-replicates --permutation-replicates
```

## Reproducibility tiers

1. **Paper audit (CPU, no downloads):**
   `cfra-analysis validate` plus `cfra-analysis figures`.
2. **Analysis rerun (CPU, feature artifacts):**
   `cfra-analysis spectra`, `geometry`, `readout-removal`,
   `readout-transfer`, `normalization`, `export-stats`, `layerwise`,
   `intervals` and `subspace-alignment`; text and FdBc tasks additionally need
   the optional assets described in `docs/analyses.md`.
3. **Full pipeline (GPU, datasets and checkpoints):**
   `40 -> 50 (x3 seeds) -> 51 -> 52/53`.

Known last-digit differences (BLAS threading, fp16 text encoding, effective
support under block masking, margin weighting) are documented in
[`docs/known_differences.md`](docs/known_differences.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `run directory does not exist` | create it before running `50` |
| `missing registered seeds` in `51` | run all three seeds or pass `--allow-partial` |
| `missing E04-5 artifact` in `53` | run `52` first, or accept a partial coverage report |
| `config/local.toml` missing | copy `config/local.example.toml` and fill paths |
| figure fonts differ | install Times New Roman; fallback is STIX/DejaVu |
| analysis task not found | run `python -m cfra.analysis --help`; the task map is in `docs/paper_map.md` |

## Tests

```bash
python -m unittest discover -s tests -t . -v
python -m ruff check .
```

The unit suite does not need datasets, model weights, or a GPU.  The GitHub
Actions workflow runs the same tests on Python 3.11 and 3.13.

## Provenance and reproducibility scope

The original working repository had accumulated ad-hoc scripts, duplicated
implementations and machine-specific paths.  This public repository is therefore
a clean, automatically reconstructed release generated by **DeepSeek V4.1 Flash
1017** within the **OpenCode 2.1.2** framework, followed by manual checks of the
key implementations.  It is not guaranteed to be a complete or numerically
identical implementation of every step in the original experiments.

The three paper seeds were produced with the original implementation on two
workstations: seed `1729` on an Intel Core i7-13650HX with an NVIDIA RTX 4060,
and seeds `2718` and `3141` on an AMD Ryzen 7 9700X with an NVIDIA RTX 5070.
Because encoder inference uses half precision and platform-specific kernels,
cross-platform reproduction is not guaranteed to be bitwise identical, even
when the same checkpoints and feature protocol are used.

## Citation and license

The manuscript citation will be added when the paper is accepted.  Software
metadata is in `CITATION.cff`.  The authors' code is released under the MIT
License; datasets and model weights remain under their original licenses.  See
`THIRD_PARTY.md` and [`docs/data.md`](docs/data.md).
