# Paper analyses

All analysis entry points are under `cfra-analysis`.  They read the persisted
`features_<model>.npz` and `layerwise_<model>_<dataset>.npz` artifacts; they do
not re-run an encoder and do not change the pipeline.

## Common interface

```bash
python -m cfra.analysis <task> \
  --run-root runs \
  --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/<task>
```

Common options:

| Option | Meaning |
|---|---|
| `--models` | comma-separated encoder ids |
| `--seeds` | subset of seeds; defaults to all `--runs` |
| `--datasets` | `dota`, `dior`, `fair1m` |
| `--perturbations` | `blur`, `brightness`, `gaussian_noise`, `patch_mask` |
| `--max-cells` | stop after this many dataset x perturbation cells (smoke runs) |
| `--out` | output directory; each task writes a `detail/` cache where useful |

The fit budget is always the paper convention: at most 15 direction-fit objects
per class in SHA-256(object_id) order, followed by removal of near-zero deltas.
Tasks that use a different budget label it in the output (`support` column).

## Task map

| Task | Paper items | Outputs |
|---|---|---|
| `spectra` | S4b, S4c-1 | `spectra_per_seed.csv`, `spectra_3seed.csv`, `residual_spectra_*`, `decomposition_*`, `removal_curves_*`, `matched16_*`, `criteria_summary.csv`, `isotropic_null.csv` |
| `geometry` | S4b cross-image cosine | `geometry_per_seed.csv`, `geometry_3seed.csv`, `geometry_by_*.csv`, `geometry_exceedance_*.csv` |
| `subspace-alignment` | S4c | `alignment_*.csv`, `pc_control_*.csv`, `alignment_validation*.csv` |
| `readout-removal` | Table IV, S10, Fig. 7a | `readout_removal_detail.csv`, `readout_removal_summary.csv`, `readout_removal_table_iv.csv`, `readout_removal_reflection.csv`, `readout_removal_dose.csv` |
| `readout-transfer` | Table III, S5e, S5i | `readout_transfer_per_seed.csv`, `readout_transfer_table_iii.csv`, `readout_transfer_s5e.csv`, `readout_transfer_s5i.csv`, `readout_transfer_fusion_weights.csv` |
| `normalization` | Table II, S7, S9 | `normalization_per_seed.csv`, `normalization_summary.csv`, `normalization_by_*.csv`, `r_over_m_*.csv` |
| `export-stats` | Table I, S4a/S4d, S8a | `rank_per_seed.csv`, `rank_3seed.csv`, `nearzero_*.csv`, `interaction_*.csv`, `permutation_*.csv` |
| `layerwise` | S11, Fig. 7b | `layerwise_depth_*.csv`, `layerwise_removal_*.csv` |
| `intervals` | S13.1-S13.4 | `intervals_rank.csv`, `intervals_operators.csv`, `intervals_readout.csv`, `intervals_regions.csv` |
| `layerwise-fdbc` | S8b | `layerwise_fdbc_detail.csv`, `layerwise_fdbc_3seed.csv`, `layerwise_fdbc_trajectory_*.csv` |
| `text-baseline` | S12 | `text_baseline_cells.csv`, `text_baseline_summary.csv`, `text_baseline_space_delta.csv` |
| `text-anchoring` | S12b | `text_anchoring_cells.csv`, `text_anchoring_summary.csv`, `text_anchoring_permutation.csv` |

## Selected commands

### Spectra and geometry

```bash
python -m cfra.analysis spectra \
  --run-root runs --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/spectra

python -m cfra.analysis geometry \
  --run-root runs --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/geometry
```

`spectra` computes M0-M6 plus the retired `w_form` reference, the M1
common-mode decomposition (`m`, `e = unit(m)`, residual `O`), top-k removal
curves and the 16-class matched control.  `geometry` reports the same-class and
different-class cosine means with a class-label permutation null.

### Readout analyses

```bash
python -m cfra.analysis readout-removal \
  --run-root runs --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/readout-removal \
  --budgets fit15,fitall

python -m cfra.analysis readout-transfer \
  --run-root runs --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/readout-transfer
```

`readout-removal` calls the public `run_e07` stage.  The default alphas are
`0.25,0.5,0.75,1.0`; `--reflection-alpha 2` produces the S10 reflection rows.
Set `--bootstrap-replicates 5000` only if you also want the per-cell pipeline
bootstrap diagnostics; the unified S13 intervals are produced by `intervals`.

### Normalization and export statistics

```bash
python -m cfra.analysis normalization \
  --run-root runs --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/normalization

python -m cfra.analysis export-stats \
  --run-root runs --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/export-stats
```

`normalization` runs C6 (M0-M6 under unit/raw/magnitude-weighted) and C7
(`r/||m||` quantile bins).  `export-stats` produces rank (including the FAIR1M
coarse variant), near-zero block-mask audits, foreground/background interaction
ratios and the Table I label-permutation controls.  Select a subset with
`--analyses rank,nearzero,interaction,permutation`.

### Layerwise

```bash
python -m cfra.analysis layerwise \
  --run-root runs --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/layerwise \
  --mode both
```

`--mode depth` runs only the common-mode/class-specific decomposition;
`--mode removal` runs only the single-direction readout removal.  For a smoke
check use `--datasets dota --perturbations blur --layers L01,final`.  The full
run reads 14 layers for every model/dataset and is CPU-heavy.

### Intervals

```bash
python -m cfra.analysis intervals \
  --run-root runs --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/intervals \
  --tables rank,operators,readout,regions
```

The defaults match the paper's final method: 5,000 shared source-image draws
per dataset and 10,000 permutation/sign-flip draws.  The source-image pool is
the union of the three seeds' confirmation images; every cell of a dataset uses
the same draw, and the three seeds are averaged at the same replicate index.
`--replicates` and `--permutations` can be lowered for smoke tests.  The four
tables are independent; run e.g. `--tables rank` for the quickest check.
`--max-cells` is rejected because the interval estimand is defined over the
full cell set.

### Layerwise FdBc interaction

```bash
python -m cfra.analysis layerwise-fdbc \
  --run-root runs --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/layerwise-fdbc \
  --fdbc 1729=features_fdbc_1729,2718=features_fdbc_2718,3141=features_fdbc_3141
```

Each FdBc directory must contain:

```text
layerwise_fdbc_<model>_<dataset>.npz
layerwise_fdbc_<model>_<dataset>.json
```

The NPZ stores `row_index` (indices into the layerwise artifact), `perturbation`
and `{layer}__fdbc` arrays; the JSON stores the extraction metadata.  The
extraction itself requires the original images, a frozen encoder and a GPU; it
is a separate pipeline step, not part of this CPU analysis.

### Text tasks

```bash
python -m cfra.analysis text-baseline \
  --run-root runs --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/text-baseline \
  --project-root . \
  --models remoteclip,siglip2,clip_shared

python -m cfra.analysis text-anchoring \
  --run-root runs --runs 1729=cfp-1729,2718=cfp-2718,3141=cfp-3141 \
  --out analysis/text-anchoring \
  --project-root . \
  --models remoteclip,siglip2,clip_shared
```

These tasks need `pip install -e ".[models]"`, a filled
`config/local.toml`, and the text-capable checkpoints.  `clip_shared` projects
the stored 768-d CLIP features with the checkpoint's bias-free
`visual_projection`; pass `--clip-projection path/to/visual_projection.pt` to
reuse a cached tensor, otherwise the projection is read from the registered
CLIP checkpoint.  Use `--templates` on `text-baseline` to score the four prompt
templates separately.  The text encoder precision is taken from the run
manifest; a full-fp32 rerun can differ from the archived fp16 values by about
`1.5e-4` macro AP.

## Validation notes

The CPU tasks were checked against the archived seed-1729 (and, where useful,
three-seed) tables for at least one representative cell:

| Task | Check |
|---|---|
| `spectra` | M1/M0 spectra, residual spectra, removal curves and decomposition match the archive to `<1e-12` |
| `geometry` | same/different/null cosines match exactly |
| `readout-transfer` | per-cell AUCs match exactly |
| `normalization` | per-operator rates and margins match to floating-point precision |
| `export-stats` | rank, near-zero and interaction rows match exactly |
| `layerwise` | depth stats match to `<1e-9`; removal accuracy/margin match to `<1e-8` |
| `intervals` | S13.1 point estimates and percentile intervals match the shared-weight archive for the checked cell |
| `layerwise-fdbc` | interaction cosines and disagreement match exactly |

Monte Carlo tasks (`geometry` permutation null, `intervals`,
`text-anchoring`) use the archived seed scheme; changing `--permutations` or
`--replicates` changes the Monte Carlo error, not the target quantity.
