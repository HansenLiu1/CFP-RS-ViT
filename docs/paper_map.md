# Paper-to-code map

This is the public, curated map.  Internal batch names are not part of the
public interface; the right-hand column is the analysis task that replaces
them.

Current status: `validate`, `figures` and `subspace-alignment` are implemented;
the remaining tasks are being ported in the order shown below.  `docs/paper_map.md`
is updated with each port, and the frozen `artifacts/paper/` tables already
allow the main tables and supplement values to be checked without feature
caches.

## Main tables

| Paper item | Content | Public task |
|---|---|---|
| Table I | own-class argmax and rank, fit15 | `export-stats` |
| Table II | M0--M6 operators, fit15 / fit50 | `normalization`, `intervals` |
| Table III | cross-perturbation transfer | `readout-transfer` |
| Table IV | readout-space removal at the final layer | `readout-removal` |
| Table V | region and geometry controls | `intervals` |

## Main figures

| Figure | Content | Public task / data |
|---|---|---|
| Fig. 1 | study overview | conceptual SVG/PNG (not a numerical plot) |
| Fig. 2 | CFP input construction | conceptual SVG/PNG; raster panels require DOTA |
| Fig. 3 | M0 / M1 / M1--M0 specificity | `figures/data/fig3_specificity.csv` |
| Fig. 4 | geometry controls and transfer | `figures/data/fig4_*.csv` |
| Fig. 5 | layerwise absolute specificity | `figures/data/fig5_*.csv` |
| Fig. 6 | common-mode and alignment | `figures/data/fig6_*.csv` |
| Fig. 7 | readout dose and layerwise response | `figures/data/fig7_*.csv` |

## Supplement

| Section | Content | Public task |
|---|---|---|
| S1 | population and exclusions | pipeline manifests |
| S2 | role split and overlap | pipeline manifests |
| S3 | M0--M6 detail | `normalization`, `intervals` |
| S4a | full own-argmax table | `export-stats` |
| S4b | spectra and cross-image cosine | `spectra`, `geometry` |
| S4c | direction geometry and subspace removal | `spectra`, `geometry`, `readout-transfer`, `subspace-alignment` |
| S4d | support and FAIR1M fine/coarse | `export-stats` |
| S5 | transfer and fusion baselines | `readout-transfer` |
| S6 | strength grid | `scripts/52`, `scripts/53` |
| S7 | normalization mechanism | `normalization` |
| S8 | factorial interaction and layerwise interaction | `export-stats`, `layerwise-fdbc` |
| S9 | common mode and masks | `normalization` |
| S10 | alpha = 2 readout | `readout-removal` |
| S11 | layerwise per model and readout | `layerwise`, `readout-removal` |
| S12 | text baseline | `text-baseline` |
| S12b | text anchoring | `text-anchoring` |
| S13 | confidence intervals | `intervals` |
| S14 | seed stability | `intervals` |
| S15 | hyperparameters | `configs/`, `docs/reproducibility.md` |
| S16 | code availability | this repository |

## Archived internal batches

The internal names `b1a`, `b1b`, `b2a`, `b2b`, `b3`, `b4`, `b5`, `b6`, `b7`,
`b8`, `b9`, `g1`, `g2` and `g3` are retained only in the internal asset
manifest.  They are mapped to the public tasks above and are not exposed as
separate entry points.
