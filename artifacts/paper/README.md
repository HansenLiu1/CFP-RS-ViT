# Frozen paper artifacts

This directory contains compact, summary-level CSV tables extracted from the
three-seed CFP runs.  They are the inputs for the Tier 1 paper audit and for
Figures 3--7.  They do not contain raw images, per-object features, checkpoints
or the full 100+ GB export tree.

- `manifest.csv` records the manuscript location, relative path, SHA-256, size,
  row count and source batch for every file.
- `provenance.json` records the run-directory names and the three split seeds.
- `tables/` contains the inputs for the main tables.
- `supplement/` contains the summary inputs for the supplement sections.

Check integrity with:

```bash
python -m cfra.analysis validate --artifacts artifacts/paper
```

The values are summary statistics derived by the analysis layer.  They are not
a substitute for rerunning the pipeline from raw data; see
`docs/reproducibility.md` for the three reproduction tiers.

Do not rename files without updating `manifest.csv`; the validation command
checks both presence and SHA-256.
