# Data directory

This directory contains only schemas and documentation.  Raw remote-sensing
imagery and annotations are not redistributed.

The pipeline reads the dataset and checkpoint roots from `config/local.toml`
and writes a fresh source manifest inside each run directory.  Do not copy an
old project's candidate manifest, crops, masks, features or deltas into a new
run.

Allowed in git:

- annotation-layout documentation;
- class registries;
- license notes;
- compact, non-identifying manifest summaries.

Not allowed in git:

- raw images, labels or crops;
- model checkpoints;
- feature tables, deltas or run caches;
- hand-picked samples that cannot be traced back to an official dataset
  identity.

See `docs/data.md` for the expected layout and `THIRD_PARTY.md` for licenses.
