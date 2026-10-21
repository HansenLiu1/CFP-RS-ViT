# Configurations

All experiment parameters are versioned TOML.  Child configurations use
`extends` and override individual fields; the resolved configuration and its
SHA-256 are written into the run manifest.

| File | Role |
|---|---|
| `base.toml` | shared protocol, sampling, model-runtime and statistics defaults |
| `main.toml` | formal reference configuration; includes the four-group release layout |
| `offline.toml` | single-machine paper-data path used by `scripts/50_run_offline.py` |
| `development_core.toml` | small DOTA/blur development run |
| `smoke.toml` | minimal engineering smoke test |
| `smoke_offline.toml` | offline E00--E05 smoke test |

The manuscript's three paper runs were produced with `offline.toml` in
single-seed run directories (`1729`, `2718`, `3141`), then exported and
aggregated with three-seed equal weight.  `main.toml` documents the formal
four-group design but is not the exact run recipe for the archived paper data;
use `offline.toml` for the paper-data path.

Machine-local paths belong in `config/local.toml`, which is ignored by git.
Do not add absolute paths to a versioned config.

`[experiment_models]` is the per-experiment execution matrix and must agree
with `registry/experiments.csv`.  `clip_shared` is a representation variant of
`clip`, not an independent model.

`smoke.toml` and `development_core.toml` use smaller sampling minima; they are
for engineering checks and do not produce paper evidence.
