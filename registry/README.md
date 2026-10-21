# Registries

The registry files are protocol metadata, not results.

| File | Purpose |
|---|---|
| `experiments.csv` | E00--E08 scientific prerequisites and evidence roles |
| `datasets.csv` | dataset version, task role and licence status |
| `models.csv` | encoder family, representation variant, checkpoint revision and hash status |
| `claims.csv` | manuscript claims, null hypotheses and allowed wording |
| `dataset_classes.json` | authoritative class-id / class-name / coarse-family mapping |

`experiments.csv` carries the protocol state; `claims.csv` carries the claim
state.  A claim is only supported by a completed, audited run with the
pre-registered evidence tier.  `checkpoint_verified` means the checkpoint
revision and SHA-256 were checked; it does not by itself validate a scientific
claim.

The public analysis layer reads `dataset_classes.json` for class names and
coarse families.  The other files document the protocol and are used by the
pipeline validator.
