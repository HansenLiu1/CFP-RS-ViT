# Tests

Run the portable unit suite without datasets, checkpoints or a GPU:

```bash
python -m unittest discover -s tests -t . -v
```

Coverage includes:

- configuration merge and registry contracts;
- source-image-disjoint role assignment;
- paired-input pixel and control invariants;
- residual-operator fitting and numerical guards;
- layerwise and retrieval interfaces;
- paper-export acceptance and release lineage;
- public direction API, readout helpers and figure-data checks.

Tests that need PyTorch or scikit-learn skip when the optional dependency is
not installed.  Real checkpoints are never loaded by the unit suite.
