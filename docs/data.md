# Data and model inputs

This repository does not redistribute images, annotations, checkpoints or
feature caches.  All inputs are read from paths supplied in
`config/local.toml`.

## Datasets

The paper uses:

- **DOTA v1.5** (primary)
- **DIOR-R** (primary)
- **FAIR1M** (boundary / fine-grained stress test)

Expected directory layout:

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

DOTA labels are the official line format:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

The first two header lines (`imagesource:` and `gsd:`) are ignored.

DIOR-R and FAIR1M labels are expected as one object per line:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Coordinates are normalized to `[0, 1]`; the pipeline multiplies them by the
image width and height.  Class ids follow `registry/dataset_classes.json`.

> **Annotation preparation.**  The public pipeline consumes the normalized
> layout above.  The official DIOR-R and FAIR1M releases distribute annotations
> in their own XML formats.  The paper's runs used a fixed normalized snapshot;
> a converter from the official XML files is a separate release task and is not
> claimed to be a one-command raw-data import yet.  If you use your own
> conversion, record the converter and its hash in the run manifest.

## Class registries

`registry/dataset_classes.json` is the authoritative class-id, class-name and
coarse-family mapping.  `registry/datasets.csv`, `registry/models.csv` and
`registry/experiments.csv` record provenance and protocol metadata.

## Model checkpoints

The five encoders are loaded by `cfra.models`:

| id | family |
|---|---|
| `remoteclip` | RemoteCLIP ViT-B/32 |
| `clip` | OpenAI CLIP ViT-B/16 |
| `siglip2` | SigLIP2 Base patch16/224 |
| `dinov2` | DINOv2 Base |
| `mae` | MAE ViT-B/16 |

The checkpoint paths are set in `config/local.toml`.  Feature artifacts record
a checkpoint SHA-256; the exporter checks that the same checkpoint was used for
all seeds of a run group.  Do not rename or modify checkpoints between seeds.

## Layerwise FdBc arrays

The normal layerwise artifact stores FcBc, FcBd and FdBd.  The S8b interaction
analysis also needs the fourth cell FdBc (foreground degraded, background
clean).  It is stored next to the run as:

```text
features_fdbc_<seed>/
  layerwise_fdbc_<model>_<dataset>.npz
  layerwise_fdbc_<model>_<dataset>.json
```

The NPZ contains `row_index` (indices into the layerwise artifact),
`perturbation`, and one `{layer}__fdbc` matrix per selected layer.  See
`docs/analyses.md` for the analysis command.  Extracting FdBc needs the original
images, a frozen encoder and a GPU; it is separate from the CPU analysis.

## Licensing

Model weights and datasets remain under their original licenses.  The MIT
license in this repository covers the authors' code only.  See
`THIRD_PARTY.md` for links and attribution.
