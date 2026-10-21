# Third-party components

This repository contains only the authors' code.  It does not redistribute
datasets, model weights, or feature caches.  The following inputs remain under
their original licenses.

## Datasets

- DOTA v1.5 -- https://captain-whu.github.io/DOTA/
- DIOR-R -- https://gcheng-nwpu.github.io/
- FAIR1M -- http://www.gaofen-challenge.com/

Users are responsible for obtaining the data and following the corresponding
license and citation requirements.

## Pretrained encoders

- RemoteCLIP -- https://github.com/ChenDelong1999/RemoteCLIP
- OpenAI CLIP -- https://github.com/openai/CLIP
- SigLIP2 -- https://huggingface.co/google/siglip2-base-patch16-224
- DINOv2 -- https://github.com/facebookresearch/dinov2
- MAE -- https://github.com/facebookresearch/mae

Model weights are not included.  Check each model card for its license.

## Python packages

The runtime depends on NumPy, pandas, Pillow, tqdm, cloudpickle, scikit-learn,
and optionally PyTorch, transformers, open_clip_torch, safetensors and
matplotlib.  Their licenses are available in the respective distributions.

## Figures

Figures 3--7 are generated from the authors' compact CSVs.  Figures 1--2 are
conceptual diagrams; any raster panels derived from DOTA imagery remain subject
to the DOTA license and are not redistributed as raw imagery.
