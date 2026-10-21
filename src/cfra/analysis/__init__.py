"""Paper-level analyses built on the CFP pipeline.

The public analysis code deliberately uses :mod:`cfra.directions` as the only
class-direction implementation.  Legacy batch scripts from the internal
research repository are not part of this package.
"""

from cfra.analysis.common import (
    DATASETS,
    MODELS,
    PERTURBATIONS,
    Features,
    RunLayout,
    cell_delta,
    cell_rows,
    common_mode,
    fit_operator,
    limit_fit,
    seed_mean,
    specificity,
    spectrum,
    unit,
    unit_delta,
)

__all__ = [
    "DATASETS",
    "MODELS",
    "PERTURBATIONS",
    "Features",
    "RunLayout",
    "cell_delta",
    "cell_rows",
    "common_mode",
    "fit_operator",
    "limit_fit",
    "seed_mean",
    "specificity",
    "spectrum",
    "unit",
    "unit_delta",
]
