"""Stage orchestration for current-run CFRA evidence."""

from .core import (
    CoreStageError,
    run_e03,
    run_e04,
    run_e04_1,
    run_e04_2,
    run_e04_3,
    run_e04_4,
    run_e04_5,
    run_e05,
    run_e07,
    write_core_results,
)
from .features import (
    FeatureSample,
    FeatureStageError,
    FeatureTable,
    extract_feature_table,
    load_feature_table,
    make_feature_samples,
)
from .features_parallel import iter_context_batches_parallel, iter_context_pair_batches_parallel
from .layerwise import (
    FeatureRecord,
    LayerwiseError,
    LayerwiseFeatureTable,
    coerce_feature_record,
    run_e06,
    run_e06_readout,
)
from .retrieval import RetrievalError, TextBaseline, TextBaselineSpec, run_e08
from .stages import (
    StageError,
    build_e00,
    build_e01,
    build_e02,
    load_current_manifests,
    load_e01_selected,
)

__all__ = [
    "CoreStageError",
    "FeatureRecord",
    "FeatureSample",
    "FeatureStageError",
    "FeatureTable",
    "LayerwiseError",
    "LayerwiseFeatureTable",
    "RetrievalError",
    "StageError",
    "TextBaseline",
    "TextBaselineSpec",
    "build_e00",
    "build_e01",
    "build_e02",
    "coerce_feature_record",
    "extract_feature_table",
    "iter_context_batches_parallel",
    "iter_context_pair_batches_parallel",
    "load_current_manifests",
    "load_e01_selected",
    "load_feature_table",
    "make_feature_samples",
    "run_e03",
    "run_e04",
    "run_e04_1",
    "run_e04_2",
    "run_e04_3",
    "run_e04_4",
    "run_e04_5",
    "run_e05",
    "run_e06",
    "run_e06_readout",
    "run_e07",
    "run_e08",
    "write_core_results",
]
