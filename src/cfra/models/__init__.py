"""Frozen encoder adapters with explicit representation-space metadata."""

from .adapters import (
    AdapterCache,
    EncodedFeatures,
    LayerwiseEncodedFeatures,
    ModelAdapterError,
    VisionEncoderAdapter,
    iter_encoded_batches,
    load_adapter,
)
from .assets import (
    ModelAsset,
    ModelAssetError,
    resolve_model_assets,
    validate_model_asset,
    verify_model_asset,
    verify_registered_model_asset,
)
from .preprocessing import ImageBatchError, preprocess_image_batch, stack_image_batch
from .runtime import ModelRuntimeConfig, ModelRuntimeConfigurationError, parse_model_runtime_config
from .specs import MODEL_SPECS, ModelSpec, PreprocessSpec, get_model_spec, primary_model_ids
from .text import (
    DEFAULT_TEXT_TEMPLATES,
    SHARED_SPACE_IDS,
    EncodedTextFeatures,
    SharedTextEncoder,
    SharedTextPrototypeBaseline,
    load_shared_text_encoder,
)

__all__ = [
    "DEFAULT_TEXT_TEMPLATES",
    "MODEL_SPECS",
    "SHARED_SPACE_IDS",
    "AdapterCache",
    "EncodedFeatures",
    "EncodedTextFeatures",
    "ImageBatchError",
    "LayerwiseEncodedFeatures",
    "ModelAdapterError",
    "ModelAsset",
    "ModelAssetError",
    "ModelRuntimeConfig",
    "ModelRuntimeConfigurationError",
    "ModelSpec",
    "PreprocessSpec",
    "SharedTextEncoder",
    "SharedTextPrototypeBaseline",
    "VisionEncoderAdapter",
    "get_model_spec",
    "iter_encoded_batches",
    "load_adapter",
    "load_shared_text_encoder",
    "parse_model_runtime_config",
    "preprocess_image_batch",
    "primary_model_ids",
    "resolve_model_assets",
    "stack_image_batch",
    "validate_model_asset",
    "verify_model_asset",
    "verify_registered_model_asset",
]
