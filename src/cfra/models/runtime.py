"""Validated execution settings for batched frozen-model inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from cfra.provenance import sha256_json

from .specs import get_model_spec


class ModelRuntimeConfigurationError(ValueError):
    """Raised when model runtime settings are incomplete or unsafe."""


@dataclass(frozen=True)
class ModelRuntimeConfig:
    offline_only: bool
    formal_approved: bool
    precision: str
    batch_size: int
    max_loaded_models: int
    pin_memory: bool
    non_blocking: bool
    encoder_input_sizes: Mapping[str, int]
    positional_interpolation: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.offline_only is not True:
            raise ModelRuntimeConfigurationError("Model runtime must be offline_only")
        if self.formal_approved not in {True, False}:
            raise ModelRuntimeConfigurationError("model_runtime.formal_approved must be boolean")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ModelRuntimeConfigurationError("model_runtime.precision must be auto, fp32, fp16, or bf16")
        _positive_integer(self.batch_size, "model_runtime.batch_size")
        _positive_integer(self.max_loaded_models, "model_runtime.max_loaded_models")
        if self.pin_memory not in {True, False}:
            raise ModelRuntimeConfigurationError("model_runtime.pin_memory must be boolean")
        if self.non_blocking not in {True, False}:
            raise ModelRuntimeConfigurationError("model_runtime.non_blocking must be boolean")

        sizes = dict(self.encoder_input_sizes)
        if not sizes:
            raise ModelRuntimeConfigurationError("model_runtime.encoder_input_sizes cannot be empty")
        for model_id, raw_size in sizes.items():
            spec = get_model_spec(model_id)
            size = _positive_integer(raw_size, f"model_runtime.encoder_input_sizes.{model_id}")
            if size % spec.patch_size:
                raise ModelRuntimeConfigurationError(
                    f"model_runtime.encoder_input_sizes.{model_id} must be divisible by patch size {spec.patch_size}"
                )
            if size != spec.checkpoint_native_input_size and not spec.supports_position_interpolation:
                raise ModelRuntimeConfigurationError(f"{model_id} cannot use a non-native encoder input size")

        frozen_sizes = MappingProxyType(sizes)
        object.__setattr__(self, "encoder_input_sizes", frozen_sizes)
        object.__setattr__(
            self,
            "fingerprint",
            sha256_json(
                {
                    "offline_only": self.offline_only,
                    "formal_approved": self.formal_approved,
                    "precision": self.precision,
                    "batch_size": self.batch_size,
                    "max_loaded_models": self.max_loaded_models,
                    "pin_memory": self.pin_memory,
                    "non_blocking": self.non_blocking,
                    "encoder_input_sizes": sizes,
                    "positional_interpolation": dict(self.positional_interpolation),
                }
            ),
        )


def _positive_integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelRuntimeConfigurationError(f"{field} must be a positive integer")
    return value


def parse_model_runtime_config(config: dict[str, Any], model_ids: Iterable[str]) -> ModelRuntimeConfig:
    section = config.get("model_runtime")
    if not isinstance(section, dict):
        raise ModelRuntimeConfigurationError("Missing [model_runtime] section")
    if section.get("offline_only") is not True:
        raise ModelRuntimeConfigurationError("Model runtime must be offline_only")
    if section.get("formal_approved") not in {True, False}:
        raise ModelRuntimeConfigurationError("model_runtime.formal_approved must be boolean")
    precision = section.get("precision")
    if precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ModelRuntimeConfigurationError("model_runtime.precision must be auto, fp32, fp16, or bf16")
    if section.get("pin_memory") not in {True, False}:
        raise ModelRuntimeConfigurationError("model_runtime.pin_memory must be boolean")
    if section.get("non_blocking") not in {True, False}:
        raise ModelRuntimeConfigurationError("model_runtime.non_blocking must be boolean")

    configured_sizes = section.get("encoder_input_sizes")
    if not isinstance(configured_sizes, dict):
        raise ModelRuntimeConfigurationError("Missing [model_runtime.encoder_input_sizes] section")

    sizes: dict[str, int] = {}
    for model_id in dict.fromkeys(model_ids):
        spec = get_model_spec(model_id)
        size = _positive_integer(configured_sizes.get(model_id), f"model_runtime.encoder_input_sizes.{model_id}")
        if size % spec.patch_size:
            raise ModelRuntimeConfigurationError(
                f"model_runtime.encoder_input_sizes.{model_id} must be divisible by patch size {spec.patch_size}"
            )
        if size != spec.checkpoint_native_input_size and not spec.supports_position_interpolation:
            raise ModelRuntimeConfigurationError(f"{model_id} cannot use a non-native encoder input size")
        sizes[model_id] = size

    interpolation = section.get("positional_interpolation")
    if interpolation is None:
        # Keep the programmatic minimal runtime contract used by unit tests and
        # downstream callers. Versioned project configs always provide this
        # table and are validated strictly below.
        interpolation = {
            "enabled": True,
            "method": "bicubic_2d_checkpoint_position_embeddings",
            "source": "implicit_default",
        }
    if not isinstance(interpolation, dict):
        raise ModelRuntimeConfigurationError("model_runtime.positional_interpolation must be a TOML table")
    if interpolation.get("enabled") is not True:
        raise ModelRuntimeConfigurationError("positional interpolation must be explicitly enabled")
    if interpolation.get("method") != "bicubic_2d_checkpoint_position_embeddings":
        raise ModelRuntimeConfigurationError("unsupported positional interpolation method")
    return ModelRuntimeConfig(
        offline_only=True,
        formal_approved=section["formal_approved"],
        precision=precision,
        batch_size=_positive_integer(section.get("batch_size"), "model_runtime.batch_size"),
        max_loaded_models=_positive_integer(
            section.get("max_loaded_models"), "model_runtime.max_loaded_models"
        ),
        pin_memory=section["pin_memory"],
        non_blocking=section["non_blocking"],
        encoder_input_sizes=sizes,
        positional_interpolation=interpolation,
    )
