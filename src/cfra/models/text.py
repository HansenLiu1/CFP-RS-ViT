"""Offline shared-space text encoders for the registered vision models."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from .adapters import (
    ModelAdapterError,
    _autocast_context,
    _effective_precision,
    _require_torch,
    resolve_device,
)
from .assets import ModelAsset, verify_registered_model_asset
from .runtime import ModelRuntimeConfig
from .specs import get_model_spec

DEFAULT_TEXT_TEMPLATES = (
    "a satellite image of a {class_name}.",
    "an overhead remote-sensing image of a {class_name}.",
    "a remote-sensing image containing a {class_name}.",
    "a centered crop of a {class_name} in a satellite image.",
)

SHARED_SPACE_IDS = {
    "remoteclip": "openclip:ViT-B-32:projected-512",
    "clip_shared": "transformers:CLIP-ViT-B-16:projected-512",
    "siglip2": "transformers:SigLIP2-base-patch16-224:text-768",
}


@dataclass(frozen=True)
class EncodedTextFeatures:
    """Text embeddings bound to a tokenizer, context length and shared space."""

    values: Any
    model_id: str
    shared_space_id: str
    dimension: int
    pooling: str
    projection: str
    text_shared_space: bool
    l2_normalized: bool
    tokenizer_id: str
    context_length: int
    texts: tuple[str, ...] = ()
    class_names: tuple[str, ...] = ()
    template_count: int = 0
    template_ensemble: str | None = None
    checkpoint_revision: str | None = None
    checkpoint_sha256: str = ""
    loader_config_sha256: str = ""
    asset_fingerprint: str = ""
    runtime_fingerprint: str = ""
    inference_precision: str = ""


def _offline_environment() -> None:
    # Explicit local-only flags protect callers that did not set the project-wide
    # offline environment before constructing an encoder.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _positive_context_length(value: Any, model_id: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelAdapterError(f"{model_id} text tower did not expose a positive context length")
    return value


def _projection_dimension(model: Any, model_id: str, expected: int) -> int:
    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "projection_dim", None),
        getattr(getattr(config, "text_config", None), "projection_dim", None),
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(model, "output_dim", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, int) and candidate > 0:
            if candidate != expected:
                raise ModelAdapterError(
                    f"{model_id} text projection dimension {candidate} does not match registered dimension {expected}"
                )
            return candidate
    projection = getattr(model, "text_projection", None)
    if projection is not None and getattr(projection, "shape", None) is not None:
        shape = tuple(projection.shape)
        candidate = int(shape[-1])
        if candidate != expected:
            raise ModelAdapterError(
                f"{model_id} text projection dimension {candidate} does not match registered dimension {expected}"
            )
        return candidate
    return expected


def _local_tokenizer(loader: Any, asset: ModelAsset, model_id: str, **kwargs: Any) -> Any:
    try:
        tokenizer = loader.from_pretrained(asset.path, local_files_only=True, **kwargs)
    except Exception as exc:
        raise ModelAdapterError(
            f"{model_id} has no usable local tokenizer; refusing to download one from the network"
        ) from exc
    if tokenizer is None or not callable(tokenizer):
        raise ModelAdapterError(f"{model_id} tokenizer is missing or not callable")
    return tokenizer


def _runtime_values(model_id: str, runtime: ModelRuntimeConfig | None) -> tuple[str, int, str]:
    if runtime is None:
        return "fp32", 32, ""
    if model_id not in runtime.encoder_input_sizes and model_id == "clip_shared" and "clip" in runtime.encoder_input_sizes:
        # Text loading does not need an image size; the alias still shares the
        # checkpoint identity with CLIP and may be absent from image runtime config.
        pass
    return runtime.precision, runtime.batch_size, runtime.fingerprint


class SharedTextEncoder:
    """Frozen official text tower paired with one registered shared image space."""

    def __init__(
        self,
        *,
        model_id: str,
        model: Any,
        tokenizer: Any,
        shared_space_id: str,
        dimension: int,
        context_length: int,
        tokenizer_id: str,
        asset: ModelAsset,
        device: Any,
        precision: str = "fp32",
        batch_size: int = 32,
        runtime_fingerprint: str = "",
        tokenizer_kind: str = "transformers",
    ) -> None:
        if model_id not in SHARED_SPACE_IDS:
            raise ModelAdapterError(f"Text encoder is unsupported for non-shared model: {model_id}")
        if shared_space_id != SHARED_SPACE_IDS[model_id]:
            raise ModelAdapterError(
                f"{model_id} text shared_space_id {shared_space_id!r} does not match {SHARED_SPACE_IDS[model_id]!r}"
            )
        if not callable(tokenizer):
            raise ModelAdapterError(f"{model_id} tokenizer is missing or not callable")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ModelAdapterError("text batch_size must be a positive integer")
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.shared_space_id = shared_space_id
        self.dimension = dimension
        self.context_length = _positive_context_length(context_length, model_id)
        self.tokenizer_id = tokenizer_id
        self.asset = asset
        self.device = device
        self.precision = precision
        self.batch_size = batch_size
        self.runtime_fingerprint = runtime_fingerprint
        self.tokenizer_kind = tokenizer_kind
        self._closed = False
        try:
            self.model.eval()
            self.model.requires_grad_(False)
            self.model.to(device)
        except Exception:
            self._closed = True
            raise

    def _tokenize(self, texts: Sequence[str]) -> Mapping[str, Any] | Any:
        if not texts:
            raise ModelAdapterError("text input cannot be empty")
        if self.tokenizer_kind == "open_clip":
            try:
                tokens = self.tokenizer(texts, context_length=self.context_length, truncate=False)
            except TypeError:
                try:
                    tokens = self.tokenizer(texts, context_length=self.context_length)
                except Exception as exc:
                    raise ModelAdapterError(f"{self.model_id} tokenizer failed for context length {self.context_length}") from exc
            except Exception as exc:
                raise ModelAdapterError(f"{self.model_id} tokenizer failed for context length {self.context_length}") from exc
            if isinstance(tokens, Mapping):
                input_ids = tokens.get("input_ids")
                if input_ids is None:
                    raise ModelAdapterError(f"{self.model_id} tokenizer did not return input_ids")
            else:
                input_ids = tokens
            return input_ids

        try:
            encoded = self.tokenizer(
                list(texts),
                padding="max_length",
                truncation=True,
                max_length=self.context_length,
                return_tensors="pt",
            )
        except Exception as exc:
            raise ModelAdapterError(
                f"{self.model_id} tokenizer failed for explicit context length {self.context_length}"
            ) from exc
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ModelAdapterError(f"{self.model_id} tokenizer must return an input_ids mapping")
        return encoded

    def _move_tokens(self, tokens: Mapping[str, Any] | Any) -> Mapping[str, Any] | Any:
        if isinstance(tokens, Mapping):
            moved = {}
            for key, value in tokens.items():
                moved[key] = value.to(self.device) if hasattr(value, "to") else value
            return moved
        return tokens.to(self.device) if hasattr(tokens, "to") else tokens

    def _text_features(self, tokens: Mapping[str, Any] | Any) -> Any:
        if self.model_id == "remoteclip":
            if not hasattr(self.model, "encode_text"):
                raise ModelAdapterError("RemoteCLIP/OpenCLIP model has no official encode_text method")
            try:
                return self.model.encode_text(tokens, normalize=False)
            except TypeError:
                try:
                    return self.model.encode_text(tokens)
                except Exception as exc:
                    raise ModelAdapterError(f"{self.model_id} official text tower failed") from exc
            except Exception as exc:
                raise ModelAdapterError(f"{self.model_id} official text tower failed") from exc
        if not hasattr(self.model, "get_text_features"):
            raise ModelAdapterError(f"{self.model_id} model has no official get_text_features method")
        if not isinstance(tokens, Mapping):
            tokens = {"input_ids": tokens}
        try:
            return self.model.get_text_features(**tokens)
        except TypeError:
            allowed = {key: value for key, value in tokens.items() if key in {"input_ids", "attention_mask"}}
            try:
                return self.model.get_text_features(**allowed)
            except Exception as nested_exc:
                raise ModelAdapterError(f"{self.model_id} official text tower rejected tokenizer outputs") from nested_exc
        except Exception as exc:
            raise ModelAdapterError(f"{self.model_id} official text tower failed") from exc

    @staticmethod
    def _unit_normalize(values: Any) -> Any:
        torch = _require_torch()
        norms = values.norm(dim=-1, keepdim=True)
        if torch.any(norms <= 1.0e-12):
            raise ModelAdapterError("text encoder returned a near-zero embedding")
        return values / norms

    def encode(
        self,
        texts: Sequence[str] | str,
        *,
        normalize: bool = True,
        output_device: str = "cpu",
    ) -> EncodedTextFeatures:
        if self._closed:
            raise ModelAdapterError(f"Text encoder is closed: {self.model_id}")
        values = [texts] if isinstance(texts, str) else list(texts)
        if not values or any(not isinstance(text, str) or not text for text in values):
            raise ModelAdapterError("texts must be a non-empty sequence of non-empty strings")
        torch = _require_torch()
        chunks = []
        with torch.inference_mode(), _autocast_context(self.device, self.precision):
            for start in range(0, len(values), self.batch_size):
                token_batch = self._move_tokens(self._tokenize(values[start : start + self.batch_size]))
                input_ids = token_batch.get("input_ids") if isinstance(token_batch, Mapping) else token_batch
                if getattr(input_ids, "ndim", None) != 2:
                    raise ModelAdapterError(f"{self.model_id} tokenizer must return [batch, context_length] input_ids")
                if input_ids.shape[0] != min(self.batch_size, len(values) - start) or input_ids.shape[1] != self.context_length:
                    raise ModelAdapterError(
                        f"{self.model_id} tokenizer returned shape {tuple(input_ids.shape)}, expected "
                        f"[{min(self.batch_size, len(values) - start)}, {self.context_length}]"
                    )
                encoded = self._text_features(token_batch)
                if getattr(encoded, "ndim", None) != 2 or encoded.shape[0] != input_ids.shape[0]:
                    raise ModelAdapterError(f"{self.model_id} text tower returned an invalid feature matrix")
                if encoded.shape[1] != self.dimension:
                    raise ModelAdapterError(
                        f"{self.model_id} text tower returned dimension {encoded.shape[1]}, expected {self.dimension}"
                    )
                if not torch.isfinite(encoded).all():
                    raise ModelAdapterError(f"{self.model_id} text tower returned non-finite features")
                chunks.append(encoded)
        features = torch.cat(chunks, dim=0)
        if normalize:
            features = self._unit_normalize(features)
        features = features.to(device=output_device, dtype=torch.float32)
        return EncodedTextFeatures(
            values=features,
            model_id=self.model_id,
            shared_space_id=self.shared_space_id,
            dimension=self.dimension,
            pooling="official_text_pooler",
            projection="official_text_projection",
            text_shared_space=True,
            l2_normalized=normalize,
            tokenizer_id=self.tokenizer_id,
            context_length=self.context_length,
            texts=tuple(values),
            checkpoint_revision=self.asset.revision,
            checkpoint_sha256=self.asset.checkpoint_sha256,
            loader_config_sha256=self.asset.loader_config_sha256,
            asset_fingerprint=self.asset.asset_fingerprint,
            runtime_fingerprint=self.runtime_fingerprint,
            inference_precision=_effective_precision(self.device, self.precision),
        )

    def encode_class_prototypes(
        self,
        class_names: Sequence[str],
        templates: Sequence[str] = DEFAULT_TEXT_TEMPLATES,
        *,
        output_device: str = "cpu",
    ) -> EncodedTextFeatures:
        if isinstance(class_names, (str, bytes)):
            raise ModelAdapterError("class_names must be a non-empty sequence of non-empty strings")
        names = list(class_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ModelAdapterError("class_names must be a non-empty sequence of non-empty strings")
        selected_templates = tuple(templates)
        if len(selected_templates) != 4 or len(set(selected_templates)) != 4:
            raise ModelAdapterError("text prototype ensemble requires exactly four templates")
        if any(not isinstance(template, str) or "{class_name}" not in template for template in selected_templates):
            raise ModelAdapterError("every text prototype template must contain the {class_name} placeholder")
        try:
            prompts = tuple(template.format(class_name=name) for name in names for template in selected_templates)
        except (IndexError, KeyError, ValueError) as exc:
            raise ModelAdapterError("text prototype templates must be valid format strings") from exc
        encoded = self.encode(prompts, normalize=True, output_device=output_device)
        torch = _require_torch()
        values = encoded.values.reshape(len(names), len(selected_templates), self.dimension)
        values = self._unit_normalize(values.mean(dim=1))
        values = values.to(device=output_device, dtype=torch.float32)
        return EncodedTextFeatures(
            values=values,
            model_id=self.model_id,
            shared_space_id=self.shared_space_id,
            dimension=self.dimension,
            pooling="template_ensemble_mean",
            projection="official_text_projection",
            text_shared_space=True,
            l2_normalized=True,
            tokenizer_id=self.tokenizer_id,
            context_length=self.context_length,
            texts=prompts,
            class_names=tuple(names),
            template_count=4,
            template_ensemble="mean_unit_normalized_text_embeddings",
            checkpoint_revision=self.asset.revision,
            checkpoint_sha256=self.asset.checkpoint_sha256,
            loader_config_sha256=self.asset.loader_config_sha256,
            asset_fingerprint=self.asset.asset_fingerprint,
            runtime_fingerprint=self.runtime_fingerprint,
            inference_precision=encoded.inference_precision,
        )

    def encode_text(
        self,
        texts: Sequence[str] | str,
        *,
        normalize: bool = True,
        output_device: str = "cpu",
    ) -> EncodedTextFeatures:
        """Explicit alias for callers that mirror the backend text-tower API."""

        return self.encode(texts, normalize=normalize, output_device=output_device)

    def encode_prototypes(
        self,
        class_names: Sequence[str],
        templates: Sequence[str] = DEFAULT_TEXT_TEMPLATES,
        *,
        output_device: str = "cpu",
    ) -> EncodedTextFeatures:
        return self.encode_class_prototypes(class_names, templates, output_device=output_device)

    def assert_shared_space(self, shared_space_id: str) -> None:
        if shared_space_id != self.shared_space_id:
            raise ModelAdapterError(
                f"Shared text space mismatch: vision={shared_space_id!r}, text={self.shared_space_id!r}"
            )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.model.to("cpu")
        finally:
            self.model = None
            self._closed = True

    @property
    def is_closed(self) -> bool:
        return self._closed

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SharedTextPrototypeBaseline:
    """E08 scorer backed by official shared-space text prototypes.

    The scorer fits only the class keys observed in ``readout_fit`` metadata. GT
    labels from confirmation candidates are never read by ``score``. Class names
    are keyed as ``dataset_id:class_id`` so dataset-local integer IDs cannot
    collide when a caller intentionally combines datasets.
    """

    def __init__(
        self,
        encoder: SharedTextEncoder,
        class_names: Mapping[str, str],
        *,
        templates: Sequence[str] = DEFAULT_TEXT_TEMPLATES,
        centered: bool = True,
        center_image_features: bool = True,
    ) -> None:
        if not isinstance(centered, (bool, np.bool_)):
            raise ModelAdapterError("shared text baseline centered must be boolean")
        if not isinstance(center_image_features, (bool, np.bool_)):
            raise ModelAdapterError("shared text baseline center_image_features must be boolean")
        self.encoder = encoder
        self.shared_space_id = encoder.shared_space_id
        self.class_names = {str(key): str(value) for key, value in class_names.items()}
        self.templates = tuple(templates)
        # Modality centering removes each modality's own mean inside the shared
        # space before cosine scoring; ``centered=False`` restores the original
        # uncentered baseline.  ``center_image_features`` only applies while
        # centering is enabled and can be disabled to keep images in raw space.
        self.centered = bool(centered)
        self.center_image_features = bool(center_image_features)
        self.space: str | None = "centered" if self.centered else "raw"
        if len(self.templates) != 4 or len(set(self.templates)) != 4:
            raise ModelAdapterError("shared text baseline requires exactly four prompt templates")

    @staticmethod
    def _fixed_center(values: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Subtract a fixed reference mean, keeping rows that would collapse.

        A query/prototype row equal to the reference mean would map to the origin,
        which has no cosine direction; such rows keep their raw values so scoring
        stays finite (matching the historical degenerate-batch guard).
        """

        centered = values - center
        norms = np.linalg.norm(centered, axis=1)
        degenerate = norms <= 1.0e-12
        if np.any(degenerate):
            centered = centered.copy()
            centered[degenerate] = values[degenerate]
        return centered

    @staticmethod
    def _field(row: Any, name: str, default: Any = None) -> Any:
        if isinstance(row, Mapping):
            return row.get(name, default)
        return getattr(row, name, default)

    def _class_key(self, row: Any) -> str:
        dataset_id = self._field(row, "dataset_id")
        class_id = self._field(row, "class_id")
        if dataset_id is None or class_id is None:
            raise ModelAdapterError("text baseline metadata requires dataset_id and class_id")
        return f"{dataset_id}:{class_id}"

    def fit(self, _features: Any, _labels: Any, metadata: Sequence[Any]) -> dict[str, Any]:
        keys = sorted({self._class_key(row) for row in metadata})
        missing = [key for key in keys if key not in self.class_names]
        if missing:
            raise ModelAdapterError(f"text baseline class mapping is missing: {missing[:5]}")
        names = [self.class_names[key] for key in keys]
        prototypes = self.encoder.encode_class_prototypes(names, self.templates, output_device="cpu")
        values = prototypes.values
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=np.float64)
        # Modality means are fixed at fit time from non-confirmation data: the
        # text prototype mean and the readout_fit image mean.  Centering by the
        # evaluation (confirmation) batch mean would let evaluation data define
        # the scoring origin and is deliberately avoided.
        text_center = values.mean(axis=0) if values.shape[0] else None
        image_center = None
        feature_matrix = np.asarray(_features, dtype=np.float64) if _features is not None else None
        if feature_matrix is not None and feature_matrix.ndim == 2 and feature_matrix.shape[0]:
            image_center = feature_matrix.mean(axis=0)
        return {
            "class_keys": tuple(keys),
            "prototypes": values,
            "shared_space_id": self.encoder.shared_space_id,
            "space": self.space,
            "centered": self.centered,
            "center_image_features": self.center_image_features,
            "text_center": text_center,
            "image_center": image_center,
        }

    def score(
        self,
        query_features: Any,
        candidate_features: Any,
        _query_metadata: Sequence[Any],
        _candidate_metadata: Sequence[Any],
        fitted: Mapping[str, Any],
    ) -> np.ndarray:
        query = np.asarray(query_features, dtype=np.float64)
        candidates = np.asarray(candidate_features, dtype=np.float64)
        prototypes = np.asarray(fitted["prototypes"], dtype=np.float64)
        if query.ndim != 2 or candidates.ndim != 2 or prototypes.ndim != 2:
            raise ModelAdapterError("text baseline features must be two-dimensional matrices")
        if query.shape[1] != prototypes.shape[1] or candidates.shape[1] != prototypes.shape[1]:
            raise ModelAdapterError("image and text embeddings are not in the same shared space")
        def unit(values: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if np.any(norms <= 1.0e-12):
                raise ModelAdapterError("text baseline received a near-zero image/profile vector")
            return values / norms
        if self.centered:
            text_center = fitted.get("text_center")
            if text_center is not None:
                prototypes = self._fixed_center(prototypes, np.asarray(text_center, dtype=np.float64))
            if self.center_image_features:
                image_center = fitted.get("image_center")
                if image_center is not None:
                    reference = np.asarray(image_center, dtype=np.float64)
                    query = self._fixed_center(query, reference)
                    candidates = self._fixed_center(candidates, reference)
                # Without a fixed readout-fit reference the image side is left
                # uncentered: defining the origin from the evaluation batch mean
                # would leak confirmation statistics into scoring.
        query_profile = unit(unit(query) @ unit(prototypes).T)
        candidate_profile = unit(unit(candidates) @ unit(prototypes).T)
        return query_profile @ candidate_profile.T


def _build_shared_encoder(
    *,
    model_id: str,
    model: Any,
    tokenizer: Any,
    asset: ModelAsset,
    device: Any,
    runtime: ModelRuntimeConfig | None,
    context_length: int,
    dimension: int,
    tokenizer_id: str,
    tokenizer_kind: str,
) -> SharedTextEncoder:
    precision, batch_size, runtime_fingerprint = _runtime_values(model_id, runtime)
    return SharedTextEncoder(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        shared_space_id=SHARED_SPACE_IDS[model_id],
        dimension=dimension,
        context_length=context_length,
        tokenizer_id=tokenizer_id,
        asset=asset,
        device=device,
        precision=precision,
        batch_size=batch_size,
        runtime_fingerprint=runtime_fingerprint,
        tokenizer_kind=tokenizer_kind,
    )


def _load_remoteclip_text(
    asset: ModelAsset,
    *,
    device: Any,
    runtime: ModelRuntimeConfig | None,
) -> SharedTextEncoder:
    try:
        import open_clip
    except ImportError as exc:
        raise ModelAdapterError("RemoteCLIP/OpenCLIP text support requires open_clip_torch") from exc
    try:
        from cfra.provenance import sha256_json

        loader_config = open_clip.get_model_config("ViT-B-32")
        if sha256_json(loader_config) != asset.loader_config_sha256:
            raise ModelAdapterError("Installed OpenCLIP ViT-B-32 configuration differs from registered identity")
        model = open_clip.create_model(
            "ViT-B-32",
            pretrained=str(asset.path),
            device="cpu",
            precision="fp32",
            weights_only=True,
            require_pretrained=True,
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
    except ModelAdapterError:
        raise
    except Exception as exc:
        raise ModelAdapterError("RemoteCLIP/OpenCLIP local model or tokenizer could not be loaded") from exc
    if tokenizer is None or not callable(tokenizer):
        raise ModelAdapterError("RemoteCLIP/OpenCLIP tokenizer is unavailable")
    context_length = _positive_context_length(getattr(model, "context_length", None), "remoteclip")
    dimension = _projection_dimension(model, "remoteclip", get_model_spec("remoteclip").dimension)
    return _build_shared_encoder(
        model_id="remoteclip",
        model=model,
        tokenizer=tokenizer,
        asset=asset,
        device=device,
        runtime=runtime,
        context_length=context_length,
        dimension=dimension,
        tokenizer_id="openclip:ViT-B-32",
        tokenizer_kind="open_clip",
    )


def _load_transformers_text(
    model_id: str,
    asset: ModelAsset,
    *,
    device: Any,
    runtime: ModelRuntimeConfig | None,
) -> SharedTextEncoder:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ModelAdapterError(f"{model_id} text support requires transformers") from exc
    tokenizer_kind = "transformers"
    try:
        tokenizer = _local_tokenizer(AutoTokenizer, asset, model_id, use_fast=True)
    except ModelAdapterError:
        if model_id != "clip_shared":
            raise
        # Some offline CLIP snapshots contain weights/config but no tokenizer
        # files. OpenCLIP ships the verified CLIP BPE tokenizer as package data;
        # using it here keeps the text path offline and records the backend.
        try:
            import open_clip

            tokenizer = open_clip.get_tokenizer("ViT-B-16")
        except Exception as exc:
            raise ModelAdapterError(
                "clip_shared has no local Transformers tokenizer and the offline OpenCLIP tokenizer is unavailable"
            ) from exc
        tokenizer_kind = "open_clip"

    try:
        if model_id == "clip_shared":
            from transformers import CLIPModel

            model, info = CLIPModel.from_pretrained(
                asset.path,
                local_files_only=True,
                use_safetensors=False,
                output_loading_info=True,
                weights_only=True,
            )
            expected_dimension = get_model_spec(model_id).dimension
            context_length = _positive_context_length(
                getattr(getattr(model.config, "text_config", None), "max_position_embeddings", None),
                model_id,
            )
        elif model_id == "siglip2":
            from transformers import SiglipModel

            model, info = SiglipModel.from_pretrained(
                asset.path,
                local_files_only=True,
                use_safetensors=True,
                output_loading_info=True,
                weights_only=True,
            )
            expected_dimension = get_model_spec(model_id).dimension
            context_length = _positive_context_length(
                getattr(getattr(model.config, "text_config", None), "max_position_embeddings", None),
                model_id,
            )
        else:
            raise ModelAdapterError(f"No official Transformers text loader is registered for {model_id}")
        _validate_transformers_text_loading_info(info, model_id)
    except ModelAdapterError:
        raise
    except (ImportError, OSError, ValueError, TypeError) as exc:
        raise ModelAdapterError(f"{model_id} local text tower could not be loaded offline") from exc

    dimension = _projection_dimension(model, model_id, expected_dimension)
    tokenizer_id = "openclip:ViT-B-16" if tokenizer_kind == "open_clip" else f"transformers:{type(tokenizer).__name__}"
    return _build_shared_encoder(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        asset=asset,
        device=device,
        runtime=runtime,
        context_length=context_length,
        dimension=dimension,
        tokenizer_id=tokenizer_id,
        tokenizer_kind=tokenizer_kind,
    )


def _validate_transformers_text_loading_info(info: Any, model_id: str) -> None:
    if not isinstance(info, Mapping):
        return
    for key in ("missing_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ModelAdapterError(f"{model_id} text checkpoint loading errors: {info[key][:5]}")
    if info.get("unexpected_keys"):
        raise ModelAdapterError(f"{model_id} text checkpoint has unexpected keys: {info['unexpected_keys'][:5]}")


def load_shared_text_encoder(
    model_id: str,
    asset: ModelAsset,
    *,
    registry_root: Path | str,
    runtime: ModelRuntimeConfig | None = None,
    device: str = "auto",
    expected_shared_space_id: str | None = None,
) -> SharedTextEncoder:
    """Load an official shared-space text tower from local checkpoint files only."""

    _offline_environment()
    try:
        spec = get_model_spec(model_id)
    except ValueError as exc:
        raise ModelAdapterError(f"Text baseline is unsupported for unknown model: {model_id!r}") from exc
    if not spec.text_shared_space or model_id not in SHARED_SPACE_IDS:
        raise ModelAdapterError(
            f"Text baseline is unsupported for {model_id}: the registered representation is not a shared text space"
        )
    if asset.asset_id != spec.asset_id:
        raise ModelAdapterError(f"Asset {asset.asset_id} cannot load text representation {model_id}")
    verify_registered_model_asset(asset, model_id, registry_root)
    shared_space_id = SHARED_SPACE_IDS[model_id]
    if expected_shared_space_id is not None and expected_shared_space_id != shared_space_id:
        raise ModelAdapterError(
            f"Requested shared_space_id {expected_shared_space_id!r} does not match {shared_space_id!r}"
        )
    resolved_device = resolve_device(device)
    if runtime is not None:
        if runtime.offline_only is not True:
            raise ModelAdapterError("Shared text encoder requires offline_only runtime")
        if runtime.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ModelAdapterError(f"Unsupported inference precision: {runtime.precision!r}")
        if resolved_device.type != "cuda" and runtime.precision in {"fp16", "bf16"}:
            raise ModelAdapterError(f"{runtime.precision} inference is only enabled for CUDA devices")

    if model_id == "remoteclip":
        return _load_remoteclip_text(asset, device=resolved_device, runtime=runtime)
    return _load_transformers_text(model_id, asset, device=resolved_device, runtime=runtime)


__all__ = [
    "DEFAULT_TEXT_TEMPLATES",
    "SHARED_SPACE_IDS",
    "EncodedTextFeatures",
    "SharedTextEncoder",
    "SharedTextPrototypeBaseline",
    "load_shared_text_encoder",
]
