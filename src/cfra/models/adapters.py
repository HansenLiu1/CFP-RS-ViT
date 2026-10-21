"""Lazy, offline-only adapters for the registered frozen vision encoders."""

from __future__ import annotations

import gc
import math
import re
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assets import ModelAsset, verify_registered_model_asset
from .preprocessing import preprocess_image_batch
from .runtime import ModelRuntimeConfig
from .specs import ModelSpec, get_model_spec


class ModelAdapterError(RuntimeError):
    """Raised when a model cannot satisfy its registered representation contract."""


@dataclass(frozen=True)
class EncodedFeatures:
    values: Any
    model_id: str
    representation_id: str
    dimension: int
    pooling: str
    projection: str
    text_shared_space: bool
    l2_normalized: bool
    encoder_input_size: int
    checkpoint_revision: str | None
    checkpoint_sha256: str
    loader_config_sha256: str
    asset_fingerprint: str
    runtime_fingerprint: str
    inference_precision: str
    shared_space_id: str | None = None


@dataclass(frozen=True)
class LayerwiseEncodedFeatures:
    """Layer-specific pooled features with an auditable representation contract.

    ``values`` keeps the selected layer names in caller order.  Every value is a
    two-dimensional tensor with shape ``[batch, dimension]``; ``metadata`` has
    one JSON-compatible representation record for the corresponding layer.
    """

    values: Mapping[str, Any]
    metadata: Mapping[str, Mapping[str, Any]]
    model_id: str = ""
    representation_id: str = ""
    encoder_input_size: int | None = None
    checkpoint_revision: str | None = None
    checkpoint_sha256: str = ""
    loader_config_sha256: str = ""
    asset_fingerprint: str = ""
    runtime_fingerprint: str = ""
    inference_precision: str = ""

    @property
    def features(self) -> Mapping[str, Any]:
        """Alias used by the pipeline layer-record contract."""

        return self.values

    @property
    def layers(self) -> Mapping[str, Any]:
        """Alias that makes the selected layer mapping explicit."""

        return self.values

    @property
    def selected_layers(self) -> tuple[str, ...]:
        return tuple(self.values.keys())

    @property
    def layer_features(self) -> Mapping[str, Any]:
        return self.values


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Model adapters require the 'models' optional dependencies") from exc
    return torch


def resolve_device(device: str) -> Any:
    """Resolve device string to torch.device (single GPU only).
    
    Args:
        device: "auto" (use cuda:0 if available, else cpu), "cuda:0", or "cpu"
        
    Returns:
        torch.device instance
    """
    torch = _require_torch()
    
    # Auto mode: use cuda:0 if available, else cpu
    if device == "auto":
        selected = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        selected = device
    
    resolved = torch.device(selected)
    
    # Validate CUDA availability
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ModelAdapterError(f"CUDA device requested but CUDA is unavailable: {device}")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ModelAdapterError(f"CUDA device index is unavailable: {device}")
    
    if resolved.type not in {"cpu", "cuda"}:
        raise ModelAdapterError(f"Unsupported model execution device: {resolved}")
    
    return resolved


def _effective_precision(device: Any, precision: str) -> str:
    return "fp16" if precision == "auto" and device.type == "cuda" else "fp32" if precision == "auto" else precision


def _autocast_context(device: Any, precision: str) -> Any:
    torch = _require_torch()
    precision = _effective_precision(device, precision)
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ModelAdapterError(f"{precision} inference is only enabled for CUDA devices")
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ModelAdapterError(f"The selected CUDA device does not support bfloat16: {device}")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ModelAdapterError(f"Unsupported inference precision: {precision!r}")


def _validate_loading_info(info: dict[str, Any], allowed_unexpected_prefixes: tuple[str, ...]) -> None:
    if info.get("missing_keys"):
        raise ModelAdapterError(f"Checkpoint has missing model keys: {info['missing_keys'][:5]}")
    if info.get("mismatched_keys"):
        raise ModelAdapterError(f"Checkpoint has mismatched model keys: {info['mismatched_keys'][:5]}")
    if info.get("error_msgs"):
        raise ModelAdapterError(f"Checkpoint loading errors: {info['error_msgs'][:5]}")
    unexpected = [
        key
        for key in info.get("unexpected_keys", [])
        if not any(key == prefix or key.startswith(prefix) for prefix in allowed_unexpected_prefixes)
    ]
    if unexpected:
        raise ModelAdapterError(f"Checkpoint has unexpected model keys: {unexpected[:5]}")


def _selected_layer_index(layer_name: str, hidden_state_count: int, block_count: int | None = None) -> tuple[str, int | None]:
    """Resolve the small public layer vocabulary to hidden-state indices."""

    normalized = str(layer_name)
    lowered = normalized.lower()
    if lowered in {"patch_emb", "patch_embedding", "embedding", "embeddings"}:
        return "patch_emb", 0
    if lowered == "final":
        return "final", hidden_state_count - 1
    match = re.fullmatch(r"(?:l|block[_-]?)(\d+)", lowered)
    if match is None:
        raise ModelAdapterError(
            f"Unsupported layer name {layer_name!r}; expected patch_emb, L01..L{{n}}, or final"
        )
    block = int(match.group(1))
    if block <= 0:
        raise ModelAdapterError(f"Transformer block numbers are one-based: {layer_name!r}")
    if block_count is not None and block > block_count:
        raise ModelAdapterError(f"Requested {layer_name!r}, but the model exposes only {block_count} transformer blocks")
    if block >= hidden_state_count:
        raise ModelAdapterError(
            f"Requested {layer_name!r}, but hidden_states contains only {hidden_state_count - 1} transformer blocks"
        )
    return "block", block


def _hidden_states_from(outputs: Any, model_id: str) -> tuple[Any, ...]:
    hidden_states = outputs.get("hidden_states") if isinstance(outputs, Mapping) else getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ModelAdapterError(f"{model_id} layerwise extraction requires output_hidden_states=True")
    try:
        result = tuple(hidden_states)
    except TypeError as exc:
        raise ModelAdapterError(f"{model_id} returned a non-sequence hidden_states value") from exc
    if not result:
        raise ModelAdapterError(f"{model_id} returned no hidden states")
    return result


def _normalise_tokens(tokens: Any, batch_size: int, model_id: str) -> Any:
    if getattr(tokens, "ndim", None) != 3:
        raise ModelAdapterError(f"{model_id} layer output must have shape [batch, tokens, dim]")
    if tokens.shape[0] == batch_size:
        return tokens
    if tokens.shape[1] == batch_size:
        return tokens.permute(1, 0, 2)
    raise ModelAdapterError(
        f"{model_id} layer output shape {tuple(tokens.shape)} does not contain batch size {batch_size}"
    )


def _pool_tokens(tokens: Any, pooling: str, has_cls_token: bool, model_id: str) -> Any:
    if pooling == "cls":
        if not has_cls_token or tokens.shape[1] < 1:
            raise ModelAdapterError(f"{model_id} does not expose a CLS token for layerwise CLS pooling")
        return tokens[:, 0, :]
    if pooling == "patch_mean":
        start = 1 if has_cls_token else 0
        if tokens.shape[1] <= start:
            raise ModelAdapterError(f"{model_id} has no patch tokens for layerwise patch pooling")
        return tokens[:, start:, :].mean(dim=1)
    raise ModelAdapterError(f"Unsupported layerwise pooling strategy: {pooling!r}")


def _grid_metadata(token_count: int, has_cls_token: bool, patch_size: int) -> dict[str, Any]:
    patch_count = token_count - 1 if has_cls_token else token_count
    side = int(math.sqrt(patch_count)) if patch_count >= 0 else 0
    square = side * side == patch_count
    return {
        "height": side if square else None,
        "width": side if square else None,
        "patch_count": int(patch_count),
        "patch_size": int(patch_size),
        "square": bool(square),
    }


def _layer_metadata(
    adapter: VisionEncoderAdapter,
    *,
    kind: str,
    block: int | None,
    tokens: Any,
    pooled: Any,
    pooling: str,
    projection: str,
    text_shared_space: bool,
    has_cls_token: bool,
) -> dict[str, Any]:
    token_count = int(tokens.shape[1])
    target_size = int(adapter.encoder_input_size)
    native_size = int(adapter.spec.checkpoint_native_input_size)
    target_grid = target_size // int(adapter.spec.patch_size)
    source_grid = native_size // int(adapter.spec.patch_size)
    return {
        "kind": kind,
        "block": None if block is None else int(block),
        "dim": int(pooled.shape[1]),
        "dimension": int(pooled.shape[1]),
        "pooling": pooling,
        "projection": projection,
        "text_shared_space": bool(text_shared_space),
        "token": {
            "count": token_count,
            "includes_cls": bool(has_cls_token),
            "pooled": (
                "cls"
                if pooling in {"cls", "post_ln_cls"}
                else "map"
                if pooling == "map_attention"
                else "patches"
            ),
        },
        "token_count": token_count,
        "grid": _grid_metadata(token_count, has_cls_token, adapter.spec.patch_size),
        "interpolation": {
            "enabled": bool(target_size != native_size),
            "source_size": native_size,
            "target_size": target_size,
            "source_grid": [source_grid, source_grid],
            "target_grid": [target_grid, target_grid],
            "mode": "position_encoding" if target_size != native_size else "none",
        },
    }


def _hidden_state_layerwise(
    adapter: VisionEncoderAdapter,
    hidden_states: tuple[Any, ...],
    selected_layers: Sequence[str],
    *,
    batch_size: int,
    has_cls_token: bool,
    internal_pooling: str,
    final_tokens: Any | None = None,
    final_pooled: Any | None = None,
    final_pooling: str | None = None,
    final_projection: str | None = None,
    final_shared_space: bool | None = None,
    block_count: int | None = None,
) -> dict[str, tuple[Any, dict[str, Any]]]:
    outputs: dict[str, tuple[Any, dict[str, Any]]] = {}
    final_tokens = hidden_states[-1] if final_tokens is None else final_tokens
    final_tokens = _normalise_tokens(final_tokens, batch_size, adapter.spec.model_id)
    for layer_name in selected_layers:
        kind, index = _selected_layer_index(layer_name, len(hidden_states), block_count)
        if kind == "patch_emb":
            tokens = _normalise_tokens(hidden_states[0], batch_size, adapter.spec.model_id)
            pooled = _pool_tokens(tokens, "patch_mean", has_cls_token, adapter.spec.model_id)
            pooling = "patch_mean"
            projection = "none"
            shared_space = False
        elif kind == "final":
            tokens = final_tokens
            pooled = final_pooled
            pooling = final_pooling or internal_pooling
            projection = final_projection if final_projection is not None else "none"
            shared_space = adapter.spec.text_shared_space if final_shared_space is None else final_shared_space
        else:
            assert index is not None
            tokens = _normalise_tokens(hidden_states[index], batch_size, adapter.spec.model_id)
            pooled = _pool_tokens(tokens, internal_pooling, has_cls_token, adapter.spec.model_id)
            pooling = internal_pooling
            projection = "none"
            shared_space = False
        if pooled is None:
            pooled = _pool_tokens(tokens, pooling, has_cls_token, adapter.spec.model_id)
        if pooled.ndim != 2 or pooled.shape[0] != batch_size:
            raise ModelAdapterError(
                f"{adapter.spec.model_id} layer {layer_name!r} returned invalid pooled shape {tuple(pooled.shape)}"
            )
        outputs[str(layer_name)] = (
            pooled,
            _layer_metadata(
                adapter,
                kind="patch_emb" if kind == "patch_emb" else kind,
                block=index if kind == "block" else None,
                tokens=tokens,
                pooled=pooled,
                pooling=pooling,
                projection=projection,
                text_shared_space=shared_space,
                has_cls_token=has_cls_token,
            ),
        )
    return outputs


def _attribute(output: Any, name: str, default: Any = None) -> Any:
    if isinstance(output, Mapping):
        return output.get(name, default)
    return getattr(output, name, default)


def _hook_tensor(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        if not output:
            return None
        return output[0]
    return output


def _forward_with_layer_hooks(
    *,
    model_id: str,
    embeddings_module: Any,
    blocks: Sequence[Any],
    forward: Any,
) -> tuple[Any, tuple[Any, ...]]:
    """Capture embedding and block outputs for backends without hidden states."""

    captured: dict[int, Any] = {}
    handles = []

    def capture_embedding(_module: Any, _inputs: Any, output: Any) -> None:
        captured[0] = _hook_tensor(output)

    try:
        handles.append(embeddings_module.register_forward_hook(capture_embedding))
        for index, block in enumerate(blocks, start=1):
            def capture_block(_module: Any, _inputs: Any, output: Any, block_index: int = index) -> None:
                captured[block_index] = _hook_tensor(output)

            handles.append(block.register_forward_hook(capture_block))
    except Exception as exc:
        for handle in handles:
            handle.remove()
        raise ModelAdapterError(
            f"{model_id} layerwise extraction requires deterministic hooks on embeddings and transformer blocks"
        ) from exc
    try:
        outputs = forward()
    except Exception as exc:
        raise ModelAdapterError(f"{model_id} deterministic layerwise hooks failed during the visual forward") from exc
    finally:
        for handle in handles:
            handle.remove()
    expected = set(range(len(blocks) + 1))
    if set(captured) != expected:
        raise ModelAdapterError(f"{model_id} hooks did not capture every transformer stage")
    return outputs, tuple(captured[index] for index in range(len(blocks) + 1))


class VisionEncoderAdapter:
    """A frozen vision encoder that returns a metadata-bound global representation."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        asset: ModelAsset,
        model: Any,
        device: Any,
        runtime: ModelRuntimeConfig,
    ) -> None:
        self.spec = spec
        self.asset = asset
        self.model = model
        self.device = device
        self.runtime = runtime
        self.precision = runtime.precision
        self.encoder_input_size = runtime.encoder_input_sizes[spec.model_id]
        self._closed = False

        try:
            self.model.eval()
            self.model.requires_grad_(False)
            self.model.to(device)
        except Exception:
            self._release_model(suppress_errors=True)
            raise

    def _release_model(self, *, suppress_errors: bool) -> None:
        torch = _require_torch()
        model = self.model
        self.model = None
        self._closed = True
        try:
            if model is not None:
                model.to("cpu")
        except Exception:
            if not suppress_errors:
                raise
        finally:
            del model
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _forward_global(self, pixel_values: Any) -> Any:
        raise NotImplementedError

    def _forward_layerwise(
        self,
        pixel_values: Any,
        selected_layers: Sequence[str],
    ) -> dict[str, tuple[Any, dict[str, Any]]]:
        raise ModelAdapterError(f"Layerwise extraction is not implemented for {self.spec.model_id}")

    def encode(
        self,
        images: Any,
        *,
        output_device: str = "cpu",
    ) -> EncodedFeatures:
        if self._closed:
            raise ModelAdapterError(f"Adapter is closed: {self.spec.model_id}")

        torch = _require_torch()
        pixel_values = preprocess_image_batch(
            images,
            preprocess=self.spec.preprocess,
            input_size=self.encoder_input_size,
            device=self.device,
            pin_memory=self.runtime.pin_memory,
            non_blocking=self.runtime.non_blocking,
        )
        if pixel_values.shape[0] > self.runtime.batch_size:
            raise ModelAdapterError(
                f"Batch contains {pixel_values.shape[0]} images, exceeding configured size {self.runtime.batch_size}"
            )
        with torch.inference_mode(), _autocast_context(self.device, self.precision):
            features = self._forward_global(pixel_values)
        if features.ndim != 2 or features.shape[0] != pixel_values.shape[0]:
            raise ModelAdapterError(
                f"{self.spec.model_id} returned invalid global feature shape: {tuple(features.shape)}"
            )
        if features.shape[1] != self.spec.dimension:
            raise ModelAdapterError(
                f"{self.spec.model_id} returned dimension {features.shape[1]}, expected {self.spec.dimension}"
            )
        features = features.to(device=output_device, dtype=torch.float32)
        if not torch.isfinite(features).all():
            raise ModelAdapterError(f"{self.spec.model_id} returned non-finite global features")

        return EncodedFeatures(
            values=features,
            model_id=self.spec.model_id,
            representation_id=self.spec.representation_id,
            dimension=self.spec.dimension,
            pooling=self.spec.pooling,
            projection=self.spec.projection,
            text_shared_space=self.spec.text_shared_space,
            shared_space_id=self.spec.shared_space_id if self.spec.text_shared_space else None,
            l2_normalized=False,
            encoder_input_size=self.encoder_input_size,
            checkpoint_revision=self.asset.revision,
            checkpoint_sha256=self.asset.checkpoint_sha256,
            loader_config_sha256=self.asset.loader_config_sha256,
            asset_fingerprint=self.asset.asset_fingerprint,
            runtime_fingerprint=self.runtime.fingerprint,
            inference_precision=_effective_precision(self.device, self.precision),
        )

    def encode_layerwise(
        self,
        images: Any,
        selected_layers: Sequence[str],
        *,
        output_device: str = "cpu",
    ) -> LayerwiseEncodedFeatures:
        """Extract pooled vectors from selected transformer layers.

        The method shares the existing image preprocessing and batch contract
        with :meth:`encode`; layerwise features are returned in caller order and
        are never silently projected into the model's text space.
        """

        if self._closed:
            raise ModelAdapterError(f"Adapter is closed: {self.spec.model_id}")
        if isinstance(selected_layers, (str, bytes)):
            raise ModelAdapterError("selected_layers must be a non-empty sequence of unique layer names")
        try:
            selected = [str(layer) for layer in selected_layers]
        except TypeError as exc:
            raise ModelAdapterError("selected_layers must be a non-empty sequence of unique layer names") from exc
        if not selected or len(set(selected)) != len(selected):
            raise ModelAdapterError("selected_layers must be a non-empty sequence of unique layer names")

        torch = _require_torch()
        pixel_values = preprocess_image_batch(
            images,
            preprocess=self.spec.preprocess,
            input_size=self.encoder_input_size,
            device=self.device,
            pin_memory=self.runtime.pin_memory,
            non_blocking=self.runtime.non_blocking,
        )
        if pixel_values.shape[0] > self.runtime.batch_size:
            raise ModelAdapterError(
                f"Batch contains {pixel_values.shape[0]} images, exceeding configured size {self.runtime.batch_size}"
            )
        with torch.inference_mode(), _autocast_context(self.device, self.precision):
            extracted = self._forward_layerwise(pixel_values, selected)
        if set(extracted) != set(selected):
            missing = [layer for layer in selected if layer not in extracted]
            raise ModelAdapterError(f"Layerwise adapter did not return requested layers: {missing}")

        values: dict[str, Any] = {}
        metadata: dict[str, Mapping[str, Any]] = {}
        for layer in selected:
            features, layer_metadata = extracted[layer]
            if features.ndim != 2 or features.shape[0] != pixel_values.shape[0]:
                raise ModelAdapterError(
                    f"{self.spec.model_id} layer {layer!r} returned invalid feature shape: {tuple(features.shape)}"
                )
            features = features.to(device=output_device, dtype=torch.float32)
            if not torch.isfinite(features).all():
                raise ModelAdapterError(f"{self.spec.model_id} layer {layer!r} returned non-finite features")
            values[layer] = features
            metadata[layer] = dict(layer_metadata)

        return LayerwiseEncodedFeatures(
            values=values,
            metadata=metadata,
            model_id=self.spec.model_id,
            representation_id=self.spec.representation_id,
            encoder_input_size=self.encoder_input_size,
            checkpoint_revision=self.asset.revision,
            checkpoint_sha256=self.asset.checkpoint_sha256,
            loader_config_sha256=self.asset.loader_config_sha256,
            asset_fingerprint=self.asset.asset_fingerprint,
            runtime_fingerprint=self.runtime.fingerprint,
            inference_precision=_effective_precision(self.device, self.precision),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._release_model(suppress_errors=False)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def __enter__(self) -> VisionEncoderAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class RemoteClipAdapter(VisionEncoderAdapter):
    def _remoteclip_embeddings(self, pixel_values: Any) -> Any:
        torch = _require_torch()
        from torch.nn import functional

        visual = self.model
        if getattr(self, "encoder_input_size", None) == self.spec.checkpoint_native_input_size and hasattr(visual, "_embeds"):
            return visual._embeds(pixel_values)
        embeddings = visual.conv1(pixel_values)
        grid_height, grid_width = embeddings.shape[-2:]
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(dtype=embeddings.dtype).expand(embeddings.shape[0], 1, -1)
        embeddings = torch.cat([class_token, embeddings], dim=1)

        positional = visual.positional_embedding
        old_grid = int(math.sqrt(positional.shape[0] - 1))
        if old_grid * old_grid != positional.shape[0] - 1:
            raise ModelAdapterError("RemoteCLIP positional embedding is not a square patch grid")
        patch_positional = positional[1:].reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_positional = functional.interpolate(
            patch_positional.to(dtype=embeddings.dtype),
            size=(grid_height, grid_width),
            mode="bicubic",
            align_corners=False,
        )
        patch_positional = patch_positional.permute(0, 2, 3, 1).reshape(1, grid_height * grid_width, -1)
        class_positional = positional[:1].to(dtype=embeddings.dtype).view(1, 1, -1)
        embeddings = embeddings + torch.cat([class_positional, patch_positional], dim=1)
        embeddings = visual.patch_dropout(embeddings)
        return visual.ln_pre(embeddings)

    def _forward_global(self, pixel_values: Any) -> Any:
        encoder_input_size = getattr(self, "encoder_input_size", None)
        if encoder_input_size is None:
            return self.model(pixel_values)
        if encoder_input_size == self.spec.checkpoint_native_input_size:
            return self.model(pixel_values)

        # OpenCLIP's visual tower has no public position-interpolation flag.
        # Rebuild its embedding path so 448-pixel crops use a deterministic
        # interpolated patch grid rather than a shape mismatch or hidden resize.
        visual = self.model
        embeddings = self._remoteclip_embeddings(pixel_values)
        embeddings = visual.transformer(embeddings)
        pooled, _ = visual._pool(embeddings)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled

    def _forward_layerwise(
        self,
        pixel_values: Any,
        selected_layers: Sequence[str],
    ) -> dict[str, tuple[Any, dict[str, Any]]]:
        visual = self.model
        transformer = getattr(visual, "transformer", None)
        blocks = getattr(transformer, "resblocks", None)
        if blocks is None:
            raise ModelAdapterError(
                "RemoteCLIP layerwise extraction requires visual.transformer.resblocks for deterministic hooks"
            )
        block_count = len(blocks)
        layer_specs = [_selected_layer_index(layer, block_count + 1, block_count) for layer in selected_layers]
        embeddings = self._remoteclip_embeddings(pixel_values)
        outputs: dict[str, tuple[Any, dict[str, Any]]] = {}
        patch_layers = [
            layer
            for layer, (kind, _) in zip(selected_layers, layer_specs)
            if kind == "patch_emb"
        ]
        if patch_layers:
            pooled = _pool_tokens(embeddings, "patch_mean", True, self.spec.model_id)
            for layer in patch_layers:
                outputs[layer] = (
                    pooled,
                    _layer_metadata(
                        self,
                        kind="patch_emb",
                        block=None,
                        tokens=embeddings,
                        pooled=pooled,
                        pooling="patch_mean",
                        projection="none",
                        text_shared_space=False,
                        has_cls_token=True,
                    ),
                )

        requested_blocks = {
            block
            for kind, block in layer_specs
            if kind == "block" and block is not None
        }
        captured: dict[int, Any] = {}
        handles = []
        try:
            for block_index in sorted(requested_blocks):
                def capture(_module: Any, _inputs: Any, output: Any, index: int = block_index) -> None:
                    captured[index] = _hook_tensor(output)

                handles.append(blocks[block_index - 1].register_forward_hook(capture))
        except Exception as exc:
            for handle in handles:
                handle.remove()
            raise ModelAdapterError(
                "RemoteCLIP layerwise extraction requires deterministic forward hooks on visual transformer blocks"
            ) from exc
        final_tokens = None
        try:
            if requested_blocks or any(kind == "final" for kind, _ in layer_specs):
                final_tokens = transformer(embeddings)
        except Exception as exc:
            raise ModelAdapterError("RemoteCLIP deterministic layerwise hooks failed during the visual forward") from exc
        finally:
            for handle in handles:
                handle.remove()

        for layer, (kind, index) in zip(selected_layers, layer_specs):
            if kind == "block":
                if index not in captured:
                    raise ModelAdapterError(f"RemoteCLIP hook did not capture requested block {index}")
                tokens = _normalise_tokens(captured[index], pixel_values.shape[0], self.spec.model_id)
                pooled = _pool_tokens(tokens, "cls", True, self.spec.model_id)
                outputs[layer] = (
                    pooled,
                    _layer_metadata(
                        self,
                        kind="block",
                        block=index,
                        tokens=tokens,
                        pooled=pooled,
                        pooling="cls",
                        projection="none",
                        text_shared_space=False,
                        has_cls_token=True,
                    ),
                )
            elif kind == "final":
                if final_tokens is None:
                    raise ModelAdapterError("RemoteCLIP final layer was not evaluated")
                raw_tokens = _normalise_tokens(final_tokens, pixel_values.shape[0], self.spec.model_id)
                pooled, _ = visual._pool(raw_tokens)
                if visual.proj is not None:
                    pooled = pooled @ visual.proj
                tokens = raw_tokens
                if not getattr(visual, "final_ln_after_pool", False) and callable(getattr(visual, "ln_post", None)):
                    tokens = visual.ln_post(tokens)
                outputs[layer] = (
                    pooled,
                    _layer_metadata(
                        self,
                        kind="final",
                        block=None,
                        tokens=tokens,
                        pooled=pooled,
                        pooling=self.spec.pooling,
                        projection=self.spec.projection,
                        text_shared_space=self.spec.text_shared_space,
                        has_cls_token=True,
                    ),
                )
        return outputs


class ClipAdapter(VisionEncoderAdapter):
    def _forward_global(self, pixel_values: Any) -> Any:
        outputs = self.model.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=self.encoder_input_size != self.spec.checkpoint_native_input_size,
        )
        pooled = outputs.pooler_output
        if self.spec.projection == "official_visual_projection":
            return self.model.visual_projection(pooled)
        return pooled

    def _forward_layerwise(
        self,
        pixel_values: Any,
        selected_layers: Sequence[str],
    ) -> dict[str, tuple[Any, dict[str, Any]]]:
        outputs = self.model.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=self.encoder_input_size != self.spec.checkpoint_native_input_size,
            output_hidden_states=True,
        )
        hidden_states = _hidden_states_from(outputs, self.spec.model_id)
        pooled = _attribute(outputs, "pooler_output")
        if pooled is None:
            raise ModelAdapterError(f"{self.spec.model_id} did not return pooler_output")
        if self.spec.projection == "official_visual_projection":
            pooled = self.model.visual_projection(pooled)
        final_tokens = _attribute(outputs, "last_hidden_state", hidden_states[-1])
        post_layernorm = getattr(self.model.vision_model, "post_layernorm", None)
        if callable(post_layernorm):
            final_tokens = post_layernorm(final_tokens)
        return _hidden_state_layerwise(
            self,
            hidden_states,
            selected_layers,
            batch_size=pixel_values.shape[0],
            has_cls_token=True,
            internal_pooling="cls",
            final_tokens=final_tokens,
            final_pooled=pooled,
            final_pooling=self.spec.pooling,
            final_projection=self.spec.projection,
            final_shared_space=self.spec.text_shared_space,
            block_count=len(hidden_states) - 1,
        )


class SiglipAdapter(VisionEncoderAdapter):
    def _forward_global(self, pixel_values: Any) -> Any:
        outputs = self.model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=self.encoder_input_size != self.spec.checkpoint_native_input_size,
        )
        if outputs.pooler_output is None:
            raise ModelAdapterError("SigLIP2 vision tower did not return its registered MAP pooler output")
        return outputs.pooler_output

    def _forward_layerwise(
        self,
        pixel_values: Any,
        selected_layers: Sequence[str],
    ) -> dict[str, tuple[Any, dict[str, Any]]]:
        vision_model = getattr(self.model, "vision_model", None)
        embeddings_module = getattr(vision_model, "embeddings", None)
        encoder = getattr(vision_model, "encoder", None)
        blocks = getattr(encoder, "layers", None)
        if embeddings_module is None or blocks is None:
            raise ModelAdapterError(
                "SigLIP2 layerwise extraction requires vision_model.embeddings and vision_model.encoder.layers"
            )
        outputs, hidden_states = _forward_with_layer_hooks(
            model_id=self.spec.model_id,
            embeddings_module=embeddings_module,
            blocks=blocks,
            forward=lambda: self.model(
                pixel_values=pixel_values,
                interpolate_pos_encoding=self.encoder_input_size != self.spec.checkpoint_native_input_size,
                output_hidden_states=True,
            ),
        )
        pooled = _attribute(outputs, "pooler_output")
        if pooled is None:
            raise ModelAdapterError("SigLIP2 vision tower did not return its registered MAP pooler output")
        final_tokens = _attribute(outputs, "last_hidden_state", hidden_states[-1])
        return _hidden_state_layerwise(
            self,
            hidden_states,
            selected_layers,
            batch_size=pixel_values.shape[0],
            has_cls_token=False,
            internal_pooling="patch_mean",
            final_tokens=final_tokens,
            final_pooled=pooled,
            final_pooling=self.spec.pooling,
            final_projection=self.spec.projection,
            final_shared_space=self.spec.text_shared_space,
            block_count=len(hidden_states) - 1,
        )


class Dinov2Adapter(VisionEncoderAdapter):
    def _forward_global(self, pixel_values: Any) -> Any:
        outputs = self.model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=self.encoder_input_size != self.spec.checkpoint_native_input_size,
        )
        return outputs.pooler_output

    def _forward_layerwise(
        self,
        pixel_values: Any,
        selected_layers: Sequence[str],
    ) -> dict[str, tuple[Any, dict[str, Any]]]:
        outputs = self.model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=self.encoder_input_size != self.spec.checkpoint_native_input_size,
            output_hidden_states=True,
        )
        hidden_states = _hidden_states_from(outputs, self.spec.model_id)
        pooled = _attribute(outputs, "pooler_output")
        if pooled is None:
            raise ModelAdapterError("DINOv2 did not return its registered CLS pooler output")
        final_tokens = _attribute(outputs, "last_hidden_state", hidden_states[-1])
        return _hidden_state_layerwise(
            self,
            hidden_states,
            selected_layers,
            batch_size=pixel_values.shape[0],
            has_cls_token=True,
            internal_pooling="cls",
            final_tokens=final_tokens,
            final_pooled=pooled,
            final_pooling=self.spec.pooling,
            final_projection=self.spec.projection,
            final_shared_space=self.spec.text_shared_space,
            block_count=len(hidden_states) - 1,
        )


class MaeAdapter(VisionEncoderAdapter):
    def _forward_no_mask(self, pixel_values: Any, *, output_hidden_states: bool = False) -> Any:
        torch = _require_torch()
        patch_size = self.spec.patch_size
        height, width = pixel_values.shape[-2:]
        if height % patch_size or width % patch_size:
            raise ModelAdapterError("MAE input dimensions must be divisible by the patch size")
        patch_count = (height // patch_size) * (width // patch_size)
        identity = torch.arange(patch_count, device=pixel_values.device, dtype=torch.float32)
        identity = identity.unsqueeze(0).expand(pixel_values.shape[0], -1)
        return self.model(
            pixel_values=pixel_values,
            noise=identity,
            interpolate_pos_encoding=self.encoder_input_size != self.spec.checkpoint_native_input_size,
            output_hidden_states=output_hidden_states,
        )

    def _forward_global(self, pixel_values: Any) -> Any:
        torch = _require_torch()
        patch_size = self.spec.patch_size
        height, width = pixel_values.shape[-2:]
        if height % patch_size or width % patch_size:
            raise ModelAdapterError("MAE input dimensions must be divisible by the patch size")
        patch_count = (height // patch_size) * (width // patch_size)
        identity = torch.arange(patch_count, device=pixel_values.device, dtype=torch.float32)
        identity = identity.unsqueeze(0).expand(pixel_values.shape[0], -1)
        outputs = self.model(
            pixel_values=pixel_values,
            noise=identity,
            interpolate_pos_encoding=self.encoder_input_size != self.spec.checkpoint_native_input_size,
        )
        expected_tokens = patch_count + 1
        if outputs.last_hidden_state.shape[1] != expected_tokens:
            raise ModelAdapterError(
                f"MAE no-mask encoder returned {outputs.last_hidden_state.shape[1]} tokens, expected {expected_tokens}"
            )
        if outputs.mask.count_nonzero().item() != 0:
            raise ModelAdapterError("MAE adapter unexpectedly masked patch tokens")
        restored = outputs.ids_restore
        expected_restore = identity.to(dtype=restored.dtype)
        if not torch.equal(restored, expected_restore):
            raise ModelAdapterError("MAE adapter did not preserve deterministic patch-token order")
        return outputs.last_hidden_state[:, 1:, :].mean(dim=1)

    def _forward_layerwise(
        self,
        pixel_values: Any,
        selected_layers: Sequence[str],
    ) -> dict[str, tuple[Any, dict[str, Any]]]:
        embeddings_module = getattr(self.model, "embeddings", None)
        encoder_layers = getattr(getattr(self.model, "encoder", None), "layer", None)
        if embeddings_module is None or encoder_layers is None:
            raise ModelAdapterError(
                "MAE layerwise extraction requires deterministic hooks on embeddings and encoder.layer"
            )
        hidden_states: dict[int, Any] = {}
        handles = []

        def capture_embeddings(_module: Any, _inputs: Any, output: Any) -> None:
            hidden_states[0] = _hook_tensor(output)

        try:
            handles.append(embeddings_module.register_forward_hook(capture_embeddings))
            for block_index, block in enumerate(encoder_layers, start=1):
                def capture_block(_module: Any, _inputs: Any, output: Any, index: int = block_index) -> None:
                    hidden_states[index] = _hook_tensor(output)

                handles.append(block.register_forward_hook(capture_block))
        except Exception as exc:
            for handle in handles:
                handle.remove()
            raise ModelAdapterError(
                "MAE layerwise extraction requires deterministic forward hooks on its transformer blocks"
            ) from exc
        try:
            outputs = self._forward_no_mask(pixel_values, output_hidden_states=False)
        except Exception as exc:
            raise ModelAdapterError("MAE deterministic layerwise hooks failed during the no-mask forward") from exc
        finally:
            for handle in handles:
                handle.remove()
        block_count = len(encoder_layers)
        if set(hidden_states) != set(range(block_count + 1)):
            raise ModelAdapterError("MAE hooks did not capture every requested encoder stage")
        hidden_state_sequence = tuple(hidden_states[index] for index in range(block_count + 1))
        last_hidden_state = _attribute(outputs, "last_hidden_state")
        if last_hidden_state is None:
            raise ModelAdapterError("MAE did not return last_hidden_state for layerwise extraction")
        last_hidden_state = _normalise_tokens(last_hidden_state, pixel_values.shape[0], self.spec.model_id)
        patch_count = last_hidden_state.shape[1] - 1
        mask = _attribute(outputs, "mask")
        ids_restore = _attribute(outputs, "ids_restore")
        if mask is None or ids_restore is None:
            raise ModelAdapterError("MAE layerwise extraction requires mask and ids_restore audit outputs")
        if mask.count_nonzero().item() != 0:
            raise ModelAdapterError("MAE layerwise extraction unexpectedly masked patch tokens")
        identity = _require_torch().arange(patch_count, device=last_hidden_state.device, dtype=ids_restore.dtype)
        identity = identity.unsqueeze(0).expand(pixel_values.shape[0], -1)
        if not _require_torch().equal(ids_restore, identity):
            raise ModelAdapterError("MAE layerwise extraction did not preserve deterministic patch-token order")
        pooled = _pool_tokens(last_hidden_state, "patch_mean", True, self.spec.model_id)
        return _hidden_state_layerwise(
            self,
            hidden_state_sequence,
            selected_layers,
            batch_size=pixel_values.shape[0],
            has_cls_token=True,
            internal_pooling="patch_mean",
            final_tokens=last_hidden_state,
            final_pooled=pooled,
            final_pooling=self.spec.pooling,
            final_projection=self.spec.projection,
            final_shared_space=self.spec.text_shared_space,
            block_count=block_count,
        )


def _load_remoteclip(spec: ModelSpec, asset: ModelAsset) -> tuple[Any, type[VisionEncoderAdapter]]:
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("RemoteCLIP requires open_clip_torch") from exc

    from cfra.provenance import sha256_json

    loader_config = open_clip.get_model_config("ViT-B-32")
    if sha256_json(loader_config) != asset.loader_config_sha256:
        raise ModelAdapterError("Installed OpenCLIP ViT-B-32 configuration differs from the registered identity")

    full_model = open_clip.create_model(
        "ViT-B-32",
        pretrained=str(asset.path),
        device="cpu",
        precision="fp32",
        weights_only=True,
        require_pretrained=True,
    )
    visual = full_model.visual
    del full_model
    if getattr(visual, "output_dim", None) != spec.dimension:
        raise ModelAdapterError("RemoteCLIP visual tower output dimension does not match the registry")
    projection = getattr(visual, "proj", None)
    if projection is None or tuple(projection.shape) != (768, spec.dimension):
        raise ModelAdapterError("RemoteCLIP visual projection does not match ViT-B-32")
    return visual, RemoteClipAdapter


def _load_clip(spec: ModelSpec, asset: ModelAsset) -> tuple[Any, type[VisionEncoderAdapter]]:
    try:
        from transformers import CLIPVisionModelWithProjection
    except ImportError as exc:
        raise RuntimeError("CLIP requires transformers") from exc

    model, info = CLIPVisionModelWithProjection.from_pretrained(
        asset.path,
        local_files_only=True,
        use_safetensors=False,
        output_loading_info=True,
        weights_only=True,
    )
    _validate_loading_info(info, ("logit_scale", "text_model.", "text_projection."))
    if model.config.hidden_size != 768 or model.config.projection_dim != 512:
        raise ModelAdapterError("CLIP checkpoint does not match ViT-B/16")
    return model, ClipAdapter


def _load_siglip(spec: ModelSpec, asset: ModelAsset) -> tuple[Any, type[VisionEncoderAdapter]]:
    try:
        from transformers import SiglipVisionModel
    except ImportError as exc:
        raise RuntimeError("SigLIP2 requires transformers") from exc

    model, info = SiglipVisionModel.from_pretrained(
        asset.path,
        local_files_only=True,
        use_safetensors=True,
        output_loading_info=True,
        weights_only=True,
    )
    _validate_loading_info(info, ("logit_bias", "logit_scale", "text_model."))
    if model.config.hidden_size != spec.dimension or model.vision_model.head is None:
        raise ModelAdapterError("SigLIP2 checkpoint lacks the registered MAP attention pooler")
    return model, SiglipAdapter


def _load_dinov2(spec: ModelSpec, asset: ModelAsset) -> tuple[Any, type[VisionEncoderAdapter]]:
    try:
        from transformers import Dinov2Model
    except ImportError as exc:
        raise RuntimeError("DINOv2 requires transformers") from exc

    model, info = Dinov2Model.from_pretrained(
        asset.path,
        local_files_only=True,
        use_safetensors=True,
        output_loading_info=True,
        weights_only=True,
    )
    _validate_loading_info(info, ())
    if model.config.hidden_size != spec.dimension or model.config.patch_size != spec.patch_size:
        raise ModelAdapterError("DINOv2 checkpoint does not match ViT-B/14")
    return model, Dinov2Adapter


def _load_mae(spec: ModelSpec, asset: ModelAsset) -> tuple[Any, type[VisionEncoderAdapter]]:
    try:
        from transformers import ViTMAEModel
    except ImportError as exc:
        raise RuntimeError("MAE requires transformers") from exc

    model, info = ViTMAEModel.from_pretrained(
        asset.path,
        local_files_only=True,
        use_safetensors=True,
        output_loading_info=True,
        weights_only=True,
    )
    _validate_loading_info(info, ("decoder.",))
    if model.config.hidden_size != spec.dimension or model.config.patch_size != spec.patch_size:
        raise ModelAdapterError("MAE checkpoint does not match ViT-B/16")
    model.config.mask_ratio = 0.0
    return model, MaeAdapter


LOADERS = {
    "open_clip": _load_remoteclip,
    "transformers_clip": _load_clip,
    "transformers_siglip": _load_siglip,
    "transformers_dinov2": _load_dinov2,
    "transformers_vit_mae": _load_mae,
}


def load_adapter(
    model_id: str,
    asset: ModelAsset,
    *,
    runtime: ModelRuntimeConfig,
    registry_root: Path | str,
    device: str = "auto",
) -> VisionEncoderAdapter:
    """Load one frozen visual tower from a validated local asset without network access."""

    spec = get_model_spec(model_id)
    verify_registered_model_asset(asset, model_id, registry_root)
    if asset.asset_id != spec.asset_id:
        raise ModelAdapterError(f"Asset {asset.asset_id} cannot load representation {model_id}")
    try:
        size = runtime.encoder_input_sizes[model_id]
    except KeyError as exc:
        raise ModelAdapterError(f"Runtime config lacks encoder input size for {model_id}") from exc
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ModelAdapterError("encoder_input_size must be a positive integer")
    if size % spec.patch_size:
        raise ModelAdapterError(f"{model_id} encoder_input_size must be divisible by patch size {spec.patch_size}")
    if size != spec.checkpoint_native_input_size and not spec.supports_position_interpolation:
        raise ModelAdapterError(f"{model_id} does not support a non-native encoder input size")
    resolved_device = resolve_device(device)
    if runtime.precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ModelAdapterError(f"Unsupported inference precision: {runtime.precision!r}")
    if resolved_device.type != "cuda" and runtime.precision in {"fp16", "bf16"}:
        raise ModelAdapterError(f"{runtime.precision} inference is only enabled for CUDA devices")

    try:
        loader = LOADERS[spec.backend]
    except KeyError as exc:
        raise ModelAdapterError(f"No runtime loader for backend: {spec.backend}") from exc
    model, adapter_type = loader(spec, asset)
    return adapter_type(
        spec=spec,
        asset=asset,
        model=model,
        device=resolved_device,
        runtime=runtime,
    )


def iter_encoded_batches(
    adapter: VisionEncoderAdapter,
    images: Any,
    *,
    output_device: str = "cpu",
) -> Iterator[EncodedFeatures]:
    """Yield vectorized model batches without retaining a persistent feature cache."""

    size = adapter.runtime.batch_size
    torch = _require_torch()
    if isinstance(images, torch.Tensor):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        for start in range(0, images.shape[0], size):
            yield adapter.encode(images[start : start + size], output_device=output_device)
        return

    iterator = iter(images)
    while True:
        batch: list[Any] = []
        for _ in range(size):
            try:
                batch.append(next(iterator))
            except StopIteration:
                break
        if not batch:
            return
        yield adapter.encode(batch, output_device=output_device)


class AdapterCache:
    """A bounded in-process model cache; it never stores sample-derived features."""

    def __init__(self, runtime: ModelRuntimeConfig, registry_root: Path | str) -> None:
        self.runtime = runtime
        self.registry_root = Path(registry_root).expanduser().resolve(strict=True)
        self.max_size = runtime.max_loaded_models
        self._adapters: OrderedDict[tuple[str, str, str, str, str, int], VisionEncoderAdapter] = OrderedDict()

    def get(
        self,
        model_id: str,
        asset: ModelAsset,
        *,
        device: str = "auto",
    ) -> VisionEncoderAdapter:
        encoder_input_size = self.runtime.encoder_input_sizes[model_id]
        key = (
            model_id,
            asset.asset_fingerprint,
            str(Path(asset.weight_path)),
            device,
            self.runtime.fingerprint,
            encoder_input_size,
        )
        adapter = self._adapters.pop(key, None)
        if adapter is not None and adapter.is_closed:
            adapter = None
        if adapter is None:
            adapter = load_adapter(
                model_id,
                asset,
                runtime=self.runtime,
                registry_root=self.registry_root,
                device=device,
            )
        self._adapters[key] = adapter
        while len(self._adapters) > self.max_size:
            _, evicted = self._adapters.popitem(last=False)
            evicted.close()
        return adapter

    def clear(self) -> None:
        adapters = list(self._adapters.values())
        self._adapters.clear()
        failures: list[Exception] = []
        for adapter in adapters:
            try:
                adapter.close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ModelAdapterError(f"Failed to release {len(failures)} cached model adapter(s)") from failures[0]

    def __enter__(self) -> AdapterCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.clear()
