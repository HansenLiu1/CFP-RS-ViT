"""Deterministic image batching and registered model preprocessing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .specs import PreprocessSpec


class ImageBatchError(ValueError):
    """Raised when an image batch does not satisfy the adapter input contract."""


def _require_runtime() -> tuple[Any, Any]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Model preprocessing requires the 'models' optional dependencies") from exc
    return np, torch


def _single_image_tensor(image: Any) -> Any:
    np, torch = _require_runtime()
    if isinstance(image, torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ImageBatchError("Tensor images must have shape [3, height, width]")
        return image.detach()

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for non-tensor image inputs") from exc

    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"), copy=True)
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise ImageBatchError(f"Unsupported image type: {type(image).__name__}")

    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageBatchError("Array images must have shape [height, width, 3]")
    return torch.from_numpy(np.array(array, copy=True, order="C")).permute(2, 0, 1)


def stack_image_batch(images: Any) -> Any:
    """Convert tensor/PIL/NumPy RGB images into a contiguous BCHW tensor."""

    _, torch = _require_runtime()
    if isinstance(images, torch.Tensor):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ImageBatchError("Tensor batches must have shape [batch, 3, height, width]")
        batch = images.detach().clone()
    else:
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
            items = list(images)
        else:
            items = [images]
        if not items:
            raise ImageBatchError("Image batch cannot be empty")
        tensors = [_single_image_tensor(image) for image in items]
        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise ImageBatchError("All images in a batch must have the same shape")
        batch = torch.stack(tensors)

    if batch.shape[-2] != batch.shape[-1]:
        raise ImageBatchError("CFRA model inputs must originate from square protocol crops")
    if batch.dtype == torch.bool:
        raise ImageBatchError("Boolean image tensors are not supported")
    if batch.is_floating_point():
        if not torch.isfinite(batch).all():
            raise ImageBatchError("Floating-point image tensors must be finite")
        minimum = float(batch.min().item())
        maximum = float(batch.max().item())
        if minimum < 0.0 or maximum > 1.0:
            raise ImageBatchError("Floating-point image tensors must be in the [0, 1] range")
    elif batch.dtype != torch.uint8:
        raise ImageBatchError("Integer image tensors must use uint8 values")
    return batch.contiguous()


def preprocess_image_batch(
    images: Any,
    *,
    preprocess: PreprocessSpec,
    input_size: int,
    device: Any,
    pin_memory: bool = True,
    non_blocking: bool = True,
) -> Any:
    """Resize and normalize a raw RGB batch on the target execution device."""

    if not isinstance(input_size, int) or isinstance(input_size, bool) or input_size <= 0:
        raise ImageBatchError("input_size must be a positive integer")

    _, torch = _require_runtime()
    from torch.nn import functional

    batch = stack_image_batch(images)
    if batch.device.type == "cpu" and device.type == "cuda" and pin_memory and not batch.is_pinned():
        batch = batch.pin_memory()
    batch = batch.to(device=device, non_blocking=non_blocking and device.type == "cuda")
    if batch.dtype == torch.uint8:
        batch = batch.to(dtype=torch.float32).div_(255.0)
    else:
        batch = batch.to(dtype=torch.float32)

    if tuple(batch.shape[-2:]) != (input_size, input_size):
        if preprocess.interpolation != "bicubic":
            raise ImageBatchError(f"Unsupported registered interpolation: {preprocess.interpolation}")
        batch = functional.interpolate(
            batch,
            size=(input_size, input_size),
            mode="bicubic",
            align_corners=False,
            antialias=preprocess.antialias,
        )
        batch.clamp_(0.0, 1.0)

    mean = batch.new_tensor(preprocess.mean).view(1, 3, 1, 1)
    std = batch.new_tensor(preprocess.std).view(1, 3, 1, 1)
    return batch.sub_(mean).div_(std)
