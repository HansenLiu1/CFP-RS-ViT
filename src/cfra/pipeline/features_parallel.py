"""CPU-parallel feature extraction with image caching.

Delegates to features._sample_context to guarantee identical behavior.
Uses a sliding window of futures to cap memory usage.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from cfra.data import DatasetManifest, ObjectRecord

# Worker globals (pruned payloads only; full manifests never cross process boundaries)
_WORKER_MANIFEST_ROOTS: dict[str, str] | None = None
_WORKER_OBJECT_INDEX: dict[str, ObjectRecord] | None = None
_WORKER_IMAGE_TO_OBJECTS: dict[tuple[str, str], tuple[ObjectRecord, ...]] | None = None
_WORKER_CONFIG: dict[str, Any] | None = None


def _init_worker(
    manifest_roots: dict[str, str],
    object_index: dict[str, ObjectRecord],
    image_to_objects: dict[tuple[str, str], tuple[ObjectRecord, ...]],
    config: dict[str, Any],
) -> None:
    """Initialize worker process state with pruned payloads."""
    global _WORKER_MANIFEST_ROOTS, _WORKER_OBJECT_INDEX, _WORKER_IMAGE_TO_OBJECTS, _WORKER_CONFIG
    _WORKER_MANIFEST_ROOTS = manifest_roots
    _WORKER_OBJECT_INDEX = object_index
    _WORKER_IMAGE_TO_OBJECTS = image_to_objects
    _WORKER_CONFIG = config


@lru_cache(maxsize=8)
def _load_cached_image(dataset_id: str, image_id: str, image_path: str) -> np.ndarray:
    """Load image with LRU caching (8 images per worker to limit memory)."""
    with Image.open(image_path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _worker_image_path(dataset_id: str, object_record: ObjectRecord) -> str:
    """Resolve the manifest-relative image path inside a worker.

    Equivalent to cfra.data.index.image_path(manifest, record) for a pruned
    manifest: the root is fixed by the E00 manifest and containment was already
    validated when the manifest was written.
    """
    root = Path(_WORKER_MANIFEST_ROOTS[dataset_id]).expanduser().resolve()
    rel = object_record.image_relpath.replace("\\", "/")
    return str(root / Path(*rel.split("/")))


def _worker_build_context(sample_dict: dict[str, Any], perturbation_overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Worker function: build context for one sample (must be module-level for pickle)."""
    assert _WORKER_MANIFEST_ROOTS is not None
    assert _WORKER_OBJECT_INDEX is not None
    assert _WORKER_IMAGE_TO_OBJECTS is not None
    assert _WORKER_CONFIG is not None

    from .features import FeatureSample, _sample_context

    sample = FeatureSample(**sample_dict)
    object_record = _WORKER_OBJECT_INDEX[sample.object_id]

    # Cached image loading
    img_path = _worker_image_path(sample.dataset_id, object_record)
    source_image = _load_cached_image(sample.dataset_id, sample.source_image_id, img_path)

    # Precomputed same-image objects
    key = (sample.dataset_id, sample.source_image_id)
    same_image_objects = _WORKER_IMAGE_TO_OBJECTS.get(key, ())

    # Delegate to existing logic
    return _sample_context(
        sample,
        manifests=None,
        object_index=_WORKER_OBJECT_INDEX,
        config=_WORKER_CONFIG,
        perturbation_overrides=perturbation_overrides,
        source_image=source_image,
        same_image_objects=same_image_objects,
    )


def _prune_worker_payloads(
    samples: tuple[Any, ...],
    manifests: dict[str, DatasetManifest],
    object_index: dict[str, ObjectRecord],
) -> tuple[
    dict[str, str],
    dict[str, ObjectRecord],
    dict[tuple[str, str], tuple[ObjectRecord, ...]],
]:
    """Keep only the objects and source images touched by the selected samples.

    Full manifests hold every raw annotation record of the datasets; only the
    selected samples (and the other objects sharing their source images, needed by
    ``sample_background_crop``) are read by the workers, so everything else is
    dropped before pickling.  Ordering follows the manifest object order, which
    matches the serial code path exactly.
    """

    needed_images: set[tuple[str, str]] = {
        (sample.dataset_id, sample.source_image_id) for sample in samples
    }
    pruned_index: dict[str, ObjectRecord] = {}
    image_to_objects: dict[tuple[str, str], list[ObjectRecord]] = {}
    for manifest in manifests.values():
        for obj in manifest.objects:
            key = (obj.dataset_id, obj.source_image_id)
            if key not in needed_images:
                continue
            pruned_index[obj.object_id] = obj
            image_to_objects.setdefault(key, []).append(obj)
    manifest_roots = {dataset_id: str(manifest.root) for dataset_id, manifest in manifests.items()}
    return (
        manifest_roots,
        pruned_index,
        {key: tuple(objs) for key, objs in image_to_objects.items()},
    )


def _worker_build_context_batch(sample_dicts: list[dict[str, Any]], perturbation_overrides: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Worker function: build contexts for a batch of samples (module-level for pickle)."""
    assert _WORKER_MANIFEST_ROOTS is not None
    assert _WORKER_OBJECT_INDEX is not None
    assert _WORKER_IMAGE_TO_OBJECTS is not None
    assert _WORKER_CONFIG is not None

    from .features import FeatureSample, _sample_context

    results: list[dict[str, Any]] = []
    for sample_dict in sample_dicts:
        sample = FeatureSample(**sample_dict)
        object_record = _WORKER_OBJECT_INDEX[sample.object_id]
        img_path = _worker_image_path(sample.dataset_id, object_record)
        source_image = _load_cached_image(sample.dataset_id, sample.source_image_id, img_path)
        key = (sample.dataset_id, sample.source_image_id)
        same_image_objects = _WORKER_IMAGE_TO_OBJECTS.get(key, ())
        results.append(
            _sample_context(
                sample,
                manifests=None,
                object_index=_WORKER_OBJECT_INDEX,
                config=_WORKER_CONFIG,
                perturbation_overrides=perturbation_overrides,
                source_image=source_image,
                same_image_objects=same_image_objects,
            )
        )
    return results


def iter_context_batches_parallel(
    samples: Sequence[Any],
    manifests: dict[str, DatasetManifest],
    object_index: dict[str, ObjectRecord],
    config: dict[str, Any],
    batch_size: int,
    cpu_workers: int | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield context batches in original sample order, built in parallel workers.

    Used by layerwise stage extraction (E06), which needs batch-aligned contexts
    to feed ``adapter.encode_layerwise``.  The workers delegate to the exact
    same ``_sample_context`` used by the serial path, so results are identical.
    """

    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    if cpu_workers is None:
        cpu_workers = max(1, min(8, os.cpu_count() or 1) - 2)
    manifest_roots, pruned_index, image_to_objects = _prune_worker_payloads(samples, manifests, object_index)
    batches: list[tuple[dict[str, Any], ...]] = [
        tuple(asdict(sample) for sample in samples[start : start + batch_size])
        for start in range(0, len(samples), batch_size)
    ]
    with ProcessPoolExecutor(
        max_workers=cpu_workers,
        initializer=_init_worker,
        initargs=(manifest_roots, pruned_index, image_to_objects, config),
        mp_context=get_context("spawn"),
    ) as executor:
        pending: dict[int, Future] = {}
        submitted = 0
        inflight = max(1, min(cpu_workers * 2, len(batches)))
        while submitted < inflight and submitted < len(batches):
            pending[submitted] = executor.submit(_worker_build_context_batch, list(batches[submitted]), None)
            submitted += 1
        for index in range(len(batches)):
            yield pending.pop(index).result()
            if submitted < len(batches):
                pending[submitted] = executor.submit(_worker_build_context_batch, list(batches[submitted]), None)
                submitted += 1


def _worker_build_pair_batch(sample_dicts: list[dict[str, Any]], perturbation_overrides: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Worker function: build only the two E06 foreground-pair crops per sample."""

    assert _WORKER_MANIFEST_ROOTS is not None
    assert _WORKER_OBJECT_INDEX is not None
    assert _WORKER_CONFIG is not None

    from .features import FeatureSample, _sample_context_pair

    results: list[dict[str, Any]] = []
    for sample_dict in sample_dicts:
        sample = FeatureSample(**sample_dict)
        object_record = _WORKER_OBJECT_INDEX[sample.object_id]
        img_path = _worker_image_path(sample.dataset_id, object_record)
        source_image = _load_cached_image(sample.dataset_id, sample.source_image_id, img_path)
        results.append(
            _sample_context_pair(
                sample,
                object_index=_WORKER_OBJECT_INDEX,
                config=_WORKER_CONFIG,
                perturbation_overrides=perturbation_overrides,
                source_image=source_image,
            )
        )
    return results


def iter_context_pair_batches_parallel(
    samples: Sequence[Any],
    manifests: dict[str, DatasetManifest],
    object_index: dict[str, ObjectRecord],
    config: dict[str, Any],
    batch_size: int,
    cpu_workers: int | None = None,
    perturbation_overrides: dict[str, Any] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield E06 pair-only context batches in original sample order.

    Same worker pool and ordering contract as
    :func:`iter_context_batches_parallel`, but each worker builds only the two
    foreground-pair crops the layerwise stage encodes, so the ~20 control and
    background crops per sample are never computed.

    ``perturbation_overrides`` is forwarded to ``_sample_context_pair`` so a
    caller (e.g. the perturbation-strength sweep) can vary one operator's
    parameters while keeping the exact same crop, RNG and paired construction.
    """

    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    if cpu_workers is None:
        cpu_workers = max(1, min(8, os.cpu_count() or 1) - 2)
    manifest_roots, pruned_index, image_to_objects = _prune_worker_payloads(samples, manifests, object_index)
    batches: list[tuple[dict[str, Any], ...]] = [
        tuple(asdict(sample) for sample in samples[start : start + batch_size])
        for start in range(0, len(samples), batch_size)
    ]
    with ProcessPoolExecutor(
        max_workers=cpu_workers,
        initializer=_init_worker,
        initargs=(manifest_roots, pruned_index, image_to_objects, config),
        mp_context=get_context("spawn"),
    ) as executor:
        pending: dict[int, Future] = {}
        submitted = 0
        inflight = max(1, min(cpu_workers * 2, len(batches)))
        while submitted < inflight and submitted < len(batches):
            pending[submitted] = executor.submit(
                _worker_build_pair_batch, list(batches[submitted]), perturbation_overrides
            )
            submitted += 1
        for index in range(len(batches)):
            yield pending.pop(index).result()
            if submitted < len(batches):
                pending[submitted] = executor.submit(
                    _worker_build_pair_batch, list(batches[submitted]), perturbation_overrides
                )
                submitted += 1


def extract_features_parallel(
    samples: tuple[Any, ...],
    manifests: dict[str, DatasetManifest],
    object_index: dict[str, ObjectRecord],
    config: dict[str, Any],
    adapter: Any,
    perturbation_overrides: dict[str, Any] | None = None,
    cpu_workers: int | None = None,
) -> tuple[dict[str, np.ndarray], list[bool], dict[str, list[bool]], list[dict[str, Any]]]:
    """Extract features in parallel using CPU workers + GPU encoding.

    Returns:
        (arrays, background_valid, control_valid, perturbation_audit)
    """
    if cpu_workers is None:
        cpu_workers = max(1, min(8, os.cpu_count() or 1) - 2)

    manifest_roots, pruned_index, image_to_objects = _prune_worker_payloads(samples, manifests, object_index)

    from .features import CONTROL_MASK_NAMES, _input_keys

    arrays: dict[str, list[np.ndarray]] = {key: [] for key in _input_keys()}
    background_valid: list[bool] = []
    control_valid: dict[str, list[bool]] = {name: [] for name in CONTROL_MASK_NAMES if name != "true_obb"}
    perturbation_audit: list[dict[str, Any]] = []

    batch_size = adapter.runtime.batch_size
    pending: dict[str, list[np.ndarray]] = {key: [] for key in _input_keys()}

    # Sliding window to cap memory. Each inflight context carries up to 21 full
    # resolution crops, so a wide window quickly exhausts physical RAM and forces
    # pagefile swapping on machines with 8-32 GB.  The window is deliberately
    # small: CPU workers are faster than the GPU encoder, so 1-2 inflight
    # context pairs are enough to keep the GPU busy without memory pressure.
    max_inflight = max(1, min(cpu_workers, 2))
    futures: list[tuple[int, Future]] = []

    with ProcessPoolExecutor(
        max_workers=cpu_workers,
        initializer=_init_worker,
        initargs=(manifest_roots, pruned_index, image_to_objects, config),
        mp_context=get_context("spawn"),
    ) as executor, tqdm(total=len(samples), desc="Parallel extraction", unit="sample") as pbar:
        sample_idx = 0

        # Fill initial window
        while sample_idx < len(samples) and len(futures) < max_inflight:
            future = executor.submit(_worker_build_context, asdict(samples[sample_idx]), perturbation_overrides)
            futures.append((sample_idx, future))
            sample_idx += 1

        # Process results as they complete
        while futures:
            _idx, future = futures.pop(0)
            context = future.result()

            background_valid.append(bool(context["_background_valid"]))
            perturbation_audit.append(dict(context["_perturbation_parameters"]))

            for name, valid in context["_control_valid"].items():
                control_valid[name].append(bool(valid))

            for key in _input_keys():
                pending[key].append(context[key])

            # Encode batch when full
            if len(pending["raw"]) >= batch_size:
                for key in _input_keys():
                    encoded = adapter.encode(pending[key])
                    if hasattr(encoded.values, "detach"):
                        arrays[key].append(encoded.values.detach().cpu().numpy().astype(np.float32))
                    else:
                        arrays[key].append(np.asarray(encoded.values, dtype=np.float32))
                    pending[key].clear()

            pbar.update(1)

            # Refill window
            if sample_idx < len(samples):
                future = executor.submit(_worker_build_context, asdict(samples[sample_idx]), perturbation_overrides)
                futures.append((sample_idx, future))
                sample_idx += 1

    # Final batch
    if pending["raw"]:
        for key in _input_keys():
            encoded = adapter.encode(pending[key])
            if hasattr(encoded.values, "detach"):
                arrays[key].append(encoded.values.detach().cpu().numpy().astype(np.float32))
            else:
                arrays[key].append(np.asarray(encoded.values, dtype=np.float32))

    # Concatenate
    stacked = {key: np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32) for key, values in arrays.items()}
    arrays.clear()

    return stacked, background_valid, control_valid, perturbation_audit
