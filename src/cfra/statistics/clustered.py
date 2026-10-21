"""Source-image clustered bootstrap and paired randomization helpers.

With ``workers > 1`` the bootstrap statistic runs inside worker processes;
every cluster draw is made on the caller's RNG in the identical sequence, so
parallel runs are bit-identical to the serial path.
"""

from __future__ import annotations

import atexit
import inspect
import pickle
import threading
from collections.abc import Callable
from typing import Any

import numpy as np


class StatisticsError(ValueError):
    """Raised when clustered inference inputs are malformed."""


def _unique_clusters(image_ids: np.ndarray) -> list[Any]:
    """Return deterministic cluster labels, including for mixed scalar dtypes."""

    values = image_ids.tolist()
    try:
        return sorted(set(values))
    except (TypeError, ValueError):
        unique: list[Any] = []
        for value in values:
            if not any(value == existing for existing in unique):
                unique.append(value)
        return unique


def _validate_confidence(confidence: float) -> float:
    level = float(confidence)
    if not 0.0 < level < 1.0:
        raise StatisticsError("confidence must be strictly between zero and one")
    return level


def _percentile_ci(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(finite, [tail, 1.0 - tail])
    return float(low), float(high)


def _can_bind(statistic: Callable[..., Any], argument_count: int) -> bool:
    try:
        inspect.signature(statistic).bind(*([None] * argument_count))
    except (TypeError, ValueError):
        return False
    return True


_CAN_BIND_CACHE: dict[tuple[Callable[..., Any], int], bool] = {}


def _cached_can_bind(statistic: Callable[..., Any], argument_count: int) -> bool:
    """Memoized signature check.

    ``clustered_bootstrap`` invokes the statistic thousands of times per call
    (5000 replicates), and ``inspect.signature().bind`` on a NumPy ufunc raises
    an exception on every attempt.  The binding contract is immutable for a
    fixed ``statistic``, so the result is cached; repeated raises (the common
    case for ufuncs) are the slowest path and are eliminated entirely.
    """

    key = (statistic, argument_count)
    cached = _CAN_BIND_CACHE.get(key)
    if cached is not None:
        return cached
    result = _can_bind(statistic, argument_count)
    _CAN_BIND_CACHE[key] = result
    return result


def _bootstrap_replicate_inputs(
    array: np.ndarray,
    labels_array: np.ndarray | None,
    image_ids: np.ndarray,
    cluster_indices: list[np.ndarray],
    rng: Any,
    *,
    relabel_clusters: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Draw one replicate on the caller's RNG and return its statistic inputs.

    Keeping every draw on the caller-provided RNG preserves the exact random
    sequence of the serial path, so parallel execution is bit-identical.
    """

    selected_positions = rng.integers(0, len(cluster_indices), size=len(cluster_indices))
    selected_indices = [cluster_indices[position] for position in selected_positions]
    indices = np.concatenate(selected_indices)
    sampled_values = array[indices]
    sampled_labels = None if labels_array is None else labels_array[indices]
    sampled_image_ids = image_ids[indices]
    if relabel_clusters:
        occurrence_ids = np.empty(len(indices), dtype=np.int64)
        offset = 0
        for draw_index, position in enumerate(selected_positions):
            cluster_length = len(cluster_indices[position])
            occurrence_ids[offset : offset + cluster_length] = draw_index
            offset += cluster_length
        sampled_image_ids = occurrence_ids
    return sampled_values, sampled_labels, sampled_image_ids


def _bootstrap_replicate(
    statistic: Callable[..., float],
    array: np.ndarray,
    labels_array: np.ndarray | None,
    image_ids: np.ndarray,
    cluster_indices: list[np.ndarray],
    rng: Any,
    *,
    relabel_clusters: bool = False,
) -> float:
    """Serial-path replicate helper (draws on the caller's RNG)."""

    return float(
        _invoke_statistic(
            statistic,
            *_bootstrap_replicate_inputs(
                array,
                labels_array,
                image_ids,
                cluster_indices,
                rng,
                relabel_clusters=relabel_clusters,
            ),
        )
    )


_WORKER_GLOBALS: dict[str, Any] = {}

_SHARED_POOL: Any = None
_SHARED_POOL_LOCK = threading.Lock()
_SHARED_POOL_WORKERS: int = 1


def _shared_pool(workers: int) -> Any:
    """Return a process-wide shared executor started lazily on first use.

    Worker processes are spawned once and reused across bootstrap calls, so the
    pool-creation cost is paid exactly once per run.  A bootstrap call carries
    its full input payload with each block, so any caller can submit to the same
    pool without worker-state conflicts.
    """

    global _SHARED_POOL, _SHARED_POOL_WORKERS
    requested = int(workers)
    if requested < 1:
        raise StatisticsError("workers must be a positive integer")
    with _SHARED_POOL_LOCK:
        if _SHARED_POOL is not None and _SHARED_POOL_WORKERS == requested:
            return _SHARED_POOL
        if _SHARED_POOL is not None:
            _SHARED_POOL.shutdown(wait=False)
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        _SHARED_POOL = ProcessPoolExecutor(
            max_workers=requested,
            mp_context=get_context("spawn"),
        )
        _SHARED_POOL_WORKERS = requested
        atexit.register(_SHARED_POOL.shutdown, wait=False)
        return _SHARED_POOL


def _serialize_statistic(statistic: Callable[..., Any]) -> bytes:
    """Serialize a statistic for worker processes.

    cloudpickle handles closures/lambdas and is the preferred serializer; a
    plain top-level function still works with the standard library pickle.
    """

    try:
        import cloudpickle
    except ImportError:
        import pickle

        try:
            return pickle.dumps(statistic)
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            raise StatisticsError(
                "clustered_bootstrap(workers>1) needs cloudpickle for non-top-level "
                f"statistics, and it is not installed: {exc}"
            ) from exc
    return cloudpickle.dumps(statistic)


def _deserialize_statistic(payload: bytes) -> Callable[..., Any]:
    try:
        import cloudpickle
    except ImportError:
        import pickle

        return pickle.loads(payload)
    return cloudpickle.loads(payload)


def _init_worker(statistic_payload: bytes, payload: dict[str, Any]) -> None:
    global _WORKER_GLOBALS
    _WORKER_GLOBALS = {
        "statistic": _deserialize_statistic(statistic_payload),
        **payload,
    }


def _worker_replicate(rep_spec: dict[str, Any]) -> tuple[int, float]:
    """Compute one replicate statistic inside a worker process.

    ``rep_spec`` carries the pre-drawn cluster positions so the random stream
    exactly matches the serial path.
    """

    position, selected_positions = rep_spec["position"], rep_spec["positions"]
    array = _WORKER_GLOBALS["array"]
    labels_array = _WORKER_GLOBALS["labels"]
    image_ids = _WORKER_GLOBALS["image_ids"]
    cluster_indices = _WORKER_GLOBALS["cluster_indices"]
    relabel_clusters = _WORKER_GLOBALS["relabel_clusters"]
    selected_indices = [cluster_indices[p] for p in selected_positions]
    indices = np.concatenate(selected_indices)
    sampled_values = array[indices]
    sampled_labels = None if labels_array is None else labels_array[indices]
    sampled_image_ids = image_ids[indices]
    if relabel_clusters:
        occurrence_ids = np.empty(len(indices), dtype=np.int64)
        offset = 0
        for draw_index, pos in enumerate(selected_positions):
            clen = len(cluster_indices[pos])
            occurrence_ids[offset : offset + clen] = draw_index
            offset += clen
        sampled_image_ids = occurrence_ids
    value = float(_invoke_statistic(_WORKER_GLOBALS["statistic"], sampled_values, sampled_labels, sampled_image_ids))
    return position, value


_WORKER_SHARED_CACHE: dict[str, tuple[Any, dict[str, Any], dict[str, np.ndarray]]] = {}

_REPLICATE_BLOCK = 1024


def _release_worker_shared() -> None:
    """Close cached segments at worker exit after dropping exported views."""

    for key in list(_WORKER_SHARED_CACHE):
        segment, context, arrays = _WORKER_SHARED_CACHE.pop(key)
        arrays.clear()
        context.clear()
        try:
            segment.close()
        except BufferError:
            pass


atexit.register(_release_worker_shared)


def _load_shared_payload(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Attach a caller shared segment once per worker and cache its objects.

    Every array in ``spec['arrays']`` is exposed as a zero-copy view over the
    segment, so a worker that processes several blocks of the same call parses
    the context and maps the arrays exactly once.
    """

    name = spec["name"]
    cached = _WORKER_SHARED_CACHE.get(name)
    if cached is not None:
        return cached[1], cached[2]
    from multiprocessing import shared_memory

    segment = shared_memory.SharedMemory(name=name)
    context = pickle.loads(bytes(segment.buf[: spec["header"]]))
    arrays = {
        key: np.ndarray(shape, dtype=np.dtype(dtype), buffer=segment.buf, offset=offset)
        for key, (dtype, shape, offset) in spec["arrays"].items()
    }
    for stale in [key for key in _WORKER_SHARED_CACHE if key != name]:
        old = _WORKER_SHARED_CACHE.pop(stale)
        old[2].clear()
        old[1].clear()
        try:
            old[0].close()
        except BufferError:
            pass
    _WORKER_SHARED_CACHE[name] = (segment, context, arrays)
    return context, arrays


def _shared_payload_map(
    worker: Callable[[dict[str, Any]], Any],
    context: dict[str, Any],
    arrays: dict[str, Any],
    blocks: list[Any],
    workers: int,
) -> list[Any]:
    """Run ``worker`` over ``blocks`` with one shared copy of the inputs.

    Context objects are pickled once and arrays are placed in the segment as
    raw bytes; each per-block payload then carries only its block description.
    The historical path re-serialized every array inside every block, so IPC
    volume grew with the worker count; here it is independent of it.  Task
    order and per-replicate math are unchanged, so results stay bit-identical.
    """

    from multiprocessing import shared_memory

    header = pickle.dumps(context, protocol=pickle.HIGHEST_PROTOCOL)
    layout: dict[str, tuple[str, tuple[int, ...], int]] = {}
    offset = len(header)
    prepared: list[tuple[np.ndarray, int]] = []
    for key, value in (arrays or {}).items():
        array = np.ascontiguousarray(value)
        layout[key] = (array.dtype.str, tuple(array.shape), offset)
        prepared.append((array, offset))
        offset += array.nbytes
    segment = shared_memory.SharedMemory(create=True, size=max(1, offset))
    try:
        segment.buf[: len(header)] = header
        for array, start in prepared:
            target = np.frombuffer(segment.buf, dtype=np.uint8, count=array.nbytes, offset=start)
            target[:] = array.view(np.uint8).reshape(-1)
            del target
        spec = {"name": segment.name, "header": len(header), "arrays": layout}
        payloads = [{"shared": spec, "block": block} for block in blocks]
        return list(_shared_pool(int(workers)).map(worker, payloads))
    finally:
        try:
            segment.close()
        except BufferError:
            pass
        try:
            segment.unlink()
        except OSError:
            pass


def _replicate_blocks(replicates: int, workers: int, *, blocks_per_worker: int = 4) -> list[tuple[int, int]]:
    """Return even ``(start, end)`` replicate ranges for the block schedulers."""

    total = int(replicates)
    if total <= 0:
        return []
    block_size = max(1, total // max(1, int(workers) * max(1, int(blocks_per_worker))))
    return [(start, min(start + block_size, total)) for start in range(0, total, block_size)]


def _worker_block(payload: dict[str, Any]) -> np.ndarray:
    """Compute a block of replicates inside a worker process.

    ``payload`` carries a ``(start, end)`` row range of the caller's pre-drawn
    cluster positions; the inputs are attached from the shared segment once per
    worker.  Rows of ``draws`` follow the serial RNG sequence, so computing the
    statistic per row yields bit-identical results to the serial path.
    """

    context, arrays = _load_shared_payload(payload["shared"])
    array = arrays["array"]
    labels_array = arrays.get("labels")
    image_ids = arrays["image_ids"]
    cluster_indices = context["cluster_indices"]
    relabel_clusters = bool(context["relabel_clusters"])
    statistic = _deserialize_statistic(context["statistic"])
    draws = arrays["draws"]
    start, end = payload["block"]
    out = np.empty(end - start, dtype=np.float64)
    for position, row in enumerate(range(start, end)):
        selected_positions = draws[row]
        selected_indices = [cluster_indices[selected] for selected in selected_positions]
        indices = np.concatenate(selected_indices)
        sampled_values = array[indices]
        sampled_labels = None if labels_array is None else labels_array[indices]
        sampled_image_ids = None
        if relabel_clusters:
            occurrence_ids = np.empty(len(indices), dtype=np.int64)
            offset = 0
            for draw_index, selected in enumerate(selected_positions):
                clen = len(cluster_indices[selected])
                occurrence_ids[offset : offset + clen] = draw_index
                offset += clen
            sampled_image_ids = occurrence_ids
        if sampled_image_ids is None:
            sampled_image_ids = image_ids[indices]
        out[position] = float(_invoke_statistic(statistic, sampled_values, sampled_labels, sampled_image_ids))
    return out


def _bootstrap_block_path(
    *,
    array: np.ndarray,
    labels_array: np.ndarray | None,
    image_ids: np.ndarray,
    clusters: list[Any],
    cluster_indices: list[np.ndarray],
    statistic: Callable[..., float],
    replicates: int,
    seed: int,
    confidence: float,
    relabel_clusters: bool = False,
    workers: int = 1,
) -> np.ndarray:
    """Bootstrap replicates with block-level process parallelism.

    All cluster draws are made in exactly the serial sequence on ``seed`` (one
    ``rng.integers`` per replicate over the cluster index space), then the
    replicate loop runs inside persistent worker processes in blocks.  The
    per-replicate statistic, sampling and relabel logic are byte-for-byte the
    same as the serial path, so the resulting sample vector is bit-identical to
    ``workers=1`` regardless of the worker count.
    """

    if int(replicates) <= 0:
        return np.empty(0, dtype=np.float64)

    rng = np.random.default_rng(seed)
    cluster_count = len(cluster_indices)
    draws = rng.integers(0, cluster_count, size=(int(replicates), cluster_count))

    statistic_payload = _serialize_statistic(statistic)
    workers = max(1, int(workers))
    blocks = _replicate_blocks(replicates, workers)
    context = {
        "cluster_indices": cluster_indices,
        "relabel_clusters": relabel_clusters,
        "statistic": statistic_payload,
    }
    arrays: dict[str, Any] = {"draws": draws, "array": array, "image_ids": image_ids}
    if labels_array is not None:
        arrays["labels"] = labels_array
    parts = _shared_payload_map(_worker_block, context, arrays, blocks, workers)
    parts = [part for part in parts if part is not None]
    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(parts)


def _invoke_statistic(
    statistic: Callable[..., Any],
    values: np.ndarray,
    labels: np.ndarray | None,
    source_image_ids: np.ndarray,
) -> float:
    """Invoke the new three-array contract while retaining the old two-array contract."""

    if labels is not None and _cached_can_bind(statistic, 3):
        result = statistic(values, labels, source_image_ids)
    elif _cached_can_bind(statistic, 2):
        result = statistic(values, source_image_ids)
    elif labels is not None:
        raise StatisticsError("statistic must accept (values, labels, source_image_ids) or (values, source_image_ids)")
    else:
        raise StatisticsError("statistic must accept (values, source_image_ids)")
    result_array = np.asarray(result)
    if result_array.ndim != 0:
        raise StatisticsError("statistic must return one scalar value")
    try:
        return float(result_array)
    except (TypeError, ValueError) as error:
        raise StatisticsError("statistic must return one scalar value") from error


def clustered_bootstrap(
    values: Any,
    *args: Any,
    labels: Any | None = None,
    source_image_ids: Any | None = None,
    statistic: Callable[..., float] | None = None,
    replicates: int = 5000,
    seed: int = 1729,
    confidence: float = 0.95,
    confidence_level: float | None = None,
    relabel_clusters: bool = False,
    workers: int = 1,
    clusters: list[Any] | None = None,
    cluster_indices: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Resample complete source-image clusters and return a percentile CI.

    The historical call is ``clustered_bootstrap(values, image_ids, statistic)``;
    its statistic receives ``(values, source_image_ids)``.  Supplying ``labels``
    enables the synchronized three-array contract
    ``statistic(values, labels, source_image_ids)``.  A two-argument statistic is
    still accepted when labels are supplied and receives the historical pair.
    The positional three-array form
    ``clustered_bootstrap(values, labels, image_ids, statistic)`` is also accepted.

    Cluster draws are represented by concatenated complete clusters, so unequal
    numbers of objects per source image remain intact.  When a statistic groups
    by image ID and therefore needs duplicate bootstrap draws to count as separate
    image observations, ``relabel_clusters`` gives each draw a unique occurrence
    ID while preserving row alignment.
    """

    if len(args) == 3:
        if labels is not None or source_image_ids is not None or statistic is not None:
            raise StatisticsError("do not mix positional and keyword clustered bootstrap inputs")
        labels, source_image_ids, statistic = args
    elif len(args) == 2:
        if source_image_ids is not None:
            raise StatisticsError("source_image_ids was supplied more than once")
        if statistic is None:
            source_image_ids, statistic = args
        else:
            if labels is not None:
                raise StatisticsError("labels was supplied more than once")
            labels, source_image_ids = args
    elif len(args) == 1:
        if source_image_ids is not None:
            if labels is None and statistic is not None:
                labels = args[0]
            else:
                raise StatisticsError("source_image_ids was supplied more than once")
        else:
            source_image_ids = args[0]
    elif len(args) != 0:
        raise StatisticsError("expected old (image_ids, statistic) or new (labels, image_ids, statistic) inputs")
    if source_image_ids is None or statistic is None:
        raise StatisticsError("source_image_ids and statistic are required")

    array = np.asarray(values)
    image_ids = np.asarray(source_image_ids)
    labels_array = None if labels is None else np.asarray(labels)
    if array.ndim == 0 or image_ids.ndim != 1 or len(array) == 0 or len(array) != len(image_ids):
        raise StatisticsError("values and source_image_ids must be non-empty and aligned")
    if labels_array is not None and (labels_array.ndim != 1 or len(labels_array) != len(array)):
        raise StatisticsError("labels must align with values")
    if not isinstance(replicates, (int, np.integer)) or int(replicates) <= 0:
        raise StatisticsError("replicates must be a positive integer")
    replicates = int(replicates)
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or int(workers) < 1:
        raise StatisticsError("workers must be a positive integer")
    workers = int(workers)
    if confidence_level is not None:
        confidence = confidence_level
    confidence = _validate_confidence(confidence)

    if clusters is None:
        clusters = _unique_clusters(image_ids)
    if cluster_indices is None:
        cluster_indices = [np.flatnonzero(image_ids == cluster) for cluster in clusters]
    rng = np.random.default_rng(seed)
    observed = _invoke_statistic(statistic, array, labels_array, image_ids)
    samples = np.empty(replicates, dtype=np.float64)
    if workers == 1:
        for replicate in range(replicates):
            samples[replicate] = _bootstrap_replicate(
                statistic,
                array,
                labels_array,
                image_ids,
                cluster_indices,
                rng,
                relabel_clusters=relabel_clusters,
            )
    else:
        samples = _bootstrap_block_path(
            array=array,
            labels_array=labels_array,
            image_ids=image_ids,
            clusters=clusters,
            cluster_indices=cluster_indices,
            statistic=statistic,
            replicates=replicates,
            seed=seed,
            confidence=confidence,
            relabel_clusters=relabel_clusters,
            workers=workers,
        )
        # The block path draws on its own RNG from ``seed`` in the identical
        # sequence, so no ``rng`` forwarding is needed after this point.
    ci_low, ci_high = _percentile_ci(samples, confidence)
    return {
        "estimate": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_level": confidence,
        "replicates": replicates,
        "source_image_count": len(clusters),
        "observation_count": len(array),
        "seed": seed,
    }


def _image_level_means(
    differences: Any,
    source_image_ids: Any,
    clusters: list[Any] | None = None,
) -> tuple[np.ndarray, list[Any]]:
    values = np.asarray(differences, dtype=np.float64)
    image_ids = np.asarray(source_image_ids)
    if values.ndim != 1 or image_ids.ndim != 1 or len(values) == 0 or len(values) != len(image_ids):
        raise StatisticsError("differences and source_image_ids must be aligned and non-empty vectors")
    if not np.all(np.isfinite(values)):
        raise StatisticsError("differences must be finite")
    if clusters is None:
        clusters = _unique_clusters(image_ids)
    image_values = np.asarray([values[image_ids == cluster].mean() for cluster in clusters], dtype=np.float64)
    return image_values, clusters


def paired_difference_ci(
    differences: Any,
    source_image_ids: Any,
    *,
    replicates: int = 10000,
    seed: int = 1729,
    confidence: float = 0.95,
    confidence_level: float | None = None,
    clusters: list[Any] | None = None,
) -> dict[str, Any]:
    """Bootstrap a paired difference after reducing each source image to one value."""

    if not isinstance(replicates, (int, np.integer)) or int(replicates) <= 0:
        raise StatisticsError("replicates must be a positive integer")
    replicates = int(replicates)
    if confidence_level is not None:
        confidence = confidence_level
    confidence = _validate_confidence(confidence)
    image_values, clusters = _image_level_means(differences, source_image_ids, clusters=clusters)
    rng = np.random.default_rng(seed)
    cluster_count = len(image_values)
    if replicates > 1:
        # Blocked draws keep the serial RNG stream and bound the temporary
        # ``replicates x clusters`` allocation; per-row means are unchanged.
        samples = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, _REPLICATE_BLOCK):
            size = min(_REPLICATE_BLOCK, replicates - start)
            draws = rng.integers(0, cluster_count, size=(size, cluster_count))
            samples[start : start + size] = image_values[draws].mean(axis=1)
    else:
        samples = np.empty(replicates, dtype=np.float64)
        for index in range(replicates):
            selected = rng.integers(0, cluster_count, size=cluster_count)
            samples[index] = float(image_values[selected].mean())
    ci_low, ci_high = _percentile_ci(samples, confidence)
    return {
        "estimate": float(image_values.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_level": confidence,
        "replicates": replicates,
        "source_image_count": len(clusters),
        "seed": seed,
    }


def paired_sign_flip(
    differences: Any,
    source_image_ids: Any,
    *,
    replicates: int = 10000,
    seed: int = 1729,
    confidence: float = 0.95,
    confidence_level: float | None = None,
) -> dict[str, Any]:
    """Permutation inference over source-image-level paired differences."""

    if not isinstance(replicates, (int, np.integer)) or int(replicates) <= 0:
        raise StatisticsError("replicates must be a positive integer")
    replicates = int(replicates)
    if confidence_level is not None:
        confidence = confidence_level
    confidence = _validate_confidence(confidence)
    image_values, clusters = _image_level_means(differences, source_image_ids)
    rng = np.random.default_rng(seed)
    observed = float(image_values.mean())
    null = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, _REPLICATE_BLOCK):
        size = min(_REPLICATE_BLOCK, replicates - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(image_values)))
        null[start : start + size] = np.mean(image_values * signs, axis=1)
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (replicates + 1))
    ci_result = paired_difference_ci(
        differences,
        source_image_ids,
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )
    return {
        "estimate": observed,
        "ci_low": ci_result["ci_low"],
        "ci_high": ci_result["ci_high"],
        "confidence_level": confidence,
        "p_value_two_sided": p_value,
        "replicates": replicates,
        "source_image_count": len(clusters),
        "seed": seed,
    }


def benjamini_hochberg(
    p_values: Any,
    *,
    q: float = 0.05,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Apply the Benjamini-Hochberg procedure and return adjusted p-values."""

    p_array = np.asarray(p_values, dtype=np.float64)
    if p_array.ndim != 1:
        raise StatisticsError("p_values must be a one-dimensional vector")
    if alpha is not None:
        if q != 0.05:
            raise StatisticsError("provide only one of q and alpha")
        q = alpha
    q = float(q)
    if not 0.0 < q <= 1.0:
        raise StatisticsError("q must be in the interval (0, 1]")
    finite = np.isfinite(p_array)
    if np.any((p_array[finite] < 0.0) | (p_array[finite] > 1.0)):
        raise StatisticsError("finite p_values must be between zero and one")
    adjusted = np.full(p_array.shape, np.nan, dtype=np.float64)
    reject = np.zeros(p_array.shape, dtype=bool)
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices):
        order = finite_indices[np.argsort(p_array[finite_indices], kind="mergesort")]
        sorted_p = p_array[order]
        ranks = np.arange(1, len(order) + 1, dtype=np.float64)
        sorted_adjusted = np.minimum.accumulate((sorted_p * len(order) / ranks)[::-1])[::-1]
        sorted_adjusted = np.clip(sorted_adjusted, 0.0, 1.0)
        adjusted[order] = sorted_adjusted
        reject[order] = sorted_adjusted <= q
    rejected_p = p_array[reject]
    critical_value = float(np.max(rejected_p)) if len(rejected_p) else float("nan")
    return {
        "q_values": adjusted,
        "adjusted_p_values": adjusted,
        "reject": reject,
        "rejections": reject,
        "q": q,
        "critical_value": critical_value,
        "threshold": critical_value,
        "tested_count": len(finite_indices),
    }


def bh_fdr(p_values: Any, *, q: float = 0.05, alpha: float | None = None) -> dict[str, Any]:
    """Alias for :func:`benjamini_hochberg`."""

    return benjamini_hochberg(p_values, q=q, alpha=alpha)


clustered_bootstrap_ci = clustered_bootstrap
paired_cluster_bootstrap = paired_difference_ci
paired_bootstrap_difference = paired_difference_ci
fdr_bh = benjamini_hochberg


def clustered_bootstrap_multi(
    values: Any,
    source_image_ids: Any,
    statistic: Callable[..., Any],
    *,
    labels: Any | None = None,
    replicates: int = 5000,
    seed: int = 1729,
    confidence: float = 0.95,
    confidence_level: float | None = None,
    relabel_clusters: bool = False,
) -> dict[str, Any]:
    """Bootstrap several statistics in a single shared sample pass.

    ``statistic`` must return a mapping (``dict``) whose values are scalar
    metrics; the observed estimate is computed once and each metric is
    resampled over the *same* cluster draw sequence, so the replicate draws
    are identical to calling :func:`clustered_bootstrap` per metric with the
    same ``seed``.  This avoids resampling -- and re-sorting the correlation
    matrix -- once per metric (e.g. E08's four retrieval metrics), which is
    the dominant cost for large candidate sets.

    The caller-RNG draw sequence and the percentile CI computation match
    :func:`clustered_bootstrap` exactly, so a multi-metric result is
    bit-identical to the per-metric calls it replaces.
    """

    image_ids = np.asarray(source_image_ids)
    array = np.asarray(values)
    labels_array = None if labels is None else np.asarray(labels)
    if array.ndim == 0 or image_ids.ndim != 1 or len(array) == 0 or len(array) != len(image_ids):
        raise StatisticsError("values and source_image_ids must be non-empty and aligned")
    if labels_array is not None and (labels_array.ndim != 1 or len(labels_array) != len(array)):
        raise StatisticsError("labels must align with values")
    if not isinstance(replicates, (int, np.integer)) or int(replicates) <= 0:
        raise StatisticsError("replicates must be a positive integer")
    replicates = int(replicates)
    if confidence_level is not None:
        confidence = confidence_level
    confidence = _validate_confidence(confidence)

    clusters = _unique_clusters(image_ids)
    cluster_indices = [np.flatnonzero(image_ids == cluster) for cluster in clusters]
    rng = np.random.default_rng(seed)

    observed = statistic(array, labels_array, image_ids)
    if not isinstance(observed, dict):
        raise StatisticsError("statistic must return a mapping of scalar metrics")
    metric_names = list(observed.keys())

    raw_observed = observed
    sample_values: dict[str, np.ndarray] = {
        name: np.empty(replicates, dtype=np.float64) for name in metric_names
    }
    for replicate in range(replicates):
        sampled_values, sampled_labels, sampled_images = _bootstrap_replicate_inputs(
            array,
            labels_array,
            image_ids,
            cluster_indices,
            rng,
            relabel_clusters=relabel_clusters,
        )
        replicate_result = statistic(sampled_values, sampled_labels, sampled_images)
        for name in metric_names:
            sample_values[name][replicate] = float(replicate_result[name])

    result: dict[str, Any] = {
        "estimate": raw_observed,
        "source_image_count": len(clusters),
        "observation_count": len(array),
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence,
        "metrics": {},
    }
    for name in metric_names:
        ci_low, ci_high = _percentile_ci(sample_values[name], confidence)
        result["metrics"][name] = {
            "estimate": float(raw_observed[name]),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence_level": confidence,
            "replicates": replicates,
            "source_image_count": len(clusters),
            "observation_count": len(array),
            "seed": seed,
        }
    return result
