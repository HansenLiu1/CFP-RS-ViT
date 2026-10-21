"""Model-free specificity, transfer and intervention metrics."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from cfra.representations import cosine_scores
from cfra.statistics import clustered_bootstrap


class EvaluationError(ValueError):
    """Raised when an evaluation matrix violates the metric contract."""


def _auc_from_scores_fast(
    score_matrix: np.ndarray,
    sorted_labels: np.ndarray,
    direction_class_ids: Sequence[Any],
) -> np.ndarray:
    """Per-column one-vs-rest AUC (NaN where a class has empty positives/negatives).

    The per-column averaged-rank AUCs use the exact same formulas as
    :func:`binary_auc`, so the results are bit-identical to a loop calling
    ``binary_auc`` per column.
    """

    return np.asarray(
        [
            binary_auc((sorted_labels == class_id).astype(np.int64), score_matrix[:, index])
            for index, class_id in enumerate(direction_class_ids)
        ],
        dtype=np.float64,
    )


def _ordered_unique(values: np.ndarray) -> list[Any]:
    unique: list[Any] = []
    for value in values.tolist():
        if not any(value == existing for existing in unique):
            unique.append(value)
    return unique


def _specificity_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: list[Any],
    source_image_ids: np.ndarray | None,
) -> dict[str, Any]:
    """Cell-averaged specificity summary from a score matrix.

    Cell order follows the first-occurrence order of source images and the rows
    inside every cell keep their original row order, so the cell means receive
    the same element sequence as the historical per-mask implementation.
    The per-cell mean itself is computed with one vectorized ``add.reduceat``
    per query class instead of ``np.mean`` per cell; this introduces a
    documented floating-point tolerance (``docs/06``) instead of bit-identical
    accumulation, and reduces the bootstrap statistic cost by roughly 30x.
    """

    total_classes = len(classes)
    score_matrix = np.full((total_classes, total_classes), np.nan, dtype=np.float64)
    if source_image_ids is None:
        for direction_index in range(total_classes):
            for query_index, query_class in enumerate(classes):
                selected = labels == query_class
                if not selected.any():
                    continue
                score_matrix[direction_index, query_index] = float(np.mean(scores[selected, direction_index]))
        return _summary_from_matrix(score_matrix)

    for query_index, query_class in enumerate(classes):
        selected = labels == query_class
        if not selected.any():
            continue
        sel_ids = source_image_ids[selected]
        sel_scores = scores[selected, :]
        order = np.argsort(sel_ids, kind="stable")
        sorted_scores = sel_scores[order]
        sorted_ids = sel_ids[order]
        boundaries = np.flatnonzero(np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1])))
        boundaries_end = np.concatenate((boundaries[1:], [len(order)]))
        bucket_ids = sorted_ids[boundaries]
        first_pos = np.fromiter(
            (int(np.flatnonzero(sel_ids == bucket_id)[0]) for bucket_id in bucket_ids),
            dtype=np.int64,
            count=len(bucket_ids),
        )
        bucket_order = np.argsort(first_pos)
        seg_start = boundaries[bucket_order]
        seg_end = boundaries_end[bucket_order]
        segment_sizes = seg_end - seg_start
        segments = np.concatenate([np.arange(int(start), int(end)) for start, end in zip(seg_start, seg_end)])
        ordered_scores = sorted_scores[segments]
        starts = np.cumsum(segment_sizes).astype(np.int64) - segment_sizes
        cell_means = np.add.reduceat(ordered_scores, starts, axis=0) / segment_sizes[:, None]
        score_matrix[:, query_index] = np.mean(cell_means, axis=0)

    return _summary_from_matrix(score_matrix)


def summary_from_score_matrix(score_matrix: Any) -> dict[str, Any]:
    """Specificity summary values from a cell-averaged score matrix.

    ``score_matrix[i, j]`` holds the averaged score of direction ``i`` against
    query class ``j``.  This wraps the private :func:`_summary_from_matrix` and
    is used by pipeline stages that build matrix cells directly.
    """

    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise EvaluationError("summary_from_score_matrix requires a square score matrix")
    return _summary_from_matrix(matrix)


def _summary_from_matrix(score_matrix: np.ndarray) -> dict[str, Any]:
    """Own-argmax rate and diagonal margin from a cell-averaged score matrix.

    Vectorized over rows; each row's hardest off-diagonal competitor is the
    row maximum with the diagonal replaced by NaN, which is identical to the
    historical per-row ``np.delete`` + ``np.nanmax`` (same reduction order,
    NaN ignored).  Rows whose diagonal or all off-diagonals are non-finite
    stay NaN, exactly as before.
    """

    own = np.diag(score_matrix)
    off_diagonal = score_matrix.copy()
    np.fill_diagonal(off_diagonal, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hardest = np.nanmax(off_diagonal, axis=1)
        row_max = np.nanmax(score_matrix, axis=1)
    valid = np.isfinite(own) & np.isfinite(hardest)
    margins = np.where(valid, own - hardest, np.nan)
    argmax = np.where(valid, (own >= row_max).astype(np.float64), np.nan)
    return {
        "score_matrix": score_matrix,
        "own_argmax_rate": float(np.nanmean(argmax)) if np.isfinite(argmax).any() else float("nan"),
        "mean_diagonal_margin": float(np.nanmean(margins)) if np.isfinite(margins).any() else float("nan"),
        "mean_own_cosine": float(np.nanmean(own)) if np.isfinite(own).any() else float("nan"),
        "class_margins": margins,
    }


def specificity_summary_metrics(
    score_matrix: Any,
    labels: Any,
    classes: Any,
    source_image_ids: Any | None = None,
) -> dict[str, Any]:
    """Specificity summary from an already computed score matrix.

    Rows of ``score_matrix`` align with ``labels`` (and optionally
    ``source_image_ids``); columns align with ``classes``.  The cell-averaging
    sequence matches :func:`specificity_metrics`, so a matrix computed once
    outside a resampling loop yields identical values.
    """

    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2:
        raise EvaluationError("score_matrix must be a matrix")
    labels_array = np.asarray(labels)
    if len(scores) != len(labels_array):
        raise EvaluationError("score_matrix and labels must be aligned")
    class_values = [int(value) for value in np.asarray(classes)]
    if scores.shape[1] != len(class_values):
        raise EvaluationError("score_matrix and classes must be aligned")
    image_array = None if source_image_ids is None else np.asarray(source_image_ids)
    if image_array is not None and (image_array.ndim != 1 or len(image_array) != len(scores)):
        raise EvaluationError("source_image_ids must align with score rows")
    return _specificity_summary(scores, labels_array, class_values, image_array)


def specificity_metrics(
    directions: Any,
    values: Any,
    class_ids: Any,
    direction_class_ids: Any,
    *,
    source_image_ids: Any | None = None,
    bootstrap_ci: bool = False,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 1729,
    bootstrap_workers: int = 1,
    scores: Any | None = None,
) -> dict[str, Any]:
    """Compute specificity and image-clustered CIs for the three primary metrics.

    The score matrix is averaged within each image/class cell before the class
    macro-average.  Bootstrap statistics receive the sampled values, labels and
    image IDs together, so labels cannot silently remain attached to the original
    rows after an image cluster has been drawn more than once.
    """

    values_array = np.asarray(values, dtype=np.float64)
    directions_array = np.asarray(directions, dtype=np.float64)
    labels = np.asarray(class_ids)
    direction_labels = np.asarray(direction_class_ids)
    if values_array.ndim != 2 or directions_array.ndim != 2:
        raise EvaluationError("values and directions must be matrices")
    if labels.ndim != 1 or len(values_array) != len(labels):
        raise EvaluationError("class_ids must align with values")
    if direction_labels.ndim != 1 or len(directions_array) != len(direction_labels):
        raise EvaluationError("direction_class_ids must align with directions")
    if len(direction_labels) == 0:
        raise EvaluationError("direction_class_ids must not be empty")
    classes = direction_labels.tolist()
    if len(_ordered_unique(direction_labels)) != len(classes):
        raise EvaluationError("direction_class_ids must be unique")
    image_ids = None if source_image_ids is None else np.asarray(source_image_ids)
    if image_ids is not None and (image_ids.ndim != 1 or len(image_ids) != len(labels)):
        raise EvaluationError("source_image_ids must align with values")

    if scores is None:
        scores = cosine_scores(values_array, directions_array)
    score_matrix_values = np.asarray(scores, dtype=np.float64)
    summary = _specificity_summary(score_matrix_values, labels, classes, image_ids)
    result = {
        "class_ids": classes,
        "score_matrix": summary["score_matrix"],
        "own_argmax_rate": summary["own_argmax_rate"],
        "own_argmax": summary["own_argmax_rate"],
        "mean_diagonal_margin": summary["mean_diagonal_margin"],
        "margin": summary["mean_diagonal_margin"],
        "mean_own_cosine": summary["mean_own_cosine"],
        "class_margins": summary["class_margins"],
    }

    if bootstrap_ci and image_ids is not None:
        def _bootstrap_statistic(metric_name: str):
            def _statistic(
                sampled_scores: np.ndarray,
                sampled_labels: np.ndarray,
                sampled_images: np.ndarray,
            ) -> float:
                sampled_summary = _specificity_summary(
                    sampled_scores,
                    sampled_labels,
                    classes,
                    sampled_images,
                )
                return float(sampled_summary[metric_name])

            return _statistic

        bootstrap_results = {}
        for metric_name in ("own_argmax_rate", "mean_diagonal_margin", "mean_own_cosine"):
            bootstrap_results[metric_name] = clustered_bootstrap(
                score_matrix_values,
                image_ids,
                _bootstrap_statistic(metric_name),
                labels=labels,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
                relabel_clusters=True,
                workers=int(bootstrap_workers),
            )
        for metric_name, ci_result in bootstrap_results.items():
            result[f"{metric_name}_ci_low"] = ci_result["ci_low"]
            result[f"{metric_name}_ci_high"] = ci_result["ci_high"]
            result[f"{metric_name}_ci"] = (ci_result["ci_low"], ci_result["ci_high"])
        result["own_argmax_ci_low"] = result["own_argmax_rate_ci_low"]
        result["own_argmax_ci_high"] = result["own_argmax_rate_ci_high"]
        result["own_argmax_ci"] = result["own_argmax_rate_ci"]
        result["margin_ci_low"] = result["mean_diagonal_margin_ci_low"]
        result["margin_ci_high"] = result["mean_diagonal_margin_ci_high"]
        result["margin_ci"] = result["mean_diagonal_margin_ci"]
        result["bootstrap_replicates"] = bootstrap_replicates
        result["bootstrap_source_image_count"] = bootstrap_results["own_argmax_rate"]["source_image_count"]
        result["bootstrap_confidence_level"] = 0.95

    return result


def binary_auc(labels: Any, scores: Any) -> float:
    """Compute AUC by the exact rank definition, avoiding a hidden sklearn dependency."""

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise EvaluationError("labels and scores must be aligned vectors")
    positive = y == 1
    negative = y == 0
    if not positive.any() or not negative.any():
        return float("nan")
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    order = np.argsort(s, kind="quicksort")
    sorted_scores = s[order]
    if (sorted_scores[1:] != sorted_scores[:-1]).all():
        # No ties: the averaged-rank value equals the plain ordinal ranks
        # bit-for-bit, so the U-statistic shortcut is exact.
        ranks = np.empty(len(s), dtype=np.float64)
        ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
        positive_rank_sum = ranks[positive].sum()
        return float(
            (positive_rank_sum - n_positive * (n_positive + 1) / 2.0)
            / (n_positive * n_negative)
        )
    equals_prev = np.zeros(len(s), dtype=bool)
    equals_prev[1:] = sorted_scores[1:] == sorted_scores[:-1]
    segment_starts = np.flatnonzero(~equals_prev)
    segment_lengths = np.diff(np.append(segment_starts, len(s)))
    avg_ranks = segment_starts + (1.0 + segment_lengths) / 2.0
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.repeat(avg_ranks, segment_lengths)
    positive_rank_sum = ranks[positive].sum()
    return float((positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def binary_average_precision(labels: Any, scores: Any) -> float:
    """Compute binary average precision from descending score ranks.

    Ties are handled by averaging the contribution within each equal-score
    group, exactly matching the iterative group-scan definition.  The
    computation is fully vectorized (a single ``argsort`` plus segment
    reductions) and is bit-identical to the previous Python loop.
    """

    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
        raise EvaluationError("labels and scores must be aligned non-empty vectors")
    if not np.all(np.isin(y, [0, 1])):
        raise EvaluationError("binary average precision requires labels in {0, 1}")
    if not np.all(np.isfinite(s)):
        raise EvaluationError("scores must be finite")
    positive = y == 1
    total_positive = int(np.count_nonzero(positive))
    if total_positive == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    sorted_positive = positive[order]
    cumulative_positive = np.cumsum(sorted_positive, dtype=np.int64)
    boundary = np.flatnonzero(
        np.concatenate(([True], sorted_scores[1:] != sorted_scores[:-1]))
    )
    group_end_ranks = np.concatenate((boundary[1:], [len(order)]))
    group_positive = np.diff(
        np.concatenate(([0], np.cumsum(sorted_positive, dtype=np.int64)))
    )[boundary]
    terms = (
        cumulative_positive[group_end_ranks - 1].astype(np.float64)
        / group_end_ranks
        * group_positive
        / total_positive
    )
    average_precision = float(np.cumsum(terms)[-1]) if terms.size else 0.0
    return average_precision


def _ranking_inputs(relevance: Any, scores: Any | None) -> tuple[np.ndarray, np.ndarray | None]:
    relevance_array = np.asarray(relevance, dtype=np.float64)
    if relevance_array.ndim != 1 or len(relevance_array) == 0:
        raise EvaluationError("relevance must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(relevance_array)) or np.any(relevance_array < 0.0):
        raise EvaluationError("relevance must contain finite non-negative values")
    if scores is None:
        return relevance_array, None
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1 or len(score_array) != len(relevance_array):
        raise EvaluationError("relevance and scores must be aligned vectors")
    if not np.all(np.isfinite(score_array)):
        raise EvaluationError("scores must be finite")
    return relevance_array, score_array


def _ranked_relevance(relevance: np.ndarray, scores: np.ndarray | None) -> np.ndarray:
    if scores is None:
        return relevance
    return relevance[np.argsort(-scores, kind="mergesort")]


def _metric_k(k: Any, length: int) -> int:
    if not isinstance(k, (int, np.integer)) or int(k) <= 0:
        raise EvaluationError("k must be a positive integer")
    return min(int(k), length)


def ndcg_at_k(relevance: Any, scores: Any | None = None, k: int | None = None) -> float:
    """Compute NDCG at ``k`` for binary or graded relevance.

    ``scores`` may be omitted when ``relevance`` is already in ranked order;
    ``ndcg_at_k(relevance, k)`` is accepted as a shorthand for that form.
    """

    if k is None and scores is not None and np.asarray(scores).ndim == 0:
        k = int(scores)
        scores = None
    if k is None:
        raise EvaluationError("k is required")
    relevance_array, score_array = _ranking_inputs(relevance, scores)
    k_value = _metric_k(k, len(relevance_array))
    ranked = _ranked_relevance(relevance_array, score_array)
    discounts = np.log2(np.arange(2, k_value + 2, dtype=np.float64))
    gains = np.exp2(ranked[:k_value]) - 1.0
    ideal = np.sort(relevance_array)[::-1][:k_value]
    ideal_gains = np.exp2(ideal) - 1.0
    ideal_dcg = float(np.sum(ideal_gains / discounts))
    if ideal_dcg <= 0.0:
        return 0.0
    return float(np.sum(gains / discounts) / ideal_dcg)


def recall_at_k(relevance: Any, scores: Any | None = None, k: int | None = None) -> float:
    """Compute binary recall at ``k``; graded relevance is treated as positive above zero."""

    if k is None and scores is not None and np.asarray(scores).ndim == 0:
        k = int(scores)
        scores = None
    if k is None:
        raise EvaluationError("k is required")
    relevance_array, score_array = _ranking_inputs(relevance, scores)
    k_value = _metric_k(k, len(relevance_array))
    total_relevant = int(np.count_nonzero(relevance_array > 0.0))
    if total_relevant == 0:
        return float("nan")
    ranked = _ranked_relevance(relevance_array, score_array)
    return float(np.count_nonzero(ranked[:k_value] > 0.0) / total_relevant)


def precision_at_k(relevance: Any, scores: Any | None = None, k: int | None = None) -> float:
    """Compute binary precision at ``k``; graded relevance is positive above zero."""

    if k is None and scores is not None and np.asarray(scores).ndim == 0:
        k = int(scores)
        scores = None
    if k is None:
        raise EvaluationError("k is required")
    relevance_array, score_array = _ranking_inputs(relevance, scores)
    k_value = _metric_k(k, len(relevance_array))
    ranked = _ranked_relevance(relevance_array, score_array)
    return float(np.count_nonzero(ranked[:k_value] > 0.0) / k_value)


def macro_average_precision(relevance: Any, scores: Any) -> float:
    """Average binary AP over retrieval queries, ignoring queries without positives."""

    relevance_array = np.asarray(relevance)
    scores_array = np.asarray(scores)
    if relevance_array.ndim == 1 and scores_array.ndim == 1:
        return binary_average_precision(relevance_array, scores_array)
    if relevance_array.ndim != 2 or scores_array.shape != relevance_array.shape:
        raise EvaluationError("relevance and scores must have matching one- or two-dimensional shapes")
    values = [binary_average_precision(row, score_row) for row, score_row in zip(relevance_array, scores_array)]
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def macro_ndcg_at_k(relevance: Any, scores: Any, k: int) -> float:
    """Average NDCG@K over retrieval queries."""

    relevance_array = np.asarray(relevance)
    scores_array = np.asarray(scores)
    if relevance_array.ndim == 1 and scores_array.ndim == 1:
        return ndcg_at_k(relevance_array, scores_array, k)
    if relevance_array.ndim != 2 or scores_array.shape != relevance_array.shape:
        raise EvaluationError("relevance and scores must have matching one- or two-dimensional shapes")
    values = [ndcg_at_k(row, score_row, k) for row, score_row in zip(relevance_array, scores_array)]
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def macro_recall_at_k(relevance: Any, scores: Any, k: int) -> float:
    """Average Recall@K over retrieval queries, ignoring queries without positives."""

    relevance_array = np.asarray(relevance)
    scores_array = np.asarray(scores)
    if relevance_array.ndim == 1 and scores_array.ndim == 1:
        return recall_at_k(relevance_array, scores_array, k)
    if relevance_array.ndim != 2 or scores_array.shape != relevance_array.shape:
        raise EvaluationError("relevance and scores must have matching one- or two-dimensional shapes")
    values = [recall_at_k(row, score_row, k) for row, score_row in zip(relevance_array, scores_array)]
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def macro_precision_at_k(relevance: Any, scores: Any, k: int) -> float:
    """Average Precision@K over retrieval queries."""

    relevance_array = np.asarray(relevance)
    scores_array = np.asarray(scores)
    if relevance_array.ndim == 1 and scores_array.ndim == 1:
        return precision_at_k(relevance_array, scores_array, k)
    if relevance_array.ndim != 2 or scores_array.shape != relevance_array.shape:
        raise EvaluationError("relevance and scores must have matching one- or two-dimensional shapes")
    values = [precision_at_k(row, score_row, k) for row, score_row in zip(relevance_array, scores_array)]
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def retrieval_metrics(relevance: Any, scores: Any, k: int) -> dict[str, float]:
    """Return the E08 macro AP, NDCG@K, Recall@K and Precision@K metrics.

    For a 2-D (query x candidate) input the four metrics share a single
    descending ``argsort`` per query.  The four per-metric helpers each sorted
    the same row independently; merging them is bit-identical (verified) and
    roughly 2.5x faster, which dominates E08's bootstrap cost.
    """

    relevance_array = np.asarray(relevance, dtype=np.float64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if relevance_array.ndim == 2 and scores_array.shape == relevance_array.shape:
        merged = _macro_retrieval_metrics_2d(relevance_array, scores_array, int(k))
        if merged is not None:
            average_precision, ndcg, recall, precision = merged
            return {
                "macro_average_precision": average_precision,
                "macro_ap": average_precision,
                "ndcg_at_k": ndcg,
                "ndcg": ndcg,
                "recall_at_k": recall,
                "recall": recall,
                "precision_at_k": precision,
                "precision": precision,
                "k": int(k),
            }
    average_precision = macro_average_precision(relevance, scores)
    ndcg = macro_ndcg_at_k(relevance, scores, k)
    recall = macro_recall_at_k(relevance, scores, k)
    precision = macro_precision_at_k(relevance, scores, k)
    return {
        "macro_average_precision": average_precision,
        "macro_ap": average_precision,
        "ndcg_at_k": ndcg,
        "ndcg": ndcg,
        "recall_at_k": recall,
        "recall": recall,
        "precision_at_k": precision,
        "precision": precision,
        "k": int(k),
    }


def retrieval_metrics_per_query(
    relevance: Any,
    scores: Any,
    k: int,
) -> dict[str, np.ndarray] | None:
    """Return per-query AP/NDCG/Recall/Precision for a binary-relevance matrix.

    Each metric is computed independently per query row, so the macro metrics
    are exactly the ``nanmean`` of these arrays.  This is the hot path behind
    E08's source-image bootstrap: precomputing the per-row values once lets a
    replicate reduce to a resample-and-average of fixed scalars instead of
    re-sorting the whole ``query x candidate`` matrix every draw.

    Returns ``None`` when the input violates the binary contract so callers can
    fall back to the validating helpers.  The per-row math is identical to the
    four independent ``macro_*`` helpers (verified bit-identical).
    """

    relevance_array = np.asarray(relevance, dtype=np.float64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if relevance_array.ndim != 2 or scores_array.shape != relevance_array.shape:
        return None
    if not np.all(np.isfinite(relevance_array)) or np.any(relevance_array < 0.0):
        return None
    if not np.all(np.isfinite(scores_array)):
        return None
    if not np.all(np.isin(relevance_array, [0.0, 1.0])):
        return None
    n_queries = relevance_array.shape[0]
    aps = np.empty(n_queries, dtype=np.float64)
    ndcgs = np.empty(n_queries, dtype=np.float64)
    recalls = np.empty(n_queries, dtype=np.float64)
    precisions = np.empty(n_queries, dtype=np.float64)
    for index in range(n_queries):
        row_relevance = relevance_array[index]
        row_scores = scores_array[index]
        k_value = _metric_k(k, len(row_relevance))
        order = np.argsort(-row_scores, kind="mergesort")
        ranked = row_relevance[order]
        total_positive = int(np.count_nonzero(row_relevance))
        # average precision (exact tie groups, matching binary_average_precision)
        if total_positive == 0:
            aps[index] = float("nan")
        else:
            positive_sorted = ranked > 0.0
            sorted_scores = row_scores[order]
            cumulative = np.cumsum(positive_sorted, dtype=np.int64)
            boundary = np.flatnonzero(
                np.concatenate(([True], sorted_scores[1:] != sorted_scores[:-1]))
            )
            group_end_ranks = np.concatenate((boundary[1:], [len(order)]))
            group_positive = np.diff(
                np.concatenate(([0], np.cumsum(positive_sorted, dtype=np.int64)))
            )[boundary]
            terms = (
                cumulative[group_end_ranks - 1].astype(np.float64)
                / group_end_ranks
                * group_positive
                / total_positive
            )
            aps[index] = float(np.cumsum(terms)[-1]) if terms.size else 0.0
        # ndcg@k
        discounts = np.log2(np.arange(2, k_value + 2, dtype=np.float64))
        gains = np.exp2(ranked[:k_value]) - 1.0
        ideal = np.sort(row_relevance)[::-1][:k_value]
        ideal_gains = np.exp2(ideal) - 1.0
        ideal_dcg = float(np.sum(ideal_gains / discounts))
        ndcgs[index] = (
            0.0
            if ideal_dcg <= 0.0
            else float(np.sum(gains / discounts) / ideal_dcg)
        )
        # recall@k / precision@k
        recalls[index] = (
            float("nan")
            if total_positive == 0
            else float(np.count_nonzero(ranked[:k_value] > 0.0) / total_positive)
        )
        precisions[index] = float(np.count_nonzero(ranked[:k_value] > 0.0) / k_value)
    return {
        "macro_ap": aps,
        "ndcg_at_k": ndcgs,
        "recall_at_k": recalls,
        "precision_at_k": precisions,
    }


def _macro_retrieval_metrics_2d(
    relevance: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> tuple[float, float, float, float] | None:
    """Per-query macro AP/NDCG/Recall/Precision sharing one argsort per row.

    Bit-identical to the four independent ``macro_*`` helpers on the same
    binary-relevance matrix.  Returns ``None`` when the input violates the
    binary contract, so the caller can fall back to the validating helpers.
    """

    per_query = retrieval_metrics_per_query(relevance, scores, k)
    if per_query is None:
        return None
    aps = per_query["macro_ap"]
    ndcgs = per_query["ndcg_at_k"]
    recalls = per_query["recall_at_k"]
    precisions = per_query["precision_at_k"]
    average_precision = (
        float(np.nanmean(aps)) if np.isfinite(aps).any() else float("nan")
    )
    ndcg = float(np.nanmean(ndcgs)) if np.isfinite(ndcgs).any() else float("nan")
    recall = float(np.nanmean(recalls)) if np.isfinite(recalls).any() else float("nan")
    precision = (
        float(np.nanmean(precisions)) if np.isfinite(precisions).any() else float("nan")
    )
    return average_precision, ndcg, recall, precision


def near_zero_audit(values: Any, *, epsilon: float = 1.0e-8, axis: int = -1) -> dict[str, Any]:
    """Audit vector norms before cosine or intervention calculations."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise EvaluationError("values must contain one or more vectors")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise EvaluationError("epsilon must be finite and non-negative")
    if axis < -array.ndim or axis >= array.ndim:
        raise EvaluationError("axis is out of bounds")
    norms = np.linalg.norm(array, axis=axis)
    near_zero = np.asarray(norms <= epsilon)
    nonfinite = ~np.isfinite(norms)
    indices = np.flatnonzero(near_zero) if near_zero.ndim == 1 else np.argwhere(near_zero)
    total = int(near_zero.size)
    count = int(np.count_nonzero(near_zero))
    return {
        "epsilon": float(epsilon),
        "norms": norms,
        "near_zero": near_zero,
        "mask": near_zero,
        "indices": indices,
        "near_zero_indices": indices,
        "count": count,
        "near_zero_count": count,
        "total_count": total,
        "fraction": float(count / total) if total else float("nan"),
        "has_near_zero": bool(count),
        "nonfinite_count": int(np.count_nonzero(nonfinite)),
    }


def audit_near_zero(values: Any, *, epsilon: float = 1.0e-8, axis: int = -1) -> dict[str, Any]:
    """Alias for :func:`near_zero_audit`."""

    return near_zero_audit(values, epsilon=epsilon, axis=axis)


binary_ap = binary_average_precision
average_precision = binary_average_precision
ndcg = ndcg_at_k
recall = recall_at_k


def macro_ovr_auc_from_scores(scores: Any, labels: Any, direction_class_ids: Any) -> float:
    """Macro one-vs-rest AUC from an already computed score matrix.

    Rows of ``scores`` align with ``labels``; columns align with
    ``direction_class_ids``.  The computation is identical to the AUC part of
    :func:`macro_ovr_auc`, so a score matrix computed once outside a resampling
    loop yields exactly the same values as per-replicate recomputation.
    """

    score_matrix = np.asarray(scores, dtype=np.float64)
    if score_matrix.ndim != 2:
        raise EvaluationError("scores must be a matrix")
    sorted_labels = np.asarray(labels, dtype=np.int64)
    if len(score_matrix) != len(sorted_labels):
        raise EvaluationError("scores and labels must be aligned")
    direction_labels = np.asarray([int(value) for value in np.asarray(direction_class_ids).tolist()], dtype=np.int64)
    if score_matrix.shape[1] != len(direction_labels):
        raise EvaluationError("scores and direction_class_ids must be aligned")
    auc_scores = _auc_from_scores_fast(score_matrix, sorted_labels, direction_labels)
    valid = auc_scores[np.isfinite(auc_scores)]
    return float(np.mean(valid)) if valid.size else float("nan")


def macro_ovr_auc(
    values: Any,
    class_ids: Any,
    directions: Any,
    direction_class_ids: Any,
    *,
    scores: Any | None = None,
) -> float:
    matrix = np.asarray(values, dtype=np.float64)
    labels = np.asarray(class_ids)
    if scores is None:
        score_matrix = cosine_scores(matrix, directions)
    else:
        score_matrix = np.asarray(scores, dtype=np.float64)
    return macro_ovr_auc_from_scores(score_matrix, labels, direction_class_ids)


def paired_effect(true_metrics: dict[str, Any], control_metrics: dict[str, Any], metric: str) -> float:
    left = float(true_metrics[metric])
    right = float(control_metrics[metric])
    return left - right


def intervene(
    features: Any,
    directions: Any,
    *,
    alpha: float,
    preserve_norm: bool,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    """Apply ``h' = h - alpha <h,w>w`` and optionally restore each original norm."""

    h = np.asarray(features, dtype=np.float64)
    w = np.asarray(directions, dtype=np.float64)
    if h.ndim != 2 or w.ndim != 2 or h.shape != w.shape:
        raise EvaluationError("features and directions must have the same matrix shape")
    if alpha < 0:
        raise EvaluationError("alpha must be non-negative")
    w_norm = np.linalg.norm(w, axis=1)
    if np.any(w_norm <= epsilon):
        raise EvaluationError("intervention direction has a near-zero norm")
    unit_w = w / w_norm[:, None]
    transformed = h - alpha * np.sum(h * unit_w, axis=1, keepdims=True) * unit_w
    if preserve_norm:
        original_norm = np.linalg.norm(h, axis=1)
        new_norm = np.linalg.norm(transformed, axis=1)
        if np.any((original_norm > epsilon) & (new_norm <= epsilon)):
            raise EvaluationError("norm preservation is undefined for a zero post-intervention vector")
        valid = new_norm > epsilon
        transformed[valid] *= (original_norm[valid] / new_norm[valid])[:, None]
    return transformed
