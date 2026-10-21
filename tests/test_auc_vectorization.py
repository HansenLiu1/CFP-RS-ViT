"""Bit-identical contract tests for the per-class AUC helper.

``_vectorized_class_auc`` is the shared primitive behind every bootstrap AUC
path in the core stage.  The reference loop below is a frozen copy of the
historical ranking implementation; these tests pin the live helper to it byte
for byte, including tie handling and classes without positives or negatives.
"""

import unittest

import numpy as np

from cfra.pipeline.core import _fast_score_auc_ci, _vectorized_class_auc
from cfra.statistics.clustered import clustered_bootstrap


def _reference_class_auc(score_matrix, labels, classes):
    """The historical per-column ranking loop, kept verbatim as a reference."""

    matrix = np.asarray(score_matrix, dtype=np.float64)
    n_rows, direction_count = matrix.shape
    label_values = np.asarray(labels, dtype=np.int64)
    positive = np.stack(
        [(label_values == int(class_id)) for class_id in classes.tolist()], axis=1
    )
    n_positive = positive.sum(axis=0, dtype=np.int64)
    n_negative = n_rows - n_positive
    valid = (n_positive > 0) & (n_negative > 0)
    order = np.argsort(matrix, axis=0, kind="quicksort")
    sorted_matrix = np.take_along_axis(matrix, order, axis=0)
    equals_prev = np.zeros((n_rows, direction_count), dtype=bool)
    equals_prev[1:] = sorted_matrix[1:] == sorted_matrix[:-1]
    ranks = np.empty((n_rows, direction_count), dtype=np.float64)
    for column in range(direction_count):
        column_starts = np.flatnonzero(~equals_prev[:, column])
        column_lengths = np.diff(np.append(column_starts, n_rows))
        average_ranks = column_starts + (1.0 + column_lengths) / 2.0
        column_ranks = np.repeat(average_ranks, column_lengths)
        ranks[order[:, column], column] = column_ranks
    positive_rank_sum = np.where(positive, ranks, 0.0).sum(axis=0)
    scores = np.full(direction_count, np.nan, dtype=np.float64)
    denom = n_positive[valid] * n_negative[valid]
    scores[valid] = (
        positive_rank_sum[valid] - n_positive[valid] * (n_positive[valid] + 1) / 2.0
    ) / denom
    return np.asarray(scores, dtype=np.float64)


class VectorizedClassAucTests(unittest.TestCase):
    def _assert_bit_identical(self, matrix, labels, classes) -> None:
        expected = _reference_class_auc(matrix, labels, classes)
        actual = _vectorized_class_auc(matrix, labels, classes)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.array_equal(np.asarray(actual), np.asarray(expected), equal_nan=True))
        self.assertEqual(
            np.ascontiguousarray(actual).tobytes(), np.ascontiguousarray(expected).tobytes()
        )

    def test_matches_reference_with_ties_and_absent_classes(self) -> None:
        rng = np.random.default_rng(20240918)
        for _ in range(40):
            n_rows = int(rng.integers(1, 40))
            n_classes = int(rng.integers(1, 6))
            classes = np.arange(n_classes, dtype=np.int64)
            matrix = np.round(rng.normal(size=(n_rows, n_classes)) * 3.0)
            labels = rng.integers(-1, n_classes + 1, size=n_rows)
            self._assert_bit_identical(matrix, labels, classes)

    def test_matches_reference_on_continuous_scores(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(20):
            n_rows = int(rng.integers(2, 60))
            n_classes = int(rng.integers(1, 8))
            classes = np.arange(n_classes, dtype=np.int64)
            matrix = rng.normal(size=(n_rows, n_classes))
            labels = rng.integers(0, n_classes, size=n_rows)
            self._assert_bit_identical(matrix, labels, classes)

    def test_all_tied_column_averages_ranks(self) -> None:
        matrix = np.ones((7, 3), dtype=np.float64)
        labels = np.asarray([0, 0, 0, 1, 1, 2, 2])
        self._assert_bit_identical(matrix, labels, np.asarray([0, 1, 2]))

    def test_class_without_positives_returns_nan(self) -> None:
        matrix = np.arange(10, dtype=np.float64).reshape(5, 2)
        labels = np.asarray([0, 0, 0, 0, 0])
        self._assert_bit_identical(matrix, labels, np.asarray([0, 1]))

    def test_class_without_negatives_returns_nan(self) -> None:
        matrix = np.arange(3, dtype=np.float64).reshape(3, 1)
        labels = np.asarray([0, 0, 0])
        self._assert_bit_identical(matrix, labels, np.asarray([0]))


class RelabelIrrelevanceTests(unittest.TestCase):
    """``relabel_clusters`` must not change results for image-id-independent statistics.

    The macro-AUC and mean statistics do not read source-image ids, so the extra
    per-replicate occurrence-id loop only adds work.  The E07 AUC/score paths
    disable relabelling; these tests pin that the resulting CI is bit-identical
    to the generic relabelled reference for both the serial and block paths.
    """

    def _fixture(self):
        rng = np.random.default_rng(4242)
        n_rows = 48
        n_classes = 4
        classes = np.arange(n_classes, dtype=np.int64)
        scores = rng.normal(size=(n_rows, n_classes)) + rng.integers(0, n_classes, size=n_rows)[:, None]
        labels = rng.integers(0, n_classes, size=n_rows)
        images = np.repeat(np.arange(8), 6)
        return scores, labels, images, classes

    def test_mean_statistic_relabel_is_bit_identical(self) -> None:
        values = np.asarray([0.1, -0.4, 0.7, 0.2, 0.0, -0.3] * 8)
        images = np.repeat(np.arange(8), 6)
        labels = np.zeros_like(values, dtype=np.int64)
        def stat(v, _l, _i):
            return float(np.nanmean(v))

        base = {"statistic": stat, "replicates": 64, "seed": 99, "confidence": 0.95, "workers": 2}
        on = clustered_bootstrap(values, labels=labels, source_image_ids=images, relabel_clusters=True, **base)
        off = clustered_bootstrap(values, labels=labels, source_image_ids=images, relabel_clusters=False, **base)
        self.assertEqual(on["ci_low"], off["ci_low"])
        self.assertEqual(on["ci_high"], off["ci_high"])
        self.assertEqual(on["estimate"], off["estimate"])

    def test_fast_auc_path_matches_relabelled_generic_path(self) -> None:
        scores, labels, images, classes = self._fixture()

        def macro_auc(values, sampled_labels, _images):
            return float(np.nanmean(_vectorized_class_auc(values, sampled_labels, classes)))

        generic = clustered_bootstrap(
            scores,
            labels=labels,
            source_image_ids=images,
            statistic=macro_auc,
            replicates=64,
            seed=123,
            confidence=0.95,
            relabel_clusters=True,
            workers=2,
        )
        fast = _fast_score_auc_ci(
            scores, labels, images, classes,
            replicates=64, seed=123, confidence=0.95, workers=2,
        )
        self.assertEqual(fast["estimate"], generic["estimate"])
        self.assertEqual(fast["ci_low"], generic["ci_low"])
        self.assertEqual(fast["ci_high"], generic["ci_high"])


if __name__ == "__main__":
    unittest.main()
