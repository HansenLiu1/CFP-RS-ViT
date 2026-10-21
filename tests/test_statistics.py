import os
import unittest
from unittest import mock

import numpy as np

from cfra.evaluation import (
    audit_near_zero,
    binary_average_precision,
    ndcg_at_k,
    recall_at_k,
    retrieval_metrics,
    specificity_metrics,
)
from cfra.evaluation.metrics import _summary_from_matrix
from cfra.pipeline.core import _cell_bootstrap_once, _statistics_settings
from cfra.statistics import (
    benjamini_hochberg,
    clustered_bootstrap,
    paired_difference_ci,
    paired_sign_flip,
)


def _reference_summary_from_matrix(score_matrix: np.ndarray) -> dict[str, object]:
    total = score_matrix.shape[0]
    own = np.diag(score_matrix)
    margins = np.full(total, np.nan, dtype=np.float64)
    argmax = np.full(total, np.nan, dtype=np.float64)
    for index, row in enumerate(score_matrix):
        off_diagonal = np.delete(row, index)
        if not np.isfinite(row[index]) or not np.isfinite(off_diagonal).any():
            continue
        hardest = float(np.nanmax(off_diagonal))
        margins[index] = row[index] - hardest
        argmax[index] = float(row[index] >= np.nanmax(row))
    return {
        "own_argmax_rate": float(np.nanmean(argmax)) if np.isfinite(argmax).any() else float("nan"),
        "mean_diagonal_margin": float(np.nanmean(margins)) if np.isfinite(margins).any() else float("nan"),
        "mean_own_cosine": float(np.nanmean(own)) if np.isfinite(own).any() else float("nan"),
        "class_margins": margins,
    }


class ClusteredBootstrapTests(unittest.TestCase):
    def test_resamples_unequal_clusters_and_keeps_labels_aligned(self) -> None:
        values = np.asarray([10, 11, 20, 30, 31, 32], dtype=np.int64)
        labels = values + 100
        image_ids = np.asarray(["a", "a", "b", "c", "c", "c"])
        calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def statistic(sampled_values, sampled_labels, sampled_images):
            calls.append((sampled_values.copy(), sampled_labels.copy(), sampled_images.copy()))
            self.assertTrue(np.array_equal(sampled_labels, sampled_values + 100))
            self.assertEqual(len(sampled_values), len(sampled_labels))
            self.assertEqual(len(sampled_values), len(sampled_images))
            return float(np.mean(sampled_values))

        result = clustered_bootstrap(
            values,
            image_ids,
            statistic,
            labels=labels,
            replicates=24,
            seed=7,
        )

        self.assertEqual(result["source_image_count"], 3)
        self.assertEqual(result["observation_count"], len(values))
        self.assertEqual(len(calls), 25)
        original_cluster_sizes = {"a": 2, "b": 1, "c": 3}
        for _, _, sampled_images in calls[1:]:
            for image_id, size in original_cluster_sizes.items():
                self.assertEqual(np.count_nonzero(sampled_images == image_id) % size, 0)

    def test_retains_historical_two_argument_statistic(self) -> None:
        values = np.asarray([1.0, 2.0, 4.0])
        image_ids = np.asarray(["a", "b", "b"])

        result = clustered_bootstrap(
            values,
            image_ids,
            lambda sampled_values, sampled_images: float(np.mean(sampled_values)),
            replicates=12,
            seed=3,
        )

        self.assertEqual(result["estimate"], np.mean(values))
        self.assertTrue(np.isfinite(result["ci_low"]))
        self.assertTrue(np.isfinite(result["ci_high"]))

    def test_accepts_new_three_array_positional_contract(self) -> None:
        values = np.asarray([1.0, 2.0, 4.0])
        labels = np.asarray([10, 20, 40])
        image_ids = np.asarray(["a", "b", "b"])

        result = clustered_bootstrap(
            values,
            labels,
            image_ids,
            lambda sampled_values, sampled_labels, sampled_images: float(
                np.mean(sampled_values + sampled_labels / 10.0)
            ),
            replicates=12,
            seed=3,
        )

        self.assertAlmostEqual(result["estimate"], 14.0 / 3.0)

    def test_precomputed_clusters_match_default_path(self) -> None:
        values = np.asarray([10, 11, 20, 30, 31, 32], dtype=np.int64)
        labels = values + 100
        image_ids = np.asarray(["a", "a", "b", "c", "c", "c"])

        def statistic(sampled_values, sampled_labels, sampled_images):
            return float(np.mean(sampled_values))

        default = clustered_bootstrap(
            values, image_ids, statistic, labels=labels, replicates=24, seed=7
        )
        cluster_ids = sorted(set(image_ids.tolist()))
        cluster_indices = [np.flatnonzero(image_ids == cluster) for cluster in cluster_ids]
        cached = clustered_bootstrap(
            values,
            image_ids,
            statistic,
            labels=labels,
            replicates=24,
            seed=7,
            clusters=cluster_ids,
            cluster_indices=cluster_indices,
        )

        self.assertEqual(default["estimate"], cached["estimate"])
        self.assertEqual(default["ci_low"], cached["ci_low"])
        self.assertEqual(default["ci_high"], cached["ci_high"])
        self.assertEqual(default["source_image_count"], cached["source_image_count"])


class PairedInferenceTests(unittest.TestCase):
    def test_image_level_difference_ci_and_sign_flip_use_equal_image_weight(self) -> None:
        differences = np.asarray([1.0, 3.0, 2.0, 4.0, 10.0, 10.0, 10.0])
        image_ids = np.asarray(["a", "a", "b", "b", "c", "c", "c"])

        ci = paired_difference_ci(differences, image_ids, replicates=100, seed=11)
        sign_flip = paired_sign_flip(differences, image_ids, replicates=100, seed=11)

        self.assertEqual(ci["source_image_count"], 3)
        self.assertEqual(ci["estimate"], np.mean([2.0, 3.0, 10.0]))
        self.assertLessEqual(ci["ci_low"], ci["estimate"])
        self.assertGreaterEqual(ci["ci_high"], ci["estimate"])
        self.assertEqual(sign_flip["source_image_count"], 3)
        self.assertTrue(0.0 < sign_flip["p_value_two_sided"] <= 1.0)
        self.assertLessEqual(sign_flip["ci_low"], sign_flip["estimate"])
        self.assertGreaterEqual(sign_flip["ci_high"], sign_flip["estimate"])

    def test_precomputed_clusters_match_default_path(self) -> None:
        differences = np.asarray([1.0, 3.0, 2.0, 4.0, 10.0, 10.0, 10.0])
        image_ids = np.asarray(["a", "a", "b", "b", "c", "c", "c"])

        default = paired_difference_ci(differences, image_ids, replicates=100, seed=11)
        cluster_ids = sorted(set(image_ids.tolist()))
        cached = paired_difference_ci(
            differences, image_ids, replicates=100, seed=11, clusters=cluster_ids
        )

        self.assertEqual(default["estimate"], cached["estimate"])
        self.assertEqual(default["ci_low"], cached["ci_low"])
        self.assertEqual(default["ci_high"], cached["ci_high"])
        self.assertEqual(default["source_image_count"], cached["source_image_count"])


class MultipleTestingTests(unittest.TestCase):
    def test_benjamini_hochberg_controls_the_declared_fdr_threshold(self) -> None:
        result = benjamini_hochberg([0.001, 0.02, 0.03, 0.8], q=0.05)

        np.testing.assert_allclose(result["q_values"], [0.004, 0.04, 0.04, 0.8])
        np.testing.assert_array_equal(result["reject"], [True, True, True, False])
        self.assertEqual(result["tested_count"], 4)


class SpecificityBootstrapTests(unittest.TestCase):
    def test_specificity_bootstrap_reports_all_primary_cis_for_unequal_clusters(self) -> None:
        directions = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        values = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        labels = np.asarray([0, 1, 0, 0, 1, 0, 1, 1, 1, 1])
        image_ids = np.asarray(["a", "a", "b", "b", "b", "c", "c", "c", "c", "c"])

        result = specificity_metrics(
            directions,
            values,
            labels,
            [0, 1],
            source_image_ids=image_ids,
            bootstrap_ci=True,
            bootstrap_replicates=40,
            bootstrap_seed=5,
        )

        self.assertEqual(result["bootstrap_source_image_count"], 3)
        for name in (
            "own_argmax_rate",
            "mean_diagonal_margin",
            "mean_own_cosine",
        ):
            self.assertIn(f"{name}_ci_low", result)
            self.assertIn(f"{name}_ci_high", result)
            self.assertTrue(np.isfinite(result[f"{name}_ci_low"]))
            self.assertTrue(np.isfinite(result[f"{name}_ci_high"]))
        self.assertEqual(result["own_argmax"], 1.0)
        self.assertEqual(result["margin"], 1.0)
        self.assertEqual(result["mean_own_cosine"], 1.0)


class RetrievalMetricTests(unittest.TestCase):
    def test_binary_ap_ndcg_and_recall_at_k(self) -> None:
        labels = np.asarray([0, 1, 0, 1])
        scores = np.asarray([0.9, 0.8, 0.7, 0.1])

        self.assertAlmostEqual(binary_average_precision(labels, scores), 0.5)
        self.assertAlmostEqual(ndcg_at_k(labels, scores, 2), 1.0 / np.log2(3.0) / (1.0 + 1.0 / np.log2(3.0)))
        self.assertEqual(recall_at_k(labels, scores, 2), 0.5)
        result = retrieval_metrics(labels[None, :], scores[None, :], k=2)
        self.assertAlmostEqual(result["macro_ap"], 0.5)
        self.assertAlmostEqual(result["ndcg_at_k"], ndcg_at_k(labels, scores, 2))
        self.assertEqual(result["recall_at_k"], 0.5)


class NearZeroAuditTests(unittest.TestCase):
    def test_near_zero_audit_reports_indices_and_fraction(self) -> None:
        result = audit_near_zero([[0.0, 0.0], [1.0, 0.0], [1.0e-9, 0.0]], epsilon=1.0e-8)

        self.assertEqual(result["near_zero_count"], 2)
        self.assertEqual(result["total_count"], 3)
        self.assertEqual(result["fraction"], 2.0 / 3.0)
        np.testing.assert_array_equal(result["indices"], [0, 2])
        self.assertTrue(result["has_near_zero"])


class SummaryVectorizationTests(unittest.TestCase):
    def test_vectorized_summary_matches_per_row_reference(self) -> None:
        rng = np.random.default_rng(0)
        for trial in range(60):
            size = int(rng.integers(1, 10))
            matrix = rng.normal(size=(size, size))
            if size > 1 and trial % 3 == 0:
                matrix[0, :] = np.nan
            if size > 2 and trial % 5 == 0:
                matrix[1, :] = np.nan
                matrix[1, 1] = 1.0
            got = _summary_from_matrix(matrix)
            reference = _reference_summary_from_matrix(matrix)
            for key in ("own_argmax_rate", "mean_diagonal_margin", "mean_own_cosine"):
                left, right = got[key], reference[key]
                self.assertTrue(
                    (left == right) or (np.isnan(left) and np.isnan(right)),
                    (trial, key, left, right),
                )
            np.testing.assert_array_equal(
                np.nan_to_num(got["class_margins"], nan=-1.0),
                np.nan_to_num(reference["class_margins"], nan=-1.0),
            )


class BootstrapWorkerSettingsTests(unittest.TestCase):
    def test_defaults_to_serial_without_env(self) -> None:
        env = dict(os.environ)
        env.pop("CFRA_BOOTSTRAP_WORKERS", None)
        with mock.patch.dict(os.environ, env, clear=True):
            settings = _statistics_settings({"bootstrap_replicates": 10})
        self.assertEqual(settings["bootstrap_workers"], 1)

    def test_reads_env_and_config_precedence(self) -> None:
        with mock.patch.dict(os.environ, {"CFRA_BOOTSTRAP_WORKERS": "7"}, clear=False):
            self.assertEqual(_statistics_settings({})["bootstrap_workers"], 7)
            self.assertEqual(
                _statistics_settings({"bootstrap_workers": 3})["bootstrap_workers"], 3
            )

    def test_cell_workers_pin_bootstrap_to_one(self) -> None:
        resolved = _cell_bootstrap_once({"bootstrap_replicates": 10, "bootstrap_workers": 9})
        self.assertEqual(resolved["bootstrap_workers"], 1)
        with mock.patch.dict(os.environ, {"CFRA_BOOTSTRAP_WORKERS": "16"}, clear=False):
            self.assertEqual(_statistics_settings(resolved)["bootstrap_workers"], 1)


if __name__ == "__main__":
    unittest.main()
