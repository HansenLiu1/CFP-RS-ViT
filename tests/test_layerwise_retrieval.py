import unittest

import numpy as np

from cfra.evaluation import retrieval_metrics
from cfra.pipeline import (
    LayerwiseError,
    RetrievalError,
    coerce_feature_record,
    run_e06,
    run_e06_readout,
    run_e08,
)
from cfra.pipeline.retrieval import _bootstrap_metrics
from cfra.statistics import clustered_bootstrap_multi


def _metadata() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role, image_prefix, class_offset in (
        ("direction_fit", "df", 0),
        ("readout_fit", "rf", 10),
        ("confirmation", "q", 20),
    ):
        for class_id in (0, 1):
            for index in range(2):
                rows.append(
                    {
                        "role": role,
                        "source_image_id": f"{image_prefix}-{class_id}-{index + class_offset}",
                        "object_id": f"{role}-{class_id}-{index}",
                        "class_id": class_id,
                        "perturbation": "blur" if index == 0 else "noise",
                    }
                )
    return rows


class LayerwiseRetrievalTests(unittest.TestCase):
    def test_e06_supports_different_layer_dimensions_and_reports_late_minus_early(self) -> None:
        metadata = _metadata()
        class_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        early = np.vstack([class_vectors[row["class_id"]] for row in metadata])
        late = np.column_stack([early[:, 0], early[:, 1], np.asarray([0.1 if row["class_id"] == 0 else -0.1 for row in metadata])])
        result = run_e06(
            {
                "model_id": "synthetic",
                "metadata": metadata,
                "layers": {
                    "patch_emb": {"features": early, "representation": {"pooling": "none", "projection": "none"}},
                    "L03": {"features": late, "representation": {"pooling": "cls", "projection": "none"}},
                },
            },
            ["patch_emb", "L03"],
        )

        self.assertEqual(result["layers"]["patch_emb"]["dimension"], 2)
        self.assertEqual(result["layers"]["L03"]["dimension"], 3)
        self.assertTrue(result["layers"]["patch_emb"]["patch_emb"])
        self.assertEqual(result["layers"]["L03"]["block"], 3)
        self.assertIn("macro_ovr_auc", result["late_minus_early"]["class_predictability"])
        self.assertEqual(result["layers"]["L03"]["class_directions"].shape, (2, 3))

    def test_e06_rejects_fit_confirmation_image_leakage(self) -> None:
        metadata = _metadata()
        metadata[0]["source_image_id"] = metadata[-1]["source_image_id"]
        values = np.ones((len(metadata), 2), dtype=np.float64)
        with self.assertRaises(LayerwiseError):
            run_e06(
                {"metadata": metadata, "layers": {"patch_emb": values, "L03": values}},
                ["patch_emb", "L03"],
            )

    def test_e06_block_trend_uses_blocks_not_endpoints(self) -> None:
        metadata = _metadata()
        class_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        base = np.vstack([class_vectors[row["class_id"]] for row in metadata])
        layers = {
            "patch_emb": {"features": base, "representation": {"pooling": "none"}},
            "L03": {"features": np.column_stack([base, np.zeros(len(base))]), "representation": {"pooling": "cls"}},
            "L06": {"features": np.column_stack([base, np.full(len(base), 0.5)]), "representation": {"pooling": "cls"}},
            "final": {"features": np.column_stack([base, np.ones(len(base))]), "representation": {"pooling": "cls"}},
        }
        result = run_e06({"metadata": metadata, "layers": layers, "model_id": "synthetic"},
                         ["patch_emb", "L03", "L06", "final"])
        self.assertEqual(result["block_layers"], ["L03", "L06"])
        self.assertEqual(result["block_trend"]["layers"], ["L03", "L06"])
        self.assertEqual(result["late_minus_early"]["early_layer"], "L03")
        self.assertEqual(result["late_minus_early"]["late_layer"], "L06")
        self.assertEqual(result["late_minus_early"]["basis"], "blocks")
        self.assertIn("macro_ovr_auc", result["block_trend"]["class_predictability"]["spearman_vs_layer_index"])

    def test_e06_readout_reports_centered_own_minus_wrong(self) -> None:
        metadata: list[dict[str, object]] = []
        for role in ("direction_fit", "readout_fit", "confirmation"):
            for class_id in (0, 1):
                for index in range(3):
                    metadata.append({
                        "role": role,
                        "source_image_id": f"{role}-{class_id}-{index}",
                        "object_id": f"{role}-{class_id}-{index}",
                        "class_id": class_id,
                        "perturbation": "blur",
                    })
        class_dir = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        common = np.full(4, 5.0)
        raw = np.vstack([common + class_dir[row["class_id"]] for row in metadata])
        clear = raw.copy()
        degraded = np.vstack([common - class_dir[row["class_id"]] for row in metadata])
        layer_arrays = {
            "L03": {"raw": raw, "clear": clear, "degraded": degraded},
            "L06": {"raw": raw + 1.0, "clear": clear, "degraded": degraded},
            "L09": {"raw": raw * 1.2, "clear": clear, "degraded": degraded},
            "final": {"raw": raw, "clear": clear, "degraded": degraded},
        }
        result = run_e06_readout(layer_arrays, metadata, ["L03", "L06", "L09", "final"])
        self.assertEqual(result["stage"], "E06_readout")
        self.assertEqual(result["block_layers"], ["L03", "L06", "L09"])
        own_minus_wrong = result["summary"]["own_minus_wrong_linear_margin_drop_by_layer"]
        self.assertEqual(own_minus_wrong["layers"], ["L03", "L06", "L09"])
        self.assertTrue(np.isfinite(own_minus_wrong["spearman_vs_layer_index"]))
        for layer in ("L03", "L06", "L09", "final"):
            for perturbation in result["layers"][layer]["by_perturbation"].values():
                alpha_keys = set(perturbation["alpha"])
                # Both the CMC-direction and raw-direction wrong-class controls
                # must be present so own-vs-wrong can be compared in each space.
                self.assertIn("wrong@1", alpha_keys)
                self.assertIn("wrong_raw@1", alpha_keys)
                self.assertIn("own_raw@1", alpha_keys)

    def test_e06_readout_centering_is_invariant_to_a_global_common_mode(self) -> None:
        metadata: list[dict[str, object]] = []
        for role in ("direction_fit", "readout_fit", "confirmation"):
            for class_id in (0, 1):
                for index in range(3):
                    metadata.append(
                        {
                            "role": role,
                            "source_image_id": f"{role}-{class_id}-{index}",
                            "object_id": f"{role}-{class_id}-{index}",
                            "class_id": class_id,
                            "perturbation": "blur",
                        }
                    )
        class_dir = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        common = np.full(4, 5.0)
        raw = np.vstack([common + class_dir[row["class_id"]] for row in metadata])
        clear = raw.copy()
        degraded = np.vstack([common - class_dir[row["class_id"]] for row in metadata])
        base = {"L03": {"raw": raw, "clear": clear, "degraded": degraded}}
        # A different global common mode (the same vector for every object) is
        # removed by the readout_fit mean, so the centered drops must not move.
        shifted = {"L03": {"raw": raw + np.asarray([0.0, 0.0, 9.0, -4.0]), "clear": clear, "degraded": degraded}}
        first = run_e06_readout(base, metadata, ["L03"])
        second = run_e06_readout(shifted, metadata, ["L03"])
        drops_first = first["layers"]["L03"]["by_perturbation"]["blur"]["alpha"]
        drops_second = second["layers"]["L03"]["by_perturbation"]["blur"]["alpha"]
        for key in ("own@1", "wrong@1", "own_raw@1", "wrong_raw@1"):
            for metric in ("linear_margin_drop", "linear_accuracy_drop", "prototype_margin_drop"):
                self.assertAlmostEqual(
                    drops_first[key][metric], drops_second[key][metric], places=9, msg=(key, metric)
                )

    def test_extract_features_prefers_paired_delta_over_raw(self) -> None:
        metadata = _metadata()
        count = len(metadata)
        clear = np.arange(count * 3, dtype=np.float64).reshape(count, 3) + 10.0
        degraded = np.ones((count, 3), dtype=np.float64)
        raw = np.full((count, 3), 99.0)
        source = {
            "metadata": metadata,
            "features": {"clear": clear, "degraded": degraded, "raw": raw},
        }

        record = coerce_feature_record(source, feature_key=None)

        np.testing.assert_allclose(record.features, clear - degraded)
        self.assertFalse(np.allclose(record.features, raw))

    def test_e08_keeps_candidates_separate_from_gt_and_returns_all_metrics(self) -> None:
        rows: list[dict[str, object]] = []
        values: list[list[float]] = []
        for role, image_prefix, candidate_flag in (
            ("direction_fit", "df", None),
            ("readout_fit", "rf", None),
        ):
            for class_id in (0, 1):
                for index in range(2):
                    rows.append(
                        {
                            "role": role,
                            "source_image_id": f"{image_prefix}-{class_id}-{index}",
                            "class_id": class_id,
                            "is_candidate": candidate_flag,
                            "is_gt": False,
                        }
                    )
                    values.append([1.0, 0.0] if class_id == 0 else [0.0, 1.0])
        for role, image_prefix, candidate_flag in (("confirmation", "query", False), ("confirmation", "gallery", True)):
            for class_id in (0, 1):
                for index in range(2):
                    rows.append(
                        {
                            "role": role,
                            "retrieval_role": "query" if not candidate_flag else "candidate",
                            "source_image_id": f"{image_prefix}-{class_id}-{index}",
                            "class_id": class_id,
                            "is_candidate": candidate_flag,
                            "is_gt": False,
                        }
                    )
                    values.append([1.0, 0.0] if class_id == 0 else [0.0, 1.0])
        result = run_e08(
            {"features": np.asarray(values, dtype=np.float64), "metadata": rows},
            feature_key="features",
            k=2,
            bootstrap_replicates=8,
            bootstrap_seed=7,
        )

        self.assertTrue(result["image_disjoint"])
        self.assertFalse(result["candidate_selection_uses_ground_truth"])
        self.assertEqual(result["query_count"], 4)
        self.assertEqual(result["candidate_count"], 4)
        for method in ("raw_prototype", "cfp_direction", "cav_like", "random"):
            metrics = result["methods"][method]
            for key in ("macro_ap", "ndcg_at_k", "recall_at_k", "precision_at_k"):
                self.assertIn(key, metrics)
            self.assertEqual(metrics["bootstrap"]["macro_ap"]["source_image_count"], 4)

    def test_e08_accepts_a_pluggable_text_scorer_without_text_encoder(self) -> None:
        rows = []
        values = []
        for role, prefix in (("direction_fit", "df"), ("readout_fit", "rf")):
            for class_id in (0, 1):
                rows.append({"role": role, "source_image_id": f"{prefix}-{class_id}", "class_id": class_id})
                values.append([1.0, 0.0] if class_id == 0 else [0.0, 1.0])
        for role, retrieval_role, prefix in (
            ("confirmation", "query", "q"),
            ("confirmation", "candidate", "c"),
        ):
            for class_id in (0, 1):
                rows.append(
                    {
                        "role": role,
                        "retrieval_role": retrieval_role,
                        "source_image_id": f"{prefix}-{class_id}",
                        "class_id": class_id,
                    }
                )
                values.append([1.0, 0.0] if class_id == 0 else [0.0, 1.0])

        def scorer(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
            return query @ candidates.T

        result = run_e08(
            {"features": np.asarray(values), "metadata": rows},
            feature_key="features",
            methods=("cfp_direction",),
            text_baseline=scorer,
            k=1,
        )
        self.assertIn("text_baseline", result["methods"])
        self.assertIn("macro_ap", result["methods"]["text_baseline"])

    def test_e08_cfp_direction_uses_m1_residualized_delta(self) -> None:
        direction_delta = {0: np.asarray([2.0, 0.0]), 1: np.asarray([2.0, 2.0])}
        rows: list[dict[str, object]] = []
        raw_values: list[np.ndarray] = []
        delta_values: list[np.ndarray] = []
        for role, prefix in (("direction_fit", "df"), ("readout_fit", "rf")):
            for class_id in (0, 1):
                rows.append({"role": role, "source_image_id": f"{prefix}-{class_id}", "class_id": class_id})
                raw_values.append(direction_delta[class_id])
                delta_values.append(direction_delta[class_id])
        for retrieval_role, prefix in (("query", "q"), ("candidate", "c")):
            for class_id in (0, 1):
                rows.append(
                    {
                        "role": "confirmation",
                        "retrieval_role": retrieval_role,
                        "source_image_id": f"{prefix}-{class_id}",
                        "class_id": class_id,
                    }
                )
                raw_values.append(direction_delta[class_id])
                delta_values.append(direction_delta[class_id])
        table = {
            "features": {
                "raw": np.asarray(raw_values, dtype=np.float64),
                "delta": np.asarray(delta_values, dtype=np.float64),
            },
            "metadata": rows,
        }

        result = run_e08(
            table,
            feature_key="raw",
            direction_feature_key="delta",
            methods=("cfp_direction", "cfp_direction_raw"),
            k=1,
        )

        for method in ("cfp_direction", "cfp_direction_raw", "cfp_direction_matched"):
            self.assertIn(method, result["methods"])

        fit_deltas = np.stack([direction_delta[0], direction_delta[1]])
        expected_raw = fit_deltas / np.linalg.norm(fit_deltas, axis=1, keepdims=True)
        # M1 origin is the mean of the per-class unit directions (runner-faithful),
        # not the mean of the per-object deltas used by the old implementation.
        class_origin = expected_raw.mean(axis=0, keepdims=True)
        centered_fit = fit_deltas - class_origin
        expected_centered = centered_fit / np.linalg.norm(centered_fit, axis=1, keepdims=True)

        fitted = result["fitted"]
        np.testing.assert_allclose(fitted["direction_raw"]["directions"], expected_raw)
        np.testing.assert_allclose(fitted["direction"]["directions"], expected_centered)
        self.assertFalse(np.allclose(expected_raw, expected_centered))

    def test_e08_rejects_overlapping_query_candidate_images(self) -> None:
        rows = [
            {"role": "direction_fit", "source_image_id": "df", "class_id": 0},
            {"role": "readout_fit", "source_image_id": "rf", "class_id": 0},
            {"role": "confirmation", "retrieval_role": "query", "source_image_id": "same", "class_id": 0},
            {"role": "confirmation", "retrieval_role": "candidate", "source_image_id": "same", "class_id": 0},
        ]
        with self.assertRaises(RetrievalError):
            run_e08({"features": np.ones((4, 2)), "metadata": rows}, feature_key="features", k=1)


class E08BootstrapBitIdenticalTests(unittest.TestCase):
    def test_per_query_precompute_matches_per_replicate_recompute(self) -> None:
        rng = np.random.default_rng(11)
        n_queries, n_candidates, k = 40, 25, 5
        relevance = (rng.random((n_queries, n_candidates)) < 0.2).astype(np.float64)
        scores = rng.random((n_queries, n_candidates))
        query_images = np.asarray([f"img-{index % 7}" for index in range(n_queries)])

        metric_names = ("macro_ap", "ndcg_at_k", "recall_at_k", "precision_at_k")
        replicates, seed = 64, 1729

        optimized = _bootstrap_metrics(
            relevance, scores, query_images, k=k, replicates=replicates, seed=seed
        )

        positions = np.arange(n_queries, dtype=np.int64)

        def reference_statistic(sampled_positions, _labels, _images):
            values = retrieval_metrics(
                relevance[sampled_positions], scores[sampled_positions], k=k
            )
            return {name: float(values[name]) for name in metric_names}

        reference = clustered_bootstrap_multi(
            positions,
            query_images,
            reference_statistic,
            replicates=replicates,
            seed=seed,
            relabel_clusters=True,
        )
        for name in metric_names:
            for key in ("estimate", "ci_low", "ci_high"):
                optimized_value = optimized[name][key]
                reference_value = reference["metrics"][name][key]
                self.assertTrue(
                    (optimized_value == reference_value)
                    or (np.isnan(optimized_value) and np.isnan(reference_value)),
                    (name, key, optimized_value, reference_value),
                )


if __name__ == "__main__":
    unittest.main()
