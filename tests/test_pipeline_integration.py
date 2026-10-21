import tempfile
import unittest
from pathlib import Path

import numpy as np

from cfra.pipeline import (
    FeatureSample,
    FeatureTable,
    run_e03,
    run_e04,
    run_e04_1,
    run_e04_2,
    run_e04_3,
    run_e04_4,
    run_e04_5,
    run_e05,
    run_e06,
    run_e07,
    run_e08,
)
from cfra.pipeline.core import write_core_statistics

def _table(
    perturbations: tuple[str, ...] = ("blur", "noise", "brightness"),
    *,
    dataset_id: str = "synthetic",
) -> FeatureTable:
    metadata: list[FeatureSample] = []
    arrays: dict[str, list[np.ndarray]] = {
        "raw": [],
        "true_clear_degraded": [],
        "true_degraded_degraded": [],
        "true_clear_clean": [],
        "true_degraded_clean": [],
        "background_clear": [],
        "background_degraded": [],
    }
    for perturbation_index, perturbation in enumerate(perturbations):
        for role_index, role in enumerate(("direction_fit", "readout_fit", "confirmation")):
            for class_id in (0, 1):
                for object_index in range(2):
                    source_image_id = f"{role}-{perturbation}-{class_id}-{object_index}"
                    object_id = f"object-{role}-{perturbation}-{class_id}-{object_index}"
                    class_vector = np.asarray([2.0, 0.0, 0.25, 0.0] if class_id == 0 else [0.0, 2.0, 0.0, 0.25])
                    perturbation_vector = np.asarray([0.0, 0.0, 0.1 * perturbation_index, 0.0])
                    raw = class_vector + perturbation_vector
                    delta = np.asarray([1.0, 0.0, 0.1, 0.0] if class_id == 0 else [0.0, 1.0, 0.0, 0.1])
                    metadata.append(
                        FeatureSample(
                            dataset_id=dataset_id,
                            object_id=object_id,
                            source_image_id=source_image_id,
                            class_id=class_id,
                            class_name=f"class-{class_id}",
                            coarse_name=f"coarse-{class_id}",
                            role=role,
                            perturbation=perturbation,
                            seed=object_index + role_index,
                            geometry=(float(class_id), float(perturbation_index), 0.0, 1.0),
                        )
                    )
                    arrays["raw"].append(raw)
                    arrays["true_clear_degraded"].append(raw + delta)
                    arrays["true_degraded_degraded"].append(raw)
                    arrays["true_clear_clean"].append(raw + delta + np.asarray([0.02, 0.0, 0.0, 0.0]))
                    arrays["true_degraded_clean"].append(raw + np.asarray([0.01, 0.0, 0.0, 0.0]))
                    arrays["background_clear"].append(raw + np.asarray([0.0, 0.0, 0.01, 0.0]))
                    arrays["background_degraded"].append(raw)

    return FeatureTable(
        model_id="synthetic",
        metadata=tuple(metadata),
        arrays={key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()},
        representation={"representation_id": "synthetic", "dimension": 4},
    )


class PipelineIntegrationTests(unittest.TestCase):
    def test_e04_2_symmetric_centering_and_finite_cluster_intervals(self) -> None:
        rows = run_e04_2(
            _table(("blur",), dataset_id="fair1m"), dataset_ids=["fair1m"],
            perturbations=["blur"], support_sizes=[2], epsilon=1e-8,
            statistics_config={"bootstrap_replicates": 32, "bootstrap_seed": 19},
        )
        for row in rows:
            if row["analysis"].endswith("readout"):
                continue
            # The two orthogonal classes become opposite unit directions after
            # symmetric centering: own cosine=1, wrong cosine=-1, margin=2.
            self.assertAlmostEqual(row["mean_diagonal_margin"], 2.0)
            self.assertTrue(np.isfinite(row["own_argmax_rate_ci"]).all())
            self.assertTrue(np.isfinite(row["mean_diagonal_margin_ci"]).all())

    def test_synthetic_pipeline_covers_e03_through_e08_without_model_loading(self) -> None:
        table = _table(("blur", "noise"))
        statistics = {"bootstrap_replicates": 4, "bootstrap_seed": 19}
        e03 = run_e03(
            table,
            dataset_ids=["synthetic"],
            perturbations=["blur"],
            statistics_config=statistics,
        )
        e04 = run_e04(
            table,
            dataset_ids=["synthetic"],
            perturbations=["blur"],
            method_ids=["raw", "euclidean_mean_subtraction"],
            normalization_variants=["unit", "raw", "magnitude_weighted"],
            epsilon=1.0e-8,
            controls={name: False for name in (
                "shifted_obb",
                "random_geometry_matched_obb",
                "label_permutation",
                "background_only",
                "boundary_only",
                "interior_eroded",
                "mask_dilation_erosion",
                "geometry_regression",
            )},
            statistics_config=statistics,
        )
        e04_1 = run_e04_1(
            table,
            dataset_ids=["synthetic"],
            perturbations=["blur"],
            epsilon=1.0e-8,
            statistics_config=statistics,
        )
        e04_2 = run_e04_2(
            _table(("blur",), dataset_id="fair1m"),
            dataset_ids=["fair1m"],
            perturbations=["blur"],
            support_sizes=[2],
            epsilon=1.0e-8,
            statistics_config=statistics,
        )
        e04_3 = run_e04_3(
            table,
            dataset_ids=["synthetic"],
            perturbations=["blur"],
            statistics_config=statistics,
        )
        e04_4 = run_e04_4(
            table,
            dataset_ids=["synthetic"],
            perturbations=["blur"],
            normalization_variants=["unit", "raw", "magnitude_weighted"],
            epsilon=1.0e-8,
            controls={name: False for name in (
                "shifted_obb",
                "random_geometry_matched_obb",
                "label_permutation",
                "background_only",
                "boundary_only",
                "interior_eroded",
                "mask_dilation_erosion",
                "geometry_regression",
            )},
            statistics_config=statistics,
        )
        e04_5 = run_e04_5(
            {"nominal": table, "low": table, "high": table},
            dataset_ids=["synthetic"],
            perturbations=["blur"],            epsilon=1.0e-8,
            method_ids=["raw"],
            normalization_variants=["unit"],
            controls={name: False for name in (
                "shifted_obb",
                "random_geometry_matched_obb",
                "label_permutation",
                "background_only",
                "boundary_only",
                "interior_eroded",
                "mask_dilation_erosion",
                "geometry_regression",
            )},
            statistics_config=statistics,
        )
        e05 = run_e05(
            table,
            dataset_ids=["synthetic"],
            perturbations=["blur", "noise"],
            epsilon=1.0e-8,
            statistics_config=statistics,
        )
        e06 = run_e06(table, ["raw"], epsilon=1.0e-8)
        confirmation_images = sorted(
            {
                sample.source_image_id
                for sample in table.metadata
                if sample.role == "confirmation"
            }
        )
        candidate_images = set(confirmation_images[: len(confirmation_images) // 2])
        candidate_mask = np.asarray(
            [sample.role == "confirmation" and sample.source_image_id in candidate_images for sample in table.metadata],
            dtype=bool,
        )
        query_mask = np.asarray(
            [sample.role == "confirmation" and sample.source_image_id not in candidate_images for sample in table.metadata],
            dtype=bool,
        )
        e08 = run_e08(
            table,
            candidate_mask=candidate_mask,
            query_mask=query_mask,
            methods=("raw_prototype", "cfp_direction"),
            k=2,
            bootstrap_replicates=4,
        )

        self.assertTrue(e03)
        self.assertTrue([row for row in e04 if row["status"] == "completed"])
        self.assertTrue(e04_1)
        self.assertTrue(e04_2)
        self.assertTrue(e04_3)
        self.assertTrue([row for row in e04_4 if row["status"] == "completed"])
        self.assertTrue([row for row in e04_5 if row["status"] == "completed"])
        self.assertTrue(e05)
        self.assertEqual(e06["stage"], "E06")
        self.assertEqual(e08["stage"], "E08")

    def test_e04_2_keeps_excluded_from_main_classes(self) -> None:
        """E04-2 must still evaluate classes excluded from the main experiments (fit-budget scaling)."""
        metadata: list[FeatureSample] = []
        arrays: dict[str, list[np.ndarray]] = {
            "raw": [],
            "true_clear_degraded": [],
            "true_degraded_degraded": [],
            "true_clear_clean": [],
            "true_degraded_clean": [],
            "background_clear": [],
            "background_degraded": [],
        }
        # 3 classes; class 2 is excluded_from_main (only 1 fit object -> supports size 1 but not 2).
        for role in ("direction_fit", "confirmation"):
            for class_id, excluded in ((0, False), (1, False), (2, True)):
                for object_index in range(2 if class_id != 2 else 1):
                    source_image_id = f"{role}-fair1m-{class_id}-{object_index}"
                    object_id = f"object-{role}-{class_id}-{object_index}"
                    class_vector = np.asarray([2.0, 0.0, 0.25, 0.0] if class_id == 0 else [0.0, 2.0, 0.0, 0.25] if class_id == 1 else [0.25, 0.25, 2.0, 0.0])
                    raw = class_vector
                    delta = class_vector * 0.1
                    metadata.append(
                        FeatureSample(
                            dataset_id="fair1m",
                            object_id=object_id,
                            source_image_id=source_image_id,
                            class_id=class_id,
                            class_name=f"class-{class_id}",
                            coarse_name=f"coarse-{class_id}",
                            role=role,
                            perturbation="blur",
                            seed=object_index,
                            geometry=(float(class_id), 0.0, 0.0, 1.0),
                            excluded_from_main=excluded,
                        )
                    )
                    arrays["raw"].append(raw)
                    arrays["true_clear_degraded"].append(raw + delta)
                    arrays["true_degraded_degraded"].append(raw)
                    arrays["true_clear_clean"].append(raw + delta)
                    arrays["true_degraded_clean"].append(raw)
                    arrays["background_clear"].append(raw)
                    arrays["background_degraded"].append(raw)
        table = FeatureTable(
            model_id="synthetic",
            metadata=tuple(metadata),
            arrays={key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()},
            representation={"representation_id": "synthetic", "dimension": 4},
        )
        rows = run_e04_2(
            table,
            dataset_ids=["fair1m"],
            perturbations=["blur"],
            support_sizes=[1, 2],
            epsilon=1.0e-8,
        )
        # class 2 (excluded_from_main) must still be evaluated by E04-2 under fit-budget scaling.
        self.assertTrue(rows)
        direction_rows = [row for row in rows if row.get("analysis") == "direction_specificity"]
        completed = [row for row in direction_rows if row.get("status") != "failed"]
        self.assertTrue(completed)
        # E04-2 evaluates the excluded-from-main class: completed specificity rows exist with
        # a finite macro AUC over a 3-class confirmation set.
        for row in completed:
            self.assertIsNotNone(row.get("macro_ovr_auc"))
        # both support sizes are reported (15/30/50 analogue: 1 and 2 here)
        reported_sizes = {row.get("support_size") for row in direction_rows}
        self.assertEqual(reported_sizes, {1, 2})

    def test_e04_controls_are_configured_and_primary_cis_are_clustered(self) -> None:
        table = _table(("blur",))
        rows = run_e04(
            table,
            dataset_ids=["synthetic"],
            perturbations=["blur"],
            method_ids=["raw"],
            normalization_variants=["unit"],
            epsilon=1.0e-8,
            controls={"shifted_obb": False},
            statistics_config={"bootstrap_replicates": 8, "bootstrap_seed": 3},
        )

        true_rows = [row for row in rows if row["control"] == "true_obb"]
        disabled_rows = [row for row in rows if row["control"] == "shifted_obb"]
        self.assertEqual(len(true_rows), 1)
        self.assertEqual(true_rows[0]["status"], "completed")
        self.assertEqual(true_rows[0]["source_image_count"], 4)
        self.assertIn("own_argmax_rate_ci", true_rows[0])
        self.assertIn("mean_diagonal_margin_ci", true_rows[0])
        self.assertIn("macro_ovr_auc_ci", true_rows[0])
        self.assertTrue(all(row["status"] == "disabled" for row in disabled_rows))
        self.assertTrue(all(row["control_enabled"] is False for row in disabled_rows))
        self.assertTrue(all("control_registered" in row for row in rows))

    def test_e05_runs_cross_perturbation_baselines_and_marks_brightness(self) -> None:
        table = _table()
        rows = run_e05(
            table,
            dataset_ids=["synthetic"],
            perturbations=["blur", "noise", "brightness"],
            epsilon=1.0e-8,
            statistics_config={"bootstrap_replicates": 8, "bootstrap_seed": 5},
        )

        cross_rows = [row for row in rows if row["transfer"] == "cross_perturbation_delta"]
        prototype_rows = [row for row in rows if row["transfer"] == "raw_prototype"]
        cav_rows = [row for row in rows if row["transfer"] == "clear_degraded_cav"]
        self.assertTrue(any(row["source_perturbation"] != row["target_perturbation"] for row in cross_rows))
        self.assertEqual(len([row for row in prototype_rows if row.get("space") == "raw"]), 9)
        self.assertEqual(len([row for row in prototype_rows if row.get("space") == "centered"]), 9)
        self.assertEqual(len(cav_rows), 18)
        self.assertTrue(all(row.get("space") in ("raw", "centered") for row in prototype_rows + cav_rows))
        brightness_rows = [row for row in rows if row["target_perturbation"] == "brightness"]
        self.assertTrue(brightness_rows)
        self.assertTrue(all(row["negative_control"] for row in brightness_rows))
        self.assertTrue(all("macro_ovr_auc_ci" in row for row in prototype_rows + cav_rows))
        self.assertTrue(all("source_fit_source_image_count" in row for row in rows))

    def test_e05_target_operator_rows_depend_on_source_directions(self) -> None:
        metadata: list[FeatureSample] = []
        arrays: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "raw", "true_clear_degraded", "true_degraded_degraded",
                "true_clear_clean", "true_degraded_clean", "background_clear", "background_degraded",
            )
        }
        for perturbation_index, perturbation in enumerate(("blur", "noise", "brightness")):
            for role in ("direction_fit", "readout_fit", "confirmation"):
                for class_id in (0, 1):
                    for object_index in range(4):
                        base = np.asarray([3.0, 0.0, 0.0, 0.0] if class_id == 0 else [0.0, 3.0, 0.0, 0.0])
                        direction = np.asarray(
                            [1.0, 0.0, 0.4 * perturbation_index, 0.0]
                            if class_id == 0
                            else [0.0, 1.0, 0.0, 0.4 * perturbation_index]
                        )
                        # Confirmation deltas carry within-class noise so AUC/margin
                        # stay below saturation and remain source-sensitive.
                        noise = np.asarray([0.0, 0.0, 0.5 * (object_index - 1.5), 0.0]) if role == "confirmation" else 0.0
                        delta = direction + noise
                        metadata.append(
                            FeatureSample(
                                dataset_id="synthetic", object_id=f"{role}-{perturbation}-{class_id}-{object_index}",
                                source_image_id=f"{role}-{perturbation}-{class_id}-{object_index}", class_id=class_id,
                                class_name=f"class-{class_id}", coarse_name=f"coarse-{class_id}", role=role,
                                perturbation=perturbation, seed=object_index + perturbation_index,
                                geometry=(float(perturbation_index), float(object_index), 0.0, 1.0),
                            )
                        )
                        arrays["raw"].append(base)
                        arrays["true_clear_degraded"].append(base + delta)
                        arrays["true_degraded_degraded"].append(base)
                        arrays["true_clear_clean"].append(base + delta)
                        arrays["true_degraded_clean"].append(base)
                        arrays["background_clear"].append(base)
                        arrays["background_degraded"].append(base)
        table = FeatureTable(
            model_id="synthetic", metadata=tuple(metadata),
            arrays={key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()},
            representation={"representation_id": "synthetic", "dimension": 4},
        )
        rows = run_e05(
            table, dataset_ids=["synthetic"], source_perturbations=["blur", "noise", "brightness"],
            target_perturbations=["blur", "noise", "brightness"], epsilon=1e-8,
        )
        target_rows = [
            row for row in rows
            if row.get("analysis") == "cross_perturbation_target_operator" and row.get("status") == "completed"
        ]
        self.assertTrue(target_rows)
        self.assertTrue(all(row["source_perturbation"] != row["target_perturbation"] for row in target_rows))
        by_target: dict[str, set[tuple[float, float]]] = {}
        for row in target_rows:
            by_target.setdefault(row["target_perturbation"], set()).add(
                (round(float(row["macro_ovr_auc"]), 9), round(float(row["mean_diagonal_margin"]), 9))
            )
        # A fixed target scored against two different source direction sets must
        # differ; otherwise the "transfer" row would just be a target self-score.
        self.assertTrue(any(len(values) > 1 for values in by_target.values()), by_target)

    def test_e07_reports_prototype_priority_recovery_and_drop_cis(self) -> None:
        table = _table(("blur",))
        rows = run_e07(
            table,
            dataset_ids=["synthetic"],
            perturbations=["blur"],
            removal_alphas=[0.0, 1.0],
            reflection_alpha=2.0,
            preserve_norm=True,
            epsilon=1.0e-8,
            statistics_config={"bootstrap_replicates": 8, "bootstrap_seed": 7},
        )

        self.assertTrue(rows)
        row = rows[0]
        self.assertIn("prototype_priority", row)
        self.assertIn("retrained_recovery", row)
        self.assertIn("retrained_recovery_ci", row)
        self.assertIn("frozen_linear_accuracy_drop_ci", row)
        self.assertIn("retrained_prototype_accuracy_drop_ci", row)
        self.assertGreaterEqual(row["confirmation_source_image_count"], 1)

    def test_e04_3_reports_space_and_class_mean_cosine(self) -> None:
        table = _table(("blur",))
        rows = run_e04_3(table, dataset_ids=["synthetic"], perturbations=["blur"])
        self.assertEqual({row["space"] for row in rows}, {"raw", "centered"})
        for row in rows:
            self.assertTrue(np.isfinite(row["clean_degraded_class_mean_delta_cosine"]))

    def test_e07_reports_raw_and_class_mean_axis_controls(self) -> None:
        table = _table(("blur",))
        rows = run_e07(
            table,
            dataset_ids=["synthetic"],
            perturbations=["blur"],
            removal_alphas=[0.0, 1.0],
            reflection_alpha=2.0,
            preserve_norm=True,
            epsilon=1.0e-8,
        )
        conditions = {row["condition"] for row in rows if row.get("status") != "missing"}
        self.assertIn("own_raw", conditions)
        self.assertIn("wrong_raw", conditions)
        self.assertIn("class_mean_axis", conditions)

    def test_e07_reports_centered_and_raw_readout_spaces(self) -> None:
        table = _table(("blur",))
        rows = run_e07(
            table,
            dataset_ids=["synthetic"],
            perturbations=["blur"],
            removal_alphas=[0.0, 1.0],
            reflection_alpha=2.0,
            preserve_norm=True,
            epsilon=1.0e-8,
        )
        observed = [row for row in rows if row.get("status") != "missing"]
        spaces = {row["space"] for row in observed}
        self.assertEqual(spaces, {"centered", "raw"})
        conditions_by_space = {
            space: {row["condition"] for row in observed if row["space"] == space}
            for space in spaces
        }
        # Both spaces evaluate the same control bank; only the origin differs.
        self.assertEqual(conditions_by_space["centered"], conditions_by_space["raw"])
        centered = next(
            row for row in observed if row["space"] == "centered" and row["condition"] == "own_class"
        )
        raw = next(
            row for row in observed if row["space"] == "raw" and row["condition"] == "own_class"
        )
        # The uncentered M0 reference keeps the common <m, w> shift, so its
        # measured projection energy differs from the centered primary result.
        self.assertNotAlmostEqual(
            centered["mean_abs_cos_h_w"], raw["mean_abs_cos_h_w"], places=6
        )

    def test_e05_target_operator_is_skipped_when_source_equals_target(self) -> None:
        table = _table(("blur", "noise"))
        rows = run_e05(
            table,
            dataset_ids=["synthetic"],
            source_perturbations=["blur", "noise"],
            target_perturbations=["blur", "noise"],
            epsilon=1.0e-8,
        )
        target_rows = [row for row in rows if row.get("analysis") == "cross_perturbation_target_operator"]
        self.assertTrue(target_rows)
        self.assertTrue(all(row["source_perturbation"] != row["target_perturbation"] for row in target_rows))
        self.assertTrue(all(row.get("space") == "unit_delta_M1_target_operator" for row in target_rows))

    def test_e06_e08_interfaces_accept_raw_feature_table_and_statistics_are_written(self) -> None:
        table = _table(("blur", "noise"))
        e06 = run_e06(table, ["raw"], epsilon=1.0e-8)
        confirmation = [index for index, sample in enumerate(table.metadata) if sample.role == "confirmation"]
        confirmation_images = sorted({table.metadata[index].source_image_id for index in confirmation})
        candidate_images = set(confirmation_images[: len(confirmation_images) // 2])
        candidate_mask = np.asarray(
            [sample.role == "confirmation" and sample.source_image_id in candidate_images for sample in table.metadata],
            dtype=bool,
        )
        query_mask = np.asarray(
            [sample.role == "confirmation" and sample.source_image_id not in candidate_images for sample in table.metadata],
            dtype=bool,
        )
        e08 = run_e08(
            table,
            feature_key="raw",
            candidate_mask=candidate_mask,
            query_mask=query_mask,
            methods=("raw_prototype", "cfp_direction"),
            k=2,
            bootstrap_replicates=8,
        )
        self.assertEqual(e06["stage"], "E06")
        self.assertEqual(e08["stage"], "E08")
        self.assertIn("macro_ap", e08["methods"]["raw_prototype"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = write_core_statistics(
                root,
                {"models": {"synthetic": {"E04": run_e04(
                    table,
                    dataset_ids=["synthetic"],
                    perturbations=["blur"],
                    method_ids=["raw"],
                    normalization_variants=["unit"],
                    epsilon=1.0e-8,
                    controls={name: False for name in ("shifted_obb", "random_geometry_matched_obb", "label_permutation", "background_only", "boundary_only", "interior_eroded", "mask_dilation_erosion", "geometry_regression")},
                    statistics_config={"bootstrap_replicates": 4},
                )}}},
                statistics_config={"bootstrap_replicates": 4},
            )
            self.assertTrue((root / "statistics" / "core_statistics.json").is_file())
            self.assertTrue((root / "statistics" / "paired_effects.json").is_file())
            self.assertTrue((root / "statistics" / "results.json").is_file())
            self.assertIn("fdr", summary)

if __name__ == "__main__":
    unittest.main()
