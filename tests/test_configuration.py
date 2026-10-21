import csv
import tempfile
import unittest
from pathlib import Path

from cfra.configuration import (
    ConfigurationError,
    configured_dataset_paths,
    configured_model_paths,
    load_experiment_config,
    validate_execution_policy,
    validate_formal_freeze,
    validate_registry_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_external_paths_are_explicit_and_scope_aware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = {
                "paths": {
                    "datasets": {"dota": str(root / "dota"), "unused": "relative/path"},
                    "models": {"clip": str(root / "clip")},
                }
            }
            self.assertEqual(configured_dataset_paths(config, ["dota"]), {"dota": root / "dota"})
            self.assertEqual(configured_model_paths(config, ["clip"]), {"clip": root / "clip"})

    def test_external_paths_reject_missing_or_relative_mappings(self) -> None:
        config = {
            "paths": {
                "datasets": {"dota": "relative/path"},
                "models": {},
            }
        }
        with self.assertRaises(ConfigurationError):
            configured_dataset_paths(config, ["dota"])
        with self.assertRaises(ConfigurationError):
            configured_model_paths(config, ["clip"])

    def test_smoke_inherits_clean_execution_policy(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "smoke.toml")
        validate_execution_policy(config)
        validate_registry_contract(config, PROJECT_ROOT / "registry")
        self.assertEqual(config["input"]["size_px"], 448)
        self.assertEqual(config["scope"]["datasets"], ["dota"])
        self.assertEqual(config["scope"]["max_source_images_per_role"], 8)
        self.assertEqual(config["sampling"]["minimum_direction_fit_images_per_class"], 2)
        self.assertNotIn("transform_support_and_query", config["methods"])

    def test_main_is_frozen_and_passes_freeze_check(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        validate_execution_policy(config)
        validate_registry_contract(config, PROJECT_ROOT / "registry")
        validate_formal_freeze(config)

    def test_rejects_removal_alpha_above_one(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "base.toml")
        config["intervention"]["removal_alphas"] = [0.0, 1.0, 2.0]
        with self.assertRaises(ConfigurationError):
            validate_execution_policy(config)

    def test_rejects_inheritance_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.toml").write_text('extends = "b.toml"\n', encoding="ascii")
            (root / "b.toml").write_text('extends = "a.toml"\n', encoding="ascii")
            with self.assertRaises(ConfigurationError):
                load_experiment_config(root / "a.toml")

    def test_rejects_clip_as_image_only_generalization(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        config["scope"]["generalization_models"] = ["clip", "mae"]
        with self.assertRaises(ConfigurationError):
            validate_registry_contract(config, PROJECT_ROOT / "registry")

    def test_rejects_shared_space_variant_as_independent_generalization(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        config["scope"]["generalization_models"] = ["clip_shared", "mae"]
        with self.assertRaises(ConfigurationError):
            validate_registry_contract(config, PROJECT_ROOT / "registry")

    def test_rejects_text_baseline_outside_shared_space(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        config["text_baseline_models"]["E08"] = ["dinov2"]
        with self.assertRaises(ConfigurationError):
            validate_registry_contract(config, PROJECT_ROOT / "registry")

    def test_rejects_run_group_without_qualification_stages(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        config["run_groups"]["layerwise"]["experiments"] = ["E00", "E01", "E06"]
        with self.assertRaises(ConfigurationError):
            validate_registry_contract(config, PROJECT_ROOT / "registry")

    def test_rejects_cross_group_derived_inputs(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        config["run_groups"]["retrieval"]["allow_cross_group_derived_inputs"] = True
        with self.assertRaises(ConfigurationError):
            validate_registry_contract(config, PROJECT_ROOT / "registry")

    def test_rejects_run_group_dependency_cycle(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        config["run_groups"]["core"]["scientific_dependencies"] = ["intervention"]
        with self.assertRaises(ConfigurationError):
            validate_registry_contract(config, PROJECT_ROOT / "registry")

    def test_rejects_unfrozen_release_when_formal_config_is_marked_frozen(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        config["release"]["status"] = "draft"
        with self.assertRaises(ConfigurationError):
            validate_formal_freeze(config)

    def test_formal_freeze_rejects_development_model_runtime(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        config["model_runtime"]["formal_approved"] = False
        with self.assertRaises(ConfigurationError):
            validate_formal_freeze(config)

    def test_main_registers_mae_in_the_model_family_gate_and_experiment_scope(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        validate_registry_contract(config, PROJECT_ROOT / "registry")

        self.assertTrue(config["run"]["frozen"])
        self.assertEqual(config["release"]["status"], "frozen")
        self.assertIn("mae", config["scope"]["models"])
        self.assertIn("mae", config["scope"]["generalization_models"])
        self.assertEqual(set(config["model_family_gate"]["required_image_only_models"]), {"dinov2", "mae"})
        self.assertEqual(
            set(config["model_family_gate"]["required_image_only_objective_families"]),
            {"teacher_student_patch_ssl", "masked_pixel_reconstruction"},
        )
        experiment_ids_with_mae = (
            "E03",
            "E04",
            "E04-1",
            "E04-2",
            "E04-3",
            "E04-4",
            "E04-5",
            "E05",
            "E06",
            "E08",
        )
        for experiment_id in experiment_ids_with_mae:
            with self.subTest(experiment_id=experiment_id):
                self.assertIn("mae", config["experiment_models"][experiment_id])
        self.assertEqual(set(config["experiment_models"]["E07"]), {"remoteclip", "clip", "siglip2"})

    def test_formal_run_groups_bind_all_registered_split_seeds(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        expected = [1729, 2718, 3141]
        self.assertEqual(config["split"]["split_seeds"], expected)
        self.assertEqual(config["robustness"]["split_seeds"], expected)
        for group_name in config["release"]["run_groups"]:
            with self.subTest(group_name=group_name):
                group = config["run_groups"][group_name]
                self.assertEqual(group["split_seeds"], expected)
                self.assertEqual(group["seed_policy"], "execute_all_registered_seeds_without_selection")

        development = load_experiment_config(PROJECT_ROOT / "configs" / "development_core.toml")
        self.assertEqual(development["split"]["split_seeds"], [1729])
        self.assertEqual(development["robustness"]["split_seeds"], [1729])

    def test_protocol_fields_are_versioned(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        self.assertEqual(config["project"]["protocol_version"], "0.4.0")
        self.assertEqual(config["input"]["size_px"], 448)
        self.assertEqual(config["model_runtime"]["encoder_input_sizes"]["mae"], 448)
        interpolation = config["model_runtime"]["positional_interpolation"]
        self.assertTrue(interpolation["enabled"])
        self.assertEqual(interpolation["input_size_px"], 448)
        self.assertEqual(interpolation["target_grid"]["mae"], 28)
        self.assertEqual(config["protocol"]["near_zero_norm_epsilon"], 1.0e-8)
        self.assertEqual(config["protocol"]["control_min_iou"], 0.0)
        self.assertEqual(config["protocol"]["control_max_iou"], 0.05)
        self.assertEqual(config["protocol"]["near_zero_comparison"], "norm_less_than_or_equal_to_epsilon")
        self.assertEqual(config["sampling"]["exclusion"]["schema_version"], "exclusion-v1")
        self.assertEqual(config["sampling"]["exclusion"]["image_dimension_reference"], "4000x4000_equivalent")
        self.assertEqual(config["protocol"]["e04_5"]["patch_budget_grid"], [0.20, 0.35, 0.50])
        self.assertEqual(config["protocol"]["e04_5"]["patch_side_grid"], [0.05, 0.10, 0.15])
        text_protocol = config["text_baseline_protocol"]
        self.assertEqual(text_protocol["template_count"], 4)
        self.assertEqual(len(text_protocol["prompt_templates"]), 4)
        self.assertEqual(text_protocol["class_name_source"], "registry/dataset_classes.json")

    def test_e07_e08_registry_models_and_controls_match_config(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "main.toml")
        with (PROJECT_ROOT / "registry" / "experiments.csv").open(encoding="utf-8", newline="") as handle:
            rows = {row["experiment_id"]: row for row in csv.DictReader(handle)}

        for experiment_id in ("E07", "E08"):
            with self.subTest(experiment_id=experiment_id):
                registry_models = set(rows[experiment_id]["models"].split(";"))
                self.assertEqual(registry_models, set(config["experiment_models"][experiment_id]))

        for experiment_id in ("E07", "E08"):
            with self.subTest(controls=experiment_id):
                registry_controls = set(rows[experiment_id]["required_controls"].split(";"))
                configured_controls = set(config["experiment_controls"][experiment_id]["controls"])
                self.assertEqual(registry_controls, configured_controls)


if __name__ == "__main__":
    unittest.main()
