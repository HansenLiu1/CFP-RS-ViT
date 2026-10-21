import csv
import re
import unittest
from pathlib import Path

from cfra.data import load_class_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ROOT = PROJECT_ROOT / "registry"


def read_rows(name: str) -> list[dict[str, str]]:
    with (REGISTRY_ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class RegistryTests(unittest.TestCase):
    def test_claims_reference_registered_experiments(self) -> None:
        experiment_ids = {row["experiment_id"] for row in read_rows("experiments.csv")}
        unknown: list[str] = []
        for claim in read_rows("claims.csv"):
            for experiment_id in claim["required_experiments"].split(";"):
                if experiment_id and experiment_id not in experiment_ids:
                    unknown.append(f"{claim['claim_id']} -> {experiment_id}")
        self.assertEqual(unknown, [], "Unknown experiment references: " + ", ".join(unknown))

    def test_experiment_prerequisites_are_registered(self) -> None:
        rows = read_rows("experiments.csv")
        experiment_ids = {row["experiment_id"] for row in rows}
        unknown: list[str] = []
        for experiment in rows:
            for prerequisite in experiment["scientific_prerequisites"].split(";"):
                if prerequisite and prerequisite != "none" and prerequisite not in experiment_ids:
                    unknown.append(f"{experiment['experiment_id']} -> {prerequisite}")
        self.assertEqual(unknown, [], "Unknown prerequisites: " + ", ".join(unknown))

    def test_all_claims_start_unverified(self) -> None:
        statuses = {row["status"] for row in read_rows("claims.csv")}
        self.assertEqual(statuses, {"unverified"})

    def test_all_experiment_protocols_are_frozen(self) -> None:
        states = {row["protocol_state"] for row in read_rows("experiments.csv")}
        self.assertEqual(states, {"frozen"})

    def test_experiment_models_are_registered(self) -> None:
        model_ids = {row["model_id"] for row in read_rows("models.csv")}
        unknown: list[str] = []
        for experiment in read_rows("experiments.csv"):
            if experiment["models"] in {"all", "none"}:
                continue
            for model_id in experiment["models"].split(";"):
                if model_id not in model_ids:
                    unknown.append(f"{experiment['experiment_id']} -> {model_id}")
        self.assertEqual(unknown, [], "Unknown experiment models: " + ", ".join(unknown))

    def test_model_variants_share_parent_independence_group(self) -> None:
        models = {row["model_id"]: row for row in read_rows("models.csv")}
        for model in models.values():
            parent_id = model["analysis_variant_of"]
            if not parent_id:
                continue
            self.assertIn(parent_id, models)
            self.assertEqual(model["independence_group"], models[parent_id]["independence_group"])

    def test_core_image_only_models_use_distinct_objectives(self) -> None:
        models = [row for row in read_rows("models.csv") if row["role"] == "core_image_only"]
        self.assertEqual({row["model_id"] for row in models}, {"dinov2", "mae"})
        self.assertTrue(all(row["pretraining_modality"] == "image_only" for row in models))
        self.assertEqual(len({row["objective_family"] for row in models}), 2)

    def test_unprojected_clip_is_still_image_text_pretrained(self) -> None:
        models = {row["model_id"]: row for row in read_rows("models.csv")}
        self.assertEqual(models["clip"]["text_shared_space"], "no")
        self.assertEqual(models["clip"]["pretraining_modality"], "image_text")

    def test_registered_core_checkpoints_have_verified_identity(self) -> None:
        models = {row["model_id"]: row for row in read_rows("models.csv")}
        for model_id in ("remoteclip", "clip", "clip_shared", "siglip2", "dinov2", "mae"):
            with self.subTest(model_id=model_id):
                model = models[model_id]
                self.assertEqual(model["status"], "checkpoint_verified")
                self.assertRegex(model["checkpoint_revision"], re.compile(r"^[0-9a-f]{40}$"))
                self.assertRegex(model["checkpoint_sha256"], re.compile(r"^[0-9a-f]{64}$"))
                self.assertRegex(model["loader_config_sha256"], re.compile(r"^[0-9a-f]{64}$"))


class ClassRegistryLoadingTests(unittest.TestCase):
    def test_registry_root_and_json_file_resolve_identically(self) -> None:
        from_root = load_class_registry(REGISTRY_ROOT)
        from_file = load_class_registry(REGISTRY_ROOT / "dataset_classes.json")

        self.assertEqual(set(from_root), set(from_file))
        for dataset_id in ("dota", "dior", "fair1m"):
            self.assertIn(dataset_id, from_root)
            self.assertTrue(from_root[dataset_id])
        self.assertEqual(from_root, from_file)


if __name__ == "__main__":
    unittest.main()
