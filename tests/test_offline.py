import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cfra.offline import (
    OfflineRunnerError,
    _e00_relevant,
    _layerwise_cache_has_readout,
    _layerwise_input_signature,
    _load_cached_e06_inputs,
    build_offline_config,
    infer_completed_stages,
    open_offline_run,
    parameters_signature,
)
from cfra.pipeline.features import _stable_seed
from cfra.resume import load_checkpoint, write_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CACHE_SPLIT_SEED = 1729


def _cache_config() -> dict:
    return {
        "scope": {"datasets": ["dota"], "perturbations": ["blur"]},
        "split": {"seed": _CACHE_SPLIT_SEED},
    }


def _write_layerwise_artifact(root: Path, model_id: str, dataset_id: str, layers: list[str], *, with_raw: bool) -> None:
    """Write a minimal loader-valid layerwise paired artifact for cache tests."""

    features = root / "intermediates" / "features"
    features.mkdir(parents=True, exist_ok=True)
    count = 6
    arrays: dict[str, np.ndarray] = {}
    for layer in layers:
        arrays[f"{layer}__clear"] = np.ones((count, 3), dtype=np.float64)
        arrays[f"{layer}__degraded"] = np.zeros((count, 3), dtype=np.float64)
        if with_raw:
            arrays[f"{layer}__raw"] = np.full((count, 3), 2.0, dtype=np.float64)
    name = f"layerwise_{model_id}_{dataset_id}"
    np.savez_compressed(features / f"{name}.npz", **arrays)
    rows = []
    for index in range(count):
        rows.append(
            {
                "role": ("direction_fit", "readout_fit", "confirmation")[index % 3],
                "class_id": index % 2,
                "perturbation": "blur",
                "dataset_id": dataset_id,
                "source_image_id": f"img-{index}",
                "object_id": f"object-{index}",
                "seed": _stable_seed(_CACHE_SPLIT_SEED, f"object-{index}", "blur"),
            }
        )
    layer_metadata = {
        layer: {
            "kind": "block", "block": 3, "dim": 3, "pooling": "cls", "projection": "none",
            "text_shared_space": False, "token": "cls", "grid": [1, 1], "interpolation": "none",
        }
        for layer in layers
    }
    metadata = {
        "schema_version": 1, "artifact_kind": "layerwise_paired_features", "model_id": model_id,
        "selected_layers": layers, "layer_metadata": layer_metadata, "sample_count": count,
        "row_metadata": rows, "array_path": f"{name}.npz", "dataset_id": dataset_id,
        "input_signature": _layerwise_input_signature(_cache_config(), model_id, dataset_id),
    }
    (features / f"{name}.json").write_text(json.dumps(metadata), encoding="utf-8")



class OfflineParametersTests(unittest.TestCase):
    def test_signature_changes_with_parameters(self) -> None:
        base = {
            "datasets": ["dota"],
            "seed": 1729,
            "models": ["clip"],
            "perturbations": ["blur"],
            "experiment_ids": ["E00", "E01"],
            "run_e04_5": False,
        }
        sig = parameters_signature(**base)
        self.assertEqual(sig, parameters_signature(**base))
        self.assertNotEqual(sig, parameters_signature(**{**base, "seed": 2718}))
        self.assertNotEqual(sig, parameters_signature(**{**base, "datasets": ["dior"]}))
        self.assertNotEqual(sig, parameters_signature(**{**base, "run_e04_5": True}))
        self.assertNotEqual(sig, parameters_signature(**{**base, "config_hash": "abc"}))

    def test_build_config_applies_parameters(self) -> None:
        config = build_offline_config(
            PROJECT_ROOT / "configs" / "offline.toml",
            datasets=["dota"],
            seed=2718,
            models=["clip"],
            perturbations=["blur"],
            experiment_ids=["E00", "E01", "E02", "E03"],
            run_e04_5=True,
        )
        self.assertEqual(config["scope"]["datasets"], ["dota"])
        self.assertEqual(config["scope"]["models"], ["clip"])
        self.assertEqual(config["scope"]["perturbations"], ["blur"])
        self.assertEqual(config["split"]["seed"], 2718)
        self.assertEqual(config["split"]["split_seeds"], [2718])
        self.assertEqual(config["statistics"]["bootstrap_seed"], 2718)
        self.assertTrue(config["experiment_controls"]["E04-5"]["enabled"])


class OfflineRunDirectoryTests(unittest.TestCase):
    def _config(self, seed: int = 1729) -> dict:
        return build_offline_config(
            PROJECT_ROOT / "configs" / "offline.toml",
            datasets=["dota"],
            seed=seed,
            models=["clip"],
            perturbations=["blur"],
            experiment_ids=["E00", "E01", "E02", "E03"],
            run_e04_5=False,
        )

    def test_missing_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "does-not-exist"
            with self.assertRaises(OfflineRunnerError):
                open_offline_run(target, self._config(), "sig")

    def test_fresh_initializes_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run1"
            target.mkdir()
            _run_root, mode, completed = open_offline_run(target, self._config(), "sigA")
            self.assertEqual(mode, "fresh")
            self.assertEqual(completed, [])
            manifest = json.loads((target / "manifest" / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "offline")
            self.assertEqual(manifest["parameter_signature"], "sigA")
            self.assertEqual((target / "STATUS").read_text(encoding="ascii").strip(), "running")

    def test_resume_reconciles_checkpoint_with_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run2"
            target.mkdir()
            open_offline_run(target, self._config(), "sigB")
            write_checkpoint(
                target,
                config_sha256="offline-trusted",
                evidence_tier="development",
                completed_stages=["E00", "E01"],
            )
            # Only stages whose artifacts are still present are trusted.
            (target / "manifest").mkdir(parents=True, exist_ok=True)
            (target / "manifest" / "e00.json").write_text("{}\n", encoding="utf-8")
            (target / "manifest" / "e01.json").write_text("{}\n", encoding="utf-8")
            _run_root, mode, completed = open_offline_run(target, self._config(), "sigB")
            self.assertEqual(mode, "resume")
            self.assertEqual(sorted(completed), ["E00", "E01"])

    def test_resume_drops_missing_artifact_for_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run2b"
            target.mkdir()
            open_offline_run(target, self._config(), "sigB2")
            write_checkpoint(
                target,
                config_sha256="offline-trusted",
                evidence_tier="development",
                completed_stages=["E00", "E01"],
            )
            (target / "manifest").mkdir(parents=True, exist_ok=True)
            (target / "manifest" / "e00.json").write_text("{}\n", encoding="utf-8")
            # E01 artifact was deleted: it must not be trusted as complete.
            _run_root, mode, completed = open_offline_run(target, self._config(), "sigB2")
            self.assertEqual(mode, "resume")
            self.assertEqual(completed, ["E00"])

    def test_analysis_only_config_change_is_soft_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run2c"
            target.mkdir()
            open_offline_run(target, self._config(), "sigS1")
            features = target / "intermediates" / "features"
            features.mkdir(parents=True, exist_ok=True)
            (features / "features_clip.npz").write_bytes(b"keep-me")
            # Same feature-affecting inputs, different signature (e.g. a new
            # analysis-only switch such as layerwise_readout).
            _run_root, mode, _completed = open_offline_run(target, self._config(), "sigS2")
            self.assertEqual(mode, "soft-reset")
            self.assertTrue((features / "features_clip.npz").is_file())

    def test_parameter_change_triggers_reset_from_e00(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run3"
            target.mkdir()
            open_offline_run(target, self._config(seed=1729), "sigC")
            write_checkpoint(
                target,
                config_sha256="offline-trusted",
                evidence_tier="development",
                completed_stages=["E00", "E01"],
            )
            (target / "intermediates").mkdir(parents=True, exist_ok=True)
            (target / "intermediates" / "dummy.npz").write_bytes(b"x")
            _run_root, mode, completed = open_offline_run(target, self._config(seed=2718), "sigD")
            self.assertEqual(mode, "reset")
            self.assertEqual(completed, [])
            self.assertFalse((target / "intermediates" / "dummy.npz").exists())
            checkpoint = load_checkpoint(target)
            self.assertEqual(checkpoint["completed_stages"], [])

    def test_reuse_e00_keeps_e00_on_seed_only_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run8"
            target.mkdir()
            open_offline_run(target, self._config(seed=1729), "sigH")
            e00 = target / "intermediates" / "e00_source_manifest"
            e00.mkdir(parents=True, exist_ok=True)
            (e00 / "dota_objects.jsonl").write_text("{}\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"CFRA_REUSE_E00": "1"}, clear=False):
                _run_root, mode, completed = open_offline_run(target, self._config(seed=2718), "sigI")
            self.assertEqual(mode, "reset")
            self.assertEqual(completed, ["E00"])
            self.assertTrue((e00 / "dota_objects.jsonl").is_file())
            self.assertEqual(load_checkpoint(target)["completed_stages"], ["E00"])

    def test_reuse_e00_refuses_when_e00_inputs_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run9"
            target.mkdir()
            open_offline_run(target, self._config(seed=1729), "sigJ")
            e00 = target / "intermediates" / "e00_source_manifest"
            e00.mkdir(parents=True, exist_ok=True)
            (e00 / "dota_objects.jsonl").write_text("{}\n", encoding="utf-8")
            other = build_offline_config(
                PROJECT_ROOT / "configs" / "offline.toml",
                datasets=["dior"],
                seed=2718,
                models=["clip"],
                perturbations=["blur"],
                experiment_ids=["E00", "E01", "E02", "E03"],
                run_e04_5=False,
            )
            with mock.patch.dict(os.environ, {"CFRA_REUSE_E00": "1"}, clear=False):
                _run_root, mode, completed = open_offline_run(target, other, "sigK")
            self.assertEqual(mode, "reset")
            self.assertEqual(completed, [])
            self.assertFalse(e00.exists())

    def test_e00_relevant_strips_seed_only_fields(self) -> None:
        base = self._config(seed=1729)
        other = self._config(seed=2718)
        self.assertEqual(_e00_relevant(base), _e00_relevant(other))
        changed = json.loads(json.dumps(base))
        changed["scope"]["datasets"] = ["dior"]
        self.assertNotEqual(_e00_relevant(base), _e00_relevant(changed))

    def test_non_empty_directory_without_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run4"
            target.mkdir()
            (target / "somefile.txt").write_text("x", encoding="ascii")
            with self.assertRaises(OfflineRunnerError):
                open_offline_run(target, self._config(), "sigE")

    def test_pre_existing_logs_dir_does_not_block_fresh_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run7"
            target.mkdir()
            (target / "logs").mkdir()
            (target / "logs" / "main.log").write_text("previous run\n", encoding="utf-8")
            _run_root, mode, completed = open_offline_run(target, self._config(), "sigG")
            self.assertEqual(mode, "fresh")
            self.assertEqual(completed, [])
            self.assertTrue((target / "logs" / "main.log").is_file())
            self.assertTrue((target / "manifest" / "run.json").is_file())

    def test_non_offline_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run5"
            target.mkdir()
            (target / "manifest").mkdir()
            (target / "manifest" / "run.json").write_text(
                json.dumps({"mode": "formal"}), encoding="utf-8"
            )
            with self.assertRaises(OfflineRunnerError):
                open_offline_run(target, self._config(), "sigF")


class OfflineInferenceTests(unittest.TestCase):
    def test_infer_completed_stages_from_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run6"
            target.mkdir()
            config = build_offline_config(
                PROJECT_ROOT / "configs" / "offline.toml",
                datasets=["dota"],
                seed=1729,
                models=["clip"],
                perturbations=["blur"],
                experiment_ids=["E00", "E01", "E02", "E03"],
                run_e04_5=False,
            )
            (target / "manifest").mkdir(parents=True)
            (target / "manifest" / "e00.json").write_text("{}", encoding="utf-8")
            (target / "manifest" / "e01.json").write_text("{}", encoding="utf-8")
            (target / "intermediates" / "e02_pixel_protocol").mkdir(parents=True)
            (target / "intermediates" / "e02_pixel_protocol" / "report.json").write_text("{}", encoding="utf-8")
            (target / "intermediates" / "features").mkdir(parents=True)
            (target / "intermediates" / "features" / "features_clip.npz").write_bytes(b"x")
            (target / "manifest" / "features_clip.json").write_text("{}", encoding="utf-8")
            (target / "intermediates" / "stage_results").mkdir(parents=True)
            (target / "intermediates" / "stage_results" / "E03.json").write_text("{}", encoding="utf-8")
            completed = infer_completed_stages(target, config)
            self.assertIn("E00", completed)
            self.assertIn("E01", completed)
            self.assertIn("E02", completed)
            self.assertIn("features:clip:nominal", completed)
            self.assertIn("E03", completed)

    def _e08_config(self, directory: Path) -> object:
        config = build_offline_config(
            PROJECT_ROOT / "configs" / "offline.toml",
            datasets=["dota", "dior"],
            seed=1729,
            models=["clip", "mae"],
            perturbations=["blur", "patch_mask"],
            experiment_ids=["E08"],
            run_e04_5=False,
        )
        (directory / "manifest").mkdir(parents=True, exist_ok=True)
        return config

    def test_infer_e08_not_complete_when_partial_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run7"
            target.mkdir()
            config = self._e08_config(target)
            (target / "metrics").mkdir(parents=True)
            (target / "metrics" / "E08_clip_dota_blur.json").write_text(
                json.dumps({"stage": "E08"}), encoding="utf-8"
            )
            completed = infer_completed_stages(target, config)
            self.assertNotIn("E08:clip", completed)
            self.assertNotIn("E08:mae", completed)

    def test_infer_e08_complete_when_all_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run8"
            target.mkdir()
            config = self._e08_config(target)
            (target / "metrics").mkdir(parents=True)
            cells = [
                (dataset_id, perturbation)
                for dataset_id in ("dota", "dior")
                for perturbation in ("blur", "patch_mask")
            ]
            for dataset_id, perturbation in cells:
                (target / "metrics" / f"E08_clip_{dataset_id}_{perturbation}.json").write_text(
                    json.dumps({"stage": "E08", "dataset_id": dataset_id, "perturbation": perturbation}),
                    encoding="utf-8",
                )
            (target / "metrics" / "E08_clip.json").write_text(
                json.dumps({"stage": "E08", "status": "completed"}), encoding="utf-8"
            )
            completed = infer_completed_stages(target, config)
            self.assertIn("E08:clip", completed)
            self.assertNotIn("E08:mae", completed)


class E06CacheReuseTests(unittest.TestCase):
    def test_cache_has_readout_requires_raw_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_layerwise_artifact(root, "clip", "dota", ["L03", "L06"], with_raw=False)
            self.assertFalse(_layerwise_cache_has_readout(root, "clip", "dota", ["L03", "L06"]))
            _write_layerwise_artifact(root, "clip", "dota", ["L03", "L06"], with_raw=True)
            self.assertTrue(_layerwise_cache_has_readout(root, "clip", "dota", ["L03", "L06"]))
            self.assertFalse(_layerwise_cache_has_readout(root, "clip", "dota", ["L03", "L09"]))

    def test_load_cached_inputs_requires_raw_for_readout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_layerwise_artifact(root, "clip", "dota", ["L03", "L06"], with_raw=False)
            self.assertIsNone(
                _load_cached_e06_inputs(
                    run_root=root, config=_cache_config(), model_id="clip", dataset_id="dota",
                    selected_layers=["L03", "L06"], readout_enabled=True,
                )
            )
            cached_no_readout = _load_cached_e06_inputs(
                run_root=root, config=_cache_config(), model_id="clip", dataset_id="dota",
                selected_layers=["L03", "L06"], readout_enabled=False,
            )
            self.assertIsNotNone(cached_no_readout)
            self.assertIsNone(cached_no_readout[2])

            _write_layerwise_artifact(root, "clip", "dota", ["L03", "L06"], with_raw=True)
            cached = _load_cached_e06_inputs(
                run_root=root, config=_cache_config(), model_id="clip", dataset_id="dota",
                selected_layers=["L03", "L06"], readout_enabled=True,
            )
            self.assertIsNotNone(cached)
            layer_inputs, metadata, readout_arrays = cached
            self.assertEqual(set(layer_inputs), {"L03", "L06"})
            self.assertEqual(set(readout_arrays), {"L03", "L06"})
            self.assertEqual(len(metadata), 6)

    def test_load_cached_inputs_rejects_stale_split_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_layerwise_artifact(root, "clip", "dota", ["L03", "L06"], with_raw=True)
            mismatched = {
                "scope": {"datasets": ["dota"], "perturbations": ["blur"]},
                "split": {"seed": _CACHE_SPLIT_SEED + 1},
            }
            self.assertIsNone(
                _load_cached_e06_inputs(
                    run_root=root, config=mismatched, model_id="clip", dataset_id="dota",
                    selected_layers=["L03", "L06"], readout_enabled=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
