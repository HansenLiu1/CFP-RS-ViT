"""Perturbation-strength sweep (E04-5) grid, cell runner and export wiring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cfra.pipeline.features import FeatureSample, FeatureTable
from cfra.pipeline.strength_sweep import (
    SWEEP_METHODS,
    StrengthSweepError,
    SweepCell,
    load_pair_table,
    nominal_cell,
    resolve_sweep_cells,
    run_sweep_cell,
    save_pair_table,
)

_CONFIG = {
    "split": {"seed": 1729},
    "perturbations": {
        "blur": {"sigma_px": 16.0},
        "patch_mask": {"target_fraction": 0.35, "patch_side_fraction": 0.10},
        "gaussian_noise": {"sigma": 0.12},
        "brightness": {"factor": 0.55},
    },
    "perturbation_strengths": {
        "blur": {"low": 8.0, "nominal": 16.0, "high": 24.0},
        "patch_mask": {"low": 0.20, "nominal": 0.35, "high": 0.50},
        "gaussian_noise": {"low": 0.06, "nominal": 0.12, "high": 0.18},
        "brightness": {"low": 0.75, "nominal": 0.55, "high": 0.35},
    },
    "protocol": {
        "e04_5": {"patch_budget_grid": [0.20, 0.35, 0.50], "patch_side_grid": [0.05, 0.10, 0.15]}
    },
    "scope": {"perturbations": ["blur", "patch_mask", "gaussian_noise", "brightness"]},
}

_STATISTICS = {"bootstrap_replicates": 4, "bootstrap_seed": 7}


def _synthetic_table(perturbation: str = "blur") -> FeatureTable:
    metadata: list[FeatureSample] = []
    clear: list[np.ndarray] = []
    degraded: list[np.ndarray] = []
    geometry: list[tuple[float, float, float, float]] = []
    for role in ("direction_fit", "confirmation"):
        for class_id in (0, 1):
            for object_index in range(4):
                base = np.asarray([2.0, 0.0, 0.25, 0.0] if class_id == 0 else [0.0, 2.0, 0.0, 0.25])
                delta = np.asarray([1.0, 0.0, 0.1, 0.0] if class_id == 0 else [0.0, 1.0, 0.0, 0.1])
                source_image_id = f"{role}-{class_id}-{object_index}"
                geom = (float(class_id), float(object_index), 0.0, 1.0)
                metadata.append(
                    FeatureSample(
                        dataset_id="synthetic",
                        object_id=f"o-{role}-{class_id}-{object_index}",
                        source_image_id=source_image_id,
                        class_id=class_id,
                        class_name=f"c{class_id}",
                        coarse_name=f"g{class_id}",
                        role=role,
                        perturbation=perturbation,
                        seed=object_index,
                        geometry=geom,
                    )
                )
                clear.append(base + delta)
                degraded.append(base)
                geometry.append(geom)
    arrays = {
        "true_clear_degraded": np.asarray(clear, dtype=np.float64),
        "true_degraded_degraded": np.asarray(degraded, dtype=np.float64),
        "geometry": np.asarray(geometry, dtype=np.float64),
    }
    return FeatureTable(model_id="synthetic", metadata=tuple(metadata), arrays=arrays, representation={})


class GridResolutionTests(unittest.TestCase):
    def test_p1_p2_cell_counts_and_overrides(self) -> None:
        cells = resolve_sweep_cells(_CONFIG, ["P1", "P2"])
        ids = [cell.variant_id for cell in cells]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(cells), 14)
        p1 = [cell for cell in cells if cell.sweep_level == "P1"]
        p2 = [cell for cell in cells if cell.sweep_level == "P2"]
        self.assertEqual(len(p1), 6)
        self.assertEqual(len(p2), 8)

        blur_low = next(cell for cell in cells if cell.variant_id == "blur_low")
        self.assertEqual(blur_low.overrides, {"blur": {"sigma_px": 8.0}})
        self.assertEqual(blur_low.strength_parameter, "sigma_px")
        self.assertEqual(blur_low.strength_value, 8.0)

        noise_high = next(cell for cell in cells if cell.variant_id == "gaussian_noise_high")
        self.assertEqual(noise_high.overrides, {"gaussian_noise": {"noise_sigma": 0.18}})
        self.assertEqual(noise_high.strength_parameter, "sigma")

        patch = next(cell for cell in cells if cell.variant_id == "patch_b020_s005")
        self.assertEqual(
            patch.overrides,
            {"patch_mask": {"patch_fraction": 0.20, "patch_side_fraction": 0.05}},
        )
        self.assertEqual(
            patch.strength_value,
            {"target_fraction": 0.20, "patch_side_fraction": 0.05},
        )
        # The nominal corner is not re-registered as a variant.
        self.assertNotIn("patch_b035_s010", ids)

    def test_p1_only_and_unknown_level(self) -> None:
        p1 = resolve_sweep_cells(_CONFIG, ["P1"])
        self.assertEqual(len(p1), 6)
        self.assertTrue(all(cell.perturbation != "patch_mask" for cell in p1))
        with self.assertRaises(StrengthSweepError):
            resolve_sweep_cells(_CONFIG, ["PX"])

    def test_nominal_cells(self) -> None:
        self.assertEqual(nominal_cell(_CONFIG, "blur").strength_value, 16.0)
        self.assertEqual(nominal_cell(_CONFIG, "gaussian_noise").strength_value, 0.12)
        self.assertEqual(nominal_cell(_CONFIG, "brightness").strength_value, 0.55)
        patch = nominal_cell(_CONFIG, "patch_mask")
        self.assertEqual(patch.strength_value, {"target_fraction": 0.35, "patch_side_fraction": 0.10})
        self.assertTrue(patch.is_nominal)


class CellRunnerTests(unittest.TestCase):
    def test_run_sweep_cell_tags_headline_rows(self) -> None:
        table = _synthetic_table("blur")
        cell = SweepCell(
            variant_id="blur_low",
            perturbation="blur",
            strength_variant="low",
            overrides={"blur": {"sigma_px": 8.0}},
            strength_parameter="sigma_px",
            strength_value=8.0,
            sweep_level="P1",
        )
        rows = run_sweep_cell(
            table,
            cell,
            dataset_ids=["synthetic"],
            epsilon=1.0e-8,
            fit_size=2,
            statistics_config=dict(_STATISTICS),
            random_seed=1729,
        )
        self.assertEqual(len(rows), len(SWEEP_METHODS))
        for row in rows:
            self.assertEqual(row["stage"], "E04-5")
            self.assertEqual(row["control"], "true_obb")
            self.assertEqual(row["normalization"], "unit")
            self.assertIn(row["method"], SWEEP_METHODS)
            self.assertEqual(row["variant_id"], "blur_low")
            self.assertEqual(row["strength_value"], 8.0)
            self.assertIn("source_metric_values", row)
            self.assertIn("own_argmax_rate", row)


class PersistenceTests(unittest.TestCase):
    def test_pair_table_roundtrip(self) -> None:
        table = _synthetic_table("blur")
        table.representation.update(
            {"variant_id": "blur_low", "strength_variant": "low", "perturbation": "blur", "model_id": "synthetic"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_pair_table(tmp, table)
            self.assertTrue(paths["npz"].is_file())
            loaded = load_pair_table(tmp, "synthetic", "blur_low", "blur")
        self.assertEqual(loaded.metadata, table.metadata)
        self.assertTrue(
            np.allclose(loaded.arrays["true_clear_degraded"], table.arrays["true_clear_degraded"])
        )
        self.assertTrue(
            np.allclose(loaded.arrays["true_degraded_degraded"], table.arrays["true_degraded_degraded"])
        )

    def test_distinct_variant_ids_do_not_collide(self) -> None:
        # Every P2 patch cell shares strength_variant="budget_side"; persistence
        # must key on the unique variant_id so the eight cells never overwrite
        # each other.
        with tempfile.TemporaryDirectory() as tmp:
            for variant_id in ("patch_b020_s005", "patch_b020_s010", "patch_b050_s015"):
                table = _synthetic_table("patch_mask")
                table.representation.update(
                    {
                        "variant_id": variant_id,
                        "strength_variant": "budget_side",
                        "perturbation": "patch_mask",
                        "model_id": "synthetic",
                    }
                )
                save_pair_table(tmp, table)
            for variant_id in ("patch_b020_s005", "patch_b020_s010", "patch_b050_s015"):
                loaded = load_pair_table(tmp, "synthetic", variant_id, "patch_mask")
                self.assertEqual(len(loaded.metadata), len(table.metadata))


class ExportGridTests(unittest.TestCase):
    def test_canonical_strength_key_matches_int_and_float(self) -> None:
        from cfra.paper_export_grid import _strength_key

        self.assertEqual(_strength_key(8), _strength_key(8.0))
        self.assertEqual(_strength_key(16.0), _strength_key(16))
        self.assertEqual(
            _strength_key({"target_fraction": 0.2, "patch_side_fraction": 0.05}),
            _strength_key({"patch_side_fraction": 0.05, "target_fraction": 0.2}),
        )

    def test_observed_m1_cell_is_not_reflagged_missing(self) -> None:
        from cfra.paper_export import PaperExport
        from cfra.paper_export_grid import expand_missing_grid

        with tempfile.TemporaryDirectory() as tmp:
            out = PaperExport([], Path(tmp) / "out")
            # A completed E04-5 M1 unit cell for one blur strength.
            for metric in ("own_argmax_rate", "mean_diagonal_margin", "macro_ovr_auc"):
                out.emit(
                    "strength_sweep",
                    {
                        "dataset": "dota",
                        "model": "clip",
                        "perturbation": "blur",
                        "split_seed": 1729,
                        "perturbation_strength": 8.0,
                        "operator": "euclidean_mean_subtraction",
                        "normalization": "unit",
                        "run_id": "one",
                        "stage": "E04-5",
                        "status": "observed",
                    },
                    metric,
                    0.7,
                )
            expand_missing_grid(out)
            observed = [
                row for row in out.tables["strength_sweep"]
                if row.get("status") != "missing"
                and row.get("perturbation") == "blur"
                and row.get("perturbation_strength") == 8.0
            ]
            missing = [
                row for row in out.tables["strength_sweep"]
                if row.get("status") == "missing"
                and row.get("perturbation") == "blur"
                and row.get("split_seed") == 1729
            ]
            self.assertEqual(len(observed), 3)
            # blur low=8 is observed; the 16/24 grid points remain missing.
            self.assertEqual(len(missing), 6)
            out.units.close()


class ExportContrastTests(unittest.TestCase):
    def _metrics(self, offset: float) -> dict:
        images = [f"img{i}" for i in range(6)]
        return {
            "source_image_id": images,
            "own_argmax_rate": [0.5 + offset for _ in images],
            "mean_diagonal_margin": [0.1 + offset for _ in images],
            "macro_ovr_auc": [0.6 + offset for _ in images],
        }

    def test_variant_vs_nominal_emits_paired_rows(self) -> None:
        from cfra.paper_export import PaperExport
        from cfra.paper_export_contrasts import apply_contrasts

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "E04-5_clip.json"
            artifact.write_text("{}", encoding="utf-8")
            out = PaperExport([], Path(tmp) / "out", bootstrap_replicates=8, permutation_replicates=19,
                              recompute_layers=False, recompute_final=False)
            out.config = {"split": {"seed": 1729}, "perturbations": {}}
            out.models = {}
            common = {
                "status": "completed",
                "dataset_id": "dota",
                "model_id": "clip",
                "perturbation": "blur",
                "method": "euclidean_mean_subtraction",
                "normalization": "unit",
                "control": "true_obb",
                "strength_value": 8.0,
            }
            rows = [
                dict(common, variant_id="nominal", perturbation_overrides={}, source_metric_values=self._metrics(0.0)),
                dict(common, variant_id="blur_low", perturbation_overrides={"blur": {"sigma_px": 8.0}},
                     source_metric_values=self._metrics(0.05)),
            ]
            apply_contrasts(out, "E04-5", artifact, rows)
            deltas = [row for row in out.tables["strength_sweep"] if str(row.get("metric", "")).startswith("delta_")]
            self.assertEqual(len(deltas), 3)
            for row in deltas:
                self.assertTrue(np.isfinite(row["estimate"]))
                self.assertEqual(row["state_a"], "nominal")
                self.assertEqual(row["state_b"], "blur_low")
                self.assertTrue(str(row["family"]).startswith("paired_strength:"))
            out.units.close()


if __name__ == "__main__":
    unittest.main()
