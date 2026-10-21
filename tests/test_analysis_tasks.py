import tempfile
import unittest
from pathlib import Path

import numpy as np

from cfra.analysis.cli import build_parser
from cfra.analysis.common import RunLayout
from cfra.analysis.matrix import class_unit_mean, project_out, same_diff, stable_seed
from cfra.analysis.tasks import export_stats, geometry, spectra
from cfra.analysis.tasks.layerwise import _stats


def _synthetic_run(root: Path) -> RunLayout:
    features = root / "run1" / "intermediates" / "features" / "features_clip.npz"
    features.parent.mkdir(parents=True)
    rng = np.random.default_rng(7)
    classes = 16
    objects_per_class = 2
    rows = classes * objects_per_class * 4
    class_ids = np.repeat(np.arange(classes, dtype=np.int64), objects_per_class * 4)
    roles = np.tile(
        np.asarray(["direction_fit", "readout_fit", "confirmation", "validation"]),
        classes * objects_per_class,
    )
    object_ids = np.asarray(
        [
            f"dota:Train:P0001:{class_id:04d}:{index:04d}"
            for class_id in range(classes)
            for index in range(objects_per_class)
            for _ in range(4)
        ]
    )
    image_ids = np.asarray([f"dota:Train:P0001:{class_id:04d}" for class_id in range(classes) for _ in range(objects_per_class * 4)])
    dim = 512
    raw = rng.normal(size=(rows, dim)).astype(np.float32)
    clear_degraded = rng.normal(size=(rows, dim)).astype(np.float32)
    degraded_degraded = rng.normal(size=(rows, dim)).astype(np.float32)
    clear_clean = (clear_degraded + rng.normal(scale=0.1, size=(rows, dim))).astype(np.float32)
    degraded_clean = (degraded_degraded + rng.normal(scale=0.1, size=(rows, dim))).astype(np.float32)
    geometry = rng.normal(size=(rows, 4)).astype(np.float64)
    geometry[:, 0] = np.log1p(100.0)
    payload = {
        "raw": raw,
        "true_clear_degraded": clear_degraded,
        "true_degraded_degraded": degraded_degraded,
        "true_clear_clean": clear_clean,
        "true_degraded_clean": degraded_clean,
        "background_clear": rng.normal(size=(rows, dim)).astype(np.float32),
        "background_degraded": rng.normal(size=(rows, dim)).astype(np.float32),
        "geometry": geometry,
        "background_valid": np.ones(rows, dtype=bool),
        "dataset_id": np.asarray(["dota"] * rows),
        "object_id": object_ids,
        "source_image_id": image_ids,
        "class_id": class_ids,
        "class_name": np.asarray([f"class-{class_id}" for class_id in class_ids]),
        "coarse_name": np.asarray([f"coarse-{class_id % 5}" for class_id in class_ids]),
        "role": roles,
        "perturbation": np.asarray(["blur"] * rows),
        "excluded_from_main": np.zeros(rows, dtype=bool),
    }
    for control in (
        "shifted_obb",
        "random_geometry_matched_obb",
        "background_only",
        "boundary_only",
        "interior_eroded",
        "mask_eroded",
        "mask_dilated",
    ):
        payload[f"control_valid_{control}"] = np.ones(rows, dtype=bool)
    np.savez_compressed(features, **payload)
    return RunLayout(root, {1729: "run1"})


class AnalysisTaskTests(unittest.TestCase):
    def test_cli_exposes_every_paper_task(self) -> None:
        parser = build_parser()
        actions = [action for action in parser._actions if getattr(action, "choices", None)]
        commands = set(actions[0].choices)
        self.assertTrue(
            {
                "spectra",
                "geometry",
                "subspace-alignment",
                "readout-removal",
                "readout-transfer",
                "normalization",
                "export-stats",
                "layerwise",
                "intervals",
                "layerwise-fdbc",
                "text-baseline",
                "text-anchoring",
            }.issubset(commands)
        )

    def test_isotropic_null_is_deterministic(self) -> None:
        first = spectra.isotropic_null(4, 16, reps=3, seed=0)
        second = spectra.isotropic_null(4, 16, reps=3, seed=0)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["iso_top1"], 0.0)
        self.assertLessEqual(first["iso_top1"], 1.0)

    def test_rank_stats_ties_do_not_push_the_own_class_down(self) -> None:
        matrix = np.asarray([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
        stats = export_stats.rank_stats(matrix)
        self.assertEqual(stats["r1"], 1.0)
        self.assertEqual(stats["r2"], 1.0)
        self.assertEqual(stats["r4"], 0.0)
        self.assertAlmostEqual(stats["margin"], 0.5)

    def test_matrix_helpers_preserve_norm_and_class_order(self) -> None:
        values = np.asarray([[3.0, 4.0], [1.0, 1.0]])
        basis = np.asarray([[1.0, 0.0]])
        projected = project_out(values, basis)
        np.testing.assert_allclose(np.linalg.norm(projected, axis=1), np.linalg.norm(values, axis=1))
        classes = np.asarray([2, 5])
        means = class_unit_mean(np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.asarray([5, 2, 5]), classes)
        np.testing.assert_allclose(means[0], [0.0, 1.0])
        same, different = same_diff(np.eye(3))
        self.assertEqual(same, 1.0)
        self.assertEqual(different, 0.0)
        self.assertEqual(stable_seed("x", 1), stable_seed("x", 1))

    def test_layerwise_class_balanced_unit_is_normalized(self) -> None:
        rng = np.random.default_rng(3)
        labels = np.repeat(np.arange(4), 4)
        values = rng.normal(size=(16, 8))
        result = _stats(values, labels, "class_balanced_unit")
        self.assertGreaterEqual(result["A_share"], -1.0)
        self.assertLessEqual(result["A_share"], 1.0)
        self.assertGreater(result["classres_over_A"], 0.0)

    def test_spectra_and_geometry_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = _synthetic_run(root / "runs")
            spectra.run(
                layout,
                root / "spectra",
                models=("clip",),
                seeds=(1729,),
                datasets=("dota",),
                perturbations=("blur",),
                max_cells=1,
                null_reps=2,
                matched16_draws=0,
            )
            self.assertTrue((root / "spectra" / "spectra_per_seed.csv").is_file())
            self.assertTrue((root / "spectra" / "removal_per_seed.csv").is_file())
            spectra_frame = np.genfromtxt(
                root / "spectra" / "spectra_per_seed.csv", delimiter=",", names=True, dtype=None, encoding="utf-8"
            )
            self.assertGreater(len(spectra_frame), 0)
            geometry.run(
                layout,
                root / "geometry",
                models=("clip",),
                seeds=(1729,),
                datasets=("dota",),
                perturbations=("blur",),
                max_cells=1,
                permutations=3,
            )
            self.assertTrue((root / "geometry" / "geometry_per_seed.csv").is_file())


if __name__ == "__main__":
    unittest.main()
