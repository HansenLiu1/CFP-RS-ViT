import unittest

import numpy as np

from cfra.data import ObjectRecord
from cfra.evaluation import intervene
from cfra.methods import fit_geometry_residualizer, fit_residual_operator
from cfra.protocols import (
    apply_perturbation,
    build_paired_inputs,
    crop_object,
    pair_pixel_invariants,
)
from cfra.representations import factorial_deltas, normalize_deltas, representation_delta


def _object(center_x: float = 8.0, center_y: float = 8.0) -> ObjectRecord:
    return ObjectRecord(
        dataset_id="synthetic",
        split="test",
        source_image_id="synthetic:test:0",
        content_group_id="content:synthetic",
        object_id="synthetic-object",
        image_relpath="image.png",
        label_relpath="label.txt",
        class_id=0,
        class_name="class",
        coarse_name="coarse",
        polygon_raw=(4.0, 4.0, 12.0, 4.0, 12.0, 12.0, 4.0, 12.0),
        polygon_visible=(4.0, 4.0, 12.0, 4.0, 12.0, 12.0, 4.0, 12.0),
        image_width=16,
        image_height=16,
        difficulty=None,
        raw_coordinate_out_of_bounds=False,
        visible_area_px2=64.0,
        visible_bbox_width_px=8.0,
        visible_bbox_height_px=8.0,
        visible_side_long_px=8.0,
        visible_side_short_px=8.0,
        center_x_px=center_x,
        center_y_px=center_y,
        orientation_rad=0.0,
        eligible=True,
    )


class FormulaTests(unittest.TestCase):
    def test_signed_delta_and_normalization(self) -> None:
        delta = representation_delta([[3.0, 4.0]], [[1.0, 1.0]])
        self.assertTrue(np.allclose(delta, [[2.0, 3.0]]))
        unit = normalize_deltas(delta, normalization="unit")
        self.assertTrue(np.allclose(unit.values, [[2.0 / np.sqrt(13.0), 3.0 / np.sqrt(13.0)]]))
        raw = normalize_deltas(delta, normalization="raw")
        self.assertTrue(np.allclose(raw.values, delta))

    def test_factorial_interaction_has_correct_sign(self) -> None:
        result = factorial_deltas(
            foreground_clear_background_clean=[[4.0, 0.0]],
            foreground_degraded_background_clean=[[1.0, 0.0]],
            foreground_clear_background_degraded=[[5.0, 0.0]],
            foreground_degraded_background_degraded=[[3.0, 0.0]],
        )
        self.assertTrue(np.allclose(result["delta_clean_background"], [[3.0, 0.0]]))
        self.assertTrue(np.allclose(result["delta_degraded_background"], [[2.0, 0.0]]))
        self.assertTrue(np.allclose(result["background_interaction"], [[1.0, 0.0]]))

    def test_alpha_one_removes_component_and_preserves_norm(self) -> None:
        features = np.asarray([[3.0, 4.0], [1.0, 2.0]])
        directions = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        transformed = intervene(features, directions, alpha=1.0, preserve_norm=True)
        self.assertTrue(np.allclose(transformed[:, 0] * directions[:, 0] + transformed[:, 1] * directions[:, 1], [0.0, 0.0]))
        self.assertTrue(np.allclose(np.linalg.norm(transformed, axis=1), np.linalg.norm(features, axis=1)))
        reflected = intervene(features[:1], directions[:1], alpha=2.0, preserve_norm=False)
        self.assertTrue(np.allclose(reflected, [[-3.0, 4.0]]))

    def test_operator_transform_is_symmetric_for_fit_and_evaluation(self) -> None:
        fit = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
        operator = fit_residual_operator("orthogonal_axis_projection", fit)
        transformed = operator.transform(fit)
        axis = fit.mean(axis=0) / np.linalg.norm(fit.mean(axis=0))
        self.assertTrue(np.allclose(transformed @ axis, 0.0, atol=1.0e-10))

    def test_geometry_residualizer_removes_registered_covariate(self) -> None:
        geometry = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        values = np.column_stack([2.0 * geometry[:, 0] + 1.0, np.ones(4)])
        residualizer = fit_geometry_residualizer(values, geometry)
        residuals = residualizer.transform(values, geometry)
        self.assertTrue(np.allclose(residuals[:, 0], 0.0, atol=1.0e-5))


class ProtocolFormulaTests(unittest.TestCase):
    def test_reflect_crop_and_pair_pixel_invariants(self) -> None:
        image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        crop = crop_object(image, _object(center_x=1.0, center_y=1.0), size_px=8, mask_supersampling=4)
        perturbation = apply_perturbation(image=crop.image, kind="blur", rng=np.random.default_rng(1), sigma_px=1.0)
        pair = build_paired_inputs(crop, perturbation)
        invariants = pair_pixel_invariants(pair, tolerance=1.0)
        self.assertTrue(invariants["outside_mask_equal_clear_vs_degraded"])
        self.assertTrue(invariants["blend_formula_exact"])


if __name__ == "__main__":
    unittest.main()
