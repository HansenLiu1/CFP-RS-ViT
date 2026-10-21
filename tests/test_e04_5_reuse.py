"""E04-5 reuse-path equivalence tests.

The optimized E04-5 path must be bit-identical to the legacy full-grid path:
cells whose perturbation parameters are unchanged copy the nominal rows
(features, seeds and statistics settings are identical), while changed cells
run the exact same ``run_e04`` code on the exact same tables.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np

from cfra.pipeline import FeatureSample, FeatureTable, run_e04, run_e04_5
from cfra.pipeline import core as core_module

_RAW_KEYS = (
    "raw",
    "true_clear_degraded",
    "true_degraded_degraded",
    "true_clear_clean",
    "true_degraded_clean",
    "background_clear",
    "background_degraded",
)


def _build_table(
    perturbations: tuple[str, ...],
    *,
    shifts: dict[str, float] | None = None,
    perturbation_parameters: dict[str, dict[str, float]],
    overrides: dict[str, object] | None = None,
) -> FeatureTable:
    shifts = shifts or {}
    metadata: list[FeatureSample] = []
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in _RAW_KEYS}
    for perturbation_index, perturbation in enumerate(perturbations):
        shift = float(shifts.get(perturbation, 0.0))
        for role_index, role in enumerate(("direction_fit", "readout_fit", "confirmation")):
            for class_id in (0, 1):
                for object_index in range(2):
                    source_image_id = f"{role}-{perturbation}-{class_id}-{object_index}"
                    object_id = f"object-{role}-{perturbation}-{class_id}-{object_index}"
                    class_vector = np.asarray([2.0, 0.0, 0.25, 0.0] if class_id == 0 else [0.0, 2.0, 0.0, 0.25])
                    perturbation_vector = np.asarray([0.0, 0.0, 0.1 * perturbation_index, 0.0])
                    raw = class_vector + perturbation_vector + shift
                    delta = np.asarray([1.0, 0.0, 0.1, 0.0] if class_id == 0 else [0.0, 1.0, 0.0, 0.1])
                    metadata.append(
                        FeatureSample(
                            dataset_id="synthetic",
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
    representation = {
        "representation_id": "synthetic",
        "dimension": 4,
        "perturbation_parameters": {key: dict(value) for key, value in perturbation_parameters.items()},
        "perturbation_overrides": dict(overrides) if overrides else {},
        "perturbation_audit_rows": [{"index": index} for index in range(len(metadata))],
    }
    return FeatureTable(
        model_id="synthetic",
        metadata=tuple(metadata),
        arrays={key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()},
        representation=representation,
    )


def _legacy_run_e04_5(tables: dict[str, FeatureTable], **kwargs: object) -> list[dict[str, object]]:
    """Independent copy of the pre-optimization E04-5 loop (full grid per variant)."""

    rows: list[dict[str, object]] = []
    for strength_variant, table in tables.items():
        strength_rows = run_e04(table, **kwargs)  # type: ignore[arg-type]
        overrides = table.representation.get("perturbation_overrides", {})
        perturbation_parameters = table.representation.get("perturbation_parameters", {})
        sensitivity_range: dict[str, object] = {
            "strength_variant": strength_variant,
            "perturbation_overrides": overrides,
            "perturbation_parameters": perturbation_parameters,
            "perturbations": list(kwargs["perturbations"]),  # type: ignore[arg-type]
            "method_ids": list(kwargs["method_ids"]),  # type: ignore[arg-type]
            "normalization_variants": list(kwargs["normalization_variants"]),  # type: ignore[arg-type]
        }
        if isinstance(overrides, dict) and "patch_mask" in overrides:
            patch_values = overrides["patch_mask"]
            if isinstance(patch_values, dict):
                sensitivity_range["patch_budget"] = patch_values.get(
                    "patch_fraction", patch_values.get("target_fraction")
                )
                sensitivity_range["patch_side_fraction"] = patch_values.get("patch_side_fraction")
        for row in strength_rows:
            row["stage"] = "E04-5"
            row["strength_variant"] = strength_variant
            row["sensitivity_range"] = sensitivity_range
            row["sensitivity_grid_complete"] = True
        rows.extend(strength_rows)
    return rows


def _canonical(rows: object) -> str:
    return json.dumps(core_module._json_safe(rows), sort_keys=True, ensure_ascii=True)


_CONTROLS_OFF = {
    name: False
    for name in (
        "shifted_obb",
        "random_geometry_matched_obb",
        "label_permutation",
        "background_only",
        "boundary_only",
        "interior_eroded",
        "mask_dilation_erosion",
        "geometry_regression",
    )
}

_STATISTICS = {"bootstrap_replicates": 4, "bootstrap_seed": 19}


class E04_5ReuseEquivalenceTests(unittest.TestCase):
    def _suite(self) -> tuple[dict[str, FeatureTable], dict[str, object]]:
        nominal_parameters = {
            "blur": {"sigma_px": 16.0},
            "patch_mask": {"patch_fraction": 0.35, "patch_side_fraction": 0.10},
            "brightness": {"brightness_factor": 0.55},
        }
        grid_parameters = {
            "blur": {"sigma_px": 16.0},
            "patch_mask": {"patch_fraction": 0.20, "patch_side_fraction": 0.05},
            "brightness": {"brightness_factor": 0.55},
        }
        high_parameters = {
            "blur": {"sigma_px": 24.0},
            "patch_mask": {"patch_fraction": 0.50, "patch_side_fraction": 0.10},
            "brightness": {"brightness_factor": 0.35},
        }
        perturbations = ("blur", "patch_mask", "brightness")
        tables = {
            "nominal": _build_table(perturbations, perturbation_parameters=nominal_parameters),
            "patch_grid": _build_table(
                perturbations,
                shifts={"patch_mask": 0.5},
                perturbation_parameters=grid_parameters,
                overrides={"patch_mask": {"patch_fraction": 0.20, "patch_side_fraction": 0.05}},
            ),
            "high": _build_table(
                perturbations,
                shifts={perturbation: 2.0 for perturbation in perturbations},
                perturbation_parameters=high_parameters,
            ),
        }
        kwargs: dict[str, object] = {
            "dataset_ids": ["synthetic"],
            "perturbations": list(perturbations),
            "epsilon": 1.0e-8,
            "method_ids": ["raw"],
            "normalization_variants": ["unit"],
            "controls": dict(_CONTROLS_OFF),
            "statistics_config": dict(_STATISTICS),
        }
        return tables, kwargs

    def test_reuse_path_is_bit_identical_to_legacy_full_grid(self) -> None:
        tables, kwargs = self._suite()
        legacy = _legacy_run_e04_5(dict(tables), **kwargs)

        with mock.patch.object(core_module, "run_e04", wraps=core_module.run_e04) as mocked:
            optimized = run_e04_5(dict(tables), **kwargs)  # type: ignore[arg-type]

        calls = [call.kwargs.get("perturbations") for call in mocked.call_args_list]
        self.assertEqual(
            calls,
            [["blur", "patch_mask", "brightness"], ["patch_mask"], ["blur", "patch_mask", "brightness"]],
            "patch grid variant must only recompute the changed perturbation cells",
        )
        self.assertEqual(_canonical(optimized), _canonical(legacy))

    def test_streamed_variant_tables_match_mapping_tables(self) -> None:
        tables, kwargs = self._suite()
        streamed = run_e04_5(iter(tables.items()), **kwargs)  # type: ignore[arg-type]
        mapped = run_e04_5(dict(tables), **kwargs)  # type: ignore[arg-type]
        self.assertEqual(_canonical(streamed), _canonical(mapped))

    def test_fully_unchanged_variant_reuses_every_cell(self) -> None:
        tables, kwargs = self._suite()
        tables["copy"] = tables["nominal"]
        with mock.patch.object(core_module, "run_e04", wraps=core_module.run_e04) as mocked:
            rows = run_e04_5(dict(tables), **kwargs)  # type: ignore[arg-type]
        calls = [call.kwargs.get("perturbations") for call in mocked.call_args_list]
        self.assertEqual(len(calls), 3, "unchanged variant must not run the grid at all")
        copied = [row for row in rows if row["strength_variant"] == "copy"]
        nominal = [row for row in rows if row["strength_variant"] == "nominal"]
        self.assertEqual(len(copied), len(nominal))

        def _strip_variant(row: dict[str, object]) -> dict[str, object]:
            cleaned = dict(core_module._json_safe(row))
            cleaned["strength_variant"] = None
            sensitivity_range = dict(cleaned.get("sensitivity_range", {}))
            sensitivity_range["strength_variant"] = None
            cleaned["sensitivity_range"] = sensitivity_range
            return cleaned

        self.assertEqual([_strip_variant(row) for row in copied], [_strip_variant(row) for row in nominal])


if __name__ == "__main__":
    unittest.main()
