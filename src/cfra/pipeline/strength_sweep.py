"""OFAT perturbation-strength sweep (E04-5) on an existing nominal run.

The registered E04-5 grid re-extracts every enabled perturbation for every
variant and evaluates the full operator/normalization/control grid.  This
module implements a much cheaper but scientifically equivalent sweep for the
perturbation-strength robustness question:

* one factor at a time: a variant changes exactly one perturbation and only
  that perturbation is extracted (the others are unchanged by construction);
* only the foreground pair (``true_clear_degraded`` / ``true_degraded_degraded``)
  is encoded, because the headline operators M0/M1/M3 never read the control or
  background crops (``cfra.methods.operators``);
* only the three headline operators at unit normalization are evaluated, with
  every spatial control disabled, so the only persisted cell is the true-OBB one.

The paired design is preserved: sample seeds are
``_stable_seed(split_seed, object_id, perturbation)`` and therefore do not depend
on the strength parameters, so a nominal and a variant cell share the same
objects, the same split roles and the same RNG stream.  Pairing variant cells
against the run's nominal table is thus object- and source-image-disjoint-exact.

The module never builds a whole feature table from raw images for the nominal
side: it reads only the two pair arrays it needs out of the persisted nominal
``features_<model>.npz``.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cfra.models import load_adapter
from cfra.pipeline.core import run_e04
from cfra.pipeline.features import (
    FeatureSample,
    FeatureTable,
    _nominal_parameters,
    _read_excluded_classes,
    make_feature_samples,
)
from cfra.pipeline.features_parallel import iter_context_pair_batches_parallel

logger = logging.getLogger("cfra")

SWEEP_STAGE = "E04-5"

# Manuscript headline operators = M0 / M1 / M3.
SWEEP_METHODS: tuple[str, ...] = (
    "raw",
    "euclidean_mean_subtraction",
    "orthogonal_axis_projection",
)
SWEEP_NORMALIZATIONS: tuple[str, ...] = ("unit",)

# Registered scalar sweep axes: perturbation -> (feature override key, export parameter name).
_SCALAR_AXES: tuple[tuple[str, str, str], ...] = (
    ("blur", "sigma_px", "sigma_px"),
    ("gaussian_noise", "noise_sigma", "sigma"),
    ("brightness", "brightness_factor", "factor"),
)

_PAIR_KEYS: tuple[str, ...] = ("true_clear_degraded", "true_degraded_degraded")

_SAMPLE_ARRAY_FIELDS: tuple[tuple[str, str], ...] = (
    ("object_id", "U"),
    ("source_image_id", "U"),
    ("dataset_id", "U"),
    ("class_id", "int64"),
    ("class_name", "U"),
    ("coarse_name", "U"),
    ("role", "U"),
    ("perturbation", "U"),
    ("seed", "uint64"),
    ("strength_variant", "U"),
    ("excluded_from_main", "bool"),
)


class StrengthSweepError(RuntimeError):
    """Raised when the strength sweep cannot build a consistent cell."""


@dataclass(frozen=True)
class SweepCell:
    """One registered strength cell for a single perturbation."""

    variant_id: str
    perturbation: str
    strength_variant: str
    overrides: dict[str, Any] | None
    strength_parameter: str
    strength_value: Any
    sweep_level: str
    is_nominal: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "perturbation": self.perturbation,
            "strength_variant": self.strength_variant,
            "overrides": self.overrides,
            "strength_parameter": self.strength_parameter,
            "strength_value": self.strength_value,
            "sweep_level": self.sweep_level,
            "is_nominal": self.is_nominal,
        }


def _short_number(value: float) -> str:
    scaled = round(float(value) * 100)
    return f"{scaled:03d}"


def _scalar_cells(config: Mapping[str, Any], perturbation: str, feature_key: str, parameter: str) -> list[SweepCell]:
    table = config.get("perturbation_strengths", {}).get(perturbation, {})
    if "low" not in table or "high" not in table:
        raise StrengthSweepError(f"perturbation_strengths.{perturbation} must define low and high")
    cells: list[SweepCell] = []
    for level in ("low", "high"):
        value = float(table[level])
        cells.append(
            SweepCell(
                variant_id=f"{perturbation}_{level}",
                perturbation=perturbation,
                strength_variant=level,
                overrides={perturbation: {feature_key: value}},
                strength_parameter=parameter,
                strength_value=value,
                sweep_level="P1",
            )
        )
    return cells


def _patch_cells(config: Mapping[str, Any]) -> list[SweepCell]:
    protocol = config.get("protocol", {}).get("e04_5", {})
    if not isinstance(protocol, Mapping):
        protocol = {}
    budgets = protocol.get("patch_budget_grid", [0.20, 0.35, 0.50])
    sides = protocol.get("patch_side_grid", [0.05, 0.10, 0.15])
    if not isinstance(budgets, Sequence) or not isinstance(sides, Sequence) or not budgets or not sides:
        raise StrengthSweepError("protocol.e04_5 patch grid must be non-empty")
    nominal = config["perturbations"]["patch_mask"]
    nominal_budget = float(nominal["target_fraction"])
    nominal_side = float(nominal["patch_side_fraction"])
    cells: list[SweepCell] = []
    for budget in budgets:
        for side in sides:
            budget = float(budget)
            side = float(side)
            if budget == nominal_budget and side == nominal_side:
                continue
            cells.append(
                SweepCell(
                    variant_id=f"patch_b{_short_number(budget)}_s{_short_number(side)}",
                    perturbation="patch_mask",
                    strength_variant="budget_side",
                    overrides={"patch_mask": {"patch_fraction": budget, "patch_side_fraction": side}},
                    strength_parameter="budget_side",
                    strength_value={"target_fraction": budget, "patch_side_fraction": side},
                    sweep_level="P2",
                )
            )
    return cells


def resolve_sweep_cells(config: Mapping[str, Any], levels: Sequence[str] = ("P1", "P2")) -> list[SweepCell]:
    """Resolve the registered sweep cells for the requested levels.

    ``P1`` is the one-factor-at-a-time low/high sweep of blur, Gaussian noise and
    brightness.  ``P2`` is the patch-mask budget x side grid (the nominal corner
    is dropped because the nominal table already covers it).
    """

    requested = {str(level) for level in levels}
    unknown = requested - {"P1", "P2"}
    if unknown:
        raise StrengthSweepError(f"unknown sweep levels: {sorted(unknown)}")
    cells: list[SweepCell] = []
    if "P1" in requested:
        for perturbation, feature_key, parameter in _SCALAR_AXES:
            cells.extend(_scalar_cells(config, perturbation, feature_key, parameter))
    if "P2" in requested:
        cells.extend(_patch_cells(config))
    seen: dict[str, SweepCell] = {}
    for cell in cells:
        if cell.variant_id in seen:
            raise StrengthSweepError(f"duplicate sweep variant id: {cell.variant_id}")
        seen[cell.variant_id] = cell
    return cells


def nominal_cell(config: Mapping[str, Any], perturbation: str) -> SweepCell:
    """Return the nominal (unswept) reference cell for one perturbation."""

    if perturbation == "patch_mask":
        patch = config["perturbations"]["patch_mask"]
        value: Any = {
            "target_fraction": float(patch["target_fraction"]),
            "patch_side_fraction": float(patch["patch_side_fraction"]),
        }
        parameter = "budget_side"
    elif perturbation == "blur":
        value = float(config["perturbations"]["blur"]["sigma_px"])
        parameter = "sigma_px"
    elif perturbation == "gaussian_noise":
        value = float(config["perturbations"]["gaussian_noise"]["sigma"])
        parameter = "sigma"
    elif perturbation == "brightness":
        value = float(config["perturbations"]["brightness"]["factor"])
        parameter = "factor"
    else:
        raise StrengthSweepError(f"unknown perturbation: {perturbation}")
    return SweepCell(
        variant_id="nominal",
        perturbation=perturbation,
        strength_variant="nominal",
        overrides=None,
        strength_parameter=parameter,
        strength_value=value,
        sweep_level="P0",
        is_nominal=True,
    )


def _config_for_perturbation(config: Mapping[str, Any], perturbation: str) -> dict[str, Any]:
    """Deep copy with the scope restricted to a single perturbation.

    The sample seed depends only on ``(split.seed, object_id, perturbation)``, so
    restricting the scope here does not change any sample's seed, and the
    extracted pair arrays are bit-identical to the nominal table's rows for the
    same perturbation.
    """

    scoped = copy.deepcopy(dict(config))
    scoped.setdefault("scope", {})["perturbations"] = [perturbation]
    return scoped


def _numpy_encoded(encoded: Any) -> np.ndarray:
    values = encoded.values
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def _extract_pair_table(
    adapter: Any,
    *,
    samples: tuple[FeatureSample, ...],
    scoped: Mapping[str, Any],
    manifests: Mapping[str, Any],
    object_index: Mapping[str, Any],
    model_id: str,
    asset: Any,
    runtime: Any,
    perturbation: str,
    strength_variant: str,
    variant_id: str | None,
    perturbation_overrides: dict[str, Any] | None,
    cpu_workers: int | None,
) -> FeatureTable:
    object_index = dict(object_index)
    clear_batches: list[np.ndarray] = []
    degraded_batches: list[np.ndarray] = []
    started = time.time()
    batches = iter_context_pair_batches_parallel(
        samples,
        dict(manifests),
        object_index,
        dict(scoped),
        batch_size=int(runtime.batch_size),
        cpu_workers=cpu_workers,
        perturbation_overrides=perturbation_overrides,
    )
    for batch in batches:
        if not batch:
            continue
        clear = adapter.encode([context["true_clear_degraded"] for context in batch])
        degraded = adapter.encode([context["true_degraded_degraded"] for context in batch])
        clear_batches.append(_numpy_encoded(clear))
        degraded_batches.append(_numpy_encoded(degraded))
    if not clear_batches:
        raise StrengthSweepError("pair extraction produced no batches")
    arrays = {
        "true_clear_degraded": np.concatenate(clear_batches, axis=0),
        "true_degraded_degraded": np.concatenate(degraded_batches, axis=0),
        "geometry": np.asarray([sample.geometry for sample in samples], dtype=np.float64),
        "background_valid": np.ones(len(samples), dtype=bool),
    }
    representation = {
        "model_id": model_id,
        "representation_id": adapter.spec.representation_id,
        "dimension": int(adapter.spec.dimension),
        "pooling": adapter.spec.pooling,
        "projection": adapter.spec.projection,
        "text_shared_space": adapter.spec.text_shared_space,
        "shared_space_id": adapter.spec.shared_space_id,
        "checkpoint_revision": asset.revision,
        "checkpoint_sha256": asset.checkpoint_sha256,
        "perturbation": perturbation,
        "strength_variant": strength_variant,
        "variant_id": variant_id or strength_variant,
        "perturbation_overrides": dict(perturbation_overrides) if perturbation_overrides else {},
        "perturbation_parameters": {
            perturbation: _nominal_parameters(dict(scoped), perturbation, strength_variant, perturbation_overrides)
        },
        "sample_count": len(samples),
        "input_keys": list(_PAIR_KEYS),
        "extract_seconds": round(time.time() - started, 3),
    }
    return FeatureTable(model_id=model_id, metadata=samples, arrays=arrays, representation=representation)


def scoped_samples(
    config: Mapping[str, Any],
    selected_by_dataset: Mapping[str, Any],
    run_root: Path | str,
    perturbation: str,
    strength_variant: str,
) -> tuple[dict[str, Any], tuple[FeatureSample, ...]]:
    """Build the single-perturbation sample list and its scoped config."""

    scoped = _config_for_perturbation(config, perturbation)
    excluded_from_main = _read_excluded_classes(run_root)
    samples = make_feature_samples(
        dict(selected_by_dataset),
        scoped,
        strength_variant=strength_variant,
        excluded_from_main=excluded_from_main,
    )
    if not samples:
        raise StrengthSweepError(f"no samples for perturbation {perturbation!r}")
    expected = {sample.perturbation for sample in samples}
    if expected != {perturbation}:
        raise StrengthSweepError(f"scoped extraction produced perturbations {sorted(expected)}")
    return scoped, samples


def extract_pair_table(
    *,
    project_root: Path | str,
    run_root: Path | str,
    config: Mapping[str, Any],
    manifests: Mapping[str, Any],
    selected_by_dataset: Mapping[str, Any],
    model_id: str,
    asset: Any,
    runtime: Any,
    perturbation: str,
    strength_variant: str,
    variant_id: str | None = None,
    perturbation_overrides: dict[str, Any] | None,
    cpu_workers: int | None = None,
    adapter: Any | None = None,
) -> FeatureTable:
    """Extract only the foreground pair for one perturbation at one strength.

    When ``adapter`` is supplied the caller owns the model lifetime, so a driver
    can extract every variant of a model with a single load.
    """

    scoped, samples = scoped_samples(config, selected_by_dataset, run_root, perturbation, strength_variant)
    object_index = {item.object_id: item for manifest in manifests.values() for item in manifest.objects}
    if adapter is not None:
        return _extract_pair_table(
            adapter,
            samples=samples,
            scoped=scoped,
            manifests=manifests,
            object_index=object_index,
            model_id=model_id,
            asset=asset,
            runtime=runtime,
            perturbation=perturbation,
            strength_variant=strength_variant,
            variant_id=variant_id,
            perturbation_overrides=perturbation_overrides,
            cpu_workers=cpu_workers,
        )
    with load_adapter(
        model_id,
        asset,
        runtime=runtime,
        registry_root=Path(project_root).resolve(strict=True) / "registry",
        device="auto",
    ) as owned:
        return _extract_pair_table(
            owned,
            samples=samples,
            scoped=scoped,
            manifests=manifests,
            object_index=object_index,
            model_id=model_id,
            asset=asset,
            runtime=runtime,
            perturbation=perturbation,
            strength_variant=strength_variant,
            variant_id=variant_id,
            perturbation_overrides=perturbation_overrides,
            cpu_workers=cpu_workers,
        )


def save_pair_table(run_root: Path | str, table: FeatureTable) -> dict[str, Path]:
    """Persist a pair-only feature table below ``intermediates/features/strength_sweep``."""

    representation = table.representation
    variant_id = str(representation.get("variant_id") or representation["strength_variant"])
    perturbation = str(representation["perturbation"])
    model_id = table.model_id
    out = Path(run_root).resolve(strict=True) / "intermediates/features/strength_sweep" / model_id
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{variant_id}__{perturbation}"
    payload: dict[str, Any] = {key: np.asarray(table.arrays[key]) for key in _PAIR_KEYS}
    payload["geometry"] = np.asarray(table.arrays["geometry"], dtype=np.float64)
    payload["background_valid"] = np.asarray(
        table.arrays.get("background_valid", np.ones(len(table.metadata), dtype=bool)), dtype=bool
    )
    for field, dtype in _SAMPLE_ARRAY_FIELDS:
        payload[field] = np.asarray([getattr(sample, field) for sample in table.metadata], dtype=dtype)
    npz_path = base.with_suffix(".npz")
    json_path = base.with_suffix(".json")
    np.savez_compressed(npz_path, **payload)
    json_path.write_text(
        json.dumps(representation, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return {"npz": npz_path, "json": json_path}


def load_pair_table(run_root: Path | str, model_id: str, variant_id: str, perturbation: str) -> FeatureTable:
    root = Path(run_root).resolve(strict=True) / "intermediates/features/strength_sweep" / model_id
    base = root / f"{variant_id}__{perturbation}"
    payload = np.load(base.with_suffix(".npz"), allow_pickle=False)
    representation = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    samples = tuple(
        FeatureSample(
            dataset_id=str(dataset_id),
            object_id=str(object_id),
            source_image_id=str(source_image_id),
            class_id=int(class_id),
            class_name=str(class_name),
            coarse_name=str(coarse_name),
            role=str(role),
            perturbation=str(pert_),
            seed=int(seed),
            geometry=tuple(float(value) for value in geometry),
            strength_variant=str(strength_variant),
            excluded_from_main=bool(excluded),
        )
        for (
            object_id,
            source_image_id,
            dataset_id,
            class_id,
            class_name,
            coarse_name,
            role,
            pert_,
            seed,
            geometry,
            strength_variant,
            excluded,
        ) in zip(
            payload["object_id"],
            payload["source_image_id"],
            payload["dataset_id"],
            payload["class_id"],
            payload["class_name"],
            payload["coarse_name"],
            payload["role"],
            payload["perturbation"],
            payload["seed"],
            payload["geometry"],
            payload["strength_variant"],
            payload["excluded_from_main"],
        )
    )
    arrays = {key: np.asarray(payload[key]) for key in (*_PAIR_KEYS, "geometry", "background_valid")}
    return FeatureTable(model_id=model_id, metadata=samples, arrays=arrays, representation=representation)


def load_nominal_pair_subset(run_root: Path | str, model_id: str, perturbation: str) -> FeatureTable:
    """Load only the pair arrays of the nominal table for one perturbation."""

    root = Path(run_root).resolve(strict=True) / "intermediates/features"
    payload = np.load(root / f"features_{model_id}.npz", allow_pickle=False)
    metadata = json.loads((root / f"features_{model_id}.json").read_text(encoding="utf-8"))
    samples = tuple(
        FeatureSample(
            dataset_id=str(dataset_id),
            object_id=str(object_id),
            source_image_id=str(source_image_id),
            class_id=int(class_id),
            class_name=str(class_name),
            coarse_name=str(coarse_name),
            role=str(role),
            perturbation=str(pert_),
            seed=int(seed),
            geometry=tuple(float(value) for value in geometry),
            strength_variant=str(strength_variant),
            excluded_from_main=bool(excluded),
        )
        for (
            dataset_id,
            object_id,
            source_image_id,
            class_id,
            class_name,
            coarse_name,
            role,
            pert_,
            seed,
            geometry,
            strength_variant,
            excluded,
        ) in zip(
            payload["dataset_id"],
            payload["object_id"],
            payload["source_image_id"],
            payload["class_id"],
            payload["class_name"],
            payload["coarse_name"],
            payload["role"],
            payload["perturbation"],
            payload["seed"],
            payload["geometry"],
            payload["strength_variant"],
            payload["excluded_from_main"] if "excluded_from_main" in payload else [False] * len(payload["object_id"]),
        )
    )
    keep = np.asarray([sample.perturbation == perturbation for sample in samples], dtype=bool)
    if not keep.any():
        raise StrengthSweepError(f"nominal table has no rows for perturbation {perturbation!r}")
    arrays = {key: np.asarray(payload[key])[keep] for key in _PAIR_KEYS}
    arrays["geometry"] = np.asarray(payload["geometry"], dtype=np.float64)[keep]
    arrays["background_valid"] = np.asarray(payload["background_valid"], dtype=np.bool_)[keep]
    selection = tuple(sample for sample, include in zip(samples, keep.tolist()) if include)
    representation = dict(metadata)
    representation["subset_perturbation"] = perturbation
    return FeatureTable(model_id=model_id, metadata=selection, arrays=arrays, representation=representation)


def run_sweep_cell(
    table: FeatureTable,
    cell: SweepCell,
    *,
    dataset_ids: Sequence[str],
    epsilon: float,
    fit_size: int | None,
    statistics_config: Mapping[str, Any] | None,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Evaluate one strength cell with the headline operators and no controls."""

    rows = run_e04(
        table,
        dataset_ids=list(dataset_ids),
        perturbations=[cell.perturbation],
        method_ids=list(SWEEP_METHODS),
        normalization_variants=list(SWEEP_NORMALIZATIONS),
        epsilon=epsilon,
        fit_size=fit_size,
        controls={},
        statistics_config=statistics_config,
        random_seed=random_seed,
    )
    tagged: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        row = dict(row)
        row["stage"] = SWEEP_STAGE
        row["strength_variant"] = cell.strength_variant
        row["variant_id"] = cell.variant_id
        row["sweep_level"] = cell.sweep_level
        row["strength_parameter"] = cell.strength_parameter
        row["strength_value"] = cell.strength_value
        row["perturbation_overrides"] = cell.overrides or {}
        row["sensitivity_grid_complete"] = True
        tagged.append(row)
    return tagged


__all__ = [
    "SWEEP_METHODS",
    "SWEEP_NORMALIZATIONS",
    "SWEEP_STAGE",
    "StrengthSweepError",
    "SweepCell",
    "extract_pair_table",
    "load_nominal_pair_subset",
    "load_pair_table",
    "nominal_cell",
    "resolve_sweep_cells",
    "run_sweep_cell",
    "save_pair_table",
    "scoped_samples",
]
