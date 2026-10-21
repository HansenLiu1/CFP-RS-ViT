"""Current-run model feature generation for E03-E05.

Features are written only below the active atomic run.  The generator reconstructs
every crop, mask and perturbation from the E00/E01 records so no cross-run sample
cache can enter a result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("cfra")

from cfra.data import DatasetManifest, ObjectRecord, image_path
from cfra.models import ModelAsset, ModelRuntimeConfig, load_adapter
from cfra.protocols import (
    ProtocolError,
    apply_perturbation,
    build_paired_inputs,
    crop_object,
    derive_control_masks,
    make_random_geometry_matched_mask,
    make_shifted_control_mask,
    replace_mask_pair,
    sample_background_crop,
)
from cfra.runs import write_manifest_json

CONTROL_MASK_NAMES = (
    "true_obb",
    "shifted_obb",
    "random_geometry_matched_obb",
    "background_only",
    "boundary_only",
    "interior_eroded",
    "mask_eroded",
    "mask_dilated",
)


class FeatureStageError(RuntimeError):
    """Raised when current-run feature extraction cannot satisfy the protocol."""


@dataclass(frozen=True)
class FeatureSample:
    dataset_id: str
    object_id: str
    source_image_id: str
    class_id: int
    class_name: str
    coarse_name: str
    role: str
    perturbation: str
    seed: int
    geometry: tuple[float, float, float, float]
    strength_variant: str = "nominal"
    excluded_from_main: bool = False


@dataclass(frozen=True)
class FeatureTable:
    model_id: str
    metadata: tuple[FeatureSample, ...]
    arrays: dict[str, np.ndarray]
    representation: dict[str, Any]


def _stable_seed(*parts: str | int) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def _read_excluded_classes(run_root: Path | str) -> dict[str, set[str]]:
    """Read the formal E01 class-level exclusion manifest (empty for non-formal runs)."""
    root = Path(run_root).expanduser().resolve(strict=True)
    path = root / "manifest" / "e01_excluded_classes.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    result: dict[str, set[str]] = {}
    for dataset_id, info in payload.items():
        if isinstance(info, dict):
            class_ids = info.get("class_ids")
            if isinstance(class_ids, list):
                result[dataset_id] = {str(value) for value in class_ids}
    return result


def _nominal_parameters(
    config: dict[str, Any],
    kind: str,
    strength_variant: str = "nominal",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    perturbations = config["perturbations"]
    strength_table = config.get("perturbation_strengths", {}).get(kind, {})
    configured_overrides = config.get("sensitivity", {}).get("perturbation_overrides", {})
    variant_entry = configured_overrides.get(strength_variant, {}) if isinstance(configured_overrides, dict) else {}
    variant_override = variant_entry.get(kind, {}) if isinstance(variant_entry, dict) else {}
    if overrides and kind in overrides:
        variant_override = {**variant_override, **overrides[kind]}
    custom_variant = bool(variant_override) or bool(overrides) or (
        isinstance(configured_overrides, dict) and strength_variant in configured_overrides
    )
    if strength_variant != "nominal" and strength_variant not in strength_table and not custom_variant:
        raise FeatureStageError(f"No {strength_variant} strength is registered for {kind}")
    fallback_variant = strength_variant if strength_variant in strength_table else "nominal"
    if kind == "blur":
        return {"sigma_px": float(variant_override.get("sigma_px", strength_table.get(fallback_variant, perturbations["blur"]["sigma_px"])))}
    if kind == "patch_mask":
        return {
            "patch_fraction": float(
                variant_override.get("patch_fraction", variant_override.get("target_fraction", strength_table.get(fallback_variant, perturbations["patch_mask"]["target_fraction"])))
            ),
            "patch_side_fraction": float(
                variant_override.get("patch_side_fraction", perturbations["patch_mask"]["patch_side_fraction"])
            ),
        }
    if kind == "gaussian_noise":
        return {"noise_sigma": float(variant_override.get("noise_sigma", strength_table.get(fallback_variant, perturbations["gaussian_noise"]["sigma"])))}
    if kind == "brightness":
        return {"brightness_factor": float(variant_override.get("brightness_factor", strength_table.get(fallback_variant, perturbations["brightness"]["factor"])))}
    raise FeatureStageError(f"Unknown configured perturbation: {kind}")


def make_feature_samples(
    selected_by_dataset: dict[str, tuple[ObjectRecord, ...]],
    config: dict[str, Any],
    *,
    strength_variant: str = "nominal",
    excluded_from_main: dict[str, set[str]] | None = None,
) -> tuple[FeatureSample, ...]:
    perturbations = [
        name for name in config["scope"]["perturbations"] if config["perturbations"][name]["enabled"]
    ]
    excluded = {dataset_id: set(class_ids) for dataset_id, class_ids in (excluded_from_main or {}).items()}
    samples: list[FeatureSample] = []
    for dataset_id, objects in selected_by_dataset.items():
        dataset_excluded = excluded.get(dataset_id, set())
        for object_record in objects:
            if object_record.role is None:
                raise FeatureStageError(f"Selected object has no E01 role: {object_record.object_id}")
            for perturbation in perturbations:
                samples.append(
                    FeatureSample(
                        dataset_id=dataset_id,
                        object_id=object_record.object_id,
                        source_image_id=object_record.source_image_id,
                        class_id=object_record.class_id,
                        class_name=object_record.class_name,
                        coarse_name=object_record.coarse_name,
                        role=object_record.role,
                        perturbation=perturbation,
                        seed=_stable_seed(config["split"]["seed"], object_record.object_id, perturbation),
                        geometry=(
                            float(np.log1p(object_record.visible_area_px2)),
                            float(np.log1p(object_record.visible_side_long_px / max(object_record.visible_side_short_px, 1.0e-8))),
                            float(np.sin(object_record.orientation_rad)),
                            float(np.cos(object_record.orientation_rad)),
                        ),
                        strength_variant=strength_variant,
                        excluded_from_main=str(object_record.class_id) in dataset_excluded,
                    )
                )
    return tuple(samples)


def _numpy_features(encoded: Any) -> np.ndarray:
    values = encoded.values
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def _load_rgb(manifest: DatasetManifest, object_record: ObjectRecord) -> np.ndarray:
    with Image.open(image_path(manifest, object_record)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _sample_context(
    sample: FeatureSample,
    *,
    manifests: dict[str, DatasetManifest] | None = None,
    object_index: dict[str, ObjectRecord],
    config: dict[str, Any],
    perturbation_overrides: dict[str, Any] | None = None,
    source_image: np.ndarray | None = None,
    same_image_objects: list[ObjectRecord] | tuple[ObjectRecord, ...] | None = None,
) -> dict[str, Any]:
    """Build every encoder input for one sample.

    ``source_image`` and ``same_image_objects`` are optional injection points so a
    caller can supply a cached decode and a precomputed per-image object list.
    Both default to the manifest-derived values, so the computation is identical
    whether or not they are provided.
    """

    object_record = object_index[sample.object_id]
    if (source_image is None or same_image_objects is None) and manifests is None:
        raise FeatureStageError(
            "_sample_context needs manifests unless source_image and same_image_objects are supplied"
        )
    if source_image is None:
        assert manifests is not None
        source_image = _load_rgb(manifests[sample.dataset_id], object_record)
    crop = crop_object(
        source_image,
        object_record,
        size_px=int(config["input"]["size_px"]),
        mask_supersampling=int(config["input"]["obb_soft_mask_supersampling"]),
    )
    rng = np.random.default_rng(sample.seed)
    perturbation = apply_perturbation(
        crop.image,
        sample.perturbation,
        rng=rng,
        **_nominal_parameters(config, sample.perturbation, sample.strength_variant, perturbation_overrides),
    )
    pair = build_paired_inputs(crop, perturbation)
    protocol = config["protocol"]
    masks: dict[str, np.ndarray] = {"true_obb": crop.mask}
    control_valid = {name: True for name in CONTROL_MASK_NAMES if name != "true_obb"}
    derived = derive_control_masks(crop.mask, int(protocol["boundary_radius_px"]))
    masks.update({name: derived[name] for name in ("background_only", "boundary_only", "interior_eroded", "mask_eroded", "mask_dilated")})
    masks["shifted_obb"] = make_shifted_control_mask(
        crop.mask,
        shift_fraction=float(protocol["control_shift_fraction"]),
        angle_rad=object_record.orientation_rad,
        direction=1,
    )
    try:
        masks["random_geometry_matched_obb"], _ = make_random_geometry_matched_mask(
            crop.mask,
            rng=rng,
            min_iou=float(protocol.get("control_min_iou", 0.0)),
            max_iou=float(protocol["control_max_iou"]),
            max_attempts=int(protocol["control_max_attempts"]),
        )
    except ProtocolError:
        control_valid["random_geometry_matched_obb"] = False
        masks["random_geometry_matched_obb"] = crop.mask.copy()

    inputs: dict[str, np.ndarray] = {
        "raw": crop.image,
        "true_clear_degraded": pair.foreground_clear_background_degraded,
        "true_degraded_degraded": pair.foreground_degraded_background_degraded,
        "true_clear_clean": pair.foreground_clear_background_clean,
        "true_degraded_clean": pair.foreground_degraded_background_clean,
    }
    for control_name, mask in masks.items():
        if control_name == "true_obb":
            continue
        control_pair = build_paired_inputs(replace_mask_pair(crop, mask), perturbation)
        inputs[f"control_{control_name}_clear_degraded"] = control_pair.foreground_clear_background_degraded
        inputs[f"control_{control_name}_degraded_degraded"] = control_pair.foreground_degraded_background_degraded

    if same_image_objects is None:
        assert manifests is not None
        same_image_objects = [
            item
            for item in manifests[sample.dataset_id].objects
            if item.source_image_id == object_record.source_image_id
        ]
    same_image_objects = list(same_image_objects)
    try:
        background = sample_background_crop(
            source_image,
            object_record,
            same_image_objects,
            size_px=int(config["input"]["size_px"]),
            rng=np.random.default_rng(_stable_seed(sample.seed, "background")),
            max_attempts=int(protocol["control_max_attempts"]),
        )
        background_perturbation = apply_perturbation(
            background.image,
            sample.perturbation,
            rng=np.random.default_rng(_stable_seed(sample.seed, "background_perturbation")),
            **_nominal_parameters(config, sample.perturbation, sample.strength_variant, perturbation_overrides),
        )
        inputs["background_clear"] = background.image
        inputs["background_degraded"] = background_perturbation.degraded
        inputs["_background_valid"] = True
    except ProtocolError:
        inputs["background_clear"] = crop.image.copy()
        inputs["background_degraded"] = crop.image.copy()
        inputs["_background_valid"] = False
    inputs["_control_valid"] = control_valid
    inputs["_perturbation_parameters"] = perturbation.parameters
    return inputs


_PAIR_INPUT_KEYS: tuple[str, ...] = ("true_clear_degraded", "true_degraded_degraded")


def _sample_context_pair(
    sample: FeatureSample,
    *,
    object_index: dict[str, ObjectRecord],
    config: dict[str, Any],
    perturbation_overrides: dict[str, Any] | None = None,
    source_image: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build only the two foreground-pair encoder inputs for one sample.

    E06 encodes exactly ``true_clear_degraded`` and ``true_degraded_degraded``
    and discards the control/background crops, so this reproduces the two
    values with the same crop, perturbation RNG and paired construction as
    :func:`_sample_context` while skipping the ~20 unused crops per sample.
    """

    object_record = object_index[sample.object_id]
    crop = crop_object(
        source_image,
        object_record,
        size_px=int(config["input"]["size_px"]),
        mask_supersampling=int(config["input"]["obb_soft_mask_supersampling"]),
    )
    rng = np.random.default_rng(sample.seed)
    perturbation = apply_perturbation(
        crop.image,
        sample.perturbation,
        rng=rng,
        **_nominal_parameters(config, sample.perturbation, sample.strength_variant, perturbation_overrides),
    )
    pair = build_paired_inputs(crop, perturbation)
    return {
        "raw": crop.image,
        "true_clear_degraded": pair.foreground_clear_background_degraded,
        "true_degraded_degraded": pair.foreground_degraded_background_degraded,
    }


def _input_keys() -> tuple[str, ...]:
    return (
        "raw",
        "true_clear_degraded",
        "true_degraded_degraded",
        "true_clear_clean",
        "true_degraded_clean",
        *(key for control in CONTROL_MASK_NAMES if control != "true_obb" for key in (f"control_{control}_clear_degraded", f"control_{control}_degraded_degraded")),
        "background_clear",
        "background_degraded",
    )


def extract_feature_table(
    *,
    project_root: Path | str,
    run_root: Path | str,
    config: dict[str, Any],
    manifests: dict[str, DatasetManifest],
    selected_by_dataset: dict[str, tuple[ObjectRecord, ...]],
    model_id: str,
    asset: ModelAsset,
    runtime: ModelRuntimeConfig,
    strength_variant: str = "nominal",
    perturbation_overrides: dict[str, Any] | None = None,
) -> FeatureTable:
    logger.info(
        f"[features] {model_id}{'/' + strength_variant if strength_variant != 'nominal' else ''} 开始: "
        f"datasets={list(selected_by_dataset)}"
    )
    extract_started = time.time()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    excluded_from_main = _read_excluded_classes(run_root)
    samples = make_feature_samples(
        selected_by_dataset,
        config,
        strength_variant=strength_variant,
        excluded_from_main=excluded_from_main,
    )
    logger.info(f"[features] {model_id} 样本规划完成: {len(samples)} 个样本")
    object_index = {item.object_id: item for manifest in manifests.values() for item in manifest.objects}
    output_root = Path(run_root).resolve(strict=True) / "intermediates/features"
    output_root.mkdir(parents=True, exist_ok=True)

    # Check for parallel extraction
    use_parallel = os.environ.get("CFRA_PARALLEL_EXTRACT", "1") == "1"
    cpu_workers_env = os.environ.get("CFRA_CPU_WORKERS")
    cpu_workers = int(cpu_workers_env) if cpu_workers_env else None

    logger.info(f"[features] {model_id} 加载模型: asset={asset.asset_id}")
    with load_adapter(
        model_id,
        asset,
        runtime=runtime,
        registry_root=Path(project_root).resolve(strict=True) / "registry",
        device="auto",
    ) as adapter:
        logger.info(f"[features] {model_id} 编码器就绪 ({len(samples)} 样本), 并行={use_parallel and len(samples) > 100}")
        if use_parallel and len(samples) > 100:
            # Use parallel extraction for large sample sets
            from .features_parallel import extract_features_parallel

            stacked, background_valid, control_valid, perturbation_audit = extract_features_parallel(
                samples=samples,
                manifests=manifests,
                object_index=object_index,
                config=config,
                adapter=adapter,
                perturbation_overrides=perturbation_overrides,
                cpu_workers=cpu_workers,
            )
            arrays = None  # Already stacked
        else:
            # Serial fallback
            arrays: dict[str, list[np.ndarray]] = {key: [] for key in _input_keys()}
            background_valid: list[bool] = []
            control_valid: dict[str, list[bool]] = {
                name: [] for name in CONTROL_MASK_NAMES if name != "true_obb"
            }
            perturbation_audit: list[dict[str, Any]] = []
            pending: dict[str, list[np.ndarray]] = {key: [] for key in _input_keys()}
            for sample in tqdm(samples, desc=f"{model_id} feature pairs", unit="sample"):
                context = _sample_context(
                    sample,
                    manifests=manifests,
                    object_index=object_index,
                    config=config,
                    perturbation_overrides=perturbation_overrides,
                )
                background_valid.append(bool(context["_background_valid"]))
                perturbation_audit.append(dict(context["_perturbation_parameters"]))
                for name, valid in context["_control_valid"].items():
                    control_valid[name].append(bool(valid))
                for key in _input_keys():
                    pending[key].append(context[key])
                if len(pending["raw"]) >= runtime.batch_size:
                    for key in _input_keys():
                        arrays[key].append(_numpy_features(adapter.encode(pending[key])))
                        pending[key].clear()
            if pending["raw"]:
                for key in _input_keys():
                    arrays[key].append(_numpy_features(adapter.encode(pending[key])))

    if arrays is not None:
        stacked = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    stacked["geometry"] = np.asarray([sample.geometry for sample in samples], dtype=np.float64)
    stacked["background_valid"] = np.asarray(background_valid, dtype=np.bool_)
    stacked["actual_changed_fraction"] = np.asarray(
        [float(item.get("actual_changed_fraction", np.nan)) for item in perturbation_audit], dtype=np.float64
    )
    stacked["actual_changed_pixels"] = np.asarray(
        [int(item.get("actual_changed_pixels", 0)) for item in perturbation_audit], dtype=np.int64
    )
    stacked["requested_region_fraction"] = np.asarray(
        [float(item.get("requested_region_fraction", np.nan)) for item in perturbation_audit], dtype=np.float64
    )
    stacked["overlap_pixels"] = np.asarray(
        [int(item.get("overlap_pixels", 0)) for item in perturbation_audit], dtype=np.int64
    )
    for name, values in control_valid.items():
        stacked[f"control_valid_{name}"] = np.asarray(values, dtype=np.bool_)
    suffix = "" if strength_variant == "nominal" else f"_{strength_variant}"
    perturbation_parameters = {
        perturbation: _nominal_parameters(
            config,
            perturbation,
            strength_variant=strength_variant,
            overrides=perturbation_overrides,
        )
        for perturbation in sorted({sample.perturbation for sample in samples})
    }
    metadata = {
        "model_id": model_id,
        "representation_id": adapter.spec.representation_id,
        "dimension": int(adapter.spec.dimension),
        "pooling": adapter.spec.pooling,
        "projection": adapter.spec.projection,
        "text_shared_space": adapter.spec.text_shared_space,
        "shared_space_id": adapter.spec.shared_space_id,
        "checkpoint_revision": asset.revision,
        "checkpoint_sha256": asset.checkpoint_sha256,
        "loader_config_sha256": asset.loader_config_sha256,
        "runtime_fingerprint": runtime.fingerprint,
        "encoder_input_size": runtime.encoder_input_sizes[model_id],
        "sample_count": len(samples),
        "input_keys": list(_input_keys()),
        "strength_variant": strength_variant,
        "perturbation_overrides": perturbation_overrides or {},
        "perturbation_parameters": perturbation_parameters,
        "perturbation_audit_fields": [
            "actual_changed_fraction",
            "actual_changed_pixels",
            "requested_region_fraction",
            "overlap_pixels",
        ],
        "perturbation_audit_rows": perturbation_audit,
    }
    npz_payload: dict[str, Any] = {**stacked}
    npz_payload["object_id"] = np.asarray([sample.object_id for sample in samples], dtype="U")
    npz_payload["source_image_id"] = np.asarray([sample.source_image_id for sample in samples], dtype="U")
    npz_payload["dataset_id"] = np.asarray([sample.dataset_id for sample in samples], dtype="U")
    npz_payload["class_id"] = np.asarray([sample.class_id for sample in samples], dtype=np.int64)
    npz_payload["role"] = np.asarray([sample.role for sample in samples], dtype="U")
    npz_payload["perturbation"] = np.asarray([sample.perturbation for sample in samples], dtype="U")
    npz_payload["class_name"] = np.asarray([sample.class_name for sample in samples], dtype="U")
    npz_payload["coarse_name"] = np.asarray([sample.coarse_name for sample in samples], dtype="U")
    npz_payload["seed"] = np.asarray([sample.seed for sample in samples], dtype=np.uint64)
    npz_payload["geometry"] = np.asarray([sample.geometry for sample in samples], dtype=np.float64)
    npz_payload["strength_variant"] = np.asarray([sample.strength_variant for sample in samples], dtype="U")
    npz_payload["excluded_from_main"] = np.asarray(
        [bool(sample.excluded_from_main) for sample in samples], dtype=np.bool_
    )
    npz_payload["background_valid"] = np.asarray(background_valid, dtype=np.bool_)
    for name, values in control_valid.items():
        npz_payload[f"control_valid_{name}"] = np.asarray(values, dtype=np.bool_)
    np.savez_compressed(output_root / f"features_{model_id}{suffix}.npz", **npz_payload)
    (output_root / f"features_{model_id}{suffix}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    write_manifest_json(run_root, f"manifest/features_{model_id}{suffix}.json", metadata)
    logger.info(
        f"[features] {model_id}{suffix or ''} 完成 (耗时 {time.time() - extract_started:.1f}s): "
        f"{len(samples)} 样本, dim={metadata['dimension']}"
    )
    return FeatureTable(model_id=model_id, metadata=samples, arrays=stacked, representation=metadata)


def load_feature_table(
    run_root: Path | str,
    model_id: str,
    *,
    strength_variant: str = "nominal",
) -> FeatureTable:
    root = Path(run_root).resolve(strict=True) / "intermediates/features"
    suffix = "" if strength_variant == "nominal" else f"_{strength_variant}"
    payload = np.load(root / f"features_{model_id}{suffix}.npz", allow_pickle=False)
    metadata = json.loads((root / f"features_{model_id}{suffix}.json").read_text(encoding="utf-8"))
    samples = tuple(
        FeatureSample(
            dataset_id=str(dataset_id),
            object_id=str(object_id),
            source_image_id=str(source_image_id),
            class_id=int(class_id),
            class_name=str(class_name),
            coarse_name=str(coarse_name),
            role=str(role),
            perturbation=str(perturbation),
            seed=int(seed),
            geometry=tuple(float(value) for value in geometry),
            strength_variant=str(sample_strength),
            excluded_from_main=bool(excluded),
        )
        for dataset_id, object_id, source_image_id, class_id, class_name, coarse_name, role, perturbation, seed, geometry, sample_strength, excluded in zip(
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
    arrays = {key: np.asarray(payload[key]) for key in metadata["input_keys"]}
    arrays["geometry"] = np.asarray(payload["geometry"], dtype=np.float64)
    arrays["background_valid"] = np.asarray(payload["background_valid"], dtype=np.bool_)
    for name in metadata.get("perturbation_audit_fields", ()):
        if name in payload:
            arrays[name] = np.asarray(payload[name])
    for name in CONTROL_MASK_NAMES:
        if name != "true_obb":
            arrays[f"control_valid_{name}"] = np.asarray(payload[f"control_valid_{name}"], dtype=np.bool_)
    return FeatureTable(model_id=model_id, metadata=samples, arrays=arrays, representation=metadata)


def extract_layerwise_feature_artifact(
    *,
    adapter: Any,
    clear_images: Any,
    degraded_images: Any,
    selected_layers: tuple[str, ...] | list[str],
    run_root: Path | str,
    artifact_name: str = "layerwise_features",
    metadata: dict[str, Any] | None = None,
    row_metadata: Any | None = None,
    output_device: str = "cpu",
) -> dict[str, Any]:
    """Extract paired layerwise features and persist them below the current run.

    The helper intentionally accepts an already loaded adapter so callers can
    keep model loading and sample reconstruction in their existing stage.  Each
    layer is written as ``clear`` and ``degraded`` arrays; no cross-run cache or
    external output path is accepted.
    """

    def _numpy_layer_values(values: Any) -> np.ndarray:
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=np.float32)

    if isinstance(selected_layers, (str, bytes)):
        raise FeatureStageError("selected_layers must be a non-empty sequence")
    try:
        selected = tuple(str(layer) for layer in selected_layers)
    except TypeError as exc:
        raise FeatureStageError("selected_layers must be a non-empty sequence") from exc
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(not layer or "/" in layer or "\\" in layer or "\x00" in layer for layer in selected)
    ):
        raise FeatureStageError("selected_layers must contain at least one unique layer")
    if (
        not isinstance(artifact_name, str)
        or not artifact_name
        or Path(artifact_name).name != artifact_name
        or artifact_name in {".", ".."}
        or artifact_name != artifact_name.strip()
        or artifact_name.endswith(".")
        or "\x00" in artifact_name
    ):
        raise FeatureStageError("artifact_name must be a single relative filename stem")
    if any(character in artifact_name for character in ("/", "\\")):
        raise FeatureStageError("artifact_name cannot contain path separators")

    try:
        root = Path(run_root).expanduser().resolve(strict=True)
        output_root = (root / "intermediates" / "features").resolve()
    except (OSError, RuntimeError) as exc:
        raise FeatureStageError("layerwise feature run root is missing or invalid") from exc
    if output_root == root or root not in output_root.parents:
        raise FeatureStageError("layerwise feature output escapes the current run")
    output_root.mkdir(parents=True, exist_ok=True)

    def _checked_target(filename: str) -> Path:
        target = output_root / filename
        resolved = target.resolve()
        if resolved == root or root not in resolved.parents:
            raise FeatureStageError(f"layerwise artifact target escapes the current run: {filename}")
        if target.exists() or target.is_symlink():
            raise FeatureStageError(f"layerwise artifact already exists and cannot be overwritten: {filename}")
        return target

    clear = adapter.encode_layerwise(clear_images, selected, output_device=output_device)
    degraded = adapter.encode_layerwise(degraded_images, selected, output_device=output_device)
    if tuple(clear.values) != tuple(selected) or tuple(degraded.values) != tuple(selected):
        raise FeatureStageError("layerwise adapter returned layers in an unexpected set")
    if clear.metadata != degraded.metadata:
        raise FeatureStageError("clear and degraded layer metadata differ")

    arrays: dict[str, Any] = {}
    layer_metadata: dict[str, Any] = {}
    sample_count: int | None = None
    for layer in selected:
        clear_values = _numpy_layer_values(clear.values[layer])
        degraded_values = _numpy_layer_values(degraded.values[layer])
        if clear_values.ndim != 2 or degraded_values.ndim != 2 or clear_values.shape != degraded_values.shape:
            raise FeatureStageError(
                f"layer {layer!r} clear/degraded arrays must have identical matrix shapes"
            )
        if not np.isfinite(clear_values).all() or not np.isfinite(degraded_values).all():
            raise FeatureStageError(f"layer {layer!r} contains non-finite values")
        if layer not in clear.metadata:
            raise FeatureStageError(f"layer {layer!r} is missing representation metadata")
        required_metadata = {
            "kind",
            "block",
            "dim",
            "pooling",
            "projection",
            "text_shared_space",
            "token",
            "grid",
            "interpolation",
        }
        missing_metadata = sorted(required_metadata - set(clear.metadata[layer]))
        if missing_metadata:
            raise FeatureStageError(f"layer {layer!r} metadata is missing: {missing_metadata}")
        if sample_count is None:
            sample_count = int(clear_values.shape[0])
        elif clear_values.shape[0] != sample_count:
            raise FeatureStageError("all selected layer arrays must have the same row count")
        arrays[f"{layer}__clear"] = clear_values
        arrays[f"{layer}__degraded"] = degraded_values
        layer_metadata[layer] = dict(clear.metadata[layer])

    serialised_rows = None
    if row_metadata is not None:
        if isinstance(row_metadata, dict):
            serialised_rows = {
                str(key): np.asarray(value).tolist() if isinstance(value, np.ndarray) else value
                for key, value in row_metadata.items()
            }
        else:
            try:
                serialised_rows = [
                    dict(row) if isinstance(row, dict) else dict(vars(row))
                    for row in row_metadata
                ]
            except (TypeError, ValueError) as exc:
                raise FeatureStageError("row_metadata must be JSON-compatible mapping or row sequence") from exc
        try:
            json.dumps(serialised_rows, ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise FeatureStageError("row_metadata must be JSON-compatible") from exc

    identity = {
        "schema_version": 1,
        "artifact_kind": "layerwise_paired_features",
        "model_id": clear.model_id,
        "representation_id": clear.representation_id,
        "selected_layers": list(selected),
        "layer_metadata": layer_metadata,
        "sample_count": int(sample_count or 0),
        "output_device": output_device,
        "checkpoint_revision": clear.checkpoint_revision,
        "checkpoint_sha256": clear.checkpoint_sha256,
        "loader_config_sha256": clear.loader_config_sha256,
        "asset_fingerprint": clear.asset_fingerprint,
        "runtime_fingerprint": clear.runtime_fingerprint,
        "inference_precision": clear.inference_precision,
        "row_metadata": serialised_rows,
        "extra_metadata": metadata or {},
    }
    if serialised_rows is not None:
        if isinstance(serialised_rows, dict):
            row_records = serialised_rows.get("records", serialised_rows.get("rows"))
            if row_records is not None and len(row_records) != identity["sample_count"]:
                raise FeatureStageError("row_metadata does not match the paired feature row count")
            if row_records is None:
                for value in serialised_rows.values():
                    if isinstance(value, (str, bytes)) or np.asarray(value).ndim == 0:
                        continue
                    if len(value) != identity["sample_count"]:
                        raise FeatureStageError("row_metadata columns do not match the paired feature row count")
        elif len(serialised_rows) != identity["sample_count"]:
            raise FeatureStageError("row_metadata does not match the paired feature row count")
    try:
        json.dumps(identity, ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise FeatureStageError("layerwise artifact metadata must be JSON-compatible") from exc
    npz_path = _checked_target(f"{artifact_name}.npz")
    json_path = _checked_target(f"{artifact_name}.json")
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(identity, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    write_manifest_json(run_root, f"manifest/{artifact_name}.json", identity)
    return {"npz": npz_path, "metadata": json_path, "manifest": root / "manifest" / f"{artifact_name}.json", **identity}
