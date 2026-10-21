"""Development and formal-stage building blocks for the CFRA pipeline."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("cfra")

import numpy as np
from PIL import Image
from tqdm import tqdm

from cfra.configuration import config_sha256, configured_dataset_paths, load_local_config
from cfra.data import (
    ROLE_NAMES,
    DatasetManifest,
    ObjectRecord,
    assign_source_image_roles,
    build_dataset_manifest,
    image_path,
    load_class_registry,
    load_dataset_manifest,
    load_jsonl,
    write_dataset_manifest,
    write_jsonl,
)
from cfra.data.index import manifest_exclusion_stats, write_schema_manifests
from cfra.guardrails import assert_clean_room_paths, assert_paths_exist, workspace_root
from cfra.models import parse_model_runtime_config, resolve_model_assets
from cfra.models.specs import get_model_spec
from cfra.protocols import (
    ProtocolError,
    apply_perturbation,
    build_paired_inputs,
    crop_object,
    derive_control_masks,
    geometry_control_feasibility,
    make_random_geometry_matched_mask,
    make_shifted_control_mask,
    object_mask,
    pair_pixel_invariants,
    replace_mask_pair,
)
from cfra.provenance import sha256_file
from cfra.runs import write_manifest_json


class StageError(RuntimeError):
    """Raised when a pipeline stage cannot produce a valid current-run artifact."""

    def __init__(self, message: str, *, failure_bundle: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.failure_bundle = failure_bundle


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stable_seed(*parts: str | int) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def _qualified_source_image_id(dataset_id: str, source_image_id: str) -> str:
    prefix = f"{dataset_id}:"
    return source_image_id if source_image_id.startswith(prefix) else prefix + source_image_id


def _run_dir(run_root: Path | str, relative: str) -> Path:
    root = Path(run_root).expanduser().resolve(strict=True)
    target = (root / relative).resolve()
    if target == root or root not in target.parents:
        raise StageError(f"Run output escapes run directory: {relative}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "Pillow", "torch", "transformers", "open_clip_torch", "tqdm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _code_identity(project_root: Path) -> dict[str, Any]:
    """Record the current checkout identity without reading historical projects."""

    result: dict[str, Any] = {"project_root": str(project_root)}
    for name, args in (
        ("git_commit", ("rev-parse", "HEAD")),
        ("git_dirty_status", ("status", "--porcelain")),
    ):
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            result[name] = "unavailable"
        else:
            value = completed.stdout.strip()
            result[name] = value if name != "git_dirty_status" else ("dirty" if value else "clean")
    return result


def _audit_geometry_control_feasibility(
    manifest: DatasetManifest,
    *,
    config: dict[str, Any],
) -> tuple[DatasetManifest, dict[str, Any]]:
    """Pre-exclude eligible objects whose registered random OBB control is impossible."""

    controls = config.get("controls", {})
    if not isinstance(controls, dict) or controls.get("random_geometry_matched_obb") is not True:
        return manifest, {
            "enabled": False,
            "policy": "not_registered",
            "checked_objects": 0,
            "excluded_objects": 0,
            "by_class": {},
        }

    input_config = config.get("input", {})
    protocol = config.get("protocol", {})
    size_px = int(input_config.get("size_px", 448))
    supersampling = int(input_config.get("obb_soft_mask_supersampling", 4))
    min_iou = float(protocol.get("control_min_iou", 0.0))
    max_iou = float(protocol.get("control_max_iou", 0.05))
    updated: list[ObjectRecord] = []
    checked = 0
    exclusions: list[dict[str, Any]] = []
    for item in manifest.objects:
        if not item.eligible:
            updated.append(item)
            continue
        checked += 1
        try:
            mask = object_mask(
                item,
                size_px=size_px,
                mask_supersampling=supersampling,
            )
            feasibility = geometry_control_feasibility(
                mask,
                min_iou=min_iou,
                max_iou=max_iou,
                max_attempts=int(protocol.get("control_max_attempts", 128)),
            )
        except (OSError, ProtocolError, ValueError) as exc:
            feasibility = {
                "feasible": False,
                "policy": "boundary_grid_3x3",
                "error": f"{type(exc).__name__}: {exc}",
                "min_iou": min_iou,
                "max_iou": max_iou,
            }
        if feasibility.get("feasible") is True:
            updated.append(item)
            continue
        excluded = replace(
            item,
            eligible=False,
            exclusion_reason="geometry_control_infeasible",
            exclusion_category="geometry_control_infeasible",
        )
        updated.append(excluded)
        exclusions.append(
            {
                "object_id": item.object_id,
                "source_image_id": item.source_image_id,
                "class_id": item.class_id,
                "class_name": item.class_name,
                "feasibility": feasibility,
            }
        )

    by_class: dict[str, int] = {}
    for item in exclusions:
        key = str(item["class_id"])
        by_class[key] = by_class.get(key, 0) + 1
    updated_manifest = replace(
        manifest,
        eligible_object_count=sum(item.eligible for item in updated),
        objects=tuple(updated),
    )
    return updated_manifest, {
        "enabled": True,
        "policy": "pre_exclude_geometry_control_infeasible",
        "placement_policy": "boundary_grid_3x3",
        "min_iou": min_iou,
        "max_iou": max_iou,
        "checked_objects": checked,
        "excluded_objects": len(exclusions),
        "by_class": by_class,
        "excluded": exclusions,
    }


def _hardware_identity() -> dict[str, Any]:
    result: dict[str, Any] = {"cuda_available": False, "gpu_names": []}
    try:
        import torch
    except ImportError:
        return result
    result["cuda_available"] = bool(torch.cuda.is_available())
    if result["cuda_available"]:
        result["gpu_names"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        result["cuda_version"] = getattr(torch.version, "cuda", None)
        result["cudnn_version"] = torch.backends.cudnn.version()
    return result


def build_e00(
    *,
    project_root: Path | str,
    run_root: Path | str,
    config: dict[str, Any],
    config_path: Path | str,
) -> tuple[dict[str, DatasetManifest], dict[str, Any]]:
    """Build raw dataset, class-registry, checkpoint and environment identity."""

    project = Path(project_root).expanduser().resolve(strict=True)
    local = load_local_config(project / "config" / "local.toml")
    dataset_ids = list(config["scope"]["datasets"])
    dataset_paths = configured_dataset_paths(local, dataset_ids)
    assert_clean_room_paths(dataset_paths.values(), workspace_root())
    assert_paths_exist(dataset_paths.values())
    class_registry_path = project / "registry" / "dataset_classes.json"
    class_registry = load_class_registry(class_registry_path)
    class_registry_sha256 = sha256_file(class_registry_path)
    sampling = config["sampling"]
    scope = config.get("scope", {})
    max_classes = scope.get("max_classes")
    manifests: dict[str, DatasetManifest] = {}
    geometry_control_audits: dict[str, dict[str, Any]] = {}
    manifest_root = _run_dir(run_root, "intermediates/e00_source_manifest")
    model_ids = [*scope.get("models", []), *scope.get("representation_variants", [])]
    logger.info(f"[E00] 开始: datasets={dataset_ids} models={model_ids}")
    for dataset_id in dataset_ids:
        started = time.time()
        logger.info(f"[E00] {dataset_id} 构建 source manifest")
        manifest = build_dataset_manifest(
            dataset_id,
            dataset_paths[dataset_id],
            class_registry=class_registry,
            class_registry_sha256=class_registry_sha256,
            minimum_side_px=float(sampling["minimum_object_side_px"]),
            maximum_side_px=float(sampling["maximum_object_side_px"]),
            maximum_image_pixels=int(sampling["maximum_image_pixels"]),
            maximum_difficulty=int(sampling["maximum_difficulty"]),
            max_classes=int(max_classes) if max_classes is not None else None,
        )
        manifest, geometry_control_audit = _audit_geometry_control_feasibility(
            manifest,
            config=config,
        )
        manifests[dataset_id] = manifest
        geometry_control_audits[dataset_id] = geometry_control_audit
        write_dataset_manifest(manifest, manifest_root)
        logger.info(
            f"[E00] {dataset_id} 完成 (耗时 {time.time() - started:.1f}s): "
            f"{manifest.object_count} objects / {manifest.eligible_object_count} eligible / "
            f"{manifest.image_count} images"
        )

    model_assets = resolve_model_assets(local, model_ids, registry_root=project / "registry")
    runtime = parse_model_runtime_config(config, model_ids)
    model_identity = {
        model_id: {
            "asset_id": asset.asset_id,
            "path": str(asset.path),
            "weight_path": str(asset.weight_path),
            "revision": asset.revision,
            "checkpoint_sha256": asset.checkpoint_sha256,
            "loader_config_sha256": asset.loader_config_sha256,
            "asset_fingerprint": asset.asset_fingerprint,
            "encoder_input_size": runtime.encoder_input_sizes[model_id],
            "representation_id": get_model_spec(model_id).representation_id,
            "dimension": get_model_spec(model_id).dimension,
            "pooling": get_model_spec(model_id).pooling,
            "projection": get_model_spec(model_id).projection,
            "text_shared_space": get_model_spec(model_id).text_shared_space,
            "shared_space_id": get_model_spec(model_id).shared_space_id,
        }
        for model_id, asset in model_assets.items()
    }
    exclusion_stats = {
        dataset_id: manifest_exclusion_stats(manifest)
        for dataset_id, manifest in manifests.items()
    }
    identity = {
        "schema_version": 3,
        "stage": "E00",
        "created_at_utc": _now(),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_sha256(config),
        "class_registry_path": str(class_registry_path),
        "class_registry_sha256": class_registry_sha256,
        "datasets": {
            dataset_id: {
                "root": manifest.root,
                "image_count": manifest.image_count,
                "labeled_image_count": manifest.labeled_image_count,
                "object_count": manifest.object_count,
                "eligible_object_count": manifest.eligible_object_count,
                "exclusion_stats": exclusion_stats[dataset_id],
                "geometry_control_audit": geometry_control_audits[dataset_id],
            }
            for dataset_id, manifest in manifests.items()
        },
        "exclusion_stats": exclusion_stats,
        "geometry_control_audit": geometry_control_audits,
        "exclusion_thresholds": {
            "minimum_side_px": float(sampling["minimum_object_side_px"]),
            "maximum_side_px": float(sampling["maximum_object_side_px"]),
            "maximum_image_pixels": int(sampling["maximum_image_pixels"]),
        },
        "models": model_identity,
        "environment": _runtime_versions(),
        "hardware": _hardware_identity(),
        "code": _code_identity(project),
        "executable": sys.executable,
    }
    model_rows = [
        {
            "model_id": model_id,
            **model_payload,
        }
        for model_id, model_payload in model_identity.items()
    ]
    write_schema_manifests(manifests, _run_dir(run_root, "manifest"), model_rows=model_rows)
    identity["manifest_schema"] = {
        "source_files": "source_files.csv",
        "images": "images.csv",
        "objects": "objects.csv",
        "splits": "splits.csv",
        "models": "models.csv",
        "exclusion_stats": "exclusion_stats.json",
        "format": "CSV",
    }
    _write_json(_run_dir(run_root, "manifest") / "exclusion_stats.json", exclusion_stats)
    _write_json(manifest_root / "identity.json", identity)
    write_manifest_json(run_root, "manifest/e00.json", identity)
    logger.info(f"[E00] 完成: {len(manifests)} 个数据集, {len(model_assets)} 个模型资产已校验")
    return manifests, {"local": local, "class_registry": class_registry, "model_assets": model_assets, "runtime": runtime}


def build_e01(
    *,
    run_root: Path | str,
    manifests: dict[str, DatasetManifest],
    config: dict[str, Any],
) -> dict[str, tuple[ObjectRecord, ...]]:
    """Assign and audit source-image roles for each dataset."""

    split = config["split"]
    sampling = config["sampling"]
    max_role_images = config.get("scope", {}).get("max_source_images_per_role")
    max_dataset_images = config.get("scope", {}).get("max_source_images_per_dataset")
    minimum_images = {
        "direction_fit": int(sampling["minimum_direction_fit_images_per_class"]),
        "readout_fit": int(sampling["minimum_readout_fit_images_per_class"]),
        "validation": int(sampling["minimum_validation_images_per_class"]),
        "confirmation": int(sampling["minimum_confirmation_images_per_class"]),
    }
    maximum_objects = {
        "direction_fit": int(sampling["direction_fit_objects_per_class"]),
        "readout_fit": int(sampling["readout_fit_objects_per_class"]),
        "validation": int(sampling.get("validation_objects_per_class", 50)),
        "confirmation": int(sampling.get("confirmation_objects_per_class", 50)),
    }
    output_root = _run_dir(run_root, "intermediates/e01_roles")
    selected_by_dataset: dict[str, tuple[ObjectRecord, ...]] = {}
    dataset_reports: dict[str, Any] = {}
    all_role_source_images: dict[str, set[str]] = {role: set() for role in ROLE_NAMES}
    all_role_content_groups: dict[str, set[str]] = {role: set() for role in ROLE_NAMES}
    logger.info(f"[E01] 开始: {len(manifests)} 个数据集, 四角色 source-image 划分")
    for dataset_id, manifest in tqdm(manifests.items(), desc="E01 source-image splits", unit="dataset"):
        started = time.time()
        selected, report = assign_source_image_roles(
            manifest,
            fractions=[float(value) for value in split["fractions"]],
            seed=int(split["seed"]),
            minimum_images_per_class=minimum_images,
            maximum_objects_per_class=maximum_objects,
            max_objects_per_image_per_class=int(sampling["max_objects_per_image_per_class"]),
            maximum_source_images_per_role=int(max_role_images) if max_role_images is not None else None,
            maximum_source_images_per_dataset=int(max_dataset_images) if max_dataset_images is not None else None,
        )
        selected_by_dataset[dataset_id] = selected
        for role in ROLE_NAMES:
            role_source_images = set(report["role_source_image_sets"][role])
            all_role_source_images[role].update(
                _qualified_source_image_id(dataset_id, image_id) for image_id in role_source_images
            )
            all_role_content_groups[role].update(report["role_content_group_sets"][role])
        report["source_image_role_sets"] = {
            role: sorted(report["role_source_image_sets"][role]) for role in ROLE_NAMES
        }
        dataset_reports[dataset_id] = report
        write_jsonl(output_root / f"{dataset_id}_objects.jsonl", (asdict(item) for item in selected))
        _write_json(output_root / f"{dataset_id}_report.json", report)
        logger.info(
            f"[E01] {dataset_id} 完成 (耗时 {time.time() - started:.1f}s): "
            f"{len(selected)} 个入选对象, "
            f"feasible={report.get('all_role_class_requirements_feasible')}"
        )
    overlaps: dict[str, list[str]] = {}
    content_overlaps: dict[str, list[str]] = {}
    for index, role in enumerate(ROLE_NAMES):
        for other_role in ROLE_NAMES[index + 1 :]:
            overlap = sorted(all_role_source_images[role] & all_role_source_images[other_role])
            if overlap:
                overlaps[f"{role}__{other_role}"] = overlap
            content_overlap = sorted(all_role_content_groups[role] & all_role_content_groups[other_role])
            if content_overlap:
                content_overlaps[f"{role}__{other_role}"] = content_overlap
    report = {
        "stage": "E01",
        "created_at_utc": _now(),
        "role_overlap": overlaps,
        "role_overlap_count": sum(len(values) for values in overlaps.values()),
        "content_group_overlap": content_overlaps,
        "content_group_overlap_count": sum(len(values) for values in content_overlaps.values()),
        "datasets": dataset_reports,
        "role_image_counts": {role: len(values) for role, values in all_role_source_images.items()},
        "role_object_counts": {
            role: sum(
                1 for selected in selected_by_dataset.values() for item in selected if item.role == role
            )
            for role in ROLE_NAMES
        },
        "class_breakdown": {
            dataset_id: dataset_report.get("role_class_breakdown", {})
            for dataset_id, dataset_report in dataset_reports.items()
        },
        "feasibility": {
            dataset_id: dataset_report["feasibility"]
            for dataset_id, dataset_report in dataset_reports.items()
        },
        "all_role_class_requirements_feasible": all(
            dataset_report["all_role_class_requirements_feasible"]
            for dataset_report in dataset_reports.values()
        ),
        "all_role_source_image_sets": {role: sorted(values) for role, values in all_role_source_images.items()},
        "source_image_disjoint": not overlaps,
        "content_group_disjoint": not content_overlaps,
    }
    _write_json(output_root / "report.json", report)
    write_manifest_json(run_root, "manifest/e01.json", report)
    role_by_image = {
        _qualified_source_image_id(dataset_id, source_image_id): role
        for dataset_id, dataset_report in dataset_reports.items()
        for role, source_image_ids in dataset_report["role_source_image_sets"].items()
        for source_image_id in source_image_ids
    }
    write_schema_manifests(
        manifests,
        _run_dir(run_root, "manifest"),
        role_by_image=role_by_image,
        seed=int(split["seed"]),
        assignment_version="source-image-content-union-v2",
    )
    if overlaps or content_overlaps:
        raise StageError(f"E01 source-image role overlap detected: {list(overlaps)[:3]}")
    if config["run"]["evidence_tier"] == "formal":
        excluded_classes: dict[str, list[str]] = {}
        for dataset_id, dataset_report in dataset_reports.items():
            if dataset_report["all_role_class_requirements_feasible"]:
                continue
            class_breakdown = dataset_report.get("role_class_breakdown", {})
            infeasible_classes: set[str] = set()
            for role, classes in class_breakdown.items():
                for class_id, info in classes.items():
                    if not info.get("feasible"):
                        infeasible_classes.add(str(class_id))
            if not infeasible_classes:
                continue
            excluded_classes[dataset_id] = sorted(infeasible_classes)
        if excluded_classes:
            report["excluded_classes"] = excluded_classes
            report["all_role_class_requirements_feasible"] = False
            _write_json(output_root / "report.json", report)
            write_manifest_json(run_root, "manifest/e01.json", report)
            summary = {
                dataset_id: {
                    "class_ids": class_ids,
                    "reason": "insufficient_role_class_capacity_under_image_disjoint_split",
                    "scope": "excluded_from_main_experiments_but_kept_for_e04_2_fit_budget_scaling",
                }
                for dataset_id, class_ids in excluded_classes.items()
            }
            write_manifest_json(run_root, "manifest/e01_excluded_classes.json", summary)
    logger.info(
        f"[E01] 完成: 总对象 {sum(len(selected) for selected in selected_by_dataset.values())}, "
        f"role_overlap={len(overlaps)} content_group_overlap={len(content_overlaps)}"
    )
    return selected_by_dataset


def _nominal_parameters(config: dict[str, Any], kind: str) -> dict[str, Any]:
    perturbations = config["perturbations"]
    if kind == "blur":
        return {"sigma_px": float(perturbations["blur"]["sigma_px"])}
    if kind == "patch_mask":
        return {
            "patch_fraction": float(perturbations["patch_mask"]["target_fraction"]),
            "patch_side_fraction": float(perturbations["patch_mask"]["patch_side_fraction"]),
            "allow_overlap": bool(perturbations["patch_mask"].get("allow_overlap", False)),
        }
    if kind == "gaussian_noise":
        return {"noise_sigma": float(perturbations["gaussian_noise"]["sigma"])}
    if kind == "brightness":
        return {"brightness_factor": float(perturbations["brightness"]["factor"])}
    raise StageError(f"Unknown configured perturbation: {kind}")


def build_e02(
    *,
    run_root: Path | str,
    manifests: dict[str, DatasetManifest],
    selected_by_dataset: dict[str, tuple[ObjectRecord, ...]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Audit the paired pixel protocol and all registered geometric mask controls."""

    output_root = _run_dir(run_root, "intermediates/e02_pixel_protocol")
    input_config = config["input"]
    protocol_config = config["protocol"]
    perturbation_names = [name for name in config["scope"]["perturbations"] if config["perturbations"][name]["enabled"]]
    configured_controls = config.get("controls", {})

    def control_enabled(name: str) -> bool:
        if not isinstance(configured_controls, dict):
            return True
        if name in configured_controls:
            return bool(configured_controls[name])
        if name in {"mask_eroded", "mask_dilated"} and "mask_dilation_erosion" in configured_controls:
            return bool(configured_controls["mask_dilation_erosion"])
        return False

    derived_control_names = ("background_only", "boundary_only", "interior_eroded", "mask_eroded", "mask_dilated")
    derived_enabled = tuple(name for name in derived_control_names if control_enabled(name))
    registered_control_names = [
        name
        for name in ("shifted_obb", "random_geometry_matched_obb", *derived_control_names)
        if control_enabled(name)
    ]

    def invariant_passes(invariant: dict[str, Any]) -> bool:
        return bool(
            invariant["outside_mask_equal_clear_vs_degraded"]
            and invariant["inside_mask_equal_clean_clear"]
            and invariant["blend_formula_exact"]
            and invariant["changed_mask_consistent_with_uint8_difference"]
        )

    def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
        left_binary = left >= 0.5
        right_binary = right >= 0.5
        union = np.logical_or(left_binary, right_binary).sum()
        return float(np.logical_and(left_binary, right_binary).sum() / union) if union else 0.0

    use_parallel = os.environ.get("CFRA_PARALLEL_E02", "1") == "1"
    cpu_workers_env = os.environ.get("CFRA_CPU_WORKERS")
    cpu_workers = int(cpu_workers_env) if cpu_workers_env else None

    logger.info(
        f"[E02] 开始: {len(selected_by_dataset)} 个数据集, {len(perturbation_names)} 种扰动, "
        f"{sum(len(items) for items in selected_by_dataset.values())} 个对象"
    )
    jobs: list[tuple[int, str, ObjectRecord, str, int]] = []
    total = sum(len(items) * len(perturbation_names) for items in selected_by_dataset.values())
    for dataset_id, selected in selected_by_dataset.items():
        logger.info(f"[E02] {dataset_id}: {len(selected)} 对象 × {len(perturbation_names)} 扰动")
        for object_record in selected:
            for perturbation_name in perturbation_names:
                seed = _stable_seed(config["split"]["seed"], object_record.object_id, perturbation_name)
                jobs.append((dataset_id, object_record.object_id, object_record, perturbation_name, seed))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    control_failures: list[dict[str, Any]] = []

    if use_parallel and total > 4:
        from .e02_parallel import audit_pairs_parallel

        rows, failures, control_failures = audit_pairs_parallel(
            jobs,
            manifests=manifests,
            config=config,
            registered_control_names=registered_control_names,
            derived_enabled=derived_enabled,
            cpu_workers=cpu_workers,
            progress_desc="E02 pixel/control audit",
        )
    else:
        with tqdm(total=total, desc="E02 pixel/control audit", unit="pair") as progress:
            for dataset_id, object_record, perturbation_name, seed in (
                (dataset_id, object_record, perturbation_name, _stable_seed(config["split"]["seed"], object_record.object_id, perturbation_name))
                for dataset_id, selected in selected_by_dataset.items()
                for object_record in selected
                for perturbation_name in perturbation_names
            ):
                manifest = manifests[dataset_id]
                with Image.open(image_path(manifest, object_record)) as image:
                    source_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
                crop = crop_object(
                    source_image,
                    object_record,
                    size_px=int(input_config["size_px"]),
                    mask_supersampling=int(input_config["obb_soft_mask_supersampling"]),
                )
                rng = np.random.default_rng(seed)
                perturbation = apply_perturbation(
                    crop.image,
                    perturbation_name,
                    rng=rng,
                    **_nominal_parameters(config, perturbation_name),
                )
                pair = build_paired_inputs(crop, perturbation)
                invariant = pair_pixel_invariants(pair, tolerance=float(protocol_config["pixel_tolerance"]))
                controls: dict[str, Any] = {}
                control_masks: dict[str, np.ndarray] = {}
                if derived_enabled:
                    try:
                        derived_masks = derive_control_masks(crop.mask, int(protocol_config["boundary_radius_px"]))
                        control_masks = {
                            name: derived_masks[name]
                            for name in derived_control_names
                            if name in derived_masks and control_enabled(name)
                        }
                        for control_name in derived_control_names:
                            if control_enabled(control_name) and control_name not in control_masks:
                                controls[control_name] = {
                                    "status": "failed",
                                    "reason": "registered_control_mask_missing",
                                    "registered": True,
                                }
                                control_failures.append(
                                    {
                                        "dataset_id": dataset_id,
                                        "object_id": object_record.object_id,
                                        "perturbation": perturbation_name,
                                        "control": control_name,
                                        "reason": "registered_control_mask_missing",
                                        "failure_type": "mask_construction",
                                    }
                                )
                    except Exception as exc:
                        for control_name in derived_control_names:
                            if control_enabled(control_name):
                                controls[control_name] = {
                                    "status": "failed",
                                    "reason": f"{type(exc).__name__}: {exc}",
                                    "registered": True,
                                }
                                control_failures.append(
                                    {
                                        "dataset_id": dataset_id,
                                        "object_id": object_record.object_id,
                                        "perturbation": perturbation_name,
                                        "control": control_name,
                                        "reason": f"{type(exc).__name__}: {exc}",
                                        "failure_type": "mask_construction",
                                    }
                                )

                for control_name, control_mask in control_masks.items():
                    try:
                        control_pair = build_paired_inputs(replace_mask_pair(crop, control_mask), perturbation)
                        control_invariant = pair_pixel_invariants(
                            control_pair,
                            tolerance=float(protocol_config["pixel_tolerance"]),
                        )
                        controls[control_name] = {
                            **control_invariant,
                            "mask_area_fraction": float(control_mask.mean()),
                            "registered": True,
                            "status": "passed" if invariant_passes(control_invariant) else "failed",
                        }
                        if not invariant_passes(control_invariant):
                            control_failures.append(
                                {
                                    "dataset_id": dataset_id,
                                    "object_id": object_record.object_id,
                                    "perturbation": perturbation_name,
                                    "control": control_name,
                                    "reason": "pixel_invariant_failure",
                                    "failed_checks": sorted(
                                        key for key in (
                                            "outside_mask_equal_clear_vs_degraded",
                                            "inside_mask_equal_clean_clear",
                                            "blend_formula_exact",
                                            "changed_mask_consistent_with_uint8_difference",
                                        )
                                        if not control_invariant[key]
                                    ),
                                    "failure_type": "pixel_invariant",
                                }
                            )
                    except Exception as exc:
                        controls[control_name] = {
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "registered": True,
                        }
                        control_failures.append(
                            {
                                "dataset_id": dataset_id,
                                "object_id": object_record.object_id,
                                "perturbation": perturbation_name,
                                "control": control_name,
                                "reason": f"{type(exc).__name__}: {exc}",
                                "failure_type": "control_audit_exception",
                            }
                        )

                if control_enabled("shifted_obb"):
                    try:
                        shifted = make_shifted_control_mask(
                            crop.mask,
                            shift_fraction=float(protocol_config["control_shift_fraction"]),
                            angle_rad=object_record.orientation_rad,
                            direction=1,
                        )
                        shifted_pair = build_paired_inputs(replace_mask_pair(crop, shifted), perturbation)
                        shifted_invariant = pair_pixel_invariants(
                            shifted_pair,
                            tolerance=float(protocol_config["pixel_tolerance"]),
                        )
                        controls["shifted_obb"] = {
                            **shifted_invariant,
                            "mask_area_fraction": float(shifted.mean()),
                            "iou": mask_iou(shifted, crop.mask),
                            "registered": True,
                            "status": "passed" if invariant_passes(shifted_invariant) else "failed",
                        }
                        if not invariant_passes(shifted_invariant):
                            control_failures.append(
                                {
                                    "dataset_id": dataset_id,
                                    "object_id": object_record.object_id,
                                    "perturbation": perturbation_name,
                                    "control": "shifted_obb",
                                    "reason": "pixel_invariant_failure",
                                    "failure_type": "pixel_invariant",
                                }
                            )
                    except Exception as exc:
                        controls["shifted_obb"] = {
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "registered": True,
                        }
                        control_failures.append(
                            {
                                "dataset_id": dataset_id,
                                "object_id": object_record.object_id,
                                "perturbation": perturbation_name,
                                "control": "shifted_obb",
                                "reason": f"{type(exc).__name__}: {exc}",
                                "failure_type": "mask_construction",
                            }
                        )

                if control_enabled("random_geometry_matched_obb"):
                    try:
                        random_mask, random_meta = make_random_geometry_matched_mask(
                            crop.mask,
                            rng=rng,
                            min_iou=float(protocol_config.get("control_min_iou", 0.0)),
                            max_iou=float(protocol_config["control_max_iou"]),
                            max_attempts=int(protocol_config["control_max_attempts"]),
                        )
                        random_pair = build_paired_inputs(replace_mask_pair(crop, random_mask), perturbation)
                        random_invariant = pair_pixel_invariants(
                            random_pair,
                            tolerance=float(protocol_config["pixel_tolerance"]),
                        )
                        controls["random_geometry_matched_obb"] = {
                            **random_meta,
                            **random_invariant,
                            "mask_area_fraction": float(random_mask.mean()),
                            "registered": True,
                            "status": "passed" if invariant_passes(random_invariant) else "failed",
                        }
                        if not invariant_passes(random_invariant):
                            control_failures.append(
                                {
                                    "dataset_id": dataset_id,
                                    "object_id": object_record.object_id,
                                    "perturbation": perturbation_name,
                                    "control": "random_geometry_matched_obb",
                                    "reason": "pixel_invariant_failure",
                                    "failure_type": "pixel_invariant",
                                }
                            )
                    except Exception as exc:
                        controls["random_geometry_matched_obb"] = {
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "registered": True,
                        }
                        control_failures.append(
                            {
                                "dataset_id": dataset_id,
                                "object_id": object_record.object_id,
                                "perturbation": perturbation_name,
                                "control": "random_geometry_matched_obb",
                                "reason": f"{type(exc).__name__}: {exc}",
                                "failure_type": "mask_construction",
                            }
                        )
                row = {
                    "dataset_id": dataset_id,
                    "object_id": object_record.object_id,
                    "source_image_id": object_record.source_image_id,
                    "class_id": object_record.class_id,
                    "class_name": object_record.class_name,
                    "role": object_record.role,
                    "perturbation": perturbation_name,
                    "seed": seed,
                    "crop_origin": [crop.origin_x, crop.origin_y],
                    "mask_area_fraction": float(crop.mask.mean()),
                    "perturbation_parameters": perturbation.parameters,
                    "invariant": invariant,
                    "controls": controls,
                    "interaction_protocol": {
                        "states": list(protocol_config["interaction_background_states"]),
                        "requires_model_features": True,
                    },
                    "protocol_audit": {
                        "pixel_tolerance": float(protocol_config["pixel_tolerance"]),
                        "pixel_tolerance_semantics": "absolute_atol_only_rtol_zero",
                        "changed_mask_semantics": "actual_uint8_rgb_difference_any_channel",
                        "registered_controls": registered_control_names,
                        "patch_allow_overlap": bool(
                            config["perturbations"].get("patch_mask", {}).get("allow_overlap", False)
                        ),
                    },
                }
                rows.append(row)
                if not invariant_passes(invariant):
                    failures.append(row)
                progress.update(1)
    write_jsonl(output_root / "audit_rows.jsonl", rows)
    empty_audit_failure = not rows
    report = {
        "stage": "E02",
        "created_at_utc": _now(),
        "row_count": len(rows),
        "failure_count": len(failures),
        "control_failure_count": len(control_failures),
        "control_failures": control_failures,
        "registered_control_failures": control_failures,
        "registered_controls": registered_control_names,
        "failed_object_ids": sorted({row["object_id"] for row in failures}),
        "all_pixel_invariants_pass": not failures,
            "all_registered_controls_available": not control_failures and not empty_audit_failure,
            "model_experiments_blocked": bool(control_failures or failures or empty_audit_failure),
        "empty_audit_failure": empty_audit_failure,
        "perturbations": perturbation_names,
        "pixel_tolerance": float(protocol_config["pixel_tolerance"]),
        "pixel_tolerance_semantics": "absolute_atol_only_rtol_zero",
        "changed_mask_semantics": "actual_uint8_rgb_difference_any_channel",
        "patch_allow_overlap": bool(config["perturbations"].get("patch_mask", {}).get("allow_overlap", False)),
    }
    _write_json(output_root / "report.json", report)
    write_manifest_json(run_root, "manifest/e02.json", report)
    if failures or control_failures or empty_audit_failure:
        failure_bundle = {
            "status": "failed",
            "stage": "E02",
            "reason": "pixel_invariant_failure" if failures else "registered_control_failure" if control_failures else "empty_audit",
            "evidence_tier": config["run"]["evidence_tier"],
            "audit_rows": "intermediates/e02_pixel_protocol/audit_rows.jsonl",
            "report": "intermediates/e02_pixel_protocol/report.json",
            "manifest": "manifest/e02.json",
            "failure_bundle": "intermediates/e02_pixel_protocol/failure_bundle.json",
            "row_count": len(rows),
            "failure_count": len(failures),
            "control_failure_count": len(control_failures),
            "control_failures": control_failures,
            "failed_object_ids": sorted(
                {row["object_id"] for row in failures}
                | {item["object_id"] for item in control_failures}
            ),
            "model_experiments_blocked": True,
        }
        report["failure_bundle"] = failure_bundle
        _write_json(output_root / "report.json", report)
        write_manifest_json(run_root, "manifest/e02.json", report)
        _write_json(output_root / "failure_bundle.json", failure_bundle)
        logger.error(
            f"[E02] 失败: pixel_failures={len(failures)}, "
            f"control_failures={len(control_failures)}, empty={empty_audit_failure}"
        )
        raise StageError(
            "E02 qualification failed; model experiments are blocked: "
            f"pixel_failures={len(failures)}, control_failures={len(control_failures)}, empty={empty_audit_failure}",
            failure_bundle=failure_bundle,
        )
    logger.info(
        f"[E02] 完成: {len(rows)} 审计对, {len(failures)} 像素失败, "
        f"{len(control_failures)} 控制失败, all_pass={not failures}"
    )
    return report


def build_e00_from_current_run(run_root: Path | str, project_root: Path | str, config: dict[str, Any], config_path: Path | str) -> dict[str, DatasetManifest]:
    manifests, _ = build_e00(project_root=project_root, run_root=run_root, config=config, config_path=config_path)
    return manifests


def load_current_manifests(run_root: Path | str, config: dict[str, Any]) -> dict[str, DatasetManifest]:
    run = Path(run_root).expanduser().resolve(strict=True)
    root = (run / "intermediates/e00_source_manifest").resolve()
    if root == run or run not in root.parents:
        raise StageError("Current manifest path escapes run directory")
    return {dataset_id: load_dataset_manifest(root, dataset_id) for dataset_id in config["scope"]["datasets"]}


def load_e01_selected(
    run_root: Path | str,
    config: dict[str, Any],
) -> dict[str, tuple[ObjectRecord, ...]]:
    """Reload the E01 role-assigned selection from the current run, if present.

    ``build_e01`` writes ``intermediates/e01_roles/<dataset>_objects.jsonl`` in
    stable manifest order, so a resumed run can reuse the exact same selection
    instead of re-running the assignment.  Raises :class:`StageError` when the
    file is missing so a caller knows it must replay E01.
    """

    run = Path(run_root).expanduser().resolve(strict=True)
    root = (run / "intermediates/e01_roles").resolve()
    if root == run or run not in root.parents:
        raise StageError("E01 role output path escapes run directory")
    selected: dict[str, tuple[ObjectRecord, ...]] = {}
    missing = []
    for dataset_id in config["scope"]["datasets"]:
        path = root / f"{dataset_id}_objects.jsonl"
        if not path.is_file():
            missing.append(dataset_id)
            continue
        try:
            rows = load_jsonl(path)
        except (OSError, ValueError) as exc:
            raise StageError(f"Cannot read E01 selection for {dataset_id}: {exc}") from exc
        selected[dataset_id] = tuple(ObjectRecord(**row) for row in rows)
    if missing:
        raise StageError(f"E01 selection is incomplete; missing datasets: {sorted(missing)}")
    return selected
