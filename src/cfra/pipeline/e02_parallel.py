"""CPU-parallel E02 pixel/control audit.

build_e02 audits every selected object x perturbation pair: crop construction,
soft-mask rasterisation, perturbation application, all registered geometric
control masks and pixel invariants.  On a 56k-pair formal run this dominated the
wall clock (6 h serial).  The audit work is pure CPU and embarrassingly
parallel, so this module executes it with a pool of worker processes while
producing exactly the same per-row payloads as the serial loop.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from cfra.data import DatasetManifest, ObjectRecord
from cfra.protocols import (
    apply_perturbation,
    build_paired_inputs,
    crop_object,
    derive_control_masks,
    make_random_geometry_matched_mask,
    make_shifted_control_mask,
    pair_pixel_invariants,
    replace_mask_pair,
)

# Constants mirrored from stages.build_e02 so the worker stays independent of the
# pipeline module (avoiding import cycles under spawn).
_DERIVED_CONTROL_NAMES = ("background_only", "boundary_only", "interior_eroded", "mask_eroded", "mask_dilated")


def _nominal_parameters(config: dict[str, Any], kind: str) -> dict[str, Any]:
    """Extract perturbation parameters (mirror of stages._nominal_parameters)."""
    perturbations = config["perturbations"]
    strength_table = config.get("perturbation_strengths", {}).get(kind, {})
    if kind == "blur":
        return {"sigma_px": float(strength_table.get("nominal", perturbations["blur"]["sigma_px"]))}
    if kind == "patch_mask":
        return {
            "patch_fraction": float(
                strength_table.get("nominal", perturbations["patch_mask"]["target_fraction"])
            ),
            "patch_side_fraction": float(perturbations["patch_mask"]["patch_side_fraction"]),
        }
    if kind == "gaussian_noise":
        return {"noise_sigma": float(strength_table.get("nominal", perturbations["gaussian_noise"]["sigma"]))}
    if kind == "brightness":
        return {"brightness_factor": float(strength_table.get("nominal", perturbations["brightness"]["factor"]))}
    raise ValueError(f"Unknown configured perturbation: {kind}")


def _stable_seed(*parts: str | int) -> int:
    import hashlib
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


@lru_cache(maxsize=16)
def _load_rgb(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _worker_audit(job: tuple[int, str, ObjectRecord, str, int],
                  config: dict[str, Any],
                  manifest_roots: dict[str, str],
                  controls_meta: dict[str, Any]) -> dict[str, Any]:
    """Audit a single (object, perturbation) pair and return the full row.

    Identical semantics to the serial loop in stages.build_e02 so downstream
    report/audit logic (failures, control_failures, report.json) is unchanged.
    """
    dataset_id, object_id, obj, perturbation_name, seed = job

    input_config = config["input"]
    protocol_config = config["protocol"]
    pixel_tolerance = float(protocol_config["pixel_tolerance"])

    # Resolve image path from manifest root + relpath
    root = Path(manifest_roots[dataset_id]).expanduser().resolve()
    image_path = root / Path(*obj.image_relpath.replace("\\", "/").split("/"))
    source_image = _load_rgb(str(image_path))

    crop = crop_object(
        source_image,
        obj,
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
    invariant = pair_pixel_invariants(pair, tolerance=pixel_tolerance)

    def invariant_passes(inv: dict[str, Any]) -> bool:
        return bool(
            inv["outside_mask_equal_clear_vs_degraded"]
            and inv["inside_mask_equal_clean_clear"]
            and inv["blend_formula_exact"]
            and inv["changed_mask_consistent_with_uint8_difference"]
        )

    def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
        left_binary = left >= 0.5
        right_binary = right >= 0.5
        union = np.logical_or(left_binary, right_binary).sum()
        return float(np.logical_and(left_binary, right_binary).sum() / union) if union else 0.0

    controls_meta = controls_meta or {}
    registered_control_names = controls_meta.get("registered_control_names", [])
    derived_control_names = tuple(
        n for n in _DERIVED_CONTROL_NAMES if n in controls_meta.get("derived_enabled", ())
    )

    controls: dict[str, Any] = {}
    control_failures: list[dict[str, Any]] = []

    if derived_control_names:
        try:
            derived_masks = derive_control_masks(crop.mask, int(protocol_config["boundary_radius_px"]))
            for control_name in derived_control_names:
                if control_name not in derived_masks:
                    controls[control_name] = {
                        "status": "failed",
                        "reason": "registered_control_mask_missing",
                        "registered": True,
                    }
                    control_failures.append({
                        "dataset_id": dataset_id,
                        "object_id": object_id,
                        "perturbation": perturbation_name,
                        "control": control_name,
                        "reason": "registered_control_mask_missing",
                        "failure_type": "mask_construction",
                    })
                    continue
                control_mask = derived_masks[control_name]
                control_pair = build_paired_inputs(replace_mask_pair(crop, control_mask), perturbation)
                control_invariant = pair_pixel_invariants(control_pair, tolerance=pixel_tolerance)
                controls[control_name] = {
                    **control_invariant,
                    "mask_area_fraction": float(control_mask.mean()),
                    "registered": True,
                    "status": "passed" if invariant_passes(control_invariant) else "failed",
                }
                if not invariant_passes(control_invariant):
                    control_failures.append({
                        "dataset_id": dataset_id,
                        "object_id": object_id,
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
                    })
        except Exception as exc:
            for control_name in derived_control_names:
                controls[control_name] = {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "registered": True,
                }
                control_failures.append({
                    "dataset_id": dataset_id,
                    "object_id": object_id,
                    "perturbation": perturbation_name,
                    "control": control_name,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "failure_type": "control_audit_exception",
                })

    if "shifted_obb" in registered_control_names:
        try:
            shifted = make_shifted_control_mask(
                crop.mask,
                shift_fraction=float(protocol_config["control_shift_fraction"]),
                angle_rad=obj.orientation_rad,
                direction=1,
            )
            shifted_pair = build_paired_inputs(replace_mask_pair(crop, shifted), perturbation)
            shifted_invariant = pair_pixel_invariants(shifted_pair, tolerance=pixel_tolerance)
            controls["shifted_obb"] = {
                **shifted_invariant,
                "mask_area_fraction": float(shifted.mean()),
                "iou": mask_iou(shifted, crop.mask),
                "registered": True,
                "status": "passed" if invariant_passes(shifted_invariant) else "failed",
            }
            if not invariant_passes(shifted_invariant):
                control_failures.append({
                    "dataset_id": dataset_id,
                    "object_id": object_id,
                    "perturbation": perturbation_name,
                    "control": "shifted_obb",
                    "reason": "pixel_invariant_failure",
                    "failure_type": "pixel_invariant",
                })
        except Exception as exc:
            controls["shifted_obb"] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "registered": True,
            }
            control_failures.append({
                "dataset_id": dataset_id,
                "object_id": object_id,
                "perturbation": perturbation_name,
                "control": "shifted_obb",
                "reason": f"{type(exc).__name__}: {exc}",
                "failure_type": "mask_construction",
            })

    if "random_geometry_matched_obb" in registered_control_names:
        try:
            random_mask, random_meta = make_random_geometry_matched_mask(
                crop.mask,
                rng=rng,
                min_iou=float(protocol_config.get("control_min_iou", 0.0)),
                max_iou=float(protocol_config["control_max_iou"]),
                max_attempts=int(protocol_config["control_max_attempts"]),
            )
            random_pair = build_paired_inputs(replace_mask_pair(crop, random_mask), perturbation)
            random_invariant = pair_pixel_invariants(random_pair, tolerance=pixel_tolerance)
            controls["random_geometry_matched_obb"] = {
                **random_meta,
                **random_invariant,
                "mask_area_fraction": float(random_mask.mean()),
                "registered": True,
                "status": "passed" if invariant_passes(random_invariant) else "failed",
            }
            if not invariant_passes(random_invariant):
                control_failures.append({
                    "dataset_id": dataset_id,
                    "object_id": object_id,
                    "perturbation": perturbation_name,
                    "control": "random_geometry_matched_obb",
                    "reason": "pixel_invariant_failure",
                    "failure_type": "pixel_invariant",
                })
        except Exception as exc:
            controls["random_geometry_matched_obb"] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "registered": True,
            }
            control_failures.append({
                "dataset_id": dataset_id,
                "object_id": object_id,
                "perturbation": perturbation_name,
                "control": "random_geometry_matched_obb",
                "reason": f"{type(exc).__name__}: {exc}",
                "failure_type": "mask_construction",
            })

    row = {
        "dataset_id": dataset_id,
        "object_id": object_id,
        "source_image_id": obj.source_image_id,
        "class_id": obj.class_id,
        "class_name": obj.class_name,
        "role": obj.role,
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
            "pixel_tolerance": pixel_tolerance,
            "pixel_tolerance_semantics": "absolute_atol_only_rtol_zero",
            "changed_mask_semantics": "actual_uint8_rgb_difference_any_channel",
            "registered_controls": registered_control_names,
            "patch_allow_overlap": bool(
                config["perturbations"].get("patch_mask", {}).get("allow_overlap", False)
            ),
        },
    }
    is_failure = not invariant_passes(invariant)
    return {"row": row, "is_failure": is_failure, "control_failures": control_failures}


def audit_pairs_parallel(
    jobs: list[tuple[int, str, ObjectRecord, str, int]],
    *,
    manifests: dict[str, DatasetManifest],
    config: dict[str, Any],
    registered_control_names: list[str],
    derived_enabled: tuple[str, ...],
    cpu_workers: int | None = None,
    progress_desc: str = "E02 pixel/control audit",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit all pairs in parallel and return (rows, failures, control_failures).

    The returned rows are in input order, matching the serial loop exactly.
    Each row is JSON-safe (numpy scalars converted), matching write_jsonl input.
    """
    if cpu_workers is None:
        cpu_workers = max(1, min(8, os.cpu_count() or 1) - 2)

    manifest_roots = {dataset_id: str(manifest.root) for dataset_id, manifest in manifests.items()}
    controls_meta = {
        "registered_control_names": list(registered_control_names),
        "derived_enabled": list(derived_enabled),
    }

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    control_failures: list[dict[str, Any]] = []

    max_inflight = cpu_workers * 4
    futures: list[Future] = []

    with ProcessPoolExecutor(
        max_workers=cpu_workers,
        mp_context=get_context("spawn"),
    ) as executor, tqdm(total=len(jobs), desc=progress_desc, unit="pair") as progress:
        idx = 0
        while idx < len(jobs) and len(futures) < max_inflight:
            futures.append(executor.submit(_worker_audit, jobs[idx], config, manifest_roots, controls_meta))
            idx += 1
        while futures:
            future = futures.pop(0)
            result = future.result()
            row = result["row"]
            rows.append(row)
            if result["is_failure"]:
                failures.append(row)
            control_failures.extend(result["control_failures"])
            progress.update(1)
            if idx < len(jobs):
                futures.append(executor.submit(_worker_audit, jobs[idx], config, manifest_roots, controls_meta))
                idx += 1

    return rows, failures, control_failures
