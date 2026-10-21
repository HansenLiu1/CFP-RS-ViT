"""Deterministic native-pixel crops, OBB masks, perturbations and controls."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from cfra.data import ObjectRecord


class ProtocolError(ValueError):
    """Raised when an image protocol invariant cannot be satisfied."""


@dataclass(frozen=True)
class CropData:
    image: np.ndarray
    mask: np.ndarray
    origin_x: int
    origin_y: int
    center_x: float
    center_y: float
    polygon_local: tuple[float, ...]


@dataclass(frozen=True)
class PerturbationResult:
    degraded: np.ndarray
    changed_mask: np.ndarray
    parameters: dict[str, Any]


@dataclass(frozen=True)
class PairedInputs:
    foreground_clear_background_clean: np.ndarray
    foreground_degraded_background_clean: np.ndarray
    foreground_clear_background_degraded: np.ndarray
    foreground_degraded_background_degraded: np.ndarray
    mask: np.ndarray
    perturbation: PerturbationResult


@dataclass(frozen=True)
class BackgroundCrop:
    image: np.ndarray
    center_x: int
    center_y: int
    attempts: int


def _reflect_indices(start: int, length: int, size: int) -> np.ndarray:
    if size <= 0:
        raise ProtocolError("Cannot reflect-index an empty image axis")
    if size == 1:
        return np.zeros(length, dtype=np.int64)
    period = 2 * size - 2
    values = (np.arange(start, start + length, dtype=np.int64) % period + period) % period
    return np.where(values < size, values, period - values)


def _polygon_points(flat: tuple[float, ...]) -> list[tuple[float, float]]:
    if len(flat) < 6 or len(flat) % 2:
        raise ProtocolError("Polygon must contain at least three x/y pairs")
    return list(zip(flat[::2], flat[1::2]))


def _rasterize_soft_mask(
    polygon_local: list[tuple[float, float]],
    size: int,
    supersampling: int,
) -> np.ndarray:
    if supersampling <= 0:
        raise ProtocolError("Mask supersampling must be positive")
    canvas_size = size * supersampling
    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(canvas)
    scaled = [(round(x * supersampling), round(y * supersampling)) for x, y in polygon_local]
    draw.polygon(scaled, fill=255)
    mask = canvas.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(mask, dtype=np.float32) / 255.0


def crop_object(
    image: np.ndarray | Image.Image,
    object_record: ObjectRecord,
    *,
    size_px: int = 448,
    mask_supersampling: int = 4,
) -> CropData:
    """Extract a fixed native-pixel crop with explicit reflect boundary handling."""

    if size_px <= 0 or size_px % 2:
        raise ProtocolError("Crop size must be a positive even integer")
    if isinstance(image, Image.Image):
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        source = np.asarray(image)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ProtocolError("Image must have shape [height, width, 3]")
        if source.dtype != np.uint8:
            if np.issubdtype(source.dtype, np.floating) and source.min() >= 0 and source.max() <= 1:
                source = np.rint(source * 255.0).astype(np.uint8)
            else:
                source = np.clip(source, 0, 255).astype(np.uint8)
    height, width = source.shape[:2]
    if (width, height) != (object_record.image_width, object_record.image_height):
        raise ProtocolError("Image dimensions do not match the indexed object record")

    origin_x = math.floor(object_record.center_x_px - size_px / 2.0)
    origin_y = math.floor(object_record.center_y_px - size_px / 2.0)
    x_indices = _reflect_indices(origin_x, size_px, width)
    y_indices = _reflect_indices(origin_y, size_px, height)
    crop = source[np.ix_(y_indices, x_indices)].copy()
    polygon = _polygon_points(object_record.polygon_visible)
    local_polygon = [(x - origin_x, y - origin_y) for x, y in polygon]
    mask = object_mask(
        object_record,
        size_px=size_px,
        mask_supersampling=mask_supersampling,
    )
    return CropData(
        image=crop,
        mask=mask,
        origin_x=origin_x,
        origin_y=origin_y,
        center_x=object_record.center_x_px,
        center_y=object_record.center_y_px,
        polygon_local=tuple(value for point in local_polygon for value in point),
    )


def object_mask(
    object_record: ObjectRecord,
    *,
    size_px: int = 448,
    mask_supersampling: int = 4,
) -> np.ndarray:
    """Rasterize the fixed-crop OBB mask without loading image pixels."""

    if size_px <= 0 or size_px % 2:
        raise ProtocolError("Crop size must be a positive even integer")
    origin_x = math.floor(object_record.center_x_px - size_px / 2.0)
    origin_y = math.floor(object_record.center_y_px - size_px / 2.0)
    polygon = _polygon_points(object_record.polygon_visible)
    local_polygon = [(x - origin_x, y - origin_y) for x, y in polygon]
    return _rasterize_soft_mask(local_polygon, size_px, mask_supersampling)


def _crop_at_center(source: np.ndarray, center_x: int, center_y: int, size_px: int) -> np.ndarray:
    height, width = source.shape[:2]
    origin_x = math.floor(center_x - size_px / 2.0)
    origin_y = math.floor(center_y - size_px / 2.0)
    x_indices = _reflect_indices(origin_x, size_px, width)
    y_indices = _reflect_indices(origin_y, size_px, height)
    return source[np.ix_(y_indices, x_indices)].copy()


def sample_background_crop(
    image: np.ndarray | Image.Image,
    object_record: ObjectRecord,
    other_objects: list[ObjectRecord] | tuple[ObjectRecord, ...],
    *,
    size_px: int,
    rng: np.random.Generator,
    max_attempts: int = 128,
) -> BackgroundCrop:
    """Sample a same-image crop whose window avoids every indexed OBB box."""

    if isinstance(image, Image.Image):
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        source = np.asarray(image, dtype=np.uint8)
    height, width = source.shape[:2]
    half = size_px / 2.0
    forbidden: list[tuple[float, float, float, float]] = []
    for item in other_objects:
        xs = item.polygon_visible[::2]
        ys = item.polygon_visible[1::2]
        if xs and ys:
            forbidden.append((min(xs), min(ys), max(xs), max(ys)))
    for attempt in range(1, max_attempts + 1):
        center_x = int(rng.integers(0, max(1, width)))
        center_y = int(rng.integers(0, max(1, height)))
        left, top, right, bottom = center_x - half, center_y - half, center_x + half, center_y + half
        if any(
            left < box_right and right > box_left and top < box_bottom and bottom > box_top
            for box_left, box_top, box_right, box_bottom in forbidden
        ):
            continue
        return BackgroundCrop(_crop_at_center(source, center_x, center_y, size_px), center_x, center_y, attempt)
    raise ProtocolError("Could not sample a same-image object-free background crop")


def _blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return image.copy()
    return np.asarray(Image.fromarray(image, mode="RGB").filter(ImageFilter.GaussianBlur(radius=float(sigma_px))))


def _non_overlapping_patch_mask(
    size: int,
    *,
    target_fraction: float,
    patch_side_fraction: float,
    rng: np.random.Generator,
    allow_overlap: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.0 <= target_fraction <= 1.0:
        raise ProtocolError("Patch target fraction must be in [0, 1]")
    if patch_side_fraction <= 0.0 or patch_side_fraction > 1.0:
        raise ProtocolError("Patch side fraction must be in (0, 1]")
    patch_side = max(1, round(size * patch_side_fraction))
    if patch_side > size:
        raise ProtocolError("Patch side cannot exceed crop size")
    desired_pixels = round(size * size * target_fraction)
    patch_count = math.ceil(desired_pixels / (patch_side * patch_side)) if desired_pixels else 0
    mask = np.zeros((size, size), dtype=bool)
    if allow_overlap:
        placements = [
            (
                int(rng.integers(0, max(1, size - patch_side + 1))),
                int(rng.integers(0, max(1, size - patch_side + 1))),
            )
            for _ in range(patch_count)
        ]
    else:
        rows = math.ceil(size / patch_side)
        cols = math.ceil(size / patch_side)
        cells = [(row * patch_side, col * patch_side) for row in range(rows) for col in range(cols)]
        rng.shuffle(cells)
        placements = cells[:patch_count]
    placement_pixels = 0
    for y0, x0 in placements:
        placement_height = min(size, y0 + patch_side) - y0
        placement_width = min(size, x0 + patch_side) - x0
        placement_pixels += placement_height * placement_width
        mask[y0 : min(size, y0 + patch_side), x0 : min(size, x0 + patch_side)] = True
    nominal_placement_pixels = placement_pixels
    return mask, {
        "target_fraction": target_fraction,
        "patch_side_px": patch_side,
        "requested_pixels": desired_pixels,
        "actual_pixels": int(mask.sum()),
        "actual_fraction": float(mask.mean()),
        "allow_overlap": bool(allow_overlap),
        "overlap_policy": "union_of_random_placements" if allow_overlap else "disjoint_grid_cells",
        "placement_count": patch_count,
        "nominal_placement_pixels": nominal_placement_pixels,
        "overlap_pixels": max(0, nominal_placement_pixels - int(mask.sum())),
        "non_overlapping_grid": not allow_overlap,
    }


def _actual_changed_mask(source: np.ndarray, degraded: np.ndarray) -> np.ndarray:
    """Return pixels whose quantized RGB value actually differs from the source."""

    return np.any(source != degraded, axis=2)


def apply_perturbation(
    image: np.ndarray,
    kind: str,
    *,
    rng: np.random.Generator,
    sigma_px: float = 16.0,
    noise_sigma: float = 0.12,
    patch_fraction: float = 0.35,
    patch_side_fraction: float = 0.10,
    allow_overlap: bool = False,
    brightness_factor: float = 0.55,
) -> PerturbationResult:
    """Apply a registered perturbation and return the exact quantized changed-pixel mask.

    ``changed_mask`` is not the requested patch region: it is ``any(degraded != source,
    axis=2)`` after uint8 quantization.  This distinction matters for constant pixels,
    clipping, and brightness factors that leave some pixels unchanged.
    """

    source = np.asarray(image, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3 or source.shape[0] != source.shape[1]:
        raise ProtocolError("Perturbations require a square RGB uint8 crop")
    if kind == "blur":
        degraded = _blur(source, sigma_px)
        parameters = {"kind": kind, "sigma_px": sigma_px}
    elif kind == "gaussian_noise":
        noise = rng.normal(0.0, noise_sigma, size=source.shape).astype(np.float32)
        degraded = np.clip(source.astype(np.float32) / 255.0 + noise, 0.0, 1.0)
        degraded = np.rint(degraded * 255.0).astype(np.uint8)
        parameters = {"kind": kind, "sigma": noise_sigma, "clip_min": 0.0, "clip_max": 1.0}
    elif kind == "patch_mask":
        requested_mask, parameters = _non_overlapping_patch_mask(
            source.shape[0],
            target_fraction=patch_fraction,
            patch_side_fraction=patch_side_fraction,
            rng=rng,
            allow_overlap=allow_overlap,
        )
        fill = np.rint(source.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
        degraded = source.copy()
        degraded[requested_mask] = fill
        parameters = {
            "kind": kind,
            **parameters,
            "fill_rgb": fill.tolist(),
            "requested_region_pixels": int(requested_mask.sum()),
            "requested_region_fraction": float(requested_mask.mean()),
        }
    elif kind == "brightness":
        if brightness_factor < 0:
            raise ProtocolError("Brightness factor must be non-negative")
        degraded = np.clip(source.astype(np.float32) * brightness_factor, 0.0, 255.0).astype(np.uint8)
        parameters = {"kind": kind, "factor": brightness_factor}
    else:
        raise ProtocolError(f"Unknown perturbation: {kind}")
    degraded = np.asarray(degraded, dtype=np.uint8)
    changed = _actual_changed_mask(source, degraded)
    parameters.update(
        {
            "changed_mask_semantics": "actual_uint8_rgb_difference_any_channel",
            "actual_changed_pixels": int(changed.sum()),
            "actual_changed_fraction": float(changed.mean()),
        }
    )
    return PerturbationResult(degraded=degraded, changed_mask=changed, parameters=parameters)


def _blend(mask: np.ndarray, foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    alpha = np.asarray(mask, dtype=np.float32)[..., None]
    return np.rint(alpha * foreground.astype(np.float32) + (1.0 - alpha) * background.astype(np.float32)).clip(0, 255).astype(
        np.uint8
    )


def build_paired_inputs(
    crop: CropData,
    perturbation: PerturbationResult,
) -> PairedInputs:
    """Build the 2x2 foreground/background factorial without an additivity assumption."""

    clean = crop.image
    degraded = perturbation.degraded
    mask = crop.mask
    return PairedInputs(
        foreground_clear_background_clean=clean.copy(),
        foreground_degraded_background_clean=_blend(mask, degraded, clean),
        foreground_clear_background_degraded=_blend(mask, clean, degraded),
        foreground_degraded_background_degraded=degraded.copy(),
        mask=mask.copy(),
        perturbation=perturbation,
    )


def _binary_mask(mask: np.ndarray) -> Image.Image:
    values = np.where(mask >= 0.5, 255, 0).astype(np.uint8)
    return Image.fromarray(values, mode="L")


def erode_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px < 0:
        raise ProtocolError("Mask erosion radius must be non-negative")
    if radius_px == 0:
        return mask.copy()
    size = 2 * radius_px + 1
    return np.asarray(_binary_mask(mask).filter(ImageFilter.MinFilter(size)), dtype=np.float32) / 255.0


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px < 0:
        raise ProtocolError("Mask dilation radius must be non-negative")
    if radius_px == 0:
        return mask.copy()
    size = 2 * radius_px + 1
    return np.asarray(_binary_mask(mask).filter(ImageFilter.MaxFilter(size)), dtype=np.float32) / 255.0


def derive_control_masks(mask: np.ndarray, radius_px: int = 3) -> dict[str, np.ndarray]:
    """Return registered OBB controls in the same crop coordinate frame."""

    binary = (mask >= 0.5).astype(np.float32)
    eroded = erode_mask(binary, radius_px)
    dilated = dilate_mask(binary, radius_px)
    return {
        "true_obb": mask.copy(),
        "background_only": 1.0 - mask,
        "boundary_only": np.clip(binary - eroded, 0.0, 1.0),
        "interior_eroded": eroded,
        "mask_eroded": eroded,
        "mask_dilated": dilated,
    }


def _translated_mask(mask: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    result = np.zeros_like(mask)
    source_y0 = max(0, -shift_y)
    source_y1 = min(mask.shape[0], mask.shape[0] - shift_y)
    source_x0 = max(0, -shift_x)
    source_x1 = min(mask.shape[1], mask.shape[1] - shift_x)
    target_y0 = max(0, shift_y)
    target_y1 = target_y0 + max(0, source_y1 - source_y0)
    target_x0 = max(0, shift_x)
    target_x1 = target_x0 + max(0, source_x1 - source_x0)
    if source_y1 > source_y0 and source_x1 > source_x0:
        result[target_y0:target_y1, target_x0:target_x1] = mask[source_y0:source_y1, source_x0:source_x1]
    return result


def _boundary_target_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return a small deterministic set of in-crop boundary placements.

    The E00 feasibility audit and the runtime sampler share this candidate set.
    It prevents a feasible control from depending on a lucky random draw while
    keeping the audit inexpensive for full-resolution manifests.
    """

    ys, xs = np.where(mask >= 0.5)
    if len(xs) == 0:
        raise ProtocolError("Cannot sample geometry control from an empty mask")
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    max_target_x = mask.shape[1] - width
    max_target_y = mask.shape[0] - height
    if max_target_x < 0 or max_target_y < 0:
        raise ProtocolError("Geometry control shape cannot fit inside the crop")
    x_values = sorted({0, max_target_x, round(max_target_x / 2)})
    y_values = sorted({0, max_target_y, round(max_target_y / 2)})
    return [
        (target_min_x - min_x, target_min_y - min_y)
        for target_min_y in y_values
        for target_min_x in x_values
    ]


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_binary = left >= 0.5
    right_binary = right >= 0.5
    union = np.logical_or(left_binary, right_binary).sum()
    return float(np.logical_and(left_binary, right_binary).sum() / union) if union else 0.0


def geometry_control_feasibility(
    mask: np.ndarray,
    *,
    min_iou: float = 0.0,
    max_iou: float = 0.05,
    max_attempts: int = 128,
) -> dict[str, Any]:
    """Audit whether the registered boundary-placement control is constructible."""

    if min_iou < 0.0 or max_iou < min_iou or max_iou > 1.0:
        raise ProtocolError("Geometry-control IoU bounds are invalid")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 0:
        raise ProtocolError("Geometry-control max_attempts must be a non-negative integer")
    all_candidates = _boundary_target_positions(np.asarray(mask, dtype=np.float32))
    candidates = all_candidates[:max_attempts]
    best_iou = float("inf")
    best_shift: tuple[int, int] | None = None
    feasible_shift: tuple[int, int] | None = None
    for shift_x, shift_y in candidates:
        candidate = _translated_mask(mask, shift_x, shift_y)
        iou = _mask_iou(mask, candidate)
        if iou < best_iou:
            best_iou = iou
            best_shift = (shift_x, shift_y)
        if min_iou <= iou <= max_iou and feasible_shift is None:
            feasible_shift = (shift_x, shift_y)
    return {
        "feasible": feasible_shift is not None,
        "policy": "boundary_grid_3x3",
        "candidate_count": len(all_candidates),
        "attempted_candidate_count": len(candidates),
        "all_candidates_within_attempt_budget": max_attempts >= len(all_candidates),
        "best_iou": None if best_shift is None else best_iou,
        "best_shift": None if best_shift is None else list(best_shift),
        "feasible_shift": None if feasible_shift is None else list(feasible_shift),
        "min_iou": min_iou,
        "max_iou": max_iou,
    }


def make_shifted_control_mask(
    mask: np.ndarray,
    *,
    shift_fraction: float,
    angle_rad: float,
    direction: int,
) -> np.ndarray:
    ys, xs = np.where(mask >= 0.5)
    if len(xs) == 0:
        raise ProtocolError("Cannot shift an empty mask")
    object_span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    distance = max(1, round(object_span * shift_fraction))
    dx = round(math.cos(angle_rad + direction * math.pi / 2.0) * distance)
    dy = round(math.sin(angle_rad + direction * math.pi / 2.0) * distance)
    return _translated_mask(mask, dx, dy)


def make_random_geometry_matched_mask(
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
    min_iou: float = 0.0,
    max_iou: float = 0.05,
    max_attempts: int = 128,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Translate the exact OBB shape to a registered in-crop location with bounded overlap."""

    ys, xs = np.where(mask >= 0.5)
    if len(xs) == 0:
        raise ProtocolError("Cannot sample geometry control from an empty mask")
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    boundary_candidates = _boundary_target_positions(mask)[:max_attempts]
    rng.shuffle(boundary_candidates)
    for candidate_index, (shift_x, shift_y) in enumerate(boundary_candidates, start=1):
        candidate = _translated_mask(mask, shift_x, shift_y)
        iou = _mask_iou(mask, candidate)
        if min_iou <= iou <= max_iou:
            return candidate, {
                "attempts": candidate_index,
                "bbox_width_px": width,
                "bbox_height_px": height,
                "iou": iou,
                "placement_policy": "boundary_grid_3x3",
            }
    # The boundary grid is the registered placement policy.  Do not add a
    # random fallback here: E00 must be able to prove that runtime construction
    # uses the same finite control family it audited.
    raise ProtocolError("Could not sample a geometry-matched mask within the registered overlap bound")


def replace_mask_pair(crop: CropData, mask: np.ndarray) -> CropData:
    if mask.shape != crop.mask.shape:
        raise ProtocolError("Replacement mask shape does not match crop")
    return replace(crop, mask=np.asarray(mask, dtype=np.float32).copy())


def pair_pixel_invariants(pair: PairedInputs, *, tolerance: float = 0.0) -> dict[str, Any]:
    if tolerance < 0.0:
        raise ProtocolError("Pixel tolerance must be non-negative")
    clean_bg = pair.foreground_clear_background_clean
    degraded_bg = pair.foreground_clear_background_degraded
    full_degraded = pair.foreground_degraded_background_degraded
    mask_threshold = 1.0e-6
    outside = pair.mask <= mask_threshold
    inside = pair.mask >= 1.0 - mask_threshold
    expected_clear_degraded = _blend(pair.mask, clean_bg, full_degraded)
    changed_mask = np.asarray(pair.perturbation.changed_mask, dtype=bool)
    actual_changed_mask = _actual_changed_mask(clean_bg, full_degraded)
    blend_tolerance = float(tolerance)
    return {
        "outside_mask_equal_clear_vs_degraded": bool(
            np.allclose(
                pair.foreground_clear_background_degraded[outside],
                full_degraded[outside],
                atol=tolerance,
                rtol=0.0,
            )
        ),
        "inside_mask_equal_clean_clear": bool(
            np.allclose(
                pair.foreground_clear_background_degraded[inside],
                pair.foreground_clear_background_clean[inside],
                atol=tolerance,
                rtol=0.0,
            )
        ),
        "blend_formula_exact": bool(
            np.allclose(
                pair.foreground_clear_background_degraded,
                expected_clear_degraded,
                atol=blend_tolerance,
                rtol=0.0,
            )
        ),
        "clean_background_changed_pixels": int(np.abs(clean_bg.astype(np.int16) - degraded_bg.astype(np.int16)).sum()),
        "mask_area_fraction": float(pair.mask.mean()),
        "perturbation_changed_fraction": float(changed_mask.mean()),
        "perturbation_changed_pixels": int(changed_mask.sum()),
        "changed_mask_consistent_with_uint8_difference": bool(np.array_equal(changed_mask, actual_changed_mask)),
        "changed_mask_semantics": "actual_uint8_rgb_difference_any_channel",
        "pixel_tolerance": float(tolerance),
        "pixel_tolerance_semantics": "absolute_atol_only_rtol_zero",
        "blend_rounding_atol": blend_tolerance,
        "blend_rounding_policy": "expected_formula_is_quantized_uint8_with_np_rint",
        "mask_threshold": mask_threshold,
    }
