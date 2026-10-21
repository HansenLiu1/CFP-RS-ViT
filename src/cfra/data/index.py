"""Clean-room indexing of the three supported raw OBB datasets.

The index is intentionally independent of any historical CFP artifact.  It stores
both the original annotation coordinates and the visible polygon used by the pixel
protocol so border-clipped annotations remain auditable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from cfra.provenance import sha256_file

ROLE_NAMES = ("direction_fit", "readout_fit", "validation", "confirmation")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class DatasetIndexError(RuntimeError):
    """Raised when a raw dataset cannot satisfy the indexing contract."""


@dataclass(frozen=True)
class ClassInfo:
    class_id: int
    name: str
    coarse: str


@dataclass(frozen=True)
class ImageRecord:
    dataset_id: str
    split: str
    source_image_id: str
    content_group_id: str
    image_relpath: str
    label_relpath: str | None
    width: int
    height: int
    image_sha256: str
    label_sha256: str | None
    status: str
    status_reason: str | None = None
    exclusion_category: str | None = None


@dataclass(frozen=True)
class ObjectRecord:
    dataset_id: str
    split: str
    source_image_id: str
    content_group_id: str
    object_id: str
    image_relpath: str
    label_relpath: str
    class_id: int
    class_name: str
    coarse_name: str
    polygon_raw: tuple[float, ...]
    polygon_visible: tuple[float, ...]
    image_width: int
    image_height: int
    difficulty: int | None
    raw_coordinate_out_of_bounds: bool
    visible_area_px2: float
    visible_bbox_width_px: float
    visible_bbox_height_px: float
    visible_side_long_px: float
    visible_side_short_px: float
    center_x_px: float
    center_y_px: float
    orientation_rad: float
    eligible: bool
    exclusion_reason: str | None = None
    role: str | None = None
    exclusion_category: str | None = None


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    root: str
    class_registry_sha256: str
    image_count: int
    labeled_image_count: int
    object_count: int
    eligible_object_count: int
    images: tuple[ImageRecord, ...]
    objects: tuple[ObjectRecord, ...]


def _stable_int(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _as_tuple(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        * 0.5
    )


def _clip_polygon(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[float, float]]:
    """Clip a polygon against the image rectangle using Sutherland-Hodgman."""

    bounds = (
        ("left", 0.0),
        ("right", float(width - 1)),
        ("top", 0.0),
        ("bottom", float(height - 1)),
    )

    def inside(point: tuple[float, float], edge: str, value: float) -> bool:
        x, y = point
        return {
            "left": x >= value,
            "right": x <= value,
            "top": y >= value,
            "bottom": y <= value,
        }[edge]

    def intersection(
        start: tuple[float, float], end: tuple[float, float], edge: str, value: float
    ) -> tuple[float, float]:
        x1, y1 = start
        x2, y2 = end
        if edge in {"left", "right"}:
            if x2 == x1:
                return value, y1
            ratio = (value - x1) / (x2 - x1)
            return value, y1 + ratio * (y2 - y1)
        if y2 == y1:
            return x1, value
        ratio = (value - y1) / (y2 - y1)
        return x1 + ratio * (x2 - x1), value

    output = points
    for edge, value in bounds:
        if not output:
            break
        clipped: list[tuple[float, float]] = []
        previous = output[-1]
        previous_inside = inside(previous, edge, value)
        for current in output:
            current_inside = inside(current, edge, value)
            if current_inside != previous_inside:
                clipped.append(intersection(previous, current, edge, value))
            if current_inside:
                clipped.append(current)
            previous = current
            previous_inside = current_inside
        output = clipped
    return output


def _polygon_metrics(
    points: list[tuple[float, float]],
) -> tuple[float, float, float, float, float, float, float, float]:
    if len(points) < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    edges = [
        math.hypot(
            points[(index + 1) % len(points)][0] - points[index][0],
            points[(index + 1) % len(points)][1] - points[index][1],
        )
        for index in range(len(points))
    ]
    side_long = max(edges[0], edges[1], edges[2], edges[3]) if len(edges) >= 4 else max(edges)
    side_short = min(edges[0], edges[1], edges[2], edges[3]) if len(edges) >= 4 else min(edges)
    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    orientation = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
    return _polygon_area(points), width, height, side_long, side_short, center_x, center_y, orientation


def load_class_registry(path: Path | str) -> dict[str, dict[int, ClassInfo]]:
    """Load ``dataset_classes.json`` from a file path or the registry root.

    Callers historically passed either the registry directory or the JSON file
    itself; both are accepted so a wrong-but-plausible path cannot silently
    degrade the E08 text baseline into an empty class registry.
    """

    registry_path = Path(path).expanduser().resolve(strict=True)
    if registry_path.is_dir():
        registry_path = registry_path / "dataset_classes.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    result: dict[str, dict[int, ClassInfo]] = {}
    for dataset_id, dataset_payload in payload.items():
        classes = dataset_payload.get("classes")
        if not isinstance(classes, list) or not classes:
            raise DatasetIndexError(f"Class registry has no classes for {dataset_id}")
        indexed: dict[int, ClassInfo] = {}
        for item in classes:
            class_id = int(item["id"])
            if class_id in indexed:
                raise DatasetIndexError(f"Duplicate class ID {dataset_id}:{class_id}")
            indexed[class_id] = ClassInfo(class_id, str(item["name"]), str(item["coarse"]))
        result[dataset_id] = indexed
    return result


def _find_image(image_dir: Path, stem: str) -> Path | None:
    candidates = sorted(path for path in image_dir.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    return candidates[0] if candidates else None


def _parse_dota_label(path: Path) -> list[tuple[Any, tuple[float, ...], int | None]]:
    parsed: list[tuple[Any, tuple[float, ...], int | None]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        parts = line.strip().split()
        if len(parts) < 10 or ":" in parts[0]:
            continue
        try:
            coordinates = tuple(float(value) for value in parts[:8])
            class_name = parts[8]
            difficulty = int(float(parts[9]))
        except (TypeError, ValueError) as exc:
            raise DatasetIndexError(f"Invalid DOTA label {path}:{line_number}") from exc
        parsed.append((class_name, coordinates, difficulty))
    return parsed


def _parse_normalized_label(path: Path) -> list[tuple[int, tuple[float, ...], int | None]]:
    parsed: list[tuple[int, tuple[float, ...], int | None]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 9:
            raise DatasetIndexError(f"Invalid normalized OBB label {path}:{line_number}")
        try:
            class_id = int(parts[0])
            coordinates = tuple(float(value) for value in parts[1:9])
        except (TypeError, ValueError) as exc:
            raise DatasetIndexError(f"Invalid normalized OBB label {path}:{line_number}") from exc
        parsed.append((class_id, coordinates, None))
    return parsed


def _parse_label(dataset_id: str, path: Path) -> list[tuple[Any, tuple[float, ...], int | None]]:
    if dataset_id == "dota":
        return _parse_dota_label(path)
    return _parse_normalized_label(path)


def _raw_to_pixel(dataset_id: str, coordinates: tuple[float, ...], width: int, height: int) -> list[tuple[float, float]]:
    points = list(zip(coordinates[::2], coordinates[1::2]))
    if dataset_id in {"dior", "fair1m"}:
        return [(x * width, y * height) for x, y in points]
    return points


def _class_info(
    dataset_id: str,
    raw_class: Any,
    registry: dict[str, dict[int, ClassInfo]],
) -> ClassInfo:
    classes = registry[dataset_id]
    if dataset_id == "dota":
        by_name = {item.name: item for item in classes.values()}
        try:
            return by_name[str(raw_class)]
        except KeyError as exc:
            raise DatasetIndexError(f"Unknown DOTA class: {raw_class!r}") from exc
    try:
        return classes[int(raw_class)]
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetIndexError(f"Unknown {dataset_id} class: {raw_class!r}") from exc


def _image_status(width: int, height: int, maximum_pixels: int) -> tuple[str, str | None]:
    if width <= 0 or height <= 0:
        return "invalid", "non_positive_dimensions"
    if width * height > maximum_pixels:
        return "excluded", "image_pixels_exceed_limit"
    return "ok", None


def _exclusion_category(reason: str | None) -> str | None:
    """Map implementation reasons to the stable audit categories used by E00."""

    return {
        "minimum_object_side": "minimum_side",
        "object_side_exceeds_fixed_crop": "oversized_crop",
        "image_pixels_exceed_limit": "image_pixel_limit",
        "smoke_class_scope": "class_scope",
        "difficulty_exceeds_limit": "difficulty_limit",
        "empty_visible_polygon": "empty_visible_polygon",
        "geometry_control_infeasible": "geometry_control_infeasible",
    }.get(reason)


EXCLUSION_CATEGORIES = (
    "minimum_side",
    "oversized_crop",
    "image_pixel_limit",
    "geometry_control_infeasible",
)


def _manifest_relative_path(root: Path, relative: str, description: str) -> Path:
    """Resolve a manifest-relative path before checking containment."""

    normalized = str(relative).replace("\\", "/")
    target = (root / Path(*normalized.split("/"))).resolve()
    if target == root or root not in target.parents:
        raise DatasetIndexError(f"{description} escapes dataset root: {relative}")
    return target


def _object_record(
    *,
    dataset_id: str,
    split: str,
    source_image_id: str,
    content_group_id: str,
    image_relpath: str,
    label_relpath: str,
    line_index: int,
    raw_class: Any,
    coordinates: tuple[float, ...],
    difficulty: int | None,
    width: int,
    height: int,
    registry: dict[str, dict[int, ClassInfo]],
    minimum_side_px: float,
    maximum_side_px: float,
    maximum_difficulty: int,
) -> ObjectRecord:
    class_info = _class_info(dataset_id, raw_class, registry)
    raw_points = _raw_to_pixel(dataset_id, coordinates, width, height)
    visible_points = _clip_polygon(raw_points, width, height)
    metrics = _polygon_metrics(visible_points)
    area, bbox_width, bbox_height, side_long, side_short, center_x, center_y, orientation = metrics
    out_of_bounds = any(
        x < 0.0 or x > width - 1 or y < 0.0 or y > height - 1 for x, y in raw_points
    )
    reason: str | None = None
    if len(visible_points) < 3 or area <= 0.0:
        reason = "empty_visible_polygon"
    elif difficulty is not None and difficulty > maximum_difficulty:
        reason = "difficulty_exceeds_limit"
    elif side_short < minimum_side_px:
        reason = "minimum_object_side"
    elif side_long > maximum_side_px:
        reason = "object_side_exceeds_fixed_crop"
    return ObjectRecord(
        dataset_id=dataset_id,
        split=split,
        source_image_id=source_image_id,
        content_group_id=content_group_id,
        object_id=f"{source_image_id}:{line_index:06d}",
        image_relpath=image_relpath,
        label_relpath=label_relpath,
        class_id=class_info.class_id,
        class_name=class_info.name,
        coarse_name=class_info.coarse,
        polygon_raw=_as_tuple(coordinates),
        polygon_visible=_as_tuple(value for point in visible_points for value in point),
        image_width=width,
        image_height=height,
        difficulty=difficulty,
        raw_coordinate_out_of_bounds=out_of_bounds,
        visible_area_px2=area,
        visible_bbox_width_px=bbox_width,
        visible_bbox_height_px=bbox_height,
        visible_side_long_px=side_long,
        visible_side_short_px=side_short,
        center_x_px=center_x,
        center_y_px=center_y,
        orientation_rad=orientation,
        eligible=reason is None,
        exclusion_reason=reason,
        exclusion_category=_exclusion_category(reason),
    )


def _iter_split_dirs(root: Path) -> Iterator[tuple[str, Path, Path | None]]:
    for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        image_dir = split_dir / "Images"
        label_dir = split_dir / "Labels"
        if image_dir.is_dir():
            yield split_dir.name, image_dir, label_dir if label_dir.is_dir() else None


def build_dataset_manifest(
    dataset_id: str,
    root: Path | str,
    *,
    class_registry: dict[str, dict[int, ClassInfo]],
    class_registry_sha256: str,
    minimum_side_px: float,
    maximum_side_px: float,
    maximum_image_pixels: int,
    maximum_difficulty: int,
    max_classes: int | None = None,
) -> DatasetManifest:
    dataset_root = Path(root).expanduser().resolve(strict=True)
    if dataset_id not in class_registry:
        raise DatasetIndexError(f"No class registry for dataset {dataset_id}")
    allowed_classes = set(sorted(class_registry[dataset_id])[:max_classes]) if max_classes else None
    images: list[ImageRecord] = []
    objects: list[ObjectRecord] = []
    image_paths: list[tuple[str, Path, Path | None]] = []
    for split, image_dir, label_dir in _iter_split_dirs(dataset_root):
        for image_path in sorted(image_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                label_path = label_dir / f"{image_path.stem}.txt" if label_dir else None
                image_paths.append((split, image_path, label_path if label_path and label_path.exists() else None))

    for split, image_path, label_path in tqdm(image_paths, desc=f"E00 {dataset_id} images", unit="image"):
        relative_image = image_path.relative_to(dataset_root).as_posix()
        relative_label = label_path.relative_to(dataset_root).as_posix() if label_path else None
        source_image_id = f"{dataset_id}:{split}:{image_path.stem}"
        image_hash = sha256_file(image_path)
        label_hash = sha256_file(label_path) if label_path else None
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            status, reason = _image_status(width, height, maximum_image_pixels)
        except Exception as exc:
            width, height = 0, 0
            status, reason = "invalid", f"image_read_error:{type(exc).__name__}"
        content_group = f"content:{image_hash}"
        image_record = ImageRecord(
            dataset_id=dataset_id,
            split=split,
            source_image_id=source_image_id,
            content_group_id=content_group,
            image_relpath=relative_image,
            label_relpath=relative_label,
            width=width,
            height=height,
            image_sha256=image_hash,
            label_sha256=label_hash,
            status=status,
            status_reason=reason,
            exclusion_category=_exclusion_category(reason),
        )
        images.append(image_record)
        if status != "ok" or label_path is None:
            continue
        parsed = _parse_label(dataset_id, label_path)
        for line_index, (raw_class, coordinates, difficulty) in enumerate(parsed):
            record = _object_record(
                dataset_id=dataset_id,
                split=split,
                source_image_id=source_image_id,
                content_group_id=content_group,
                image_relpath=relative_image,
                label_relpath=relative_label or "",
                line_index=line_index,
                raw_class=raw_class,
                coordinates=coordinates,
                difficulty=difficulty,
                width=width,
                height=height,
                registry=class_registry,
                minimum_side_px=minimum_side_px,
                maximum_side_px=maximum_side_px,
                maximum_difficulty=maximum_difficulty,
            )
            if allowed_classes is not None and record.class_id not in allowed_classes:
                record = replace(
                    record,
                    eligible=False,
                    exclusion_reason="smoke_class_scope",
                    exclusion_category="class_scope",
                )
            objects.append(record)
    return DatasetManifest(
        dataset_id=dataset_id,
        root=str(dataset_root),
        class_registry_sha256=class_registry_sha256,
        image_count=len(images),
        labeled_image_count=sum(image.label_relpath is not None for image in images),
        object_count=len(objects),
        eligible_object_count=sum(object.eligible for object in objects),
        images=tuple(images),
        objects=tuple(objects),
    )


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def manifest_exclusion_stats(manifest: DatasetManifest) -> dict[str, Any]:
    """Return dataset/class exclusion counts without collapsing protocol reasons."""

    def empty_bucket() -> dict[str, Any]:
        return {
            "total_objects": 0,
            "eligible_objects": 0,
            "excluded_objects": 0,
            "total_count": 0,
            "excluded_count": 0,
            "excluded_fraction": 0.0,
            "affected_class_count": 0,
            "affected_source_image_count": 0,
            "by_reason": {},
            "by_category": {},
        }

    def add(bucket: dict[str, Any], item: ObjectRecord) -> None:
        bucket["total_objects"] += 1
        bucket["total_count"] += 1
        if item.eligible:
            bucket["eligible_objects"] += 1
            return
        bucket["excluded_objects"] += 1
        bucket["excluded_count"] += 1
        if item.exclusion_reason:
            bucket["by_reason"][item.exclusion_reason] = bucket["by_reason"].get(item.exclusion_reason, 0) + 1
        if item.exclusion_category:
            bucket["by_category"][item.exclusion_category] = bucket["by_category"].get(item.exclusion_category, 0) + 1

    by_class: dict[str, dict[str, Any]] = {}
    for item in manifest.objects:
        key = str(item.class_id)
        bucket = by_class.setdefault(key, {"class_id": item.class_id, "class_name": item.class_name, **empty_bucket()})
        add(bucket, item)

    for bucket in by_class.values():
        bucket["excluded_fraction"] = (
            bucket["excluded_count"] / bucket["total_count"] if bucket["total_count"] else 0.0
        )
        bucket["affected_class_count"] = int(bucket["excluded_count"] > 0)
        bucket["affected_source_image_count"] = len(
            {
                item.source_image_id
                for item in manifest.objects
                if item.class_id == bucket["class_id"] and not item.eligible
            }
        )

    image_by_reason = Counter(image.status_reason for image in manifest.images if image.status_reason)
    image_by_category = Counter(image.exclusion_category for image in manifest.images if image.exclusion_category)
    object_by_reason = Counter(item.exclusion_reason for item in manifest.objects if item.exclusion_reason)
    object_by_category = Counter(item.exclusion_category for item in manifest.objects if item.exclusion_category)
    excluded_objects = [item for item in manifest.objects if not item.eligible]
    excluded_images = [image for image in manifest.images if image.status != "ok"]
    excluded_image_ids = {image.source_image_id for image in excluded_images}
    object_summary = {
        "total_count": len(manifest.objects),
        "excluded_count": len(excluded_objects),
        "excluded_fraction": len(excluded_objects) / len(manifest.objects) if manifest.objects else 0.0,
        "affected_class_count": len({item.class_id for item in excluded_objects}),
        "affected_source_image_count": len({item.source_image_id for item in excluded_objects}),
        "by_reason": dict(object_by_reason),
    }
    image_summary = {
        "total_count": len(manifest.images),
        "excluded_count": len(excluded_images),
        "excluded_fraction": len(excluded_images) / len(manifest.images) if manifest.images else 0.0,
        "affected_class_count": len(
            {
                item.class_id
                for item in manifest.objects
                if item.source_image_id in excluded_image_ids
            }
        ),
        "affected_source_image_count": len({image.source_image_id for image in excluded_images}),
        "by_reason": dict(image_by_reason),
    }
    return {
        "dataset_id": manifest.dataset_id,
        "images": {
            **image_summary,
            "total": len(manifest.images),
            "ok": sum(image.status == "ok" for image in manifest.images),
            "excluded_or_invalid": sum(image.status != "ok" for image in manifest.images),
            "by_reason": dict(image_by_reason),
            "by_category": {
                category: int(image_by_category.get(category, 0))
                for category in EXCLUSION_CATEGORIES
            } | {
                category: int(count)
                for category, count in image_by_category.items()
                if category not in EXCLUSION_CATEGORIES
            },
        },
        "objects": {
            **object_summary,
            "total": len(manifest.objects),
            "eligible": sum(item.eligible for item in manifest.objects),
            "excluded": sum(not item.eligible for item in manifest.objects),
            "by_reason": dict(object_by_reason),
            "by_category": {
                category: int(object_by_category.get(category, 0))
                for category in EXCLUSION_CATEGORIES
            } | {
                category: int(count)
                for category, count in object_by_category.items()
                if category not in EXCLUSION_CATEGORIES
            },
        },
        "by_class": by_class,
        "by_dataset_class": {manifest.dataset_id: by_class},
    }


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_schema_manifests(
    manifests: dict[str, DatasetManifest],
    output_root: Path | str,
    *,
    model_rows: Iterable[dict[str, Any]] | None = None,
    role_by_image: dict[str, str] | None = None,
    seed: int | None = None,
    assignment_version: str = "e00-unassigned",
) -> None:
    """Write the five CSV artifacts documented in ``data/manifests/README.md``."""

    root = Path(output_root).expanduser().resolve()
    role_by_image = role_by_image or {}
    source_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for dataset_id in sorted(manifests):
        manifest = manifests[dataset_id]
        objects_by_image = Counter(item.source_image_id for item in manifest.objects)
        eligible_objects_by_image = Counter(item.source_image_id for item in manifest.objects if item.eligible)
        dataset_root = Path(manifest.root).expanduser().resolve(strict=True)
        for image in manifest.images:
            image_file = image_path(manifest, image)
            source_rows.append(
                {
                    "dataset": dataset_id,
                    "relative_path": image.image_relpath,
                    "file_type": "image",
                    "size_bytes": image_file.stat().st_size,
                    "sha256": image.image_sha256,
                    "annotation_pair": image.label_relpath or "",
                    "source_image_id": image.source_image_id,
                    "split": image.split,
                }
            )
            if image.label_relpath:
                label_path = _manifest_relative_path(dataset_root, image.label_relpath, "Manifest label path")
                source_rows.append(
                    {
                        "dataset": dataset_id,
                        "relative_path": image.label_relpath,
                        "file_type": "label",
                        "size_bytes": label_path.stat().st_size,
                        "sha256": image.label_sha256 or "",
                        "annotation_pair": image.image_relpath,
                        "source_image_id": image.source_image_id,
                        "split": image.split,
                    }
                )
            role = role_by_image.get(
                image.source_image_id,
                role_by_image.get(f"{dataset_id}:{image.source_image_id}", ""),
            )
            image_rows.append(
                {
                    "dataset": dataset_id,
                    "image_id": image.source_image_id,
                    "content_group_id": image.content_group_id,
                    "split": image.split,
                    "relative_path": image.image_relpath,
                    "label_relpath": image.label_relpath or "",
                    "width": image.width,
                    "height": image.height,
                    "image_sha256": image.image_sha256,
                    "label_sha256": image.label_sha256 or "",
                    "status": image.status,
                    "status_reason": image.status_reason or "",
                    "exclusion_category": image.exclusion_category or "",
                    "role": role,
                    "object_count": objects_by_image[image.source_image_id],
                    "eligible_object_count": eligible_objects_by_image[image.source_image_id],
                }
            )
            split_rows.append(
                {
                    "dataset": dataset_id,
                    "image_id": image.source_image_id,
                    "content_group_id": image.content_group_id,
                    "role": role,
                    "seed": "" if seed is None else seed,
                    "assignment_version": assignment_version,
                }
            )
        for item in manifest.objects:
            object_rows.append(
                {
                    "dataset": dataset_id,
                    "image_id": item.source_image_id,
                    "object_id": item.object_id,
                    "split": item.split,
                    "relative_path": item.image_relpath,
                    "class_id": item.class_id,
                    "class_name": item.class_name,
                    "coarse_name": item.coarse_name,
                    "polygon_raw": json.dumps(item.polygon_raw, separators=(",", ":")),
                    "polygon_visible": json.dumps(item.polygon_visible, separators=(",", ":")),
                    "visible_area_px2": item.visible_area_px2,
                    "visible_side_long_px": item.visible_side_long_px,
                    "visible_side_short_px": item.visible_side_short_px,
                    "orientation_rad": item.orientation_rad,
                    "eligible": item.eligible,
                    "exclusion_reason": item.exclusion_reason or "",
                    "exclusion_category": item.exclusion_category or "",
                    "role": role_by_image.get(
                        item.source_image_id,
                        role_by_image.get(f"{dataset_id}:{item.source_image_id}", item.role or ""),
                    ),
                }
            )

    _write_csv(
        root / "source_files.csv",
        ["dataset", "relative_path", "file_type", "size_bytes", "sha256", "annotation_pair", "source_image_id", "split"],
        source_rows,
    )
    _write_csv(
        root / "images.csv",
        [
            "dataset", "image_id", "content_group_id", "split", "relative_path", "label_relpath", "width", "height",
            "image_sha256", "label_sha256", "status", "status_reason", "exclusion_category", "role", "object_count",
            "eligible_object_count",
        ],
        image_rows,
    )
    _write_csv(
        root / "objects.csv",
        [
            "dataset", "image_id", "object_id", "split", "relative_path", "class_id", "class_name", "coarse_name",
            "polygon_raw", "polygon_visible", "visible_area_px2", "visible_side_long_px", "visible_side_short_px",
            "orientation_rad", "eligible", "exclusion_reason", "exclusion_category", "role",
        ],
        object_rows,
    )
    _write_csv(
        root / "splits.csv",
        ["dataset", "image_id", "content_group_id", "role", "seed", "assignment_version"],
        split_rows,
    )
    # E01 rewrites the data/split tables after E00. Preserve the E00 model
    # identity table when that later call does not carry model rows.
    models_path = root / "models.csv"
    if model_rows is not None or not models_path.is_file():
        _write_csv(
            models_path,
            [
                "model_id", "asset_id", "revision", "checkpoint_sha256", "loader_config_sha256", "asset_fingerprint",
                "path", "weight_path", "representation_id", "dimension", "pooling", "projection", "text_shared_space",
                "shared_space_id",
                "encoder_input_size",
            ],
            model_rows or (),
        )


def write_dataset_manifest(manifest: DatasetManifest, output_root: Path | str) -> None:
    root = Path(output_root).expanduser().resolve()
    write_jsonl(root / f"{manifest.dataset_id}_images.jsonl", (asdict(row) for row in manifest.images))
    write_jsonl(root / f"{manifest.dataset_id}_objects.jsonl", (asdict(row) for row in manifest.objects))
    exclusion_stats = manifest_exclusion_stats(manifest)
    summary = {
        "dataset_id": manifest.dataset_id,
        "root": manifest.root,
        "class_registry_sha256": manifest.class_registry_sha256,
        "image_count": manifest.image_count,
        "labeled_image_count": manifest.labeled_image_count,
        "object_count": manifest.object_count,
        "eligible_object_count": manifest.eligible_object_count,
        "excluded_objects_by_reason": dict(
            Counter(object.exclusion_reason for object in manifest.objects if object.exclusion_reason)
        ),
        "raw_coordinate_out_of_bounds_count": sum(
            object.raw_coordinate_out_of_bounds for object in manifest.objects
        ),
        "exclusion_stats": exclusion_stats,
    }
    (root / f"{manifest.dataset_id}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_dataset_manifest(output_root: Path | str, dataset_id: str) -> DatasetManifest:
    """Load only the current run's machine-readable E00 manifest."""

    root = Path(output_root).expanduser().resolve(strict=True)
    summary = json.loads((root / f"{dataset_id}_summary.json").read_text(encoding="utf-8"))
    images = tuple(
        ImageRecord(**{**row, "exclusion_category": row.get("exclusion_category")})
        for row in load_jsonl(root / f"{dataset_id}_images.jsonl")
    )
    objects = tuple(
        ObjectRecord(
            **{
                **row,
                "polygon_raw": tuple(row["polygon_raw"]),
                "polygon_visible": tuple(row["polygon_visible"]),
                "exclusion_category": row.get("exclusion_category"),
            }
        )
        for row in load_jsonl(root / f"{dataset_id}_objects.jsonl")
    )
    return DatasetManifest(
        dataset_id=dataset_id,
        root=str(summary["root"]),
        class_registry_sha256=str(summary["class_registry_sha256"]),
        image_count=int(summary["image_count"]),
        labeled_image_count=int(summary["labeled_image_count"]),
        object_count=int(summary["object_count"]),
        eligible_object_count=int(summary["eligible_object_count"]),
        images=images,
        objects=objects,
    )


def image_path(manifest: DatasetManifest, record: ImageRecord | ObjectRecord) -> Path:
    """Resolve a path recorded in the current E00 manifest."""

    root = Path(manifest.root).expanduser().resolve(strict=True)
    return _manifest_relative_path(root, record.image_relpath, "Manifest image path")


def _role_target_counts(
    total_images: int,
    fractions: list[float],
    maximum_per_role: int | None,
    maximum_per_dataset: int | None,
) -> dict[str, int]:
    if len(fractions) != len(ROLE_NAMES) or any(value < 0.0 for value in fractions):
        raise DatasetIndexError("Role fractions must contain four non-negative values")
    fraction_total = sum(fractions)
    if fraction_total <= 0.0:
        raise DatasetIndexError("Role fractions must have a positive sum")
    if maximum_per_role is not None and maximum_per_role < 0:
        raise DatasetIndexError("maximum_source_images_per_role must be non-negative")
    if maximum_per_dataset is not None and maximum_per_dataset < 0:
        raise DatasetIndexError("maximum_source_images_per_dataset must be non-negative")
    dataset_budget = total_images if maximum_per_dataset is None else min(total_images, maximum_per_dataset)
    if dataset_budget < 0:
        raise DatasetIndexError("maximum_source_images_per_dataset must be non-negative")
    role_capacity = {
        role: total_images if maximum_per_role is None else max(0, maximum_per_role) for role in ROLE_NAMES
    }
    raw = {
        role: dataset_budget * fractions[index] / fraction_total
        for index, role in enumerate(ROLE_NAMES)
    }
    targets = {
        role: min(role_capacity[role], int(math.floor(raw[role])))
        for role in ROLE_NAMES
    }
    target_budget = min(dataset_budget, sum(role_capacity.values()))
    while sum(targets.values()) < target_budget:
        available = [role for role in ROLE_NAMES if targets[role] < role_capacity[role]]
        if not available:
            break
        role = max(
            available,
            key=lambda candidate: (
                raw[candidate] - targets[candidate],
                role_capacity[candidate] - targets[candidate],
                -ROLE_NAMES.index(candidate),
            ),
        )
        targets[role] += 1
    return targets


def _source_content_units(objects: list[ObjectRecord]) -> dict[str, list[ObjectRecord]]:
    """Build atomic units so neither a source image nor a content duplicate can split roles."""

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for item in objects:
        source_key = f"source:{item.source_image_id}"
        content_key = f"content:{item.content_group_id}"
        union(source_key, content_key)

    grouped: dict[str, list[ObjectRecord]] = defaultdict(list)
    for item in objects:
        grouped[find(f"source:{item.source_image_id}")].append(item)

    units: dict[str, list[ObjectRecord]] = {}
    for items in grouped.values():
        source_ids = sorted({item.source_image_id for item in items})
        content_ids = sorted({item.content_group_id for item in items})
        unit_key = "unit:" + "|".join((*source_ids, *content_ids))
        units[unit_key] = sorted(items, key=lambda item: item.object_id)
    return units


def assign_source_image_roles(
    manifest: DatasetManifest,
    *,
    fractions: list[float],
    seed: int,
    minimum_images_per_class: dict[str, int],
    maximum_objects_per_class: dict[str, int],
    max_objects_per_image_per_class: int,
    maximum_source_images_per_role: int | None = None,
    maximum_source_images_per_dataset: int | None = None,
) -> tuple[tuple[ObjectRecord, ...], dict[str, Any]]:
    eligible = [object for object in manifest.objects if object.eligible]
    if max_objects_per_image_per_class <= 0:
        raise DatasetIndexError("max_objects_per_image_per_class must be positive")
    if any(value < 0 for value in maximum_objects_per_class.values()):
        raise DatasetIndexError("maximum_objects_per_class values must be non-negative")
    if maximum_source_images_per_role is not None and maximum_source_images_per_role < 0:
        raise DatasetIndexError("maximum_source_images_per_role must be non-negative")
    if maximum_source_images_per_dataset is not None and maximum_source_images_per_dataset < 0:
        raise DatasetIndexError("maximum_source_images_per_dataset must be non-negative")
    units = _source_content_units(eligible)
    unit_images = {unit: {item.source_image_id for item in items} for unit, items in units.items()}
    unit_content = {unit: {item.content_group_id for item in items} for unit, items in units.items()}
    total_images = len({item.source_image_id for item in eligible})
    targets = _role_target_counts(
        total_images,
        fractions,
        maximum_source_images_per_role,
        maximum_source_images_per_dataset,
    )
    role_capacities = {
        role: total_images if maximum_source_images_per_role is None else max(0, maximum_source_images_per_role)
        for role in ROLE_NAMES
    }
    dataset_capacity = total_images if maximum_source_images_per_dataset is None else max(0, maximum_source_images_per_dataset)
    assigned: dict[str, str] = {}
    role_source_images: dict[str, set[str]] = {role: set() for role in ROLE_NAMES}
    role_content_groups: dict[str, set[str]] = {role: set() for role in ROLE_NAMES}
    role_class_images: dict[str, dict[int, set[str]]] = {
        role: defaultdict(set) for role in ROLE_NAMES
    }

    class_unit_counts = Counter(
        class_id
        for items in units.values()
        for class_id in {item.class_id for item in items}
    )
    ordered_units = sorted(
        units,
        key=lambda unit: (
            -sum(1.0 / max(1, class_unit_counts[item.class_id]) for item in units[unit]),
            _stable_int(seed, unit),
            unit,
        )
    )
    unassigned_units: list[dict[str, Any]] = []
    for unit in ordered_units:
        unit_source_images = unit_images[unit]
        unit_content_groups = unit_content[unit]
        group_classes = {item.class_id for item in units[unit]}
        available = [
            role
            for role in ROLE_NAMES
            if len(role_source_images[role]) + len(unit_source_images) <= targets[role]
            and len(role_source_images[role]) + len(unit_source_images) <= role_capacities[role]
            and sum(len(values) for values in role_source_images.values()) + len(unit_source_images)
            <= dataset_capacity
        ]
        if not available:
            unassigned_units.append(
                {
                    "unit": unit,
                    "source_image_ids": sorted(unit_source_images),
                    "content_group_ids": sorted(unit_content_groups),
                    "reason": "no_role_or_dataset_capacity",
                }
            )
            continue
        role = max(
            available,
            key=lambda candidate: (
                sum(
                    max(
                        0,
                        minimum_images_per_class[candidate]
                        - len(role_class_images[candidate][class_id]),
                    )
                    for class_id in group_classes
                ),
                targets[candidate] - len(role_source_images[candidate]),
                -len(role_source_images[candidate]),
                -ROLE_NAMES.index(candidate),
            ),
        )
        assigned[unit] = role
        role_source_images[role].update(unit_source_images)
        role_content_groups[role].update(unit_content_groups)
        for class_id in group_classes:
            role_class_images[role][class_id].update(unit_source_images)

    selected: list[ObjectRecord] = []
    feasibility: dict[str, dict[str, Any]] = {}
    by_role_class: dict[tuple[str, int], list[ObjectRecord]] = defaultdict(list)
    for unit, role in assigned.items():
        for object in units[unit]:
            by_role_class[(role, object.class_id)].append(object)

    classes = sorted({object.class_id for object in eligible})
    role_class_breakdown: dict[str, dict[str, dict[str, Any]]] = {role: {} for role in ROLE_NAMES}
    for role in ROLE_NAMES:
        for class_id in classes:
            candidates = by_role_class[(role, class_id)]
            by_image: dict[str, list[ObjectRecord]] = defaultdict(list)
            for candidate in candidates:
                by_image[candidate.source_image_id].append(candidate)
            image_ids = sorted(by_image, key=lambda value: (_stable_int(seed + 17, value), value))
            minimum_images = max(0, minimum_images_per_class[role])
            maximum_objects = max(0, maximum_objects_per_class[role])
            chosen: list[ObjectRecord] = []
            for image_id in image_ids[:minimum_images]:
                if len(chosen) >= maximum_objects:
                    break
                image_candidates = sorted(
                    by_image[image_id], key=lambda item: (_stable_int(seed + 23, item.object_id), item.object_id)
                )
                chosen.append(image_candidates[0])
            remaining = [
                candidate
                for candidate in sorted(candidates, key=lambda item: (_stable_int(seed + 29, item.object_id), item.object_id))
                if candidate.object_id not in {item.object_id for item in chosen}
            ]
            image_counts = Counter(item.source_image_id for item in chosen)
            for candidate in remaining:
                if len(chosen) >= maximum_objects:
                    break
                if image_counts[candidate.source_image_id] >= max_objects_per_image_per_class:
                    continue
                chosen.append(candidate)
                image_counts[candidate.source_image_id] += 1
            for object in chosen:
                selected.append(replace(object, role=role))
            maximum_selectable_objects = sum(
                min(max_objects_per_image_per_class, len(items)) for items in by_image.values()
            )
            maximum_selectable_images = min(len(image_ids), maximum_objects)
            class_result = {
                "available_objects": len(candidates),
                "available_images": len(image_ids),
                "selected_objects": len(chosen),
                "selected_images": len({item.source_image_id for item in chosen}),
                "required_objects": maximum_objects,
                "required_images": minimum_images,
                "maximum_selectable_objects": maximum_selectable_objects,
                "maximum_selectable_images": maximum_selectable_images,
                "feasible": (
                    maximum_selectable_images >= minimum_images
                    and maximum_selectable_objects >= maximum_objects
                ),
            }
            feasibility[f"{role}:{class_id}"] = class_result
            role_class_breakdown[role][str(class_id)] = class_result

    selected_ids = {object.object_id for object in selected}
    selected_objects = tuple(sorted(selected, key=lambda item: item.object_id))
    role_object_counts = {
        role: sum(1 for item in selected_objects if item.role == role) for role in ROLE_NAMES
    }
    role_selected_image_counts = {
        role: len({item.source_image_id for item in selected_objects if item.role == role}) for role in ROLE_NAMES
    }
    role_feasibility = {
        role: {
            "class_count": len(classes),
            "feasible_class_count": sum(
                role_class_breakdown[role][str(class_id)]["feasible"] for class_id in classes
            ),
            "all_classes_feasible": all(
                role_class_breakdown[role][str(class_id)]["feasible"] for class_id in classes
            ),
        }
        for role in ROLE_NAMES
    }
    role_sets = [role_source_images[role] for role in ROLE_NAMES]
    role_content_sets = [role_content_groups[role] for role in ROLE_NAMES]
    report = {
        "dataset_id": manifest.dataset_id,
        "seed": seed,
        "assignment_version": "source-image-content-union-v2",
        "role_targets": targets,
        "role_capacities": role_capacities,
        "maximum_source_images_per_role": maximum_source_images_per_role,
        "maximum_source_images_per_dataset": maximum_source_images_per_dataset,
        "dataset_image_count": total_images,
        "assigned_dataset_image_count": sum(len(values) for values in role_source_images.values()),
        "dataset_capacity_used": sum(len(values) for values in role_source_images.values()),
        "dataset_capacity": dataset_capacity,
        "unassigned_unit_count": len(unassigned_units),
        "unassigned_units": unassigned_units,
        "role_group_counts": {role: sum(assigned_role == role for assigned_role in assigned.values()) for role in ROLE_NAMES},
        "role_image_counts": {role: len(role_source_images[role]) for role in ROLE_NAMES},
        "role_selected_image_counts": role_selected_image_counts,
        "role_object_counts": role_object_counts,
        "role_source_image_sets": {role: sorted(values) for role, values in role_source_images.items()},
        "role_content_group_sets": {role: sorted(values) for role, values in role_content_groups.items()},
        "role_class_breakdown": role_class_breakdown,
        "class_breakdown": role_class_breakdown,
        "role_feasibility": role_feasibility,
        "role_overlap_count": sum(
            len(left & right)
            for index, left in enumerate(role_sets)
            for right in role_sets[index + 1 :]
        ),
        "content_group_overlap_count": sum(
            len(left & right)
            for index, left in enumerate(role_content_sets)
            for right in role_content_sets[index + 1 :]
        ),
        "selected_object_count": len(selected_ids),
        "feasibility": feasibility,
        "all_role_class_requirements_feasible": all(item["feasible"] for item in feasibility.values()),
        "source_image_disjoint": sum(len(values) for values in role_source_images.values())
        == len(set().union(*role_source_images.values())) if role_source_images else True,
        "content_group_disjoint": sum(len(values) for values in role_content_groups.values())
        == len(set().union(*role_content_groups.values())) if role_content_groups else True,
    }
    return selected_objects, report
