"""Raw dataset indexing and source-image split assignment."""

from .index import (
    ROLE_NAMES,
    ClassInfo,
    DatasetIndexError,
    DatasetManifest,
    ImageRecord,
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

__all__ = [
    "ROLE_NAMES",
    "ClassInfo",
    "DatasetIndexError",
    "DatasetManifest",
    "ImageRecord",
    "ObjectRecord",
    "assign_source_image_roles",
    "build_dataset_manifest",
    "image_path",
    "load_class_registry",
    "load_dataset_manifest",
    "load_jsonl",
    "write_dataset_manifest",
    "write_jsonl",
]
