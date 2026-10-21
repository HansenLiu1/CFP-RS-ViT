"""Resolution and structural validation of local public checkpoints."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cfra.configuration import configured_model_paths
from cfra.guardrails import assert_clean_room_paths
from cfra.provenance import sha256_file, sha256_json

from .specs import get_model_spec

REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelAssetError(RuntimeError):
    """Raised when a configured checkpoint cannot satisfy its adapter contract."""


@dataclass(frozen=True)
class ModelAsset:
    model_id: str
    asset_id: str
    path: Path
    weight_path: Path
    loader_config_path: Path | None
    revision: str | None
    checkpoint_sha256: str
    loader_config_sha256: str
    asset_fingerprint: str


def _require_within(path: Path, root: Path, description: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ModelAssetError(f"{description} escapes its configured checkpoint root: {path}") from exc


def _resolve_checkpoint_file(root: Path, filename: str, model_id: str) -> Path:
    try:
        path = (root / filename).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ModelAssetError(f"{model_id} checkpoint lacks {filename}: {root}") from exc
    assert_clean_room_paths([path])
    _require_within(path, root, f"{model_id} {filename}")
    if not path.is_file():
        raise ModelAssetError(f"{model_id} checkpoint input is not a file: {path}")
    return path


def _revision_from_path(path: Path, containment_root: Path) -> str | None:
    candidates = (path.name, path.parent.name)
    for candidate in candidates:
        if REVISION_PATTERN.fullmatch(candidate):
            return candidate

    manifest_root = path if path.is_dir() else path.parent
    try:
        manifest_path = (manifest_root / "download_manifest.json").resolve(strict=True)
    except FileNotFoundError:
        return None
    assert_clean_room_paths([manifest_path])
    _require_within(manifest_path, containment_root, "checkpoint download_manifest.json")
    if manifest_path.is_file():
        try:
            revision = json.loads(manifest_path.read_text(encoding="utf-8")).get("revision")
        except (json.JSONDecodeError, OSError) as exc:
            raise ModelAssetError(f"Cannot read checkpoint manifest: {manifest_path}") from exc
        if isinstance(revision, str) and REVISION_PATTERN.fullmatch(revision):
            return revision
    return None


def validate_model_asset(
    model_id: str,
    path: Path | str,
    *,
    expected_revision: str | None = None,
    expected_sha256: str | None = None,
    expected_loader_config_sha256: str | None = None,
) -> ModelAsset:
    spec = get_model_spec(model_id)
    candidate = Path(path).expanduser().resolve(strict=True)
    assert_clean_room_paths([candidate])

    if spec.asset_kind == "file":
        if not candidate.is_file() or candidate.name != spec.weight_filename:
            raise ModelAssetError(f"{model_id} requires an explicit checkpoint file: {candidate}")
        weight_path = candidate
        loader_config_path = None
        if spec.loader_config_sha256 is None:
            raise ModelAssetError(f"{model_id} lacks a registered backend configuration identity")
        loader_config_sha256 = spec.loader_config_sha256
        containment_root = candidate.parent
    elif spec.asset_kind == "directory":
        if not candidate.is_dir():
            raise ModelAssetError(f"{model_id} requires a local checkpoint directory: {candidate}")
        if spec.weight_filename is None:
            raise ModelAssetError(f"{model_id} does not register an exact weight filename")
        loader_config_path = _resolve_checkpoint_file(candidate, "config.json", model_id)
        weight_path = _resolve_checkpoint_file(candidate, spec.weight_filename, model_id)
        loader_config_sha256 = sha256_file(loader_config_path)
        containment_root = candidate
    else:
        raise ModelAssetError(f"Unsupported asset kind for {model_id}: {spec.asset_kind}")

    actual_sha256 = sha256_file(weight_path)
    revision = _revision_from_path(candidate, containment_root)
    normalized_expected = expected_sha256.lower() if expected_sha256 else None
    if normalized_expected is not None:
        if not SHA256_PATTERN.fullmatch(normalized_expected):
            raise ModelAssetError(f"{model_id} has an invalid registered checkpoint SHA-256")
        if actual_sha256 != normalized_expected:
            raise ModelAssetError(
                f"{model_id} checkpoint SHA-256 mismatch: expected {normalized_expected}, got {actual_sha256}"
            )
    normalized_config = expected_loader_config_sha256.lower() if expected_loader_config_sha256 else None
    if normalized_config is not None:
        if not SHA256_PATTERN.fullmatch(normalized_config):
            raise ModelAssetError(f"{model_id} has an invalid registered loader-config SHA-256")
        if loader_config_sha256 != normalized_config:
            raise ModelAssetError(
                f"{model_id} loader-config SHA-256 mismatch: expected {normalized_config}, "
                f"got {loader_config_sha256}"
            )
    if expected_revision is not None:
        if not REVISION_PATTERN.fullmatch(expected_revision):
            raise ModelAssetError(f"{model_id} has an invalid registered checkpoint revision")
        if revision != expected_revision:
            raise ModelAssetError(
                f"{model_id} checkpoint revision mismatch: expected {expected_revision}, got {revision}"
            )

    asset_fingerprint = sha256_json(
        {
            "asset_id": spec.asset_id,
            "checkpoint_revision": revision,
            "checkpoint_sha256": actual_sha256,
            "loader_config_sha256": loader_config_sha256,
        }
    )

    return ModelAsset(
        model_id=model_id,
        asset_id=spec.asset_id,
        path=candidate,
        weight_path=weight_path,
        loader_config_path=loader_config_path,
        revision=revision,
        checkpoint_sha256=actual_sha256,
        loader_config_sha256=loader_config_sha256,
        asset_fingerprint=asset_fingerprint,
    )


def verify_model_asset(asset: ModelAsset) -> None:
    """Recheck clean-room location and content identity at the final loading boundary."""

    rebuilt = validate_model_asset(asset.model_id, asset.path)
    if rebuilt != asset:
        raise ModelAssetError(f"Checkpoint identity changed after validation: {asset.path}")


def verify_registered_model_asset(
    asset: ModelAsset,
    model_id: str,
    registry_root: Path | str,
) -> None:
    """Verify a loaded asset against the versioned model registry at the final boundary."""

    rebuilt = resolve_registered_model_asset(model_id, asset.path, registry_root)
    if rebuilt != asset:
        raise ModelAssetError(f"{model_id} asset identity does not match its canonical registered identity")


def _registered_models(registry_root: Path | str) -> dict[str, dict[str, str]]:
    path = Path(registry_root).expanduser().resolve(strict=True) / "models.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["model_id"]: row for row in rows}


def resolve_registered_model_asset(
    model_id: str,
    path: Path | str,
    registry_root: Path | str,
) -> ModelAsset:
    models = _registered_models(registry_root)
    try:
        row = models[model_id]
    except KeyError as exc:
        raise ModelAssetError(f"Model is absent from registry/models.csv: {model_id}") from exc
    if row.get("status") != "checkpoint_verified":
        raise ModelAssetError(f"Model checkpoint is not verified in registry/models.csv: {model_id}")
    return validate_model_asset(
        model_id,
        path,
        expected_revision=row.get("checkpoint_revision") or None,
        expected_sha256=row.get("checkpoint_sha256") or None,
        expected_loader_config_sha256=row.get("loader_config_sha256") or None,
    )


def resolve_model_assets(
    local_config: dict[str, Any],
    model_ids: Iterable[str],
    *,
    registry_root: Path | str,
) -> dict[str, ModelAsset]:
    requested = list(dict.fromkeys(model_ids))
    asset_ids = list(dict.fromkeys(get_model_spec(model_id).asset_id for model_id in requested))
    configured = configured_model_paths(local_config, asset_ids)
    assert_clean_room_paths(configured.values())

    assets: dict[str, ModelAsset] = {}
    validated: dict[str, ModelAsset] = {}
    for model_id in requested:
        spec = get_model_spec(model_id)
        if spec.asset_id not in validated:
            validated[spec.asset_id] = resolve_registered_model_asset(
                model_id,
                configured[spec.asset_id],
                registry_root,
            )
        base = validated[spec.asset_id]
        assets[model_id] = ModelAsset(
            model_id=model_id,
            asset_id=base.asset_id,
            path=base.path,
            weight_path=base.weight_path,
            loader_config_path=base.loader_config_path,
            revision=base.revision,
            checkpoint_sha256=base.checkpoint_sha256,
            loader_config_sha256=base.loader_config_sha256,
            asset_fingerprint=base.asset_fingerprint,
        )
    return assets
