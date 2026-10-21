"""Creation of immutable, non-resumable run directories."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RUN_SUBDIRECTORIES = (
    "manifest",
    "intermediates",
    "metrics",
    "statistics",
    "tables",
    "figures",
    "logs",
)


class RunInitializationError(RuntimeError):
    """Raised when an immutable run cannot be safely initialized."""


class GroupManifestError(ValueError):
    """Raised when a formal group manifest is malformed or cannot be changed."""


VALID_RUN_STATUSES = {"running", "completed", "audited", "failed", "invalid"}
RUN_TRANSITIONS = {
    "running": {"completed", "failed", "invalid"},
    "completed": {"audited", "invalid"},
    "audited": {"invalid"},
    "failed": set(),
    "invalid": set(),
}


def create_run(
    *,
    project_root: Path | str,
    run_id: str,
    evidence_tier: str,
    resolved_config: dict[str, Any],
    resolved_config_sha256: str,
    allow_resume: bool = False,
) -> Path:
    """Create a new run exactly once and write its initial manifest."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RunInitializationError(f"Invalid run_id: {run_id!r}")
    if evidence_tier not in {"development", "formal"}:
        raise RunInitializationError(f"Invalid evidence tier: {evidence_tier!r}")

    project = Path(project_root).expanduser().resolve(strict=True)
    runs_root = (project / "artifacts" / "runs").resolve()
    if runs_root == project or project not in runs_root.parents:
        raise RunInitializationError("Run output root escapes project root")
    runs_root.mkdir(parents=True, exist_ok=True)
    run_root = (runs_root / run_id).resolve()
    if run_root == runs_root or runs_root not in run_root.parents:
        raise RunInitializationError(f"Run path escapes run root: {run_id}")
    try:
        run_root.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise RunInitializationError(f"Run already exists and cannot be overwritten: {run_root}") from exc

    try:
        for relative in RUN_SUBDIRECTORIES:
            (run_root / relative).mkdir()
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "evidence_tier": evidence_tier,
            "status": "running",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "allow_resume": bool(allow_resume),
            "allow_cross_run_cache": False,
            "resolved_config_sha256": resolved_config_sha256,
            "resolved_config": resolved_config,
        }
        (run_root / "manifest" / "run.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_root / "STATUS").write_text("running\n", encoding="ascii")
    except Exception:
        (run_root / "STATUS").write_text("failed\n", encoding="ascii")
        raise

    return run_root


def open_or_create_run(
    *,
    run_dir: Path | str,
    evidence_tier: str,
    resolved_config: dict[str, Any],
    resolved_config_sha256: str,
    allow_resume: bool,
) -> tuple[Path, str, dict[str, Any] | None]:
    """Open an explicitly chosen run directory for a fresh start or a resume.

    Unlike :func:`create_run`, the caller names the directory.  The state is
    decided by :func:`cfra.resume.detect_run_state`, and any state that is not
    unambiguously empty or resumable raises instead of guessing.

    Returns ``(run_root, mode, checkpoint)`` where ``mode`` is ``"fresh"`` or
    ``"resume"``.
    """

    from .resume import (
        detect_run_state,
        validate_checkpoint_compatibility,
        write_checkpoint,
    )

    if evidence_tier not in {"development", "formal"}:
        raise RunInitializationError(f"Invalid evidence tier: {evidence_tier!r}")

    run_root = Path(run_dir).expanduser().resolve()
    state, checkpoint = detect_run_state(run_root)

    if state != "empty" and not allow_resume:
        raise RunInitializationError(
            f"Run directory is not empty and execution.allow_resume is false: {run_root}. "
            "Choose an empty directory or enable resume for this evidence tier."
        )

    if state == "empty":
        run_root.mkdir(parents=True, exist_ok=True)
        for relative in RUN_SUBDIRECTORIES:
            (run_root / relative).mkdir(exist_ok=True)
        manifest = {
            "schema_version": 1,
            "run_id": run_root.name,
            "evidence_tier": evidence_tier,
            "status": "running",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "allow_resume": bool(allow_resume),
            "allow_cross_run_cache": False,
            "resolved_config_sha256": resolved_config_sha256,
            "resolved_config": resolved_config,
        }
        write_manifest_json(run_root, "manifest/run.json", manifest)
        (run_root / "STATUS").write_text("running\n", encoding="ascii")
        write_checkpoint(
            run_root,
            config_sha256=resolved_config_sha256,
            evidence_tier=evidence_tier,
            completed_stages=[],
        )
        return run_root, "fresh", None

    if state == "resumable":
        assert checkpoint is not None
        compatible, errors = validate_checkpoint_compatibility(
            checkpoint,
            config_sha256=resolved_config_sha256,
            strict_code_version=True,
        )
        if not compatible:
            raise RunInitializationError(
                "Existing run cannot be resumed because it no longer matches this "
                f"configuration or code version: {'; '.join(errors)}. "
                "Use a new run directory instead of mixing incompatible stages."
            )
        if checkpoint.get("evidence_tier") != evidence_tier:
            raise RunInitializationError(
                f"Existing run has evidence_tier={checkpoint.get('evidence_tier')!r}, "
                f"but {evidence_tier!r} was requested"
            )
        manifest = read_run_manifest(run_root)
        if manifest.get("status") in {"failed", "invalid"}:
            raise RunInitializationError(
                f"Run manifest status is terminal: {manifest.get('status')!r}"
            )
        manifest["status"] = "running"
        manifest["allow_resume"] = True
        manifest.setdefault("resume_events", []).append(datetime.now(UTC).isoformat())
        write_manifest_json(run_root, "manifest/run.json", manifest)
        (run_root / "STATUS").write_text("running\n", encoding="ascii")
        return run_root, "resume", checkpoint

    reasons = {
        "invalid_no_checkpoint": (
            "the directory already contains files but has no checkpoint.json, so its "
            "contents cannot be attributed to a known stage sequence"
        ),
        "invalid_corrupted": "checkpoint.json could not be parsed",
        "invalid_schema": "checkpoint.json uses an unsupported schema version",
        "invalid_status": "the run is marked failed or invalid and needs manual review",
    }
    raise RunInitializationError(
        f"Cannot use run directory {run_root}: {reasons.get(state, state)}."
    )


def read_run_manifest(run_root: Path | str) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve(strict=True)
    return json.loads((root / "manifest" / "run.json").read_text(encoding="utf-8"))


def write_manifest_json(run_root: Path | str, relative_path: str, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact inside an already-created run atomically enough for one process."""

    root = Path(run_root).expanduser().resolve(strict=True)
    target = (root / relative_path).resolve()
    if target == root or root not in target.parents:
        raise RunInitializationError(f"Manifest path escapes run: {relative_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def update_run_status(run_root: Path | str, status: str, *, reason: str | None = None) -> None:
    if status not in VALID_RUN_STATUSES:
        raise RunInitializationError(f"Unknown run status: {status}")
    root = Path(run_root).expanduser().resolve(strict=True)
    manifest = read_run_manifest(root)
    current = str(manifest.get("status", ""))
    if status != current and status not in RUN_TRANSITIONS.get(current, set()):
        raise RunInitializationError(f"Invalid run transition: {current} -> {status}")
    manifest["status"] = status
    if reason is not None:
        manifest["status_reason"] = reason
    write_manifest_json(root, "manifest/run.json", manifest)
    (root / "STATUS").write_text(status + "\n", encoding="ascii")


def write_group_manifest(
    project_root: Path | str,
    manifest: Mapping[str, Any],
    *,
    relative_path: str = "artifacts/runs/group-manifest.json",
) -> Path:
    """Write a group manifest atomically, refusing to replace an existing one.

    Formal orchestration records its complete seed plan before the first atomic
    run is created.  A manifest is therefore an immutable plan, just like the
    run directories it points at; later status changes use
    :func:`update_group_manifest`.
    """

    project = Path(project_root).expanduser().resolve(strict=True)
    target = (project / relative_path).resolve()
    if target == project or project not in target.parents:
        raise GroupManifestError(f"Group manifest escapes project root: {relative_path}")
    if target.exists():
        raise GroupManifestError(f"Group manifest already exists and cannot be overwritten: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def update_group_manifest(manifest_path: Path | str, manifest: Mapping[str, Any]) -> Path:
    """Atomically update an existing group manifest without replacing its file."""

    target = Path(manifest_path).expanduser().resolve(strict=True)
    if not target.is_file():
        raise GroupManifestError(f"Group manifest does not exist: {target}")
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def read_group_manifest(manifest_path: Path | str) -> dict[str, Any]:
    """Read a JSON group manifest and require an object payload."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroupManifestError(f"Could not read group manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise GroupManifestError("Group manifest must be a JSON object")
    return payload
