"""Stage-level resume support with hash-chain verification."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA_VERSION = 1


class ResumeError(RuntimeError):
    """Raised when resume state is invalid or inconsistent."""


def _get_code_version() -> str:
    """Get git commit hash or fallback identifier."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def detect_run_state(run_dir: Path | str) -> tuple[str, dict[str, Any] | None]:
    """Detect run directory state.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Tuple of (state, checkpoint_data)
        
    Possible states:
        - "empty": Directory doesn't exist or is empty
        - "resumable": Valid checkpoint exists and can be resumed
        - "invalid_no_checkpoint": Directory exists but no valid checkpoint
        - "invalid_config_mismatch": Config hash doesn't match
        - "invalid_code_mismatch": Code version doesn't match
        - "invalid_status": Run status is terminal (failed/invalid)
        - "invalid_schema": Checkpoint schema is unsupported
        - "invalid_corrupted": Checkpoint file is corrupted
    """
    run_path = Path(run_dir).expanduser().resolve()
    
    # Check if directory exists and is not empty
    if not run_path.exists():
        return ("empty", None)
    
    if not any(run_path.iterdir()):
        return ("empty", None)
    
    # Try to load checkpoint
    checkpoint_path = run_path / "checkpoint.json"
    if not checkpoint_path.is_file():
        return ("invalid_no_checkpoint", None)
    
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ("invalid_corrupted", None)
    
    # Validate schema
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return ("invalid_schema", checkpoint)
    
    # Check run status
    status_file = run_path / "STATUS"
    if status_file.is_file():
        status = status_file.read_text(encoding="ascii").strip()
        if status in {"failed", "invalid"}:
            return ("invalid_status", checkpoint)
    
    return ("resumable", checkpoint)


def validate_checkpoint_compatibility(
    checkpoint: dict[str, Any],
    config_sha256: str,
    code_version: str | None = None,
    strict_code_version: bool = False,
) -> tuple[bool, list[str]]:
    """Validate if checkpoint is compatible with current config/code.
    
    Args:
        checkpoint: Loaded checkpoint data
        config_sha256: Current config hash
        code_version: Current code version (auto-detected if None)
        strict_code_version: Whether to enforce code version match
        
    Returns:
        Tuple of (is_compatible, error_messages)
    """
    errors = []
    
    # Check config hash
    checkpoint_config = checkpoint.get("config_sha256", "")
    if checkpoint_config != config_sha256:
        errors.append(
            f"Config mismatch: checkpoint={checkpoint_config[:8]}..., current={config_sha256[:8]}..."
        )
    
    # Check code version if requested
    if strict_code_version:
        if code_version is None:
            code_version = _get_code_version()
        
        checkpoint_code = checkpoint.get("code_version", "")
        if checkpoint_code != code_version:
            errors.append(
                f"Code version mismatch: checkpoint={checkpoint_code}, current={code_version}"
            )
    
    return (len(errors) == 0, errors)


def write_checkpoint(
    run_dir: Path | str,
    config_sha256: str,
    evidence_tier: str,
    completed_stages: list[str],
    stage_artifacts: dict[str, dict[str, Any]] | None = None,
    code_version: str | None = None,
) -> Path:
    """Write or update checkpoint file atomically.
    
    Args:
        run_dir: Run directory path
        config_sha256: Config hash
        evidence_tier: "development" or "formal"
        completed_stages: List of completed stage IDs
        stage_artifacts: Optional dict of {stage_id: {path, sha256, ...}}
        code_version: Code version (auto-detected if None)
        
    Returns:
        Path to written checkpoint file
    """
    run_path = Path(run_dir).expanduser().resolve()
    checkpoint_path = run_path / "checkpoint.json"
    
    if code_version is None:
        code_version = _get_code_version()
    
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_dir": str(run_path),
        "config_sha256": config_sha256,
        "code_version": code_version,
        "evidence_tier": evidence_tier,
        "completed_stages": list(completed_stages),
        "stage_artifacts": stage_artifacts or {},
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    
    # Atomic write
    tmp_path = checkpoint_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(run_dir: Path | str) -> dict[str, Any]:
    """Load checkpoint from run directory.
    
    Args:
        run_dir: Run directory path
        
    Returns:
        Checkpoint data
        
    Raises:
        ResumeError: If checkpoint doesn't exist or is invalid
    """
    run_path = Path(run_dir).expanduser().resolve()
    checkpoint_path = run_path / "checkpoint.json"
    
    if not checkpoint_path.is_file():
        raise ResumeError(f"No checkpoint found in {run_path}")
    
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeError(f"Corrupted checkpoint in {run_path}: {exc}") from exc
    
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ResumeError(
            f"Unsupported checkpoint schema version: {checkpoint.get('schema_version')}"
        )
    
    return checkpoint


def stage_completed(checkpoint: dict[str, Any], stage_id: str) -> bool:
    """Check if a stage is marked as completed in checkpoint."""
    return stage_id in checkpoint.get("completed_stages", [])


def add_completed_stage(
    run_dir: Path | str,
    stage_id: str,
    artifact_path: Path | str | None = None,
    artifact_sha256: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Add a completed stage to checkpoint.
    
    Args:
        run_dir: Run directory path
        stage_id: Stage identifier (e.g., "E00", "features:clip:nominal")
        artifact_path: Optional path to stage output artifact
        artifact_sha256: Optional SHA256 hash of artifact
        metadata: Optional additional metadata
    """
    checkpoint = load_checkpoint(run_dir)
    
    if stage_id not in checkpoint["completed_stages"]:
        checkpoint["completed_stages"].append(stage_id)
    
    if artifact_path is not None or artifact_sha256 is not None or metadata is not None:
        artifact_info: dict[str, Any] = {}
        if artifact_path is not None:
            artifact_info["path"] = str(artifact_path)
        if artifact_sha256 is not None:
            artifact_info["sha256"] = artifact_sha256
        if metadata is not None:
            artifact_info["metadata"] = metadata
        
        checkpoint["stage_artifacts"][stage_id] = artifact_info
    
    checkpoint["updated_at_utc"] = datetime.now(UTC).isoformat()
    
    # Atomic write
    run_path = Path(run_dir).expanduser().resolve()
    checkpoint_path = run_path / "checkpoint.json"
    tmp_path = checkpoint_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(checkpoint_path)


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_stage_result(run_dir: Path | str, stage_id: str, result: dict[str, Any]) -> Path:
    """Write stage result to JSON for resumption."""
    run_path = Path(run_dir).expanduser().resolve()
    results_dir = run_path / "intermediates" / "stage_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = results_dir / f"{stage_id}.json"
    tmp_path = result_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(result_path)
    return result_path


def load_stage_result(run_dir: Path | str, stage_id: str) -> dict[str, Any] | None:
    """Load stage result from JSON if it exists."""
    run_path = Path(run_dir).expanduser().resolve()
    result_path = run_path / "intermediates" / "stage_results" / f"{stage_id}.json"
    
    if not result_path.is_file():
        return None
    
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
