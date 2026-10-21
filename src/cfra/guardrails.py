"""Path guardrails used by the public pipeline.

The internal research repository used this module to reject outputs from
retired projects.  Those project names are not part of the public release, so
the default list is empty.  The hook remains available for users who want to
forbid additional directories through ``CFRA_LEGACY_ROOTS``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

LEGACY_RELATIVE_ROOTS: tuple[Path, ...] = ()


class CleanRoomViolation(RuntimeError):
    """Raised when a runtime input crosses a configured path boundary."""


def project_root() -> Path:
    """Return the package root without depending on the current directory."""

    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    """Return the root used for relative legacy-path configuration."""

    return project_root()


def _configured_legacy_roots() -> tuple[Path, ...]:
    raw = os.environ.get("CFRA_LEGACY_ROOTS", "")
    return tuple(Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip())


def legacy_roots(root: Path | None = None) -> tuple[Path, ...]:
    """Return absolute paths that may not be used as runtime inputs."""

    base = Path(root) if root is not None else workspace_root()
    configured = tuple(base / relative for relative in LEGACY_RELATIVE_ROOTS)
    return configured + _configured_legacy_roots()


def _canonical(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve(strict=False)))


def _is_within(candidate: Path, root: Path) -> bool:
    candidate_text = _canonical(candidate)
    root_text = _canonical(root)
    try:
        return os.path.commonpath((candidate_text, root_text)) == root_text
    except ValueError:
        return False


def assert_clean_room_paths(paths: Iterable[Path | str], root: Path | None = None) -> None:
    """Reject paths inside configured legacy roots."""

    forbidden = legacy_roots(root)
    if not forbidden:
        return
    violations: list[tuple[Path, Path]] = []
    for raw_path in paths:
        candidate = Path(raw_path)
        for legacy_root in forbidden:
            if _is_within(candidate, legacy_root):
                violations.append((candidate, legacy_root))
                break
    if violations:
        details = "; ".join(f"{path} is inside {legacy}" for path, legacy in violations)
        raise CleanRoomViolation(f"Legacy runtime input rejected: {details}")


def assert_paths_exist(paths: Iterable[Path | str]) -> None:
    """Require every configured external input to exist."""

    missing = [str(Path(path)) for path in paths if not Path(path).expanduser().exists()]
    if missing:
        raise FileNotFoundError("Configured inputs do not exist: " + "; ".join(missing))
