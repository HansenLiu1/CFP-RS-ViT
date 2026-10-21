"""Run auditing and frozen evidence-release validation.

This module is intentionally independent of the experiment stages.  It reads
the immutable files already present in a run and refuses to infer scientific
evidence from a missing or partially written artifact.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cfra.configuration import config_for_seed, config_sha256, load_experiment_config
from cfra.provenance import sha256_file, sha256_json
from cfra.runs import read_run_manifest, update_run_status


class AuditError(RuntimeError):
    """Raised for an unusable run path or malformed audit request."""


class ReleaseError(ValueError):
    """Raised when a release contains ineligible or unregistered evidence."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPERIMENT = re.compile(r"E\d+(?:-\d+)?")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_QUALIFICATION = ("E00", "E01", "E02")
_INELIGIBLE_EVIDENCE_STATUSES = {
    "skipped",
    "partial",
    "incomplete",
    "interface-only",
    "interface_only",
    "not-run",
    "not_run",
    "not-executed",
    "not_executed",
}
_STAGE_CONTRACTS = ("E06", "E08")
_LINEAGE_CANDIDATES = (
    "manifest/lineage.json",
    "lineage.json",
    "tables/summary.lineage.json",
    "tables/lineage.json",
    "reports/lineage.json",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if isinstance(item, (str, int)) and str(item)]


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _normalise_experiment_id(value: str) -> str:
    return value.upper().replace("_", "-")


def _experiment_ids_from_node(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            key_text = str(key)
            if _EXPERIMENT.fullmatch(key_text):
                found.add(_normalise_experiment_id(key_text))
            if key_text.lower() in {"experiment_id", "experiment", "stage", "experiment_ids"}:
                if isinstance(value, str):
                    found.update(_normalise_experiment_id(item) for item in _EXPERIMENT.findall(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            found.update(_normalise_experiment_id(part) for part in _EXPERIMENT.findall(item))
            found.update(_experiment_ids_from_node(value))
    elif isinstance(node, list):
        for item in node:
            found.update(_experiment_ids_from_node(item))
    return found


_STRUCTURAL_METRIC_KEYS = {
    "config_sha256",
    "created_at_utc",
    "dataset_id",
    "experiment",
    "experiment_id",
    "generated_at_utc",
    "model_id",
    "models",
    "result_path",
    "run_id",
    "schema_version",
    "stage",
    "status",
}


def _looks_like_metric_record(node: Mapping[str, Any]) -> bool:
    return any(
        key not in _STRUCTURAL_METRIC_KEYS and _is_scalar(value)
        for key, value in node.items()
    )


def _has_metric_record(node: Any) -> bool:
    if isinstance(node, dict):
        return _looks_like_metric_record(node) or any(_has_metric_record(value) for value in node.values())
    if isinstance(node, list):
        return any(_has_metric_record(item) for item in node)
    return False


def _metric_record_ids(node: Any, inherited: str | None = None) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        context = inherited
        for key, value in node.items():
            key_text = str(key)
            if _EXPERIMENT.fullmatch(key_text):
                context = _normalise_experiment_id(key_text)
            elif key_text.lower() in {"experiment_id", "experiment", "stage"} and isinstance(value, str):
                matches = _EXPERIMENT.findall(value)
                if len(matches) == 1:
                    context = _normalise_experiment_id(matches[0])
        for key, value in node.items():
            key_text = str(key)
            child_context = context
            if _EXPERIMENT.fullmatch(key_text):
                child_context = _normalise_experiment_id(key_text)
            elif key_text.lower() in {"experiment_id", "experiment", "stage"} and isinstance(value, str):
                matches = _EXPERIMENT.findall(value)
                if len(matches) == 1:
                    child_context = _normalise_experiment_id(matches[0])
            found.update(_metric_record_ids(value, child_context))
        if context and _looks_like_metric_record(node):
            found.add(context)
    elif isinstance(node, list):
        for item in node:
            found.update(_metric_record_ids(item, inherited))
    return found


def _experiment_statuses_from_node(node: Any, inherited: str | None = None) -> dict[str, set[str]]:
    statuses: dict[str, set[str]] = {}
    if isinstance(node, dict):
        context = inherited
        for key, value in node.items():
            key_text = str(key)
            if _EXPERIMENT.fullmatch(key_text):
                context = _normalise_experiment_id(key_text)
            elif key_text.lower() in {"experiment_id", "experiment", "stage"} and isinstance(value, str):
                matches = _EXPERIMENT.findall(value)
                if len(matches) == 1:
                    context = _normalise_experiment_id(matches[0])
        for key, value in node.items():
            key_text = str(key)
            child_context = context
            if _EXPERIMENT.fullmatch(key_text):
                child_context = _normalise_experiment_id(key_text)
            elif key_text.lower() in {"experiment_id", "experiment", "stage"} and isinstance(value, str):
                matches = _EXPERIMENT.findall(value)
                if matches:
                    child_context = _normalise_experiment_id(matches[0])
            if key_text.lower() == "status" and isinstance(value, str) and context:
                statuses.setdefault(context, set()).add(value.lower())
            nested = _experiment_statuses_from_node(value, child_context)
            for experiment_id, values in nested.items():
                statuses.setdefault(experiment_id, set()).update(values)
    elif isinstance(node, list):
        for item in node:
            nested = _experiment_statuses_from_node(item, inherited)
            for experiment_id, values in nested.items():
                statuses.setdefault(experiment_id, set()).update(values)
    return statuses


def _metric_files(root: Path) -> list[Path]:
    metrics_root = root / "metrics"
    if not metrics_root.exists():
        return []
    return sorted(
        path
        for path in metrics_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )


def _stage_manifest_files(root: Path) -> list[Path]:
    manifest_root = root / "manifest"
    if not manifest_root.exists():
        return []
    return sorted(
        path
        for path in manifest_root.rglob("*.json")
        if path.is_file() and re.fullmatch(r"e(?:06|08)_.+", path.stem.lower())
    )


def _experiment_markers(node: Any, inherited: str | None = None) -> dict[str, dict[str, Any]]:
    """Collect status markers while retaining the experiment context."""

    markers: dict[str, dict[str, Any]] = {}
    if isinstance(node, dict):
        context = inherited
        for key, value in node.items():
            key_text = str(key)
            if _EXPERIMENT.fullmatch(key_text):
                context = _normalise_experiment_id(key_text)
            elif key_text.lower() in {"experiment_id", "experiment", "stage"} and isinstance(value, str):
                matches = _EXPERIMENT.findall(value)
                if len(matches) == 1:
                    context = _normalise_experiment_id(matches[0])

        if context:
            bucket = markers.setdefault(
                context,
                {
                    "statuses": set(),
                    "execution_statuses": set(),
                    "interface_called": set(),
                    "configured_enabled": set(),
                    "layer_output_statuses": set(),
                    "interface_only": False,
                    "evidence_eligible": set(),
                },
            )
            status = node.get("status")
            if isinstance(status, str):
                bucket["statuses"].add(status.strip().lower())
            execution_status = node.get("execution_status")
            if isinstance(execution_status, str):
                bucket["execution_statuses"].add(execution_status.strip().lower())
            for field in ("interface_called", "configured_enabled", "evidence_eligible"):
                if field in node and isinstance(node[field], bool):
                    bucket[field].add(node[field])
            layer_status = node.get("layer_output_status")
            if isinstance(layer_status, str):
                bucket["layer_output_statuses"].add(layer_status.strip().lower())
            if node.get("interface_only") is True or node.get("interface-only") is True:
                bucket["interface_only"] = True

        for key, value in node.items():
            key_text = str(key)
            child_context = context
            if _EXPERIMENT.fullmatch(key_text):
                child_context = _normalise_experiment_id(key_text)
            elif key_text.lower() in {"experiment_id", "experiment", "stage"} and isinstance(value, str):
                matches = _EXPERIMENT.findall(value)
                if len(matches) == 1:
                    child_context = _normalise_experiment_id(matches[0])
            nested = _experiment_markers(value, child_context)
            for experiment_id, child in nested.items():
                target = markers.setdefault(
                    experiment_id,
                    {
                        "statuses": set(),
                        "execution_statuses": set(),
                        "interface_called": set(),
                        "configured_enabled": set(),
                        "layer_output_statuses": set(),
                        "interface_only": False,
                        "evidence_eligible": set(),
                    },
                )
                for field in (
                    "statuses",
                    "execution_statuses",
                    "interface_called",
                    "configured_enabled",
                    "layer_output_statuses",
                    "evidence_eligible",
                ):
                    target[field].update(child[field])
                target["interface_only"] = target["interface_only"] or child["interface_only"]
    elif isinstance(node, list):
        for item in node:
            nested = _experiment_markers(item, inherited)
            for experiment_id, child in nested.items():
                target = markers.setdefault(
                    experiment_id,
                    {
                        "statuses": set(),
                        "execution_statuses": set(),
                        "interface_called": set(),
                        "configured_enabled": set(),
                        "layer_output_statuses": set(),
                        "interface_only": False,
                        "evidence_eligible": set(),
                    },
                )
                for field in (
                    "statuses",
                    "execution_statuses",
                    "interface_called",
                    "configured_enabled",
                    "evidence_eligible",
                ):
                    target[field].update(child[field])
                target["layer_output_statuses"].update(child["layer_output_statuses"])
                target["interface_only"] = target["interface_only"] or child["interface_only"]
    return markers


def _marker_is_ineligible(marker: Mapping[str, Any]) -> bool:
    statuses = set(marker.get("statuses", set())) | set(marker.get("execution_statuses", set()))
    statuses.update(marker.get("layer_output_statuses", set()))
    return bool(
        statuses & _INELIGIBLE_EVIDENCE_STATUSES
        or marker.get("interface_only") is True
        or False in set(marker.get("evidence_eligible", set()))
    )


def _metric_evidence(root: Path) -> tuple[set[str], dict[str, set[str]], list[str], dict[str, dict[str, Any]]]:
    experiments: set[str] = set()
    statuses: dict[str, set[str]] = {}
    errors: list[str] = []
    markers: dict[str, dict[str, Any]] = {}
    for path in [*_metric_files(root), *_stage_manifest_files(root)]:
        stem_matches = _EXPERIMENT.findall(path.stem.upper().replace("_", "-"))
        try:
            if path.suffix.lower() == ".jsonl":
                payloads = []
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                    if not line.strip():
                        continue
                    try:
                        payloads.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        errors.append(f"invalid JSONL: {path.relative_to(root).as_posix()}:{line_number} ({exc.msg})")
            else:
                payloads = [_read_json(path)]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"unreadable metrics file: {path.relative_to(root).as_posix()} ({exc})")
            continue
        for payload in payloads:
            payload_experiments = _metric_record_ids(payload)
            experiments.update(payload_experiments)
            if len(stem_matches) == 1 and _has_metric_record(payload):
                experiments.add(_normalise_experiment_id(stem_matches[0]))
            for experiment_id, values in _experiment_statuses_from_node(payload).items():
                statuses.setdefault(experiment_id, set()).update(values)
            for experiment_id, marker in _experiment_markers(payload).items():
                target = markers.setdefault(
                    experiment_id,
                    {
                        "statuses": set(),
                        "execution_statuses": set(),
                        "interface_called": set(),
                        "configured_enabled": set(),
                        "layer_output_statuses": set(),
                        "interface_only": False,
                        "evidence_eligible": set(),
                    },
                )
                for field in (
                    "statuses",
                    "execution_statuses",
                    "interface_called",
                    "configured_enabled",
                    "layer_output_statuses",
                    "evidence_eligible",
                ):
                    target[field].update(marker[field])
                target["interface_only"] = target["interface_only"] or marker["interface_only"]
    experiments.difference_update(
        experiment_id for experiment_id, marker in markers.items() if _marker_is_ineligible(marker)
    )
    return experiments, statuses, errors, markers


def _configured_experiments(
    manifest: Mapping[str, Any],
    required_experiments: Iterable[str] | None,
    *,
    group_name: str | None = None,
) -> list[str]:
    if required_experiments is not None:
        configured = _as_strings(list(required_experiments))
    else:
        resolved = manifest.get("resolved_config", {})
        configured = []
        if isinstance(resolved, dict):
            release = resolved.get("release", {})
            if isinstance(release, dict):
                configured = _as_strings(release.get("required_experiments"))
            if not configured and group_name:
                groups = resolved.get("run_groups", {})
                if isinstance(groups, dict) and isinstance(groups.get(group_name), dict):
                    configured = _as_strings(groups[group_name].get("experiments"))
            if not configured:
                scope = resolved.get("scope", {})
                if isinstance(scope, dict):
                    configured = _as_strings(scope.get("experiment_ids"))
    result = {_normalise_experiment_id(item) for item in configured if _EXPERIMENT.fullmatch(str(item))}
    result.update(_QUALIFICATION)
    return sorted(result, key=lambda item: (int(re.search(r"\d+", item).group()), item))


def _resolve_relative(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if root not in target.parents:
        raise ValueError(f"path escapes run root: {relative}")
    return target


def _check(checks: list[dict[str, Any]], name: str, passed: bool, details: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def _qualification_check(
    root: Path,
    manifest: Mapping[str, Any],
    expected_config_sha256: str,
    checks: list[dict[str, Any]],
) -> None:
    run_manifest_config_hash = str(manifest.get("resolved_config_sha256", ""))
    for experiment_id, relative in (
        ("E00", "manifest/e00.json"),
        ("E01", "manifest/e01.json"),
        ("E02", "manifest/e02.json"),
    ):
        path = root / relative
        if not path.is_file():
            _check(checks, f"{experiment_id}_manifest", False, f"missing {relative}")
            continue
        try:
            payload = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _check(checks, f"{experiment_id}_manifest", False, f"unreadable {relative}: {exc}")
            continue
        stage_ok = str(payload.get("stage", "")).upper() == experiment_id
        _check(checks, f"{experiment_id}_stage", stage_ok, f"stage={payload.get('stage')!r}")
        _check(
            checks,
            f"{experiment_id}_evidence_status",
            _payload_evidence_eligible(payload),
            "qualification manifest is eligible evidence"
            if _payload_evidence_eligible(payload)
            else "qualification manifest is skipped, partial, interface-only or otherwise ineligible",
        )
        if experiment_id == "E00":
            config_ok = payload.get("config_sha256") == run_manifest_config_hash
            _check(checks, "E00_config_hash", config_ok, "E00 config hash matches run manifest")
            dataset_ok = isinstance(payload.get("datasets"), dict) and bool(payload.get("datasets"))
            _check(checks, "E00_source_identity", dataset_ok, "E00 contains dataset source identity")
        elif experiment_id == "E01":
            disjoint_ok = payload.get("source_image_disjoint") is True
            overlap_ok = payload.get("role_overlap_count", 0) == 0 and not payload.get("role_overlap")
            content_ok = payload.get("content_group_disjoint") is True and payload.get("content_group_overlap_count", 0) == 0
            _check(checks, "E01_source_image_disjoint", disjoint_ok and overlap_ok, "E01 role split is source-image disjoint")
            _check(checks, "E01_content_group_disjoint", content_ok, "E01 content groups are disjoint")
        else:
            required_e02_fields = {
                "all_pixel_invariants_pass",
                "all_registered_controls_available",
                "failure_count",
                "control_failure_count",
                "row_count",
            }
            complete = required_e02_fields.issubset(payload)
            pixel_ok = complete and payload.get("all_pixel_invariants_pass") is True
            controls_ok = complete and payload.get("all_registered_controls_available") is True
            no_failures = complete and payload.get("failure_count") == 0 and payload.get("control_failure_count") == 0
            row_ok = complete and isinstance(payload.get("row_count"), int) and payload.get("row_count", 0) > 0
            _check(checks, "E02_complete", complete, "E02 required completeness fields are present")
            _check(checks, "E02_pixel_invariants", pixel_ok, "E02 pixel invariants pass")
            _check(checks, "E02_registered_controls", controls_ok, "E02 registered controls are complete")
            _check(checks, "E02_no_failures", no_failures, "E02 has no failed rows or controls")
            _check(checks, "E02_rows", row_ok, "E02 has at least one audited row")

    hash_ok = (
        bool(_SHA256.fullmatch(run_manifest_config_hash))
        and bool(expected_config_sha256)
        and bool(_SHA256.fullmatch(str(expected_config_sha256)))
        and run_manifest_config_hash == expected_config_sha256
    )
    _check(checks, "run_config_hash", hash_ok, "run manifest config hash is present and consistent")


def _lineage_candidates(root: Path) -> list[Path]:
    return [root / relative for relative in _LINEAGE_CANDIDATES if (root / relative).is_file()]


def _payload_evidence_eligible(payload: Mapping[str, Any]) -> bool:
    statuses = {
        str(payload.get(field, "")).strip().lower()
        for field in ("status", "execution_status", "layer_output_status")
        if isinstance(payload.get(field), str)
    }
    return not (
        statuses & _INELIGIBLE_EVIDENCE_STATUSES
        or payload.get("interface_only") is True
        or payload.get("interface-only") is True
        or payload.get("evidence_eligible") is False
    )


def _stage_contract_check(
    expected_experiments: Iterable[str],
    markers: Mapping[str, Mapping[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    required = set(expected_experiments) & set(_STAGE_CONTRACTS)
    for experiment_id in sorted(required):
        marker = markers.get(experiment_id, {})
        statuses = set(marker.get("statuses", set()))
        interface_called = set(marker.get("interface_called", set()))
        configured_enabled = set(marker.get("configured_enabled", set()))
        layer_statuses = set(marker.get("layer_output_statuses", set()))
        status_ok = statuses == {"completed"}
        interface_ok = interface_called == {True}
        configured_ok = not configured_enabled or configured_enabled == {True}
        layer_ok = not layer_statuses or layer_statuses == {"completed"}
        _check(
            checks,
            f"{experiment_id}_completed",
            status_ok,
            f"{experiment_id} status is completed" if status_ok else f"{experiment_id} statuses={sorted(statuses)}",
        )
        _check(
            checks,
            f"{experiment_id}_interface_called",
            interface_ok and configured_ok,
            f"{experiment_id} interface was called" if interface_ok else f"{experiment_id} interface_called={sorted(interface_called, key=str)}",
        )
        if configured_enabled:
            _check(
                checks,
                f"{experiment_id}_configured",
                configured_ok,
                f"{experiment_id} is configured for execution"
                if configured_ok
                else f"{experiment_id} configured_enabled={sorted(configured_enabled, key=str)}",
            )
        _check(
            checks,
            f"{experiment_id}_layer_output",
            layer_ok,
            f"{experiment_id} has no skipped layer output" if layer_ok else f"{experiment_id} layer_output_status=skipped",
        )


def _lineage_entries(payload: Any) -> list[Mapping[str, Any]] | None:
    if not isinstance(payload, list) or not payload:
        return None
    if not all(isinstance(item, Mapping) for item in payload):
        return None
    return payload


def _validate_lineage_file_entries(
    root: Path,
    entries: Any,
    label: str,
    problems: list[str],
) -> bool:
    valid_entries = _lineage_entries(entries)
    if valid_entries is None:
        problems.append(f"{label} is missing or empty")
        return False
    valid = True
    for item in valid_entries:
        if not isinstance(item.get("path"), str) or not _SHA256.fullmatch(str(item.get("sha256", ""))):
            problems.append(f"malformed {label} entry")
            valid = False
            continue
        try:
            path = _resolve_relative(root, str(item["path"]))
        except ValueError as exc:
            problems.append(f"{label}: {exc}")
            valid = False
            continue
        if not path.is_file():
            problems.append(f"missing {label} file: {item['path']}")
            valid = False
        elif sha256_file(path) != item["sha256"]:
            problems.append(f"{label} hash mismatch: {item['path']}")
            valid = False
    return valid


def _lineage_provenance(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    payloads: list[Mapping[str, Any]] = [manifest]
    for relative in (
        "manifest/e00.json",
        "manifest/source_manifest.json",
        "intermediates/e00_source_manifest/identity.json",
    ):
        path = root / relative
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            value = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping):
            payloads.append(value)

    result: dict[str, Any] = {}
    code_keys = {
        "code",
        "code_identity",
        "code_provenance",
        "code_sha256",
        "git_commit",
        "commit",
        "git_dirty",
        "dirty_status",
        "package_version",
    }
    environment_keys = {
        "environment",
        "environment_identity",
        "environment_provenance",
        "environment_sha256",
        "python_version",
        "platform",
        "cuda",
        "cudnn",
        "gpu",
        "driver",
    }
    for output_key, keys in (("code", code_keys), ("environment", environment_keys)):
        for payload in payloads:
            selected = {str(key): payload[key] for key in keys if key in payload}
            if output_key in payload and payload[output_key] not in (None, "", {}, []):
                result[output_key] = payload[output_key]
                break
            if selected:
                result[output_key] = selected
                break
    return result


def _validate_lineage(root: Path, manifest: Mapping[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = _lineage_candidates(root)
    if not candidates:
        _check(checks, "lineage_present", False, "no lineage JSON found")
        return None

    expected_run_id = str(manifest.get("run_id", ""))
    expected_config_hash = str(manifest.get("resolved_config_sha256", ""))
    valid_payload: dict[str, Any] | None = None
    problems: list[str] = []
    for path in candidates:
        try:
            payload = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"{path.relative_to(root).as_posix()}: {exc}")
            continue
        if not isinstance(payload, dict):
            problems.append(f"{path.relative_to(root).as_posix()}: lineage must be an object")
            continue
        if payload.get("run_id") != expected_run_id:
            problems.append(f"{path.relative_to(root).as_posix()}: run_id mismatch")
            continue
        if payload.get("config_sha256") != expected_config_hash:
            problems.append(f"{path.relative_to(root).as_posix()}: config hash mismatch")
            continue
        if not _SHA256.fullmatch(expected_config_hash):
            problems.append(f"{path.relative_to(root).as_posix()}: config hash is missing or malformed")
            continue
        source_files = payload.get("source_files")
        if not _validate_lineage_file_entries(root, source_files, "source_files", problems):
            continue
        if not _validate_lineage_file_entries(root, payload.get("source_manifest"), "source_manifest", problems):
            continue
        if not _validate_lineage_file_entries(root, payload.get("statistics_files"), "statistics_files", problems):
            continue
        source_problem = False
        for item in source_files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not _SHA256.fullmatch(str(item.get("sha256", ""))):
                problems.append(f"{path.relative_to(root).as_posix()}: malformed source file entry")
                source_problem = True
                break
            try:
                source_path = _resolve_relative(root, str(item["path"]))
            except ValueError as exc:
                problems.append(f"{path.relative_to(root).as_posix()}: {exc}")
                source_problem = True
                break
            if not source_path.is_file():
                problems.append(f"{path.relative_to(root).as_posix()}: missing source {item['path']}")
                source_problem = True
                break
            actual_hash = sha256_file(source_path)
            if actual_hash != item["sha256"]:
                problems.append(f"{path.relative_to(root).as_posix()}: hash mismatch for {item['path']}")
                source_problem = True
                break
        if source_problem:
            continue
        recorded_lineage_hash = str(payload.get("lineage_sha256", ""))
        if not _SHA256.fullmatch(recorded_lineage_hash):
            problems.append(f"{path.relative_to(root).as_posix()}: lineage_sha256 is missing or malformed")
            continue
        without_hash = dict(payload)
        without_hash.pop("lineage_sha256", None)
        if recorded_lineage_hash != sha256_json(without_hash):
            problems.append(f"{path.relative_to(root).as_posix()}: lineage hash mismatch")
            continue
        output_problem = False
        for item in payload.get("outputs", []):
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not _SHA256.fullmatch(str(item.get("sha256", ""))):
                problems.append(f"{path.relative_to(root).as_posix()}: malformed output entry")
                output_problem = True
                break
            try:
                output_path = _resolve_relative(root, str(item["path"]))
            except ValueError as exc:
                problems.append(f"{path.relative_to(root).as_posix()}: {exc}")
                output_problem = True
                break
            if not output_path.is_file() or sha256_file(output_path) != item["sha256"]:
                problems.append(f"{path.relative_to(root).as_posix()}: output hash mismatch for {item['path']}")
                output_problem = True
                break
        if output_problem:
            continue
        expected_provenance = _lineage_provenance(root, manifest)
        provenance_problem = False
        for field, expected in expected_provenance.items():
            if payload.get(field) != expected:
                problems.append(f"{path.relative_to(root).as_posix()}: {field} provenance mismatch")
                provenance_problem = True
        if provenance_problem:
            continue
        valid_payload = payload
        break

    _check(
        checks,
        "lineage_integrity",
        valid_payload is not None,
        "lineage source hashes match" if valid_payload is not None else "; ".join(problems),
    )
    return valid_payload


def _validate_audit_payload(
    root: Path,
    payload: Mapping[str, Any] | None,
    *,
    run_id: str,
    config_sha256_value: str,
) -> tuple[bool, str]:
    if not isinstance(payload, Mapping):
        return False, "audit report is not a JSON object"
    if payload.get("passed") is not True:
        return False, "audit report did not pass"
    if payload.get("run_id") != run_id:
        return False, "audit report run_id mismatch"
    if payload.get("config_sha256") != config_sha256_value:
        return False, "audit report config hash mismatch"
    checks = payload.get("checks")
    if not isinstance(checks, list) or not all(isinstance(item, Mapping) and item.get("passed") is True for item in checks):
        return False, "audit report contains a failed or malformed check"
    if payload.get("status_after") != "audited":
        return False, "audit report did not record audited status"
    checked_files = payload.get("checked_files")
    if not isinstance(checked_files, list) or not checked_files:
        return False, "audit report has no checked_files"
    for item in checked_files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not _SHA256.fullmatch(str(item.get("sha256", ""))):
            return False, "audit report has a malformed checked file"
        try:
            path = _resolve_relative(root, str(item["path"]))
        except ValueError as exc:
            return False, str(exc)
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            return False, f"audit checked file hash mismatch: {item['path']}"
    recorded_hash = str(payload.get("lineage_sha256", ""))
    expected_hash = sha256_json(
        {
            "run_id": run_id,
            "config_sha256": config_sha256_value,
            "checked_files": checked_files,
        }
    )
    if recorded_hash != expected_hash:
        return False, "audit report lineage hash mismatch"
    return True, "audit report is internally consistent"


def _inspect_run(
    root: Path,
    *,
    required_experiments: Iterable[str] | None,
    required_files: Iterable[str] | None,
    require_audited: bool,
    require_formal: bool,
    expected_config_sha256: str | None,
    group_name: str | None = None,
    require_formal_seed: bool = False,
    expected_group_id: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    manifest_path = root / "manifest" / "run.json"
    manifest: dict[str, Any] = {}
    try:
        payload = _read_json(manifest_path)
        if isinstance(payload, dict):
            manifest = payload
            _check(checks, "run_manifest_readable", True, str(manifest_path.relative_to(root).as_posix()))
        else:
            _check(checks, "run_manifest_readable", False, "run manifest must be a JSON object")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _check(checks, "run_manifest_readable", False, str(exc))

    run_id = str(manifest.get("run_id", ""))
    _check(checks, "run_manifest_schema", manifest.get("schema_version") == 1, "run manifest schema_version is 1")
    
    # Stage-level resume is permitted under the hash-chain policy; the lineage is
    # still auditable because checkpoint.json binds every completed stage to config
    # and code hashes, and resumed runs record all resume events.
    allow_resume = manifest.get("allow_resume")
    if allow_resume is True:
        _check(
            checks,
            "run_manifest_resume_auditable",
            (manifest.get("resume_events") is not None) or (root / "checkpoint.json").is_file(),
            "resumed run has checkpoint.json or resume_events recorded"
        )
    else:
        _check(checks, "run_manifest_no_resume", allow_resume is False, "run manifest disallows resume")
    
    _check(checks, "run_manifest_no_cross_run_cache", manifest.get("allow_cross_run_cache") is False, "run manifest disallows cross-run cache")
    _check(checks, "run_manifest_config", isinstance(manifest.get("resolved_config"), dict), "resolved config is present")
    _check(checks, "run_id", bool(run_id) and run_id == root.name and bool(_RUN_ID.fullmatch(run_id)), f"run_id={run_id!r}")
    evidence_tier = str(manifest.get("evidence_tier", ""))
    if require_formal:
        _check(checks, "formal_evidence", evidence_tier == "formal", f"evidence_tier={evidence_tier!r}")
    else:
        _check(checks, "evidence_tier_present", evidence_tier in {"development", "formal"}, f"evidence_tier={evidence_tier!r}")

    status_from_manifest = str(manifest.get("status", ""))
    status_path = root / "STATUS"
    try:
        status_from_file = status_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        status_from_file = ""
        _check(checks, "STATUS_readable", False, str(exc))
    else:
        _check(checks, "STATUS_readable", True, f"STATUS={status_from_file!r}")
    _check(checks, "STATUS_matches_manifest", bool(status_from_file) and status_from_file == status_from_manifest, "STATUS agrees with run manifest")
    allowed_statuses = {"audited"} if require_audited else {"completed", "audited"}
    _check(checks, "run_status", status_from_manifest in allowed_statuses, f"status={status_from_manifest!r}")

    resolved_config = manifest.get("resolved_config", {})
    if not isinstance(resolved_config, dict):
        resolved_config = {}
    frozen = manifest.get("frozen") is True or resolved_config.get("run", {}).get("frozen") is True
    release_section = resolved_config.get("release", {})
    release_status = release_section.get("status") if isinstance(release_section, dict) else None
    if require_formal:
        run_config_tier = resolved_config.get("run", {}).get("evidence_tier")
        _check(checks, "resolved_config_evidence", run_config_tier in {None, "formal"}, f"resolved run evidence_tier={run_config_tier!r}")
        _check(checks, "frozen_config", frozen, "formal run has an explicit frozen flag")
        _check(checks, "frozen_release", release_status == "frozen", f"release.status={release_status!r}")

    stored_hash = str(manifest.get("resolved_config_sha256", ""))
    expected_hash = expected_config_sha256 or stored_hash
    if expected_config_sha256 is not None:
        expected_hash_valid = bool(_SHA256.fullmatch(str(expected_config_sha256)))
        _check(
            checks,
            "expected_config_hash",
            expected_hash_valid and stored_hash == str(expected_config_sha256),
            "run config hash matches the registered expected hash"
            if expected_hash_valid and stored_hash == str(expected_config_sha256)
            else "run config hash differs from the registered expected hash",
        )
    resolved_config_hash = config_sha256(resolved_config) if isinstance(resolved_config, dict) else ""
    _check(
        checks,
        "resolved_config_hash",
        bool(_SHA256.fullmatch(stored_hash)) and resolved_config_hash == stored_hash,
        "resolved config content matches its recorded SHA256"
        if bool(_SHA256.fullmatch(stored_hash)) and resolved_config_hash == stored_hash
        else "resolved config content does not match resolved_config_sha256",
    )
    formal_seed_path = root / "manifest" / "formal_seed.json"
    formal_seed_payload: dict[str, Any] | None = None
    if require_formal_seed and not formal_seed_path.is_file():
        _check(
            checks,
            "formal_seed_manifest",
            False,
            "formal run is missing manifest/formal_seed.json",
        )
    if formal_seed_path.is_file():
        try:
            candidate = _read_json(formal_seed_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _check(checks, "formal_seed_manifest", False, f"unreadable formal seed manifest: {exc}")
        else:
            if isinstance(candidate, dict):
                formal_seed_payload = candidate
                required_seed_experiments = required_experiments
                seed_experiments = candidate.get("required_experiments")
                seed_manifest_ok = (
                    candidate.get("run_id") == manifest.get("run_id")
                    and candidate.get("config_sha256") == stored_hash
                    and candidate.get("allow_resume") in (False, True)
                    and candidate.get("allow_cross_run_cache") is False
                    and (
                        not isinstance(required_seed_experiments, list)
                        or seed_experiments == required_seed_experiments
                    )
                )
                if require_formal_seed:
                    seed_manifest_ok = seed_manifest_ok and bool(candidate.get("group_id")) and isinstance(
                        candidate.get("seed"), int
                    )
                if expected_group_id is not None:
                    seed_manifest_ok = seed_manifest_ok and candidate.get("group_id") == expected_group_id
                if expected_seed is not None:
                    seed_manifest_ok = seed_manifest_ok and candidate.get("seed") == expected_seed
                if expected_group_id is not None and expected_seed is not None:
                    expected_formal_run_id = f"formal-{expected_group_id}-seed-{expected_seed}"
                    seed_manifest_ok = seed_manifest_ok and manifest.get("run_id") == expected_formal_run_id
                _check(
                    checks,
                    "formal_seed_manifest",
                    seed_manifest_ok,
                    "formal seed manifest binds this run and disallows resume"
                    if seed_manifest_ok
                    else "formal seed manifest does not bind this run/config or allows resume",
                )
            else:
                _check(checks, "formal_seed_manifest", False, "formal seed manifest must be a JSON object")
    if formal_seed_payload is not None and isinstance(formal_seed_payload.get("required_experiments"), list):
        required_experiments = formal_seed_payload["required_experiments"]
    _qualification_check(root, manifest, expected_hash, checks)

    metric_paths = _metric_files(root)
    _check(checks, "metrics_present", bool(metric_paths), "metrics contains JSON/JSONL files")
    metric_experiments, metric_statuses, metric_errors, metric_markers = _metric_evidence(root)
    _check(checks, "metrics_parseable", not metric_errors, "metric JSON/JSONL files parse")
    expected_experiment_ids = _configured_experiments(manifest, required_experiments, group_name=group_name)
    present_experiments = set(metric_experiments)
    observed_experiments = set(present_experiments) | set(metric_statuses) | set(metric_markers)
    unexpected_experiments = sorted(observed_experiments - set(expected_experiment_ids))
    _check(
        checks,
        "exact_experiment_membership",
        not unexpected_experiments,
        "metric evidence matches the registered experiment set"
        if not unexpected_experiments
        else "unexpected experiment evidence: " + ", ".join(unexpected_experiments),
    )
    blocked_experiments = {
        experiment_id
        for experiment_id, marker in metric_markers.items()
        if _marker_is_ineligible(marker)
    }
    for experiment_id in _QUALIFICATION:
        qualification_path = root / "manifest" / f"{experiment_id.lower()}.json"
        if not qualification_path.is_file() or experiment_id in blocked_experiments:
            continue
        try:
            qualification_payload = _read_json(qualification_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            qualification_payload = {}
        if isinstance(qualification_payload, Mapping) and _payload_evidence_eligible(qualification_payload):
            present_experiments.add(experiment_id)
    missing_experiments = sorted(set(expected_experiment_ids) - present_experiments)
    _check(checks, "required_experiments", not missing_experiments, "all required experiments have evidence" if not missing_experiments else "missing: " + ", ".join(missing_experiments))
    blocked_required = sorted(set(expected_experiment_ids) & blocked_experiments)
    _check(
        checks,
        "required_experiment_evidence_status",
        not blocked_required,
        "required experiments are eligible evidence"
        if not blocked_required
        else "ineligible evidence: " + ", ".join(blocked_required),
    )
    failed_experiments = sorted(
        experiment_id
        for experiment_id, statuses in metric_statuses.items()
        if experiment_id in expected_experiment_ids
        and statuses
        and statuses.intersection(_INELIGIBLE_EVIDENCE_STATUSES | {"failed", "error", "invalid"})
    )
    _check(
        checks,
        "required_experiment_status",
        not failed_experiments,
        "required metric experiments contain no failed or partial records"
        if not failed_experiments
        else "failed or partial records: " + ", ".join(failed_experiments),
    )
    _stage_contract_check(expected_experiment_ids, metric_markers, checks)

    statistics_root = root / "statistics"
    statistics_files = [path for path in statistics_root.rglob("*") if path.is_file()] if statistics_root.exists() else []
    nonempty_statistics = [path for path in statistics_files if path.stat().st_size > 0]
    _check(checks, "statistics_present", bool(nonempty_statistics), "statistics contains non-empty files")
    statistics_errors: list[str] = []
    for path in nonempty_statistics:
        if path.suffix.lower() in {".json", ".jsonl"}:
            try:
                if path.suffix.lower() == ".json":
                    _read_json(path)
                else:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            json.loads(line)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                statistics_errors.append(f"{path.relative_to(root).as_posix()}: {exc}")
    _check(checks, "statistics_parseable", not statistics_errors, "statistics files parse" if not statistics_errors else "; ".join(statistics_errors))

    required_paths: list[Path] = []
    for relative in required_files or ():
        try:
            required_path = _resolve_relative(root, str(relative))
            required_ok = required_path.is_file() and required_path.stat().st_size > 0
            detail = str(relative) if required_ok else f"missing or empty: {relative}"
            if required_ok:
                required_paths.append(required_path)
        except (OSError, ValueError) as exc:
            required_ok = False
            detail = f"{relative}: {exc}"
        _check(checks, f"required_file:{relative}", required_ok, detail)

    lineage = _validate_lineage(root, manifest, checks)
    audit_path = root / "manifest" / "audit.json"
    audit_payload: dict[str, Any] | None = None
    if require_audited:
        try:
            audit_candidate = _read_json(audit_path)
            if isinstance(audit_candidate, dict):
                audit_payload = audit_candidate
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            audit_payload = None
        audit_ok, audit_detail = _validate_audit_payload(
            root,
            audit_payload,
            run_id=run_id,
            config_sha256_value=stored_hash,
        )
        _check(checks, "audit_report", audit_ok, audit_detail)

    all_passed = all(item["passed"] for item in checks)
    lineage_paths: list[Path] = []
    if isinstance(lineage, Mapping):
        for field in ("source_files", "source_manifest", "statistics_files", "outputs"):
            for item in lineage.get(field, []) if isinstance(lineage.get(field), list) else []:
                if isinstance(item, Mapping) and isinstance(item.get("path"), str):
                    try:
                        lineage_path = _resolve_relative(root, str(item["path"]))
                    except ValueError:
                        continue
                    if lineage_path.is_file():
                        lineage_paths.append(lineage_path)
    checked_paths = [
        path
        for path in [manifest_path, status_path, formal_seed_path, root / "manifest/e00.json", root / "manifest/e01.json", root / "manifest/e02.json", *metric_paths, *_stage_manifest_files(root), *nonempty_statistics, *required_paths, *lineage_paths, *_lineage_candidates(root), audit_path if require_audited else None]
        if path is not None and path.is_file()
    ]
    checked_paths = list(dict.fromkeys(checked_paths))
    return {
        "passed": all_passed,
        "checks": checks,
        "errors": [item["details"] for item in checks if not item["passed"]],
        "manifest": manifest,
        "lineage": lineage,
        "audit": audit_payload,
        "run_id": run_id,
        "status": status_from_manifest,
        "evidence_tier": evidence_tier,
        "config_sha256": stored_hash,
        "required_experiments": expected_experiment_ids,
        "checked_paths": checked_paths,
    }


def audit_run(
    run_root: Path | str,
    *,
    required_experiments: Iterable[str] | None = None,
    required_files: Iterable[str] | None = None,
    output_path: Path | str | None = None,
    mark_audited: bool = True,
    require_formal: bool = False,
    require_formal_seed: bool = False,
    expected_config_sha256: str | None = None,
    expected_group_id: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Audit one run and mark it audited only after every check passes.

    ``require_formal`` is used by the formal group auditor.  The default keeps
    development QC useful without allowing a development run to enter a
    release; formal auditing additionally requires a frozen formal config and a
    bound ``formal_seed.json`` manifest.
    """

    root = Path(run_root).expanduser().resolve(strict=True)
    inspection = _inspect_run(
        root,
        required_experiments=required_experiments,
        required_files=required_files,
        require_audited=False,
        require_formal=require_formal,
        expected_config_sha256=expected_config_sha256,
        require_formal_seed=require_formal_seed,
        expected_group_id=expected_group_id,
        expected_seed=expected_seed,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "run_audit",
        "run_id": inspection["run_id"] or root.name,
        "status_before": inspection["status"],
        "status_after": inspection["status"],
        "evidence_tier": inspection["evidence_tier"],
        "config_sha256": inspection["config_sha256"],
        "required_experiments": inspection["required_experiments"],
        "passed": inspection["passed"],
        "checks": inspection["checks"],
        "checked_files": [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            for path in inspection["checked_paths"]
        ],
        "generated_at_utc": _now(),
    }
    report["lineage_sha256"] = sha256_json(
        {
            "run_id": report["run_id"],
            "config_sha256": report["config_sha256"],
            "checked_files": report["checked_files"],
        }
    )
    if output_path is None:
        target = root / "manifest" / "audit.json"
    else:
        candidate = Path(output_path).expanduser()
        target = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if output_path is not None and root not in target.parents:
        raise AuditError("Audit report must be written inside the run directory")
    _write_json(target, report)

    if inspection["passed"] and mark_audited and inspection["status"] == "completed":
        try:
            update_run_status(root, "audited")
        except Exception as exc:
            report["passed"] = False
            report["status_after"] = inspection["status"]
            report["status_update_error"] = f"{type(exc).__name__}: {exc}"
        else:
            report["status_after"] = "audited"
        report["checked_files"] = [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            for path in inspection["checked_paths"]
            if path.is_file()
        ]
        report["lineage_sha256"] = sha256_json(
            {
                "run_id": report["run_id"],
                "config_sha256": report["config_sha256"],
                "checked_files": report["checked_files"],
            }
        )
        _write_json(target, report)
        try:
            _update_formal_group_after_audit(root, report["run_id"])
        except AuditError as exc:
            report["passed"] = False
            report["group_manifest_update_error"] = str(exc)
            _write_json(target, report)

    return report


def _load_manifest_source(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return deepcopy(dict(source))
    path = Path(source).expanduser().resolve(strict=True)
    try:
        payload = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"Could not read release/run-group manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ReleaseError("Release/run-group manifest must be a JSON object")
    return payload


def _group_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("manifest_type") == "formal_run_group":
        return _formal_group_entries(payload)
    raw = payload.get("run_groups")
    if raw is None:
        raw = payload.get("groups")
    if raw is None:
        raw = payload.get("runs")
    if raw is None:
        return []
    entries: list[dict[str, Any]] = []

    def add_entry(group_id: str, value: Any, group_fields: Mapping[str, Any] | None = None) -> None:
        if isinstance(value, str):
            value = {"run_id": value}
        if not isinstance(value, Mapping):
            return
        entry = dict(group_fields or {})
        entry.update(dict(value))
        entry.setdefault("group_id", group_id)
        if group_fields:
            if any(key in group_fields for key in ("experiments", "required_experiments")):
                entry["_declared_group_experiments"] = _as_strings(
                    group_fields.get("experiments", group_fields.get("required_experiments"))
                )
            if any(key in group_fields for key in ("seeds", "split_seeds", "seed_ids")):
                entry["_declared_group_seeds"] = _seed_values(
                    group_fields.get("seeds", group_fields.get("split_seeds", group_fields.get("seed_ids")))
                )
            if "scientific_dependencies" in group_fields or "dependencies" in group_fields:
                entry["_declared_group_dependencies"] = _as_strings(
                    group_fields.get("scientific_dependencies", group_fields.get("dependencies"))
                )
            group_hash = _config_hash_from_entry(group_fields)
            if group_hash:
                entry["_declared_group_config_sha256"] = group_hash
            manifest_hash = _manifest_hash_from_entry(group_fields)
            if manifest_hash:
                entry["_declared_group_manifest_hash"] = manifest_hash
        entries.append(entry)

    if isinstance(raw, dict):
        if any(key in raw for key in ("run_id", "id", "run_path", "path", "run_root")):
            raw = {str(payload.get("group_id", "atomic")): raw}
        for group_id, value in raw.items():
            if isinstance(value, str):
                add_entry(str(group_id), value)
            elif isinstance(value, dict):
                nested = _nested_run_values(value)
                if nested is None:
                    add_entry(str(group_id), value)
                else:
                    group_fields = {key: item for key, item in value.items() if key not in _NESTED_RUN_KEYS}
                    for child in nested:
                        add_entry(str(group_id), child, group_fields)
    elif isinstance(raw, list):
        for index, value in enumerate(raw):
            if isinstance(value, str):
                add_entry(f"atomic-{index + 1}", value)
            elif isinstance(value, dict):
                group_id = str(value.get("group_id", f"atomic-{index + 1}"))
                nested = _nested_run_values(value)
                if nested is None:
                    add_entry(group_id, value)
                else:
                    group_fields = {key: item for key, item in value.items() if key not in _NESTED_RUN_KEYS}
                    for child in nested:
                        add_entry(group_id, child, group_fields)
    return entries


_NESTED_RUN_KEYS = ("runs", "members", "seed_runs", "run_entries", "atomic_runs", "seeds")


def _nested_run_values(value: Mapping[str, Any]) -> list[Any] | None:
    for key in _NESTED_RUN_KEYS:
        nested = value.get(key)
        if nested is not None:
            if not isinstance(nested, list):
                raise ReleaseError(f"Group {value.get('group_id', '?')} has malformed {key}")
            return nested
    return None


def _seed_values(value: Any) -> list[int | str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[int | str] = []
    for item in values:
        if isinstance(item, Mapping) and "seed" in item:
            result.extend(_seed_values(item["seed"]))
            continue
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            continue
        text = str(item).strip()
        if text:
            result.append(int(text) if text.isdigit() else text)
    return result


def _entry_seed_values(entry: Mapping[str, Any]) -> list[int | str]:
    for key in ("seed", "split_seed", "seed_id"):
        if key in entry:
            return _seed_values(entry[key])
    for key in ("seeds", "split_seeds", "seed_ids"):
        if key in entry:
            return _seed_values(entry[key])
    return []


def _config_hash_from_entry(entry: Mapping[str, Any]) -> str:
    for key in ("config_sha256", "config_hash", "group_config_sha256", "group_config_hash"):
        value = entry.get(key)
        if value:
            return str(value)
    return ""


def _manifest_hash_from_entry(entry: Mapping[str, Any]) -> str:
    for key in ("manifest_hash", "manifest_sha256", "group_manifest_hash", "group_manifest_sha256"):
        value = entry.get(key)
        if value:
            return str(value)
    return ""


def _registration_source(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("frozen_group_manifest") or payload.get("formal_group_manifest")
    return nested if isinstance(nested, Mapping) else payload


def _registration_fields(payload: Mapping[str, Any]) -> tuple[str, str]:
    nested = payload.get("pre_registration")
    nested_mapping = nested if isinstance(nested, Mapping) else {}
    pre_registered_at = payload.get("pre_registered_at", nested_mapping.get("pre_registered_at", ""))
    manifest_hash = payload.get(
        "manifest_hash",
        payload.get("manifest_sha256", nested_mapping.get("manifest_hash", nested_mapping.get("manifest_sha256", ""))),
    )
    return str(pre_registered_at or ""), str(manifest_hash or "")


def _canonical_manifest_hash(payload: Mapping[str, Any]) -> str:
    without_hash = deepcopy(dict(payload))
    without_hash.pop("manifest_hash", None)
    without_hash.pop("manifest_sha256", None)
    nested = without_hash.get("pre_registration")
    if isinstance(nested, Mapping):
        nested_without_hash = dict(nested)
        nested_without_hash.pop("manifest_hash", None)
        nested_without_hash.pop("manifest_sha256", None)
        without_hash["pre_registration"] = nested_without_hash
    return sha256_json(without_hash)


def _canonical_registration_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only fields that are fixed before execution for mutable group plans."""

    projection = deepcopy(dict(payload))
    projection.pop("manifest_hash", None)
    projection.pop("manifest_sha256", None)
    projection.pop("status", None)
    projection.pop("audit_status", None)
    seeds = projection.get("seeds")
    if isinstance(seeds, list):
        projected_seeds: list[Any] = []
        for item in seeds:
            if isinstance(item, Mapping):
                projected = dict(item)
                for key in ("status", "audit_status", "audit_path", "failure_reason"):
                    projected.pop(key, None)
                projected_seeds.append(projected)
            else:
                projected_seeds.append(item)
        projection["seeds"] = projected_seeds
    nested = projection.get("pre_registration")
    if isinstance(nested, Mapping):
        nested_projection = dict(nested)
        nested_projection.pop("manifest_hash", None)
        nested_projection.pop("manifest_sha256", None)
        projection["pre_registration"] = nested_projection
    return projection


def _manifest_hash_matches(payload: Mapping[str, Any], recorded_hash: str) -> bool:
    if not _SHA256.fullmatch(recorded_hash):
        return False
    return recorded_hash in {
        _canonical_manifest_hash(payload),
        sha256_json(_canonical_registration_projection(payload)),
    }


def _registration_timestamp_valid(payload: Mapping[str, Any], value: str) -> bool:
    if not value:
        return False
    try:
        registered_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if registered_at.tzinfo is None:
        return False
    for field in ("generated_at_utc", "results_generated_at_utc", "execution_started_at_utc"):
        comparison = payload.get(field)
        if not isinstance(comparison, str) or not comparison:
            continue
        try:
            comparison_at = datetime.fromisoformat(comparison.replace("Z", "+00:00"))
        except ValueError:
            return False
        if comparison_at.tzinfo is not None and registered_at > comparison_at:
            return False
    return True


def _validate_registration_manifest(payload: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    source = _registration_source(payload)
    pre_registered_at, manifest_hash = _registration_fields(source)
    timestamp_ok = _registration_timestamp_valid(source, pre_registered_at)
    hash_ok = _manifest_hash_matches(source, manifest_hash)
    _check(
        checks,
        "pre_registered_at",
        timestamp_ok,
        "frozen group manifest records pre_registered_at" if timestamp_ok else "pre_registered_at is missing or malformed",
    )
    _check(
        checks,
        "group_manifest_hash",
        hash_ok,
        "frozen group manifest hash matches its pre-registration content"
        if hash_ok
        else "manifest_hash is missing, malformed, or does not match the frozen group manifest",
    )


def _validate_formal_group_registration(payload: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    """Validate the immutable group snapshot before inspecting seed runs."""

    snapshot = payload.get("frozen_group_manifest")
    if not isinstance(snapshot, Mapping):
        _check(checks, "formal_group_snapshot_present", False, "formal group manifest lacks frozen_group_manifest")
        return
    snapshot_hash = str(snapshot.get("manifest_hash", ""))
    snapshot_without_hash = dict(snapshot)
    snapshot_without_hash.pop("manifest_hash", None)
    snapshot_ok = bool(_SHA256.fullmatch(snapshot_hash)) and snapshot_hash == sha256_json(snapshot_without_hash)
    _check(
        checks,
        "formal_group_snapshot_hash",
        snapshot_ok and payload.get("manifest_hash") == snapshot_hash,
        "formal group snapshot hash is internally consistent"
        if snapshot_ok and payload.get("manifest_hash") == snapshot_hash
        else "formal group snapshot hash is missing or inconsistent",
    )
    for field in (
        "group_id",
        "config_sha256",
        "required_experiments",
        "dependencies",
        "seed_policy",
        "from_raw_inputs",
        "allow_cross_group_derived_inputs",
    ):
        _check(
            checks,
            f"formal_group:{field}",
            payload.get(field) == snapshot.get(field),
            f"formal group {field} matches its frozen snapshot"
            if payload.get(field) == snapshot.get(field)
            else f"formal group {field} differs from its frozen snapshot",
        )
    timestamp, _ = _registration_fields(snapshot)
    _check(
        checks,
        "formal_group_pre_registration",
        _registration_timestamp_valid(snapshot, timestamp),
        "formal group records a valid pre-registration timestamp"
        if _registration_timestamp_valid(snapshot, timestamp)
        else "formal group pre-registration timestamp is missing or invalid",
    )
    planned = snapshot.get("seeds")
    current = payload.get("seeds")
    if not isinstance(planned, list) or not isinstance(current, list):
        _check(checks, "formal_group_seed_plan", False, "formal group seed plan is missing")
        return
    planned_projection = {
        (item.get("seed"), item.get("run_id"), item.get("run_path"), item.get("config_sha256"))
        for item in planned
        if isinstance(item, Mapping)
    }
    current_projection = {
        (item.get("seed"), item.get("run_id"), item.get("run_path"), item.get("config_sha256"))
        for item in current
        if isinstance(item, Mapping)
    }
    _check(
        checks,
        "formal_group_seed_plan",
        planned_projection == current_projection and len(current_projection) == len(current),
        "all current seed entries match the frozen group plan"
        if planned_projection == current_projection and len(current_projection) == len(current)
        else "current seed entries differ from the frozen group plan",
    )


def _validate_formal_release_groups_registration(
    payload: Mapping[str, Any],
    project_root: Path,
    checks: list[dict[str, Any]],
) -> None:
    """Validate each immutable source group referenced by an aggregate plan."""

    source_paths = payload.get("source_group_manifests")
    if not isinstance(source_paths, list) or not source_paths:
        _check(checks, "formal_release_groups_sources", False, "aggregate manifest has no source_group_manifests")
        return
    aggregate_hash = str(payload.get("config_sha256", ""))
    all_valid = True
    for relative in source_paths:
        if not isinstance(relative, str):
            all_valid = False
            checks.append({"name": "formal_release_groups_source", "passed": False, "details": "malformed source group path"})
            continue
        try:
            path = (project_root / relative).resolve()
            if project_root not in path.parents or not path.is_file():
                raise ReleaseError(f"missing source group manifest: {relative}")
            source = _read_json(path)
            if not isinstance(source, Mapping) or source.get("manifest_type") != "formal_run_group":
                raise ReleaseError(f"source is not a formal_run_group manifest: {relative}")
            source_hash = str(source.get("config_sha256", ""))
            snapshot = source.get("frozen_group_manifest")
            snapshot_hash = str(snapshot.get("manifest_hash", "")) if isinstance(snapshot, Mapping) else ""
            if source_hash != aggregate_hash or source.get("status") != "audited" or source.get("audit_status") != "audited":
                raise ReleaseError(f"source group is not audited or has a config mismatch: {relative}")
            if not _SHA256.fullmatch(snapshot_hash):
                raise ReleaseError(f"source group snapshot hash is missing: {relative}")
            snapshot_without_hash = dict(snapshot)
            snapshot_without_hash.pop("manifest_hash", None)
            if snapshot_hash != sha256_json(snapshot_without_hash) or source.get("manifest_hash") != snapshot_hash:
                raise ReleaseError(f"source group snapshot hash is invalid: {relative}")
            if source.get("group_id") != snapshot.get("group_id"):
                raise ReleaseError(f"source group ID differs from its snapshot: {relative}")
            if source.get("required_experiments") != snapshot.get("required_experiments"):
                raise ReleaseError(f"source group experiment membership differs from its snapshot: {relative}")
            seed_entries = source.get("seeds")
            if not isinstance(seed_entries, list) or not seed_entries or any(
                not isinstance(item, Mapping) or item.get("status") != "audited" or item.get("audit_status") != "audited"
                for item in seed_entries
            ):
                raise ReleaseError(f"source group has incomplete audited seeds: {relative}")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ReleaseError, ValueError) as exc:
            all_valid = False
            checks.append({"name": "formal_release_groups_source", "passed": False, "details": str(exc)})
    _check(
        checks,
        "formal_release_groups_sources",
        all_valid,
        "all source group manifests are audited and hash-consistent" if all_valid else "one or more source group manifests are invalid",
    )


def _formal_group_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand one immutable multi-seed group plan into release seed entries."""

    if payload.get("schema_version") != 1:
        raise ReleaseError("Unsupported formal run-group manifest schema")
    group_id = payload.get("group_id")
    if not isinstance(group_id, str) or not group_id:
        raise ReleaseError("Formal run-group manifest requires group_id")
    if payload.get("status") != "audited":
        raise ReleaseError(
            f"Formal run-group manifest status={payload.get('status')!r}; expected audited"
        )
    experiments = payload.get("required_experiments") or payload.get("experiments")
    if not isinstance(experiments, list) or not experiments or any(not isinstance(item, str) or not item for item in experiments):
        raise ReleaseError("Formal run-group manifest requires required experiments")
    dependencies = payload.get("dependencies", payload.get("scientific_dependencies", []))
    if not isinstance(dependencies, list) or any(not isinstance(item, str) or not item for item in dependencies):
        raise ReleaseError("Formal run-group dependencies must be a list of strings")
    group_config_hash = str(payload.get("config_sha256", ""))
    if not _SHA256.fullmatch(group_config_hash):
        raise ReleaseError("Formal run-group config_sha256 must be a SHA256 digest")
    raw_seeds = payload.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ReleaseError("Formal run-group manifest requires a non-empty seeds list")

    entries: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    seen_runs: set[str] = set()
    for seed_entry in raw_seeds:
        if not isinstance(seed_entry, dict):
            raise ReleaseError("Every formal group seed entry must be an object")
        seed = seed_entry.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ReleaseError("Formal group seed must be an integer")
        if seed in seen_seeds:
            raise ReleaseError(f"Duplicate formal group seed: {seed}")
        run_id = seed_entry.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in seen_runs:
            raise ReleaseError(f"Duplicate or missing formal group run_id: {run_id!r}")
        expected_run_id = f"formal-{group_id}-seed-{seed}"
        if run_id != expected_run_id:
            raise ReleaseError(f"Formal group seed {seed} has unexpected run_id: {run_id!r}")
        seed_config_hash = str(seed_entry.get("config_sha256", ""))
        if not _SHA256.fullmatch(seed_config_hash):
            raise ReleaseError(f"Malformed config_sha256 for formal group seed {seed}")
        seed_status = seed_entry.get("status")
        if seed_status != "audited":
            raise ReleaseError(
                f"Formal group seed {seed} is not audited: status={seed_status!r}"
            )
        if seed_entry.get("audit_status") != "audited":
            raise ReleaseError(f"Formal group seed {seed} does not have an audited status")
        run_path = seed_entry.get("run_path") or seed_entry.get("run_root")
        if not isinstance(run_path, str) or not run_path:
            raise ReleaseError(f"Formal group seed {seed} lacks run_path")
        expected_run_path = f"artifacts/runs/{expected_run_id}"
        if Path(run_path).as_posix() != expected_run_path:
            raise ReleaseError(f"Formal group seed {seed} has unexpected run_path: {run_path!r}")
        entries.append(
            {
                "group_id": f"{group_id}-seed-{seed}",
                "parent_group_id": group_id,
                "seed": seed,
                "run_id": run_id,
                "run_path": run_path,
                "config_sha256": seed_config_hash,
                "group_config_sha256": group_config_hash,
                "experiments": list(experiments),
                "dependencies": list(dependencies),
                "seed_status": seed_status,
                "seed_audit_status": seed_entry.get("audit_status", "not_audited"),
                "pre_registered": True,
            }
        )
        seen_seeds.add(seed)
        seen_runs.add(run_id)
    return entries


def _project_root_from_inputs(
    project_root: Path | str | None,
    release_source: Path | str | Mapping[str, Any] | None,
    entries: Iterable[Mapping[str, Any]],
) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve(strict=True)
    if release_source is not None and not isinstance(release_source, Mapping):
        source_path = Path(release_source).expanduser().resolve()
        for parent in (source_path, *source_path.parents):
            if parent.name == "releases" and parent.parent.name == "artifacts":
                return parent.parent.parent
    for entry in entries:
        path_value = entry.get("run_root") or entry.get("run_path") or entry.get("path")
        if path_value:
            path = Path(str(path_value)).expanduser()
            if path.is_absolute():
                resolved = path.resolve()
                for parent in (resolved, *resolved.parents):
                    if parent.name == "runs" and parent.parent.name == "artifacts":
                        return parent.parent.parent
    return Path.cwd().resolve()


def _formal_group_payload_as_release(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a group plan for validation without changing the source plan."""

    entries = _formal_group_entries(payload)
    release_payload: dict[str, Any] = {
        "schema_version": 1,
        "release_id": f"group-{payload['group_id']}",
        "status": "frozen",
        "aggregation_policy": "frozen_manifest_only",
        "config_sha256": payload.get("config_sha256"),
        "required_experiments": payload.get("required_experiments") or payload.get("experiments"),
        "run_groups": entries,
        "created_at_utc": payload.get("created_at_utc", _now()),
        "frozen_group_manifest": deepcopy(dict(payload)),
    }
    release_payload["lineage"] = None
    return release_payload


def _update_formal_group_after_audit(root: Path, run_id: str) -> None:
    """Reflect a successful seed audit in its owning group manifest."""

    seed_manifest_path = root / "manifest" / "formal_seed.json"
    if not seed_manifest_path.is_file():
        return
    try:
        seed_manifest = _read_json(seed_manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"Formal seed manifest is unreadable: {seed_manifest_path}") from exc
    if not isinstance(seed_manifest, dict):
        raise AuditError(f"Formal seed manifest is not an object: {seed_manifest_path}")
    group_manifest_value = seed_manifest.get("group_manifest_path")
    if not isinstance(group_manifest_value, str) or not group_manifest_value:
        raise AuditError("Formal seed manifest lacks group_manifest_path")
    project_root = root.parents[2]
    group_manifest_path = (project_root / group_manifest_value).resolve()
    if project_root not in group_manifest_path.parents or not group_manifest_path.is_file():
        raise AuditError(f"Formal group manifest is missing: {group_manifest_path}")
    try:
        group_payload = _read_json(group_manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"Formal group manifest is unreadable: {group_manifest_path}") from exc
    if not isinstance(group_payload, dict) or group_payload.get("manifest_type") != "formal_run_group":
        raise AuditError("Formal group manifest has an invalid schema")
    seeds = group_payload.get("seeds")
    if not isinstance(seeds, list):
        raise AuditError("Formal group manifest has no seeds list")
    matched = False
    for entry in seeds:
        if isinstance(entry, dict) and entry.get("run_id") == run_id:
            entry["status"] = "audited"
            entry["audit_status"] = "audited"
            entry["audit_path"] = "manifest/audit.json"
            matched = True
            break
    if not matched:
        raise AuditError(f"Run {run_id} is not registered in formal group manifest")
    if all(isinstance(entry, dict) and entry.get("status") == "audited" and entry.get("audit_status") == "audited" for entry in seeds):
        group_payload["status"] = "audited"
        group_payload["audit_status"] = "audited"
    _write_json(group_manifest_path, group_payload)


def _resolve_run_entry(project_root: Path, entry: Mapping[str, Any]) -> Path:
    path_value = entry.get("run_root") or entry.get("run_path") or entry.get("path")
    run_id = entry.get("run_id") or entry.get("id")
    if path_value:
        path = Path(str(path_value)).expanduser()
        root = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    elif run_id:
        root = (project_root / "artifacts" / "runs" / str(run_id)).resolve()
    else:
        raise ReleaseError(f"Run-group entry lacks run_id/path: {entry}")
    runs_root = (project_root / "artifacts" / "runs").resolve()
    if runs_root not in root.parents:
        raise ReleaseError(f"Run is outside project artifacts/runs: {root}")
    if not root.is_dir():
        raise ReleaseError(f"Run directory does not exist: {root}")
    return root


def _config_for_release(config: Mapping[str, Any] | Path | str | None, config_path: Path | str | None) -> tuple[dict[str, Any] | None, str | None]:
    if config is not None and config_path is not None:
        raise ReleaseError("provide config or config_path, not both")
    if config_path is not None:
        loaded = load_experiment_config(config_path)
        return loaded, config_sha256(loaded)
    if isinstance(config, (str, Path)):
        loaded = load_experiment_config(config)
        return loaded, config_sha256(loaded)
    if config is not None:
        loaded = dict(config)
        return loaded, config_sha256(loaded)
    return None, None


def _release_experiments(config: Mapping[str, Any] | None, explicit: Iterable[str] | None) -> list[str]:
    if explicit is not None:
        values = _as_strings(list(explicit))
    elif config:
        release = config.get("release", {})
        values = _as_strings(release.get("required_experiments")) if isinstance(release, dict) else []
        if not values:
            scope = config.get("scope", {})
            values = _as_strings(scope.get("experiment_ids")) if isinstance(scope, dict) else []
    else:
        values = []
    result = {_normalise_experiment_id(value) for value in values if _EXPERIMENT.fullmatch(value)}
    result.update(_QUALIFICATION)
    return sorted(result, key=lambda item: (int(re.search(r"\d+", item).group()), item))


def _entry_experiments(
    entry: Mapping[str, Any],
    config: Mapping[str, Any] | None,
    global_experiments: Iterable[str],
) -> list[str]:
    explicit = entry.get("required_experiments") or entry.get("experiments")
    if explicit:
        return _release_experiments(None, _as_strings(explicit))
    group_id = str(entry.get("group_id", ""))
    if config and group_id:
        groups = config.get("run_groups", {})
        if isinstance(groups, dict) and isinstance(groups.get(group_id), dict):
            group_experiments = _as_strings(groups[group_id].get("experiments"))
            if group_experiments:
                return _release_experiments(None, group_experiments)
    return list(global_experiments)


def _entry_group_experiments(entry: Mapping[str, Any]) -> set[str]:
    declared = entry.get("_declared_group_experiments")
    if declared is None:
        declared = entry.get("experiments") or entry.get("required_experiments")
    return {
        _normalise_experiment_id(value)
        for value in _as_strings(declared)
        if _EXPERIMENT.fullmatch(str(value))
    }


def _entry_group_seeds(entry: Mapping[str, Any]) -> set[int | str]:
    if any(key in entry for key in ("seed", "split_seed", "seed_id")):
        return set(_entry_seed_values(entry))
    declared = entry.get("_declared_group_seeds")
    return set(_seed_values(declared)) if declared is not None else set(_entry_seed_values(entry))


def _entry_group_dependencies(entry: Mapping[str, Any]) -> set[str]:
    declared = entry.get("_declared_group_dependencies")
    if declared is None:
        declared = entry.get("scientific_dependencies", entry.get("dependencies", []))
    return set(_as_strings(declared))


def _run_manifest_seed_values(run_manifest: Mapping[str, Any]) -> set[int | str]:
    values: list[int | str] = []
    for key in ("seed", "split_seed", "seed_id", "seeds", "split_seeds", "seed_ids"):
        if key in run_manifest:
            values.extend(_seed_values(run_manifest[key]))
    resolved_config = run_manifest.get("resolved_config")
    if isinstance(resolved_config, Mapping):
        split = resolved_config.get("split")
        if isinstance(split, Mapping):
            values.extend(_seed_values(split.get("seed")))
            values.extend(_seed_values(split.get("split_seeds")))
        run = resolved_config.get("run")
        if isinstance(run, Mapping):
            values.extend(_seed_values(run.get("seed")))
    return set(values)


def _audited_seed_values(run_manifest: Mapping[str, Any], audit_payload: Mapping[str, Any] | None) -> set[int | str]:
    values: list[int | str] = []
    for payload in (run_manifest, audit_payload or {}):
        for key in ("audited_seeds", "audited_seed_ids", "seeds", "split_seeds", "seed_ids"):
            if key in payload:
                values.extend(_seed_values(payload[key]))
        seed_audit = payload.get("seed_audit")
        if isinstance(seed_audit, Mapping):
            for seed, audit in seed_audit.items():
                if isinstance(audit, Mapping):
                    status = str(audit.get("status", audit.get("audit_status", ""))).lower()
                    if status in {"audited", "completed", "passed", "pass", "ok"}:
                        values.extend(_seed_values(seed))
                elif audit is True:
                    values.extend(_seed_values(seed))
    return set(values)


def _config_group_specs(config: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if not config:
        return {}
    groups = config.get("run_groups")
    return {
        str(group_id): value
        for group_id, value in groups.items()
        if isinstance(groups, Mapping) and isinstance(value, Mapping)
    } if isinstance(groups, Mapping) else {}


def _transitive_group_dependencies(group_id: str, groups: Mapping[str, Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    pending = list(_as_strings(groups.get(group_id, {}).get("scientific_dependencies", [])))
    while pending:
        dependency = pending.pop()
        if dependency in result:
            continue
        result.add(dependency)
        pending.extend(_as_strings(groups.get(dependency, {}).get("scientific_dependencies", [])))
    return result


def _validate_group_manifest_contract(
    payload: Mapping[str, Any],
    entries: list[dict[str, Any]],
    config: Mapping[str, Any] | None,
    checks: list[dict[str, Any]],
) -> dict[str, set[int | str]]:
    group_specs = _config_group_specs(config)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        group_id = str(entry.get("parent_group_id") or entry.get("group_id", ""))
        grouped.setdefault(group_id, []).append(entry)

    configured_group_order = []
    if config:
        release = config.get("release", {})
        if isinstance(release, Mapping):
            configured_group_order = _as_strings(release.get("run_groups", []))
    declared_groups = set(grouped)
    if configured_group_order:
        expected_groups = set(configured_group_order)
        _check(
            checks,
            "exact_group_membership",
            declared_groups == expected_groups,
            "manifest groups exactly match formal config"
            if declared_groups == expected_groups
            else f"group mismatch: missing={sorted(expected_groups - declared_groups)}, extra={sorted(declared_groups - expected_groups)}",
        )

    covered_seeds: dict[str, set[int | str]] = {}
    for group_id, group_entries in grouped.items():
        config_group = group_specs.get(group_id)
        configured_experiments = (
            {
                _normalise_experiment_id(value)
                for value in _as_strings(config_group.get("experiments", []))
                if _EXPERIMENT.fullmatch(str(value))
            }
            if config_group
            else set()
        )
        configured_experiment_list = (
            [
                _normalise_experiment_id(value)
                for value in _as_strings(config_group.get("experiments", []))
                if _EXPERIMENT.fullmatch(str(value))
            ]
            if config_group
            else []
        )
        declared_experiment_sets = {
            frozenset(_entry_group_experiments(entry))
            for entry in group_entries
            if _entry_group_experiments(entry)
        }
        if config_group:
            exact_experiments = all(value == configured_experiments for value in declared_experiment_sets)
            _check(
                checks,
                f"group:{group_id}:experiments",
                exact_experiments and bool(declared_experiment_sets),
                "group experiments exactly match formal config"
                if exact_experiments and declared_experiment_sets
                else f"group experiment membership mismatch for {group_id}",
            )
        elif len(declared_experiment_sets) > 1:
            _check(checks, f"group:{group_id}:experiments_consistent", False, "group entries disagree on experiment membership")

        configured_seeds = set(_seed_values(config_group.get("split_seeds", []))) if config_group else set()
        declared_seeds: set[int | str] = set()
        for entry in group_entries:
            declared_seeds.update(_entry_group_seeds(entry))
        if configured_seeds:
            _check(
                checks,
                f"group:{group_id}:seeds_declared",
                declared_seeds == configured_seeds,
                "all formal group seeds are declared"
                if declared_seeds == configured_seeds
                else f"seed membership mismatch: missing={sorted(configured_seeds - declared_seeds, key=str)}, extra={sorted(declared_seeds - configured_seeds, key=str)}",
            )
            duplicate_seeds = [
                seed
                for seed in declared_seeds
                if sum(seed in _entry_group_seeds(entry) for entry in group_entries) != 1
            ]
            _check(
                checks,
                f"group:{group_id}:unique_seeds",
                not duplicate_seeds,
                "each formal seed is listed exactly once"
                if not duplicate_seeds
                else f"duplicate seeds: {sorted(duplicate_seeds, key=str)}",
            )
        for entry in group_entries:
            if not any(key in entry for key in ("seed", "split_seed", "seed_id")):
                continue
            seed_values = _entry_group_seeds(entry)
            if config_group and len(seed_values) == 1:
                seed_value = next(iter(seed_values))
                if isinstance(seed_value, int):
                    try:
                        expected_seed_hash = config_sha256(
                            config_for_seed(
                                dict(config),
                                seed_value,
                                required_experiments=configured_experiment_list,
                            )
                        )
                    except (TypeError, ValueError):
                        expected_seed_hash = ""
                    _check(
                        checks,
                        f"group:{group_id}:seed:{seed_value}:config_hash",
                        str(entry.get("config_sha256", "")) == expected_seed_hash,
                        "seed config hash matches the registered seed projection"
                        if str(entry.get("config_sha256", "")) == expected_seed_hash
                        else "seed config hash differs from the registered seed projection",
                    )
            seed_status = str(entry.get("seed_status", entry.get("status", ""))).lower()
            audit_status = str(entry.get("seed_audit_status", entry.get("audit_status", ""))).lower()
            _check(
                checks,
                f"group:{group_id}:seed:{entry.get('seed', '?')}:audited",
                seed_status == "audited" and audit_status == "audited",
                "seed is audited" if seed_status == "audited" and audit_status == "audited" else "seed is not audited",
            )
        covered_seeds[group_id] = set()

        expected_dependencies = set(_as_strings(config_group.get("scientific_dependencies", []))) if config_group else set()
        declared_dependencies = set().union(*(_entry_group_dependencies(entry) for entry in group_entries)) if group_entries else set()
        if config_group:
            _check(
                checks,
                f"group:{group_id}:dependencies",
                declared_dependencies == expected_dependencies,
                "group dependencies exactly match formal config"
                if declared_dependencies == expected_dependencies
                else f"dependency mismatch: expected={sorted(expected_dependencies)}, actual={sorted(declared_dependencies)}",
            )
            unknown_dependencies = expected_dependencies - declared_groups
            _check(
                checks,
                f"group:{group_id}:dependency_presence",
                not unknown_dependencies,
                "all direct dependencies are present"
                if not unknown_dependencies
                else "missing dependencies: " + ", ".join(sorted(unknown_dependencies)),
            )

    if group_specs and configured_group_order:
        for group_id in configured_group_order:
            closure = _transitive_group_dependencies(group_id, group_specs)
            missing = closure - declared_groups
            _check(
                checks,
                f"group:{group_id}:dependency_closure",
                not missing,
                "dependency closure is fully registered" if not missing else "missing transitive dependencies: " + ", ".join(sorted(missing)),
            )
        for group_id in configured_group_order:
            config_group = group_specs.get(group_id)
            expected_seeds = set(_seed_values(config_group.get("split_seeds", []))) if config_group else set()
            observed_seeds = {
                seed
                for entry in grouped.get(group_id, [])
                for seed in _entry_group_seeds(entry)
            }
            if expected_seeds:
                _check(
                    checks,
                    f"group:{group_id}:seed_coverage",
                    observed_seeds == expected_seeds,
                    "all configured seeds are represented"
                    if observed_seeds == expected_seeds
                    else f"seed coverage mismatch: missing={sorted(expected_seeds - observed_seeds, key=str)}, extra={sorted(observed_seeds - expected_seeds, key=str)}",
                )
    return covered_seeds


def _release_payload_lineage(payload: Mapping[str, Any], project_root: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    source_runs: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    statistics_files: list[dict[str, Any]] = []
    code_by_run: dict[str, Any] = {}
    environment_by_run: dict[str, Any] = {}
    config_hashes: set[str] = set()
    for entry in entries:
        root = _resolve_run_entry(project_root, entry)
        run_manifest = read_run_manifest(root)
        relative_run = root.relative_to(project_root).as_posix()
        audit_path = root / "manifest/audit.json"
        if not audit_path.is_file():
            raise ReleaseError(f"Run lacks audit report: {audit_path}")
        source_runs.append(
            {
                "group_id": entry.get("parent_group_id") or entry.get("group_id"),
                "seed": entry.get("seed"),
                "run_id": run_manifest.get("run_id"),
                "run_path": relative_run,
                "config_sha256": run_manifest.get("resolved_config_sha256"),
                "group_config_sha256": entry.get("group_config_sha256") or payload.get("config_sha256"),
                "audit_sha256": sha256_file(audit_path),
            }
        )
        run_config_hash = str(run_manifest.get("resolved_config_sha256", ""))
        if _SHA256.fullmatch(run_config_hash):
            config_hashes.add(run_config_hash)
        for relative in ("manifest/run.json", "manifest/audit.json", "manifest/e00.json", "manifest/e01.json", "manifest/e02.json"):
            path = root / relative
            if path.is_file():
                source_files.append(
                    {
                        "run_id": run_manifest.get("run_id"),
                        "path": path.relative_to(project_root).as_posix(),
                        "sha256": sha256_file(path),
                    }
                )
        try:
            audit_payload = _read_json(audit_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReleaseError(f"Run audit report is unreadable: {audit_path}") from exc
        for item in audit_payload.get("checked_files", []) if isinstance(audit_payload, dict) else []:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ReleaseError(f"Run audit report has a malformed checked file: {audit_path}")
            path = _resolve_relative(root, str(item["path"]))
            source_files.append(
                {
                    "run_id": run_manifest.get("run_id"),
                    "path": path.relative_to(project_root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
        for lineage_path in _lineage_candidates(root):
            try:
                run_lineage = _read_json(lineage_path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(run_lineage, Mapping):
                continue
            run_id = str(run_manifest.get("run_id", ""))
            for field, target in (("source_manifest", source_manifests), ("statistics_files", statistics_files)):
                values = run_lineage.get(field)
                if not isinstance(values, list):
                    continue
                for item in values:
                    if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                        continue
                    path = _resolve_relative(root, str(item["path"]))
                    target.append(
                        {
                            "run_id": run_id,
                            "path": path.relative_to(project_root).as_posix(),
                            "sha256": sha256_file(path),
                        }
                    )
            if "code" in run_lineage:
                code_by_run[run_id] = run_lineage["code"]
            if "environment" in run_lineage:
                environment_by_run[run_id] = run_lineage["environment"]
            break
    source_files = list({(item["run_id"], item["path"]): item for item in source_files}.values())
    source_manifests = list({(item["run_id"], item["path"]): item for item in source_manifests}.values())
    statistics_files = list({(item["run_id"], item["path"]): item for item in statistics_files}.values())
    lineage: dict[str, Any] = {
        "config_sha256": payload.get("config_sha256"),
        "config_hashes": sorted(config_hashes),
        "source_runs": source_runs,
        "source_files": source_files,
        "source_manifest": source_manifests,
        "statistics_files": statistics_files,
    }
    if code_by_run:
        lineage["code"] = code_by_run
    if environment_by_run:
        lineage["environment"] = environment_by_run
    return lineage


def _validate_release_lineage(payload: Mapping[str, Any], project_root: Path, checks: list[dict[str, Any]]) -> None:
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        _check(checks, "release_lineage", False, "release lineage is missing")
        return
    source_runs = lineage.get("source_runs")
    source_files = lineage.get("source_files")
    source_ok = (
        isinstance(source_runs, list)
        and bool(source_runs)
        and isinstance(source_files, list)
        and bool(source_files)
        and isinstance(lineage.get("source_manifest"), list)
        and bool(lineage.get("source_manifest"))
        and isinstance(lineage.get("statistics_files"), list)
        and bool(lineage.get("statistics_files"))
    )
    problems: list[str] = []
    for label in ("source_files", "source_manifest", "statistics_files"):
        entries = lineage.get(label)
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not _SHA256.fullmatch(str(item.get("sha256", ""))):
                problems.append(f"malformed release {label} entry")
                break
            path = (project_root / str(item["path"])).resolve()
            if project_root not in path.parents or not path.is_file():
                problems.append(f"missing release {label} file: {item['path']}")
                break
            if sha256_file(path) != item["sha256"]:
                problems.append(f"release {label} hash mismatch: {item['path']}")
                break
    if lineage.get("config_sha256") != payload.get("config_sha256"):
        problems.append("release lineage config hash mismatch")
    actual_run_hashes: set[str] = set()
    for item in source_runs if isinstance(source_runs, list) else []:
        if not isinstance(item, Mapping):
            problems.append("malformed release source run entry")
            continue
        run_id = str(item.get("run_id", ""))
        run_path = item.get("run_path")
        if not isinstance(run_path, str) or not run_path:
            problems.append(f"release source run path is missing: {run_id}")
            continue
        group_hash = str(item.get("group_config_sha256") or item.get("config_sha256") or "")
        if not _SHA256.fullmatch(group_hash) or group_hash != str(payload.get("config_sha256", "")):
            problems.append(f"release source run group config hash mismatch: {run_id}")
        run_hash = str(item.get("config_sha256", ""))
        if not _SHA256.fullmatch(run_hash):
            problems.append(f"release source run config hash is malformed: {run_id}")
        else:
            actual_run_hashes.add(run_hash)
        try:
            run_root = _resolve_run_entry(project_root, {"run_path": run_path})
            run_manifest = read_run_manifest(run_root)
        except (OSError, ReleaseError, json.JSONDecodeError) as exc:
            problems.append(f"release source run is unreadable: {run_id} ({exc})")
            continue
        if run_manifest.get("run_id") != run_id:
            problems.append(f"release source run_id mismatch: {run_id}")
        expected_provenance = _lineage_provenance(run_root, run_manifest)
        for field in ("code", "environment"):
            recorded = lineage.get(field)
            if field not in expected_provenance:
                continue
            if not isinstance(recorded, Mapping) or recorded.get(run_id) != expected_provenance[field]:
                problems.append(f"release source {field} provenance mismatch: {run_id}")
    recorded_run_hashes = lineage.get("config_hashes")
    if not isinstance(recorded_run_hashes, list) or {str(value) for value in recorded_run_hashes} != actual_run_hashes:
        problems.append("release lineage config_hashes do not match source runs")
    _check(checks, "release_lineage", source_ok and not problems, "release source hashes match" if source_ok and not problems else "; ".join(problems) or "release lineage is empty")


def validate_release(
    release: Path | str | Mapping[str, Any] | None = None,
    *,
    project_root: Path | str | None = None,
    run_roots: Iterable[Path | str] | None = None,
    run_group_manifest: Path | str | Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | Path | str | None = None,
    config_path: Path | str | None = None,
    required_experiments: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate a frozen release manifest or a pre-registered run-group list."""

    config_value, configured_hash = _config_for_release(config, config_path)
    release_payload: dict[str, Any] | None = None
    release_source: Path | str | Mapping[str, Any] | None = release
    registration_payload: Mapping[str, Any] | None = None
    if run_group_manifest is not None:
        release_payload = _load_manifest_source(run_group_manifest)
        # Keep the immutable pre-registration snapshot separate from the
        # temporary release payload, which may receive computed lineage below.
        registration_payload = deepcopy(release_payload)
        release_source = run_group_manifest
    elif release is not None and not isinstance(release, Mapping):
        source_path = Path(release).expanduser().resolve()
        if source_path.is_file():
            release_payload = _load_manifest_source(source_path)
            registration_payload = deepcopy(release_payload)
        elif source_path.is_dir() and (source_path / "manifest/run.json").is_file():
            run_roots = [source_path, *(list(run_roots or []))]
            release_source = None
        else:
            raise ReleaseError(f"Release source is neither a manifest nor a run directory: {release}")
    elif isinstance(release, Mapping):
        release_payload = _load_manifest_source(release)
        registration_payload = deepcopy(release_payload)

    if release_payload is not None and release_payload.get("manifest_type") == "formal_run_group":
        release_payload = _formal_group_payload_as_release(release_payload)

    if (
        registration_payload is not None
        and isinstance(registration_payload.get("formal_group_manifest"), Mapping)
    ):
        registration_payload = registration_payload["formal_group_manifest"]

    if release_payload is not None and run_roots is not None:
        raise ReleaseError("provide release/run-group manifest or run_roots, not both")
    if release_payload is None:
        raise ReleaseError("A frozen run-group manifest with explicit pre-registration proof is required")

    if (
        configured_hash
        and release_payload.get("manifest_type") == "formal_run_group"
        and release_payload.get("config_sha256") != configured_hash
    ):
        raise ReleaseError("Formal group config hash does not match the supplied config")

    entries = _group_entries(release_payload)
    if not entries:
        raise ReleaseError("Release manifest has no pre-registered run groups")
    root = _project_root_from_inputs(project_root, release_source, entries)
    if release_payload.get("release_id") == "unmaterialized" and release_payload.get("lineage") is None:
        if not release_payload.get("config_sha256"):
            config_hashes = {
                str(read_run_manifest(_resolve_run_entry(root, entry)).get("resolved_config_sha256", ""))
                for entry in entries
            }
            if len(config_hashes) == 1:
                release_payload["config_sha256"] = next(iter(config_hashes))
        release_payload["lineage"] = _release_payload_lineage(release_payload, root, entries)
        release_payload["lineage_sha256"] = sha256_json(release_payload)
    elif release_payload.get("release_id", "").startswith("group-") and release_payload.get("lineage") is None:
        release_payload["lineage"] = _release_payload_lineage(release_payload, root, entries)
        release_payload["lineage_sha256"] = sha256_json(release_payload)
    checks: list[dict[str, Any]] = []
    if registration_payload is None:
        registration_payload = release_payload
    if registration_payload.get("manifest_type") == "formal_run_group":
        top_level_hash = str(registration_payload.get("manifest_hash", ""))
        frozen_snapshot = registration_payload.get("frozen_group_manifest")
        frozen_snapshot_hash = (
            str(frozen_snapshot.get("manifest_hash", ""))
            if isinstance(frozen_snapshot, Mapping)
            else ""
        )
        _check(
            checks,
            "formal_group_manifest_snapshot_hash",
            bool(_SHA256.fullmatch(top_level_hash)) and top_level_hash == frozen_snapshot_hash,
            "formal group manifest snapshot hash matches its frozen content",
        )
        _check(
            checks,
            "formal_group_manifest_status",
            registration_payload.get("status") == "audited" and registration_payload.get("audit_status") == "audited",
            "formal group manifest and all seeds are audited",
        )
    elif registration_payload.get("manifest_type") == "formal_release_groups":
        aggregate_hash = str(registration_payload.get("manifest_hash", ""))
        aggregate_without_hash = dict(registration_payload)
        aggregate_without_hash.pop("manifest_hash", None)
        aggregate_ok = bool(_SHA256.fullmatch(aggregate_hash)) and aggregate_hash == sha256_json(aggregate_without_hash)
        _check(
            checks,
            "formal_release_groups_hash",
            aggregate_ok,
            "aggregate group manifest hash matches its pre-registration content"
            if aggregate_ok
            else "aggregate group manifest hash is missing or inconsistent",
        )
        _check(
            checks,
            "formal_release_groups_status",
            registration_payload.get("status") == "frozen" and registration_payload.get("audit_status") == "audited",
            "aggregate group manifest is audited"
            if registration_payload.get("status") == "audited" and registration_payload.get("audit_status") == "audited"
            else "aggregate group manifest is not audited",
        )
        _validate_formal_release_groups_registration(registration_payload, root, checks)
    if isinstance(registration_payload.get("formal_group_manifest"), Mapping):
        registration_payload = registration_payload["formal_group_manifest"]
        _check(
            checks,
            "formal_group_manifest_status",
            (
                registration_payload.get("status") == "audited"
                and registration_payload.get("audit_status") == "audited"
            )
            or (
                registration_payload.get("manifest_type") == "formal_release_groups"
                and registration_payload.get("status") == "frozen"
                and registration_payload.get("audit_status") == "audited"
            ),
            "formal group manifest and all seeds are audited",
        )
    if registration_payload.get("manifest_type") == "formal_run_group":
        _validate_formal_group_registration(registration_payload, checks)
        frozen_group = registration_payload.get("frozen_group_manifest")
        if isinstance(frozen_group, Mapping):
            frozen_hash = str(frozen_group.get("manifest_hash", ""))
            frozen_without_hash = dict(frozen_group)
            frozen_without_hash.pop("manifest_hash", None)
            _check(
                checks,
                "formal_group_manifest_hash",
                bool(_SHA256.fullmatch(frozen_hash)) and frozen_hash == sha256_json(frozen_without_hash),
                "formal group pre-registration hash matches frozen content",
            )
    elif registration_payload.get("manifest_type") == "formal_release_groups":
        _validate_formal_release_groups_registration(registration_payload, root, checks)
    else:
        _validate_registration_manifest(registration_payload, checks)
    if release_payload.get("status") != "frozen":
        _check(checks, "release_status", False, f"release status={release_payload.get('status')!r}; expected frozen")
    else:
        _check(checks, "release_status", True, "release status is frozen")
    if release_payload.get("aggregation_policy") != "frozen_manifest_only":
        _check(checks, "aggregation_policy", False, "release aggregation policy must be frozen_manifest_only")
    else:
        _check(checks, "aggregation_policy", True, "release uses frozen manifest aggregation")
    if configured_hash:
        _check(checks, "release_config_hash", release_payload.get("config_sha256") == configured_hash, "release config hash matches supplied config")
    release_config_hash = str(release_payload.get("config_sha256") or "")
    _check(checks, "release_config_hash_present", bool(release_config_hash) and bool(_SHA256.fullmatch(release_config_hash)), "release config hash is a SHA256 digest")
    group_seed_sets = _validate_group_manifest_contract(release_payload, entries, config_value, checks)
    group_hashes: dict[str, set[str]] = {}
    if config_value:
        config_run = config_value.get("run", {})
        config_release = config_value.get("release", {})
        config_is_formal_frozen = (
            isinstance(config_run, dict)
            and config_run.get("evidence_tier") == "formal"
            and config_run.get("frozen") is True
            and isinstance(config_release, dict)
            and config_release.get("status") == "frozen"
        )
        _check(checks, "supplied_config_frozen", config_is_formal_frozen, "supplied release config is formal and frozen")

    global_experiments = _release_experiments(config_value, required_experiments or release_payload.get("required_experiments"))
    group_ids = [str(entry.get("group_id", "")) for entry in entries]
    group_memberships = [
        (
            str(entry.get("parent_group_id") or entry.get("group_id", "")),
            tuple(sorted(_entry_group_seeds(entry), key=str)) or None,
        )
        for entry in entries
    ]
    _check(
        checks,
        "unique_group_ids",
        all(group_ids) and len(group_memberships) == len(set(group_memberships)),
        "run group seed memberships are unique",
    )
    if config_value:
        configured_groups = config_value.get("release", {}).get("run_groups", []) if isinstance(config_value.get("release", {}), dict) else []
        if configured_groups:
            registered_group_ids = {
                str(entry.get("parent_group_id") or entry.get("group_id", "")) for entry in entries
            }
            missing_groups = sorted(set(_as_strings(configured_groups)) - registered_group_ids)
            _check(checks, "pre_registered_groups", not missing_groups, "all configured run groups are registered" if not missing_groups else "missing: " + ", ".join(missing_groups))

    seen_runs: set[str] = set()
    run_results: list[dict[str, Any]] = []
    for entry in entries:
        try:
            run_root = _resolve_run_entry(root, entry)
        except ReleaseError as exc:
            _check(checks, f"run:{entry.get('group_id', '?')}", False, str(exc))
            continue
        group_experiments = _entry_experiments(entry, config_value, global_experiments)
        if not config_value and not entry.get("required_experiments") and not entry.get("experiments"):
            run_manifest = read_run_manifest(run_root)
            group_experiments = _configured_experiments(
                run_manifest,
                None,
                group_name=str(entry.get("group_id", "")),
            )
        expected_hash = str(entry.get("config_sha256") or release_payload.get("config_sha256") or configured_hash or "")
        parent_group_id = str(entry.get("parent_group_id") or entry.get("group_id", ""))
        entry_seed = entry.get("seed")
        inspection = _inspect_run(
            run_root,
            required_experiments=group_experiments,
            required_files=None,
            require_audited=True,
            require_formal=True,
            expected_config_sha256=expected_hash or None,
            group_name=str(entry.get("group_id", "")),
            require_formal_seed=(entry.get("seed") is not None or entry.get("parent_group_id") is not None),
            expected_group_id=parent_group_id if entry.get("parent_group_id") is not None else None,
            expected_seed=entry_seed if isinstance(entry_seed, int) and not isinstance(entry_seed, bool) else None,
        )
        run_id = inspection["run_id"] or run_root.name
        entry_group_hash = str(entry.get("group_config_sha256") or entry.get("_declared_group_config_sha256") or "")
        if entry_group_hash:
            group_hashes.setdefault(parent_group_id, set()).add(entry_group_hash)
            _check(
                checks,
                f"group:{parent_group_id}:config_hash",
                entry_group_hash == release_config_hash,
                "all group entries use the release config hash"
                if entry_group_hash == release_config_hash
                else f"group config hash differs from release for {parent_group_id}",
            )
        seed = entry.get("seed")
        if seed is not None:
            run_manifest = inspection["manifest"]
            run_seeds = _run_manifest_seed_values(run_manifest)
            seed_value = _seed_values(seed)
            seed_ok = bool(seed_value) and (not run_seeds or set(seed_value).issubset(run_seeds))
            _check(
                checks,
                f"run:{run_id}:seed",
                seed_ok,
                "run seed matches its frozen group membership"
                if seed_ok
                else f"run seed {seed!r} is missing from run manifest {sorted(run_seeds, key=str)}",
            )
            if parent_group_id in group_seed_sets:
                group_seed_sets[parent_group_id].add(seed_value[0] if seed_value else seed)
        if release_config_hash and entry.get("group_config_sha256"):
            _check(
                checks,
                f"run:{run_id}:group_config_hash",
                entry.get("group_config_sha256") == release_config_hash,
                "seed group config hash matches release" if entry.get("group_config_sha256") == release_config_hash else "seed group config hash differs from release",
            )
        elif release_config_hash:
            _check(
                checks,
                f"run:{run_id}:config_hash",
                inspection["config_sha256"] == release_config_hash,
                "run config hash matches release" if inspection["config_sha256"] == release_config_hash else "run config hash differs from release",
            )
        duplicate = run_id in seen_runs
        seen_runs.add(run_id)
        _check(checks, f"run:{run_id}:unique", not duplicate, "run is listed once" if not duplicate else "duplicate run")
        _check(checks, f"run:{run_id}:eligible", inspection["passed"], "formal frozen audited run passes audit contract" if inspection["passed"] else "; ".join(inspection["errors"]))
        run_results.append(
            {
                "group_id": parent_group_id,
                "run_id": run_id,
                "run_root": run_root,
                "config_sha256": inspection["config_sha256"],
                "passed": inspection["passed"],
                "errors": inspection["errors"],
            }
        )

    for group_id, hashes in group_hashes.items():
        _check(
            checks,
            f"group:{group_id}:unified_config_hash",
            len(hashes) == 1 and next(iter(hashes), "") == release_config_hash,
            "group config hash is unified" if len(hashes) == 1 and next(iter(hashes), "") == release_config_hash else "group config hash mismatch",
        )

    _validate_release_lineage(release_payload, root, checks)
    recorded_release_hash = release_payload.get("lineage_sha256")
    if recorded_release_hash:
        without_hash = dict(release_payload)
        without_hash.pop("lineage_sha256", None)
        _check(checks, "release_lineage_hash", recorded_release_hash == sha256_json(without_hash), "release lineage hash matches manifest")
    else:
        _check(checks, "release_lineage_hash", False, "release lineage hash is missing")
    valid = all(item["passed"] for item in checks) and bool(run_results) and all(item["passed"] for item in run_results)
    result = {
        "valid": valid,
        "release_id": release_payload.get("release_id"),
        "status": release_payload.get("status"),
        "config_sha256": release_payload.get("config_sha256"),
        "checks": checks,
        "runs": run_results,
    }
    if not valid:
        failures = [item["details"] for item in checks if not item["passed"]]
        failures.extend(error for item in run_results for error in item["errors"])
        raise ReleaseError("Release validation failed: " + " | ".join(failures))
    return result


def create_release(
    project_root: Path | str,
    release_id: str | None = None,
    run_roots: Iterable[Path | str] | None = None,
    *,
    run_group_manifest: Path | str | Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | Path | str | None = None,
    config_path: Path | str | None = None,
    required_experiments: Iterable[str] | None = None,
    output_root: Path | str | None = None,
) -> Path:
    """Create a frozen release manifest from pre-registered audited runs."""

    project = Path(project_root).expanduser().resolve(strict=True)
    if release_id is None:
        release_id = "release-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not _RUN_ID.fullmatch(release_id):
        raise ReleaseError(f"Invalid release_id: {release_id!r}")
    config_value, configured_hash = _config_for_release(config, config_path)
    if run_group_manifest is not None:
        group_payload = _load_manifest_source(run_group_manifest)
        if (
            configured_hash
            and group_payload.get("manifest_type") == "formal_run_group"
            and group_payload.get("config_sha256") != configured_hash
        ):
            raise ReleaseError("Formal group config hash does not match the supplied config")
        entries = _group_entries(group_payload)
    else:
        raise ReleaseError("run_group_manifest with explicit pre-registration proof is required")
    if not entries:
        raise ReleaseError("No run groups were supplied")

    global_experiments = _release_experiments(config_value, required_experiments)
    normalised_entries: list[dict[str, Any]] = []
    for entry in entries:
        normalised = dict(entry)
        run_root = _resolve_run_entry(project, normalised)
        run_manifest = read_run_manifest(run_root)
        normalised["run_id"] = str(run_manifest.get("run_id") or normalised.get("run_id") or run_root.name)
        normalised["run_path"] = run_root.relative_to(project).as_posix()
        normalised.pop("run_root", None)
        entry_experiments = _entry_experiments(normalised, config_value, global_experiments)
        if not config_value and not normalised.get("required_experiments") and not normalised.get("experiments"):
            entry_experiments = _configured_experiments(
                run_manifest,
                None,
                group_name=str(normalised.get("group_id", "")),
            )
        normalised.setdefault("experiments", entry_experiments)
        normalised.setdefault("config_sha256", run_manifest.get("resolved_config_sha256"))
        normalised_entries.append(normalised)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "release_id": release_id,
        "status": "frozen",
        "aggregation_policy": "frozen_manifest_only",
        "config_sha256": configured_hash or (normalised_entries[0].get("config_sha256") if len({entry.get('config_sha256') for entry in normalised_entries}) == 1 else None),
        "required_experiments": global_experiments,
        "run_groups": normalised_entries,
        "created_at_utc": _now(),
    }
    if run_group_manifest is not None and isinstance(group_payload, Mapping) and group_payload.get("manifest_type") == "formal_run_group" or run_group_manifest is not None and isinstance(group_payload, Mapping) and group_payload.get("manifest_type") == "formal_release_groups":
        payload["formal_group_manifest"] = deepcopy(dict(group_payload))
    elif run_group_manifest is not None and isinstance(group_payload, Mapping):
        payload["frozen_group_manifest"] = deepcopy(dict(group_payload))
    payload["lineage"] = _release_payload_lineage(payload, project, normalised_entries)
    payload["lineage_sha256"] = sha256_json(payload)

    # Validate before materializing the file so an invalid release cannot be advertised.
    validate_release(payload, project_root=project, config=config_value, required_experiments=required_experiments)
    destination_root = Path(output_root).expanduser().resolve() if output_root is not None else project / "artifacts" / "releases"
    destination_root.mkdir(parents=True, exist_ok=True)
    release_directory = destination_root / release_id
    release_directory.mkdir(parents=False, exist_ok=False)
    destination = release_directory / "release.json"
    if destination.exists():
        raise ReleaseError(f"Release already exists and cannot be overwritten: {destination}")
    _write_json(destination, payload)
    return destination


__all__ = ["AuditError", "ReleaseError", "audit_run", "create_release", "validate_release"]
