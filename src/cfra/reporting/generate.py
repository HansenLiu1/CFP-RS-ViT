"""Generate machine-readable summaries from one immutable run.

The reporting layer deliberately treats metric files as data rather than as a
Python API.  It accepts both row-oriented JSON/JSONL and the nested model and
experiment bundles currently emitted by the pipeline.
"""

from __future__ import annotations

import csv
import html
import json
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cfra.provenance import sha256_file, sha256_json
from cfra.runs import read_run_manifest


class ReportingError(RuntimeError):
    """Raised when a run cannot be converted into a report."""


_METADATA_COLUMNS = (
    "run_id",
    "evidence_tier",
    "status",
    "config_sha256",
    "source_file",
    "source_sha256",
    "record_sha256",
    "lineage_sha256",
)
_EXCLUDED_SOURCE_NAMES = {"lineage.json", "audit.json"}
_EXPERIMENT_KEY = re.compile(r"^E\d+(?:-\d+)?$")
_SOURCE_MANIFEST_NAMES = {
    "source_files.csv",
    "images.csv",
    "objects.csv",
    "splits.csv",
    "models.csv",
    "source_manifest.json",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportingError(f"Could not read JSON metrics file: {path}") from exc


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _contains_nested_collection(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            isinstance(item, list) or (isinstance(item, dict) and _contains_nested_collection(item))
            for item in value.values()
        )
    if isinstance(value, list):
        return any(isinstance(item, (dict, list)) for item in value)
    return False


def _csv_safe_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _flatten_scalars(value: Any, prefix: str) -> dict[str, Any]:
    """Flatten metadata-like nested dictionaries without splitting metric rows."""

    if _is_scalar(value):
        return {prefix: value}
    if isinstance(value, list):
        return {prefix: json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))}
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}_{key}" if prefix else str(key)
            if _contains_nested_collection(child):
                result[child_prefix] = json.dumps(child, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
            else:
                result.update(_flatten_scalars(child, child_prefix))
        return result
    return {prefix: str(value)}


def _context_field(parent_key: str, item_key: str) -> str:
    if parent_key == "models":
        return "model_id"
    if parent_key in {"experiments", "stages"} or _EXPERIMENT_KEY.fullmatch(item_key):
        return "experiment_id"
    if parent_key in {"results", "records", "rows", "metrics", "data"}:
        return "record_group"
    return f"context_{parent_key}" if parent_key else "record_group"


def _extract_records(node: Any, context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Extract leaf metric rows while retaining model/experiment context."""

    inherited = dict(context or {})
    if isinstance(node, list):
        records: list[dict[str, Any]] = []
        for item in node:
            records.extend(_extract_records(item, inherited))
        return records

    if not isinstance(node, dict):
        return [{**inherited, "value": node}]

    direct: dict[str, Any] = {}
    children: list[tuple[str, Any]] = []
    for key, value in node.items():
        key = str(key)
        if _is_scalar(value):
            direct[key] = value
        elif isinstance(value, list):
            if all(_is_scalar(item) for item in value):
                direct[key] = value
            else:
                children.append((key, value))
        elif isinstance(value, dict):
            if _contains_nested_collection(value):
                children.append((key, value))
            else:
                direct.update(_flatten_scalars(value, key))
        else:
            direct[key] = str(value)

    merged = {**inherited, **direct}
    if not children:
        return [merged] if merged else []

    records: list[dict[str, Any]] = []
    for parent_key, child in children:
        if isinstance(child, dict):
            for item_key, item in child.items():
                field = _context_field(parent_key, str(item_key))
                child_context = {**merged, field: item_key}
                records.extend(_extract_records(item, child_context))
        else:
            field = "experiment_id" if _EXPERIMENT_KEY.fullmatch(parent_key) else "record_group"
            child_context = {**merged, field: parent_key}
            records.extend(_extract_records(child, child_context))
    return records


def _metric_files(run_root: Path, metrics_paths: Iterable[Path | str] | Path | str | None) -> list[Path]:
    if metrics_paths is None:
        paths = [
            path
            for path in (run_root / "metrics").rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        ] if (run_root / "metrics").exists() else []
    elif isinstance(metrics_paths, (str, Path)):
        paths = [Path(metrics_paths)]
    else:
        paths = [Path(path) for path in metrics_paths]

    result: list[Path] = []
    for path in paths:
        target = path if path.is_absolute() else run_root / path
        target = target.resolve(strict=True)
        if target.suffix.lower() not in {".json", ".jsonl"}:
            raise ReportingError(f"Metrics input must be JSON or JSONL: {target}")
        if run_root not in target.parents:
            raise ReportingError(f"Metrics input escapes the run directory: {target}")
        result.append(target)
    unique = {path: None for path in result}
    return sorted(unique)


def load_metric_records(
    run_root: Path | str,
    metrics_paths: Iterable[Path | str] | Path | str | None = None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Load JSON/JSONL metrics and return records plus the exact source files."""

    root = Path(run_root).expanduser().resolve(strict=True)
    files = _metric_files(root, metrics_paths)
    if not files:
        raise ReportingError(f"No JSON/JSONL metrics found under {root / 'metrics'}")

    records: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".jsonl":
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as exc:
                raise ReportingError(f"Could not read JSONL metrics file: {path}") from exc
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReportingError(f"Invalid JSONL at {relative}:{line_number}") from exc
                extracted = _extract_records(payload)
                for record_index, record in enumerate(extracted):
                    records.append(
                        {
                            "_record": record,
                            "_source_file": relative,
                            "_source_path": path,
                            "_line_number": line_number,
                            "_record_index": record_index,
                        }
                    )
        else:
            payload = _read_json(path)
            for record_index, record in enumerate(_extract_records(payload)):
                records.append(
                    {
                        "_record": record,
                        "_source_file": relative,
                        "_source_path": path,
                        "_line_number": "",
                        "_record_index": record_index,
                    }
                )

    if not records:
        raise ReportingError("Metrics files contained no JSON records")
    return records, files


def _source_files(root: Path, metric_files: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set(metric_files)
    manifest_root = root / "manifest"
    if manifest_root.exists():
        paths.update(
            path
            for path in manifest_root.rglob("*")
            if path.name not in {*_EXCLUDED_SOURCE_NAMES, "run.json"} and path.is_file()
        )
    statistics_root = root / "statistics"
    if statistics_root.exists():
        paths.update(path for path in statistics_root.rglob("*") if path.is_file())
    for relative in (
        "intermediates/e00_source_manifest/identity.json",
        "intermediates/e01_roles/report.json",
        "intermediates/e02_pixel_protocol/report.json",
        "intermediates/e02_pixel_protocol/audit_rows.jsonl",
    ):
        path = root / relative
        if path.is_file():
            paths.add(path)
    return sorted(paths)


def _source_manifest_files(root: Path) -> list[Path]:
    """Return the raw-input manifest files that bind a report to its source."""

    candidates: set[Path] = set()
    manifest_root = root / "manifest"
    if manifest_root.exists():
        candidates.update(
            path
            for path in manifest_root.rglob("*")
            if path.is_file()
            and path.name not in {*_EXCLUDED_SOURCE_NAMES, "run.json"}
            and (path.name in _SOURCE_MANIFEST_NAMES or "source_manifest" in path.name.lower())
        )
    source_identity_root = root / "intermediates" / "e00_source_manifest"
    if source_identity_root.exists():
        candidates.update(path for path in source_identity_root.rglob("*") if path.is_file())
    return sorted(candidates)


def _read_optional_json(path: Path) -> Any | None:
    try:
        return _read_json(path)
    except ReportingError:
        return None


def _provenance_sections(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Collect code/environment identity without inventing absent provenance."""

    payloads: list[Mapping[str, Any]] = [manifest]
    e00_path = root / "manifest" / "e00.json"
    if e00_path.is_file():
        payload = _read_optional_json(e00_path)
        if isinstance(payload, Mapping):
            payloads.append(payload)
    for path in _source_manifest_files(root):
        if path.suffix.lower() != ".json":
            continue
        payload = _read_optional_json(path)
        if isinstance(payload, Mapping):
            payloads.append(payload)

    result: dict[str, Any] = {}
    aliases = {
        "code": (
            "code",
            "code_identity",
            "code_provenance",
            "code_sha256",
            "git_commit",
            "commit",
            "git_dirty",
            "dirty_status",
            "package_version",
        ),
        "environment": (
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
        ),
    }
    for output_key, keys in aliases.items():
        for payload in payloads:
            value: Any | None = None
            for key in keys:
                if key in payload:
                    value = payload[key]
                    break
            if value not in (None, "", {}, []):
                if any(key in payload for key in keys[3:]):
                    result[output_key] = {
                        str(key): payload[key]
                        for key in keys[3:]
                        if key in payload and payload[key] not in (None, "", {}, [])
                    }
                else:
                    result[output_key] = value
                break
    return result


def _relative_entry(root: Path, path: Path) -> dict[str, str]:
    relative = path.relative_to(root).as_posix()
    return {"path": relative, "sha256": sha256_file(path)}


def _normalise_row(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(metadata)
    for key, value in record.items():
        key = str(key)
        if key in row:
            row[f"metric_{key}"] = value
        else:
            row[key] = value
    return row


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = list(_METADATA_COLUMNS)
    for key in sorted({str(key) for row in rows for key in row} - set(columns)):
        columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_safe_value(row.get(key)) for key in columns})
    temporary.replace(path)


def _markdown_value(value: Any) -> str:
    text = str(_csv_safe_value(value))
    return text.replace("|", "\\|").replace("\n", " ")


def _write_markdown(path: Path, rows: list[Mapping[str, Any]], source_entries: list[Mapping[str, str]], metadata: Mapping[str, Any]) -> None:
    metric_columns = sorted({str(key) for row in rows for key in row if key not in _METADATA_COLUMNS})
    display_columns = [*(_METADATA_COLUMNS[:4]), *metric_columns]
    lines = [
        "# Run metric summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| run_id | {_markdown_value(metadata['run_id'])} |",
        f"| evidence_tier | {_markdown_value(metadata['evidence_tier'])} |",
        f"| status | {_markdown_value(metadata['status'])} |",
        f"| config_sha256 | {_markdown_value(metadata['config_sha256'])} |",
        f"| record_count | {len(rows)} |",
        f"| source_file_count | {len(source_entries)} |",
        "",
        "## Metrics",
        "",
    ]
    if rows:
        lines.append("| " + " | ".join(display_columns) + " |")
        lines.append("| " + " | ".join("---" for _ in display_columns) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_markdown_value(row.get(key, "")) for key in display_columns) + " |")
    else:
        lines.append("No metric records were found.")
    lines.extend(["", "## Source files", "", "| Path | SHA256 |", "| --- | --- |"])
    for entry in source_entries:
        lines.append(f"| {_markdown_value(entry['path'])} | `{entry['sha256']}` |")
    lines.append("")
    _write_text(path, "\n".join(lines))


def _workflow_svg(metadata: Mapping[str, Any]) -> str:
    run_id = html.escape(str(metadata["run_id"]), quote=True)
    evidence_tier = html.escape(str(metadata["evidence_tier"]), quote=True)
    status = html.escape(str(metadata["status"]), quote=True)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="300" viewBox="0 0 960 300" role="img" aria-labelledby="title description" data-run-id="{run_id}">
  <title id="title">CFRA auditable reporting workflow</title>
  <desc id="description">E00, E01 and E02 qualification evidence flows into metrics, statistics and lineage before CSV, Markdown and figure outputs.</desc>
  <rect width="960" height="300" fill="#f7f8fa"/>
  <g font-family="Arial, sans-serif" font-size="16" fill="#17202a" text-anchor="middle">
    <g stroke="#34495e" stroke-width="2" fill="#ffffff">
      <rect x="35" y="80" width="130" height="64" rx="6"/><rect x="190" y="80" width="130" height="64" rx="6"/>
      <rect x="345" y="80" width="130" height="64" rx="6"/><rect x="500" y="80" width="130" height="64" rx="6"/>
      <rect x="655" y="35" width="130" height="64" rx="6"/><rect x="655" y="125" width="130" height="64" rx="6"/>
      <rect x="810" y="80" width="115" height="64" rx="6"/>
    </g>
    <text x="100" y="108">Run manifest</text><text x="100" y="130">+ raw identity</text>
    <text x="255" y="108">E00 / E01</text><text x="255" y="130">qualification</text>
    <text x="410" y="108">E02</text><text x="410" y="130">pixel controls</text>
    <text x="565" y="108">Metrics</text><text x="565" y="130">+ statistics</text>
    <text x="720" y="63">CSV</text><text x="720" y="85">+ Markdown</text>
    <text x="720" y="153">Figure source</text><text x="720" y="175">+ SVG</text>
    <text x="867" y="108">Lineage</text><text x="867" y="130">SHA256</text>
    <g stroke="#7f8c8d" stroke-width="2" fill="none" marker-end="url(#arrow)">
      <path d="M165 112H190"/><path d="M320 112H345"/><path d="M475 112H500"/><path d="M630 95H655"/><path d="M630 130H655"/><path d="M785 95H810"/><path d="M785 157H810"/>
    </g>
  </g>
  <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#7f8c8d"/></marker></defs>
  <text x="35" y="235" font-family="Arial, sans-serif" font-size="13" fill="#34495e">run_id={run_id}</text>
  <text x="35" y="255" font-family="Arial, sans-serif" font-size="13" fill="#34495e">evidence_tier={evidence_tier}; status={status}</text>
</svg>
'''


def _write_figure_manifest(path: Path, metadata: Mapping[str, Any], outputs: Mapping[str, str]) -> None:
    lines = [
        "# Figure manifest",
        "",
        "This manifest is generated from the run metrics; it contains no hand-entered metric values.",
        "",
        f"- run_id: `{metadata['run_id']}`",
        f"- evidence_tier: `{metadata['evidence_tier']}`",
        f"- status: `{metadata['status']}`",
        f"- config_sha256: `{metadata['config_sha256']}`",
        "",
        "| Artifact | SHA256 |",
        "| --- | --- |",
    ]
    for relative, digest in sorted(outputs.items()):
        lines.append(f"| `{relative}` | `{digest}` |")
    lines.append("")
    _write_text(path, "\n".join(lines))


def generate_report(
    run_root: Path | str,
    *,
    metrics_paths: Iterable[Path | str] | Path | str | None = None,
    output_root: Path | str | None = None,
) -> dict[str, Any]:
    """Generate CSV/Markdown summaries, figure data, SVG and SHA256 lineage."""

    root = Path(run_root).expanduser().resolve(strict=True)
    try:
        manifest = read_run_manifest(root)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ReportingError(f"Run manifest is not readable: {root / 'manifest/run.json'}") from exc

    run_id = str(manifest.get("run_id", root.name))
    evidence_tier = str(manifest.get("evidence_tier", ""))
    status = str(manifest.get("status", ""))
    config_sha256 = str(manifest.get("resolved_config_sha256", ""))
    if not run_id or not evidence_tier or not status or not config_sha256:
        raise ReportingError("Run manifest lacks run_id, evidence_tier, status or resolved_config_sha256")

    raw_records, metric_files = load_metric_records(root, metrics_paths)
    source_paths = _source_files(root, metric_files)
    source_entries = [_relative_entry(root, path) for path in source_paths]
    source_manifest_paths = _source_manifest_files(root)
    statistics_paths = sorted(
        path
        for path in (root / "statistics").rglob("*")
        if path.is_file() and path.stat().st_size > 0
    ) if (root / "statistics").exists() else []
    if not source_manifest_paths:
        raise ReportingError("Run has no source manifest files to bind report lineage")
    if not statistics_paths:
        raise ReportingError("Run has no statistics files to bind report lineage")
    source_manifest_entries = [_relative_entry(root, path) for path in source_manifest_paths]
    statistics_entries = [_relative_entry(root, path) for path in statistics_paths]
    source_hash_by_path = {entry["path"]: entry["sha256"] for entry in source_entries}
    metadata = {
        "run_id": run_id,
        "evidence_tier": evidence_tier,
        "status": status,
        "config_sha256": config_sha256,
    }
    rows: list[dict[str, Any]] = []
    for item in raw_records:
        record = dict(item["_record"])
        source_file = str(item["_source_file"])
        record_hash = sha256_json(record)
        row_lineage_hash = sha256_json(
            {
                "run_id": run_id,
                "config_sha256": config_sha256,
                "source_file": source_file,
                "source_sha256": source_hash_by_path[source_file],
                "record_sha256": record_hash,
                "line_number": item["_line_number"],
                "record_index": item["_record_index"],
            }
        )
        row_metadata = {
            **metadata,
            "source_file": source_file,
            "source_sha256": source_hash_by_path[source_file],
            "record_sha256": record_hash,
            "lineage_sha256": row_lineage_hash,
        }
        rows.append(_normalise_row(record, row_metadata))

    rows.sort(key=lambda row: (str(row["source_file"]), str(row["record_sha256"])))
    lineage_sha256 = sha256_json(
        {"run_id": run_id, "config_sha256": config_sha256, "source_files": source_entries}
    )
    for row in rows:
        row["lineage_sha256"] = lineage_sha256

    if output_root is None:
        tables_root = root / "tables"
    else:
        output_path = Path(output_root).expanduser()
        tables_root = (root / output_path).resolve() if not output_path.is_absolute() else output_path.resolve()
        if root not in tables_root.parents:
            raise ReportingError("output_root must be inside the run directory")
    figures_root = root / "figures"
    tables_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    summary_csv = tables_root / "summary.csv"
    summary_md = tables_root / "summary.md"
    figure_source = figures_root / "metric_source.csv"
    workflow_svg = figures_root / "workflow.svg"
    figure_manifest = figures_root / "manifest.md"
    lineage_path = tables_root / "summary.lineage.json"

    _write_csv(summary_csv, rows)
    _write_markdown(summary_md, rows, source_entries, metadata)
    _write_csv(figure_source, rows)
    _write_text(workflow_svg, _workflow_svg(metadata))
    figure_outputs = {
        figure_source.relative_to(root).as_posix(): sha256_file(figure_source),
        workflow_svg.relative_to(root).as_posix(): sha256_file(workflow_svg),
    }
    _write_figure_manifest(figure_manifest, metadata, figure_outputs)

    output_paths = [summary_csv, summary_md, figure_source, workflow_svg, figure_manifest]
    output_entries = [_relative_entry(root, path) for path in output_paths]
    lineage_payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "run_report",
        "run_id": run_id,
        "evidence_tier": evidence_tier,
        "status": status,
        "config_sha256": config_sha256,
        "source_files": source_entries,
        "source_manifest": source_manifest_entries,
        "statistics_files": statistics_entries,
        "outputs": output_entries,
        "generated_at_utc": _now(),
        "generator": "cfra.reporting.generate",
    }
    lineage_payload.update(_provenance_sections(root, manifest))
    lineage_payload["lineage_sha256"] = sha256_json(lineage_payload)
    _write_text(lineage_path, json.dumps(lineage_payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")

    return {
        "run_id": run_id,
        "evidence_tier": evidence_tier,
        "status": status,
        "config_sha256": config_sha256,
        "record_count": len(rows),
        "source_files": source_entries,
        "lineage_sha256": lineage_sha256,
        "outputs": {
            "summary_csv": summary_csv,
            "summary_markdown": summary_md,
            "figure_source_csv": figure_source,
            "workflow_svg": workflow_svg,
            "figure_manifest": figure_manifest,
            "lineage": lineage_path,
        },
    }


__all__ = ["ReportingError", "generate_report", "load_metric_records"]
