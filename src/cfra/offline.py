"""Offline trusted-state runner with artifact-based stage resume.

This module is the single-machine, offline-only execution path.  Unlike the
formal clean-room machinery in ``cfra.runs`` / ``cfra.resume``, it assumes
that every intermediate state inside one run directory is trustworthy:

* run directories must already exist (the runner never creates them);
* completion is tracked by ``checkpoint.json`` plus artifact presence;
* config/code hash checks are not enforced;
* changing the resolved parameters (datasets/seed/models) invalidates all
  derived outputs and restarts from E00;
* the formal orchestration (scripts/30-32, registry claim state machine) is
  deliberately untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cfra.configuration import config_sha256, load_experiment_config
from cfra.data import load_class_registry
from cfra.logging_utils import log_stage_complete, log_stage_start
from cfra.models import (
    SharedTextPrototypeBaseline,
    load_adapter,
    load_shared_text_encoder,
    parse_model_runtime_config,
    resolve_model_assets,
)
from cfra.pipeline import (
    build_e00,
    build_e01,
    build_e02,
    extract_feature_table,
    iter_context_pair_batches_parallel,
    load_current_manifests,
    load_e01_selected,
    load_feature_table,
    make_feature_samples,
    run_e03,
    run_e04,
    run_e04_1,
    run_e04_2,
    run_e04_3,
    run_e04_4,
    run_e04_5,
    run_e05,
    run_e06,
    run_e06_readout,
    run_e07,
    run_e08,
    write_core_results,
)
from cfra.pipeline.core import write_core_statistics
from cfra.resume import (
    add_completed_stage,
    load_stage_result,
    write_checkpoint,
    write_stage_result,
)
from cfra.sensitivity import resolve_sensitivity_variants
from cfra.runs import RUN_SUBDIRECTORIES, write_manifest_json

logger = logging.getLogger("cfra")

OFFLINE_SCHEMA_VERSION = 1
SIGNATURE_FIELDS = ("datasets", "seed", "models", "perturbations", "experiment_ids", "run_e04_5")


class OfflineRunnerError(RuntimeError):
    """Raised when the offline runner cannot safely proceed."""


def _enabled_perturbations(config: dict[str, Any]) -> list[str]:
    return [
        name
        for name in config["scope"]["perturbations"]
        if bool(config.get("perturbations", {}).get(name, {}).get("enabled", True))
    ]


def _sensitivity_variants(config: dict[str, Any]) -> list[tuple[str, str, dict[str, Any] | None]]:
    """Resolve the registered sensitivity grid from the package implementation."""

    return resolve_sensitivity_variants(config)


def _numpy_encoded(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _json_safe(value.tolist())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _read_json_file(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parameters_signature(
    *,
    datasets: Sequence[str],
    seed: int,
    models: Sequence[str],
    perturbations: Sequence[str],
    experiment_ids: Sequence[str],
    run_e04_5: bool,
    config_hash: str = "",
) -> str:
    """Stable hash of the run-affecting parameters (all inputs to the pipeline).

    ``config_hash`` folds the entire resolved configuration (sampling budgets,
    max classes, statistics, controls, layer selection, ...) into the signature
    so that editing the template config also triggers a reset from E00.
    """

    payload = {
        "datasets": list(datasets),
        "seed": int(seed),
        "models": list(models),
        "perturbations": list(perturbations),
        "experiment_ids": list(experiment_ids),
        "run_e04_5": bool(run_e04_5),
        "config_hash": config_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_offline_config(
    config_path: Path | str,
    *,
    datasets: Sequence[str],
    seed: int,
    models: Sequence[str],
    perturbations: Sequence[str],
    experiment_ids: Sequence[str],
    run_e04_5: bool,
) -> dict[str, Any]:
    """Resolve the offline template config and apply the requested parameters."""

    config = load_experiment_config(config_path)
    scope = config.setdefault("scope", {})
    scope["datasets"] = list(datasets)
    scope["primary_datasets"] = list(datasets)
    scope["boundary_datasets"] = ["fair1m"] if "fair1m" in datasets else []
    scope["models"] = list(models)
    scope["experiment_ids"] = list(experiment_ids)
    if perturbations:
        scope["perturbations"] = list(perturbations)
    split = config.setdefault("split", {})
    split["seed"] = int(seed)
    split["split_seeds"] = [int(seed)]
    robustness = config.get("robustness")
    if isinstance(robustness, dict):
        robustness["split_seeds"] = [int(seed)]
    statistics = config.get("statistics")
    if isinstance(statistics, dict):
        statistics["bootstrap_seed"] = int(seed)
    controls = config.setdefault("experiment_controls", {})
    controls.setdefault("E04-5", {})["enabled"] = bool(run_e04_5)
    return config


def _layerwise_cache_has_readout(
    root: Path, model_id: str, dataset_id: str, selected_layers: list[str]
) -> bool:
    """Whether a layerwise cache carries the clean ``__raw`` arrays the readout needs."""

    artifact = root / "intermediates" / "features" / f"layerwise_{model_id}_{dataset_id}.npz"
    if not artifact.is_file():
        return False
    try:
        with np.load(artifact, allow_pickle=False) as payload:
            keys = set(payload.files)
    except (OSError, ValueError):
        return False
    if selected_layers:
        return all(f"{layer}__raw" in keys for layer in selected_layers)
    return any(key.endswith("__raw") for key in keys)


def infer_completed_stages(run_root: Path | str, config: dict[str, Any]) -> list[str]:
    """Rebuild the completed-stage list from artifact presence (trusted state).

    Only artifacts that are already complete are trusted; a partially written
    stage file is treated as absent.
    """

    root = Path(run_root).expanduser().resolve()
    completed: list[str] = []
    e00 = root / "manifest" / "e00.json"
    e01 = root / "manifest" / "e01.json"
    e02 = root / "intermediates" / "e02_pixel_protocol" / "report.json"
    if e00.is_file():
        completed.append("E00")
    if e01.is_file():
        completed.append("E01")
    if e02.is_file():
        completed.append("E02")

    models = [str(item) for item in config["scope"].get("models", [])]
    features_root = root / "intermediates" / "features"
    for model_id in models:
        feature_path = features_root / f"features_{model_id}.npz"
        feature_manifest = root / "manifest" / f"features_{model_id}.json"
        if feature_path.is_file() and feature_manifest.is_file():
            completed.append(f"features:{model_id}:nominal")

    stage_results = root / "intermediates" / "stage_results"
    for stage_id in ("E03", "E04", "E04-1", "E04-2", "E04-3", "E04-4", "E04-5", "E05", "E07"):
        if (stage_results / f"{stage_id}.json").is_file():
            completed.append(stage_id)
        for model_id in models:
            if (stage_results / f"{stage_id}_{model_id}.json").is_file():
                completed.append(f"{stage_id}:{model_id}")

    experiments = {str(item) for item in config["scope"].get("experiment_ids", [])}
    if "E06" in experiments:
        readout_enabled = bool(config.get("robustness", {}).get("layerwise_readout", True))
        selected_layers = list(config.get("robustness", {}).get("selected_layers", []))
        for model_id in models:
            metric_ok = all(
                (root / "metrics" / f"E06_{model_id}_{dataset_id}.json").is_file()
                for dataset_id in config["scope"].get("datasets", [])
            )
            if metric_ok and readout_enabled:
                # A pre-readout E06 cache has no ``__raw`` layer arrays, so the
                # centered readout has never run.  Require the arrays to exist
                # before treating E06 as complete, otherwise enabling
                # layerwise_readout would never regenerate the readout table.
                metric_ok = all(
                    _layerwise_cache_has_readout(root, model_id, dataset_id, selected_layers)
                    for dataset_id in config["scope"].get("datasets", [])
                )
            if metric_ok:
                completed.append(f"E06:{model_id}")
    if "E08" in experiments:
        e08_datasets = [
            str(item) for item in config["scope"].get("datasets", [])
        ]
        e08_perturbations = [
            str(item) for item in config["scope"].get("perturbations", [])
        ]
        for model_id in models:
            all_cells_done = all(
                (root / "metrics" / f"E08_{model_id}_{dataset_id}_{perturbation}.json").is_file()
                for dataset_id in e08_datasets
                for perturbation in e08_perturbations
            )
            if all_cells_done and (root / "metrics" / f"E08_{model_id}.json").is_file():
                completed.append(f"E08:{model_id}")
        text_models = config.get("text_baseline_protocol", {}).get("models", [])
        for text_model in text_models:
            if (root / "metrics" / f"E08_text_{text_model}.json").is_file():
                completed.append(f"E08text:{text_model}")

    if (root / "statistics" / "core_statistics.json").is_file() and (
        root / "metrics" / "core_results.json"
    ).is_file():
        completed.append("results")
    return completed


def _safe_to_initialize(run_root: Path) -> bool:
    """A directory is initializable only if it contains at most empty standard dirs.

    ``logs`` (populated by ``setup_logger`` before the run is opened) is always
    tolerated; other standard run directories must be empty.
    """

    for entry in run_root.iterdir():
        if not entry.is_dir():
            return False
        if entry.name == "logs":
            continue
        if entry.name in RUN_SUBDIRECTORIES and not any(entry.iterdir()):
            continue
        return False
    return True


_SEED_ONLY_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("split", ("seed", "split_seeds", "seed_policy")),
    ("robustness", ("split_seeds", "seed_policy")),
    ("statistics", ("bootstrap_seed",)),
)


def _e00_relevant(config: Mapping[str, Any]) -> dict[str, Any]:
    """Config with seed-only fields removed, for cross-seed E00 reuse checks.

    E00 indexes the raw datasets and does not depend on the split/bootstrap
    seed; every other field (datasets, sampling, input, controls, protocol,
    perturbations) can change its output.
    """

    stripped = _json_safe(dict(config))
    for section, keys in _SEED_ONLY_KEYS:
        node = stripped.get(section)
        if isinstance(node, dict):
            for key in keys:
                node.pop(key, None)
    return stripped


_ANALYSIS_ONLY_SECTIONS: tuple[str, ...] = (
    "statistics",
    "robustness",
    "intervention",
    "methods",
    "retrieval",
    "text_baseline_protocol",
    "experiment_controls",
    "reporting",
    "figures",
    "export",
)
_ANALYSIS_ONLY_SCOPE_KEYS: tuple[str, ...] = ("experiment_ids",)


def _feature_relevant(config: Mapping[str, Any]) -> dict[str, Any]:
    """Config with analysis-only fields removed, for feature-cache reuse checks.

    Only feature-affecting settings (datasets, sampling, input, controls,
    perturbations, protocol input handling, split) may invalidate the cached
    encoder features.  Analysis-only sections -- statistics, robustness
    (including the new ``layerwise_readout`` switch), interventions, methods,
    retrieval, text baselines and experiment toggles -- change derived outputs
    but not the encoded features, so they must not trigger a feature reset.
    """

    stripped = _json_safe(dict(config))
    for section in _ANALYSIS_ONLY_SECTIONS:
        stripped.pop(section, None)
    scope = stripped.get("scope")
    if isinstance(scope, dict):
        for key in _ANALYSIS_ONLY_SCOPE_KEYS:
            scope.pop(key, None)
    return stripped


def _same_feature_inputs(previous_config: Any, config: Mapping[str, Any]) -> bool:
    """Whether the cached encoder features remain valid for ``config``."""

    if not isinstance(previous_config, Mapping) or not previous_config:
        return False
    return _feature_relevant(previous_config) == _feature_relevant(config)


def _can_reuse_e00(run_root: Path, previous_config: Any, config: Mapping[str, Any]) -> bool:
    """Whether ``CFRA_REUSE_E00=1`` may carry E00 across a seed-only reset."""

    if os.environ.get("CFRA_REUSE_E00", "0") != "1":
        return False
    if not (run_root / "intermediates" / "e00_source_manifest").is_dir():
        return False
    if not isinstance(previous_config, Mapping) or not previous_config:
        return False
    return _e00_relevant(previous_config) == _e00_relevant(config)


def _clear_run_outputs(run_root: Path, *, preserve_e00: bool = False) -> None:
    """Delete derived outputs, optionally preserving reusable input caches.

    Encoder feature caches (``intermediates/features``) cost hours of GPU time
    to rebuild, so they are stashed and restored unless the caller explicitly
    opts in to dropping them with ``CFRA_ALLOW_FEATURE_RESET=1``.
    """

    stashes: list[tuple[Path, Path]] = []

    def _stash(relative: str) -> None:
        target = run_root / relative
        if not target.is_dir():
            return
        stash = run_root / f".reuse_stash_{relative.replace('/', '_')}"
        if stash.exists():
            shutil.rmtree(stash, ignore_errors=True)
        shutil.move(str(target), str(stash))
        stashes.append((target, stash))

    if preserve_e00:
        _stash("intermediates/e00_source_manifest")
    if os.environ.get("CFRA_ALLOW_FEATURE_RESET", "0") != "1":
        _stash("intermediates/features")

    for relative in ("intermediates", "metrics", "statistics", "tables", "figures"):
        target = run_root / relative
        if target.is_dir():
            shutil.rmtree(target)

    for target, stash in stashes:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stash), str(target))


def _initialize_offline_run(run_root: Path, config: dict[str, Any], signature: str) -> None:
    for relative in RUN_SUBDIRECTORIES:
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": OFFLINE_SCHEMA_VERSION,
        "run_id": run_root.name,
        "evidence_tier": "development",
        "mode": "offline",
        "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "allow_resume": True,
        "allow_cross_run_cache": True,
        "parameter_signature": signature,
        "resolved_config_sha256": config_sha256(config),
        "resolved_config": config,
    }
    write_manifest_json(run_root, "manifest/run.json", manifest)
    (run_root / "STATUS").write_text("running\n", encoding="ascii")
    write_checkpoint(
        run_root,
        config_sha256=config_sha256(config),
        evidence_tier="development",
        completed_stages=[],
    )


def open_offline_run(
    run_dir: Path | str,
    config: dict[str, Any],
    signature: str,
) -> tuple[Path, str, list[str]]:
    """Open an explicitly named offline run directory.

    Modes:
    * ``fresh``    -- directory exists and is empty; initialized now;
    * ``resume``   -- directory holds an offline run with the same parameter
                      signature; completed stages are loaded from the checkpoint
                      or inferred from artifacts;
    * ``reset``    -- directory holds an offline run with a different parameter
                      signature; derived outputs are cleared and E00 restarts.

    A directory that does not exist is always rejected.
    """

    run_root = Path(run_dir).expanduser().resolve()
    if not run_root.exists():
        raise OfflineRunnerError(
            f"Run directory does not exist and will not be created: {run_root}. "
            "Create (or reuse) an existing empty directory first."
        )
    if not run_root.is_dir():
        raise OfflineRunnerError(f"Run directory is not a directory: {run_root}")

    run_manifest_path = run_root / "manifest" / "run.json"
    if run_manifest_path.is_file():
        try:
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OfflineRunnerError(f"Run manifest is corrupted: {run_manifest_path}") from exc
        if run_manifest.get("mode") != "offline":
            raise OfflineRunnerError(
                f"Run directory was not created by the offline runner: {run_root}"
            )
        previous = str(run_manifest.get("parameter_signature", ""))
        if previous != signature:
            previous_config = run_manifest.get("resolved_config")
            hard_reset = os.environ.get("CFRA_HARD_RESET", "0") == "1"
            features_changed = not _same_feature_inputs(previous_config, config)
            if hard_reset or features_changed:
                reuse_e00 = _can_reuse_e00(run_root, previous_config, config)
                if features_changed and not hard_reset:
                    logger.warning(
                        "[offline] 参数签名变化且涉及特征输入，执行重置；"
                        "特征缓存仅在 CFRA_ALLOW_FEATURE_RESET=1 时清除"
                    )
                else:
                    logger.info(f"[offline] 参数签名变化，执行重置: {run_root}")
                _clear_run_outputs(run_root, preserve_e00=reuse_e00)
                completed = ["E00"] if reuse_e00 else []
                mode = "reset"
            else:
                # Analysis-only config change (e.g. enabling layerwise_readout):
                # keep the expensive feature caches and recompute only the
                # stages whose artifacts are absent.
                logger.info(
                    "[offline] 参数签名变化仅涉及分析配置；保留特征缓存，按产物缺失补算 (soft reset)"
                )
                completed = infer_completed_stages(run_root, config)
                mode = "soft-reset"
            write_checkpoint(
                run_root,
                config_sha256=config_sha256(config),
                evidence_tier="development",
                completed_stages=completed,
            )
            run_manifest["parameter_signature"] = signature
            run_manifest["resolved_config_sha256"] = config_sha256(config)
            run_manifest["resolved_config"] = config
            run_manifest["resume_events"] = run_manifest.get("resume_events", [])
            run_manifest["resume_events"].append(
                {
                    "at_utc": datetime.now(UTC).isoformat(),
                    "kind": "soft_reset" if mode == "soft-reset" else "reset_parameters",
                    "features_changed": bool(features_changed),
                }
            )
            write_manifest_json(run_root, "manifest/run.json", run_manifest)
            (run_root / "STATUS").write_text("running\n", encoding="ascii")
            return run_root, mode, completed

        # Artifacts are the source of truth: reconcile the checkpoint against
        # what is actually on disk so a deleted/missing stage is recomputed
        # automatically on the next run.
        completed = infer_completed_stages(run_root, config)
        write_checkpoint(
            run_root,
            config_sha256=config_sha256(config),
            evidence_tier="development",
            completed_stages=completed,
        )
        run_manifest["status"] = "running"
        run_manifest["resolved_config_sha256"] = config_sha256(config)
        run_manifest["resolved_config"] = config
        run_manifest["resume_events"] = run_manifest.get("resume_events", [])
        run_manifest["resume_events"].append(
            {"at_utc": datetime.now(UTC).isoformat(), "kind": "resume"}
        )
        write_manifest_json(run_root, "manifest/run.json", run_manifest)
        (run_root / "STATUS").write_text("running\n", encoding="ascii")
        return run_root, "resume", completed
    if not _safe_to_initialize(run_root):
        raise OfflineRunnerError(
            f"Run directory is non-empty but has no offline run manifest: {run_root}"
        )
    _initialize_offline_run(run_root, config, signature)
    return run_root, "fresh", []


def _reload_offline_run(run_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = run_root / "manifest" / "run.json"
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_manifest["resolved_config"] = config
    run_manifest["resolved_config_sha256"] = config_sha256(config)
    write_manifest_json(run_root, "manifest/run.json", run_manifest)
    return run_manifest


def _stage_done(completed: list[str], stage_id: str) -> bool:
    return stage_id in completed


def _legacy_stage_covers_model(stage_id: str, model_id: str) -> bool:
    """Legacy whole-stage checkpoints (e.g. ``E03``) cover every registered model.

    Before model-level resume was introduced, a completed core stage was recorded
    as one entry (e.g. ``E03``) covering all models.  Such an entry is honoured as
    all-models-complete for the same stage id.
    """

    return True


def _core_stage(
    stage_id: str,
    table: Any,
    *,
    datasets: list[str],
    perturbations: list[str],
    config: dict[str, Any],
    seed: int,
    run_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    statistics_config = config.get("statistics", {})
    controls_config = config.get("controls", {})
    method_ids = list(config["methods"]["ids"])
    normalizations = list(config["protocol"]["normalization_variants"])
    epsilon = float(config["protocol"]["near_zero_norm_epsilon"])
    if stage_id == "E03":
        return run_e03(
            table,
            dataset_ids=datasets,
            perturbations=perturbations,
            statistics_config=statistics_config,
            random_seed=seed,
        )
    if stage_id == "E04":
        cell_workers = max(1, int(os.environ.get("CFRA_CELL_WORKERS", "1")))
        if cell_workers > 1 and run_root is not None:
            from cfra.pipeline.core import run_e04_parallel

            return run_e04_parallel(
                table,
                run_root=run_root,
                dataset_ids=datasets,
                perturbations=perturbations,
                method_ids=method_ids,
                normalization_variants=normalizations,
                epsilon=epsilon,
                controls=controls_config,
                statistics_config=statistics_config,
                random_seed=seed,
                workers=cell_workers,
            )
        return run_e04(
            table,
            dataset_ids=datasets,
            perturbations=perturbations,
            method_ids=method_ids,
            normalization_variants=normalizations,
            epsilon=epsilon,
            controls=controls_config,
            statistics_config=statistics_config,
            random_seed=seed,
        )
    if stage_id == "E04-1":
        return run_e04_1(
            table,
            dataset_ids=datasets,
            perturbations=perturbations,
            epsilon=epsilon,
            fit_size=int(config["protocol"]["primary_fit_size"]),
            statistics_config=statistics_config,
            random_seed=seed,
        )
    if stage_id == "E04-2":
        if "fair1m" not in datasets:
            return [
                {
                    "stage": "E04-2",
                    "status": "skipped",
                    "reason": "E04-2 is registered only for FAIR1M; the current scope excludes it",
                }
            ]
        return run_e04_2(
            table,
            dataset_ids=["fair1m"],
            perturbations=perturbations,
            support_sizes=[int(value) for value in config["protocol"]["support_size_variants"]],
            epsilon=epsilon,
            statistics_config=statistics_config,
            random_seed=seed,
        )
    if stage_id == "E04-3":
        return run_e04_3(
            table,
            dataset_ids=datasets,
            perturbations=perturbations,
            statistics_config=statistics_config,
            random_seed=seed,
        )
    if stage_id == "E04-4":
        return run_e04_4(
            table,
            dataset_ids=datasets,
            perturbations=perturbations,
            normalization_variants=normalizations,
            epsilon=epsilon,
            fit_size=int(config["protocol"]["primary_fit_size"]),
            controls=controls_config,
            statistics_config=statistics_config,
            random_seed=seed,
        )
    if stage_id == "E04-5":
        return run_e04_5(
            {"nominal": table},
            dataset_ids=datasets,
            perturbations=perturbations,
            epsilon=epsilon,
            method_ids=method_ids,
            normalization_variants=normalizations,
            fit_size=int(config["protocol"]["primary_fit_size"]),
            controls=controls_config,
            statistics_config=statistics_config,
            random_seed=seed,
        )
    if stage_id == "E05":
        return run_e05(
            table,
            dataset_ids=datasets,
            perturbations=perturbations,
            epsilon=epsilon,
            statistics_config=statistics_config,
        )
    if stage_id == "E07":
        cell_workers = max(1, int(os.environ.get("CFRA_CELL_WORKERS", "1")))
        if cell_workers > 1 and run_root is not None:
            from cfra.pipeline.core import run_e07_parallel

            return run_e07_parallel(
                table,
                run_root=run_root,
                dataset_ids=datasets,
                perturbations=perturbations,
                removal_alphas=[float(value) for value in config["intervention"]["removal_alphas"]],
                reflection_alpha=float(config["intervention"]["reflection_alpha"]),
                preserve_norm=bool(config["intervention"]["preserve_feature_norm"]),
                epsilon=epsilon,
                statistics_config=statistics_config,
                random_seed=seed,
                workers=cell_workers,
            )
        return run_e07(
            table,
            dataset_ids=datasets,
            perturbations=perturbations,
            removal_alphas=[float(value) for value in config["intervention"]["removal_alphas"]],
            reflection_alpha=float(config["intervention"]["reflection_alpha"]),
            preserve_norm=bool(config["intervention"]["preserve_feature_norm"]),
            epsilon=epsilon,
            statistics_config=statistics_config,
            random_seed=seed,
        )
    raise OfflineRunnerError(f"Unknown core stage: {stage_id}")


def _retrieval_masks(
    table: Any,
    *,
    dataset_id: str | None = None,
    perturbation: str | None = None,
) -> tuple[list[bool], list[bool]]:
    confirmation_images = sorted(
        {
            sample.source_image_id
            for sample in table.metadata
            if sample.role == "confirmation"
            and (dataset_id is None or sample.dataset_id == dataset_id)
            and (perturbation is None or sample.perturbation == perturbation)
        },
        key=str,
    )
    if len(confirmation_images) < 2:
        raise OfflineRunnerError("E08 requires at least two confirmation source images")
    split = max(1, len(confirmation_images) // 2)
    candidate_images = set(confirmation_images[:split])
    query_images = set(confirmation_images[split:])
    return (
        [
            sample.role == "confirmation"
            and sample.source_image_id in candidate_images
            and (dataset_id is None or sample.dataset_id == dataset_id)
            and (perturbation is None or sample.perturbation == perturbation)
            for sample in table.metadata
        ],
        [
            sample.role == "confirmation"
            and sample.source_image_id in query_images
            and (dataset_id is None or sample.dataset_id == dataset_id)
            and (perturbation is None or sample.perturbation == perturbation)
            for sample in table.metadata
        ],
    )


def _load_cached_e06_inputs(
    *,
    run_root: Path,
    config: dict[str, Any],
    model_id: str,
    dataset_id: str,
    selected_layers: list[str],
    readout_enabled: bool,
    expected_samples: Sequence[Any] | None = None,
) -> tuple[dict[str, Any], list[Any], dict[str, dict[str, np.ndarray]] | None] | None:
    """Rebuild E06 inputs from an existing layerwise artifact when it is complete.

    Returns ``(layer_inputs, metadata, readout_arrays)`` or ``None`` when the
    cache is absent, malformed or does not match the current configuration.  A
    cache hit lets E06 and its centered readout be recomputed on CPU without
    loading an encoder.

    The cache is validated against the current configuration through the input
    signature (model, dataset, crop geometry, nominal perturbation strengths and
    the input-affecting protocol values), its recorded ``sample_count``, the
    per-row perturbation seeds and, when ``expected_samples`` is supplied, the
    exact object/perturbation row order.  Every rejection is logged with its
    reason so a fall back to encoder inference is never unexplained.
    """

    from cfra.pipeline.layerwise import LayerwiseError, load_layerwise_feature_artifact

    signature = _layerwise_input_signature(config, model_id, dataset_id)
    artifact_name = f"layerwise_{model_id}_{dataset_id}"
    artifact_path = run_root / "intermediates" / "features" / f"{artifact_name}.npz"
    if not artifact_path.is_file():
        logger.info(f"[E06] {model_id}/{dataset_id}: cache rejected: artifact {artifact_path.name} is absent")
        return None
    try:
        table = load_layerwise_feature_artifact(run_root, artifact_name=artifact_name)
    except (LayerwiseError, OSError, ValueError) as exc:
        logger.info(f"[E06] {model_id}/{dataset_id}: cache rejected: artifact could not be loaded ({exc})")
        return None
    metadata = table.metadata
    if not isinstance(metadata, (list, tuple)) or not metadata:
        logger.info(f"[E06] {model_id}/{dataset_id}: cache rejected: artifact carries no row metadata")
        return None
    if getattr(table, "model_id", None) not in (None, model_id):
        logger.info(
            f"[E06] {model_id}/{dataset_id}: cache rejected: artifact model_id "
            f"{getattr(table, 'model_id', None)!r} does not match the requested model"
        )
        return None
    representation = table.representation if isinstance(table.representation, dict) else {}
    manifest_dataset = representation.get("dataset_id")
    if manifest_dataset not in (None, dataset_id):
        logger.info(
            f"[E06] {model_id}/{dataset_id}: cache rejected: artifact dataset_id "
            f"{manifest_dataset!r} does not match the requested dataset"
        )
        return None
    if representation.get("input_signature") != signature:
        logger.info(
            f"[E06] {model_id}/{dataset_id}: cache rejected: input signature changed "
            f"(crop, nominal perturbation strength or protocol settings differ)"
        )
        return None
    if not _cached_e06_matches_config(metadata, config):
        logger.info(
            f"[E06] {model_id}/{dataset_id}: cache rejected: cached rows do not match the "
            f"current perturbation scope or split seed"
        )
        return None
    if expected_samples is not None and not _cached_e06_rows_match(metadata, expected_samples):
        logger.info(
            f"[E06] {model_id}/{dataset_id}: cache rejected: cached row count/order "
            f"({len(metadata)}) does not match the current manifest"
        )
        return None
    layer_inputs: dict[str, Any] = {}
    readout_arrays: dict[str, dict[str, np.ndarray]] = {}
    for layer in selected_layers:
        record = table.layers.get(layer)
        if record is None:
            logger.info(f"[E06] {model_id}/{dataset_id}: cache rejected: layer {layer!r} is not cached")
            return None
        features = record["features"]
        if "clear" not in features or "degraded" not in features:
            logger.info(f"[E06] {model_id}/{dataset_id}: cache rejected: layer {layer!r} lacks paired arrays")
            return None
        layer_inputs[layer] = {
            "features": {"clear": features["clear"], "degraded": features["degraded"]},
            "metadata": metadata,
            "representation": record["representation"],
            "model_id": model_id,
        }
        if readout_enabled:
            if "raw" not in features:
                logger.info(
                    f"[E06] {model_id}/{dataset_id}: cache rejected: layer {layer!r} lacks the "
                    f"clean raw arrays the readout intervention needs"
                )
                return None
            readout_arrays[layer] = {
                "raw": features["raw"],
                "clear": features["clear"],
                "degraded": features["degraded"],
            }
    return layer_inputs, list(metadata), (readout_arrays if readout_enabled else None)


def _cached_e06_matches_config(metadata: Sequence[Any], config: dict[str, Any]) -> bool:
    """Whether cached layerwise rows were produced under the current config.

    Compares the cached perturbation set, dataset ids and per-row perturbation
    seeds (derived from the split seed, object id and perturbation) so a cache
    built under a different split or perturbation scope is rejected instead of
    silently feeding stale features into E06.
    """

    from cfra.pipeline.features import _stable_seed

    def field(row: Any, name: str) -> Any:
        if isinstance(row, Mapping):
            return row.get(name)
        return getattr(row, name, None)

    scope = config.get("scope", {})
    expected_perturbations = set(scope.get("perturbations", []))
    expected_datasets = set(scope.get("datasets", []))
    split_seed = config.get("split", {}).get("seed")
    cached_perturbations: set[Any] = set()
    cached_datasets: set[Any] = set()
    for row in metadata:
        perturbation = field(row, "perturbation")
        dataset_id = field(row, "dataset_id")
        object_id = field(row, "object_id")
        seed_value = field(row, "seed")
        if perturbation is None or dataset_id is None or object_id is None or seed_value is None:
            return False
        cached_perturbations.add(perturbation)
        cached_datasets.add(dataset_id)
        if split_seed is not None:
            expected = _stable_seed(int(split_seed), object_id, perturbation)
            if int(seed_value) != expected:
                return False
    if expected_perturbations and cached_perturbations != expected_perturbations:
        return False
    if expected_datasets:
        return cached_datasets.issubset(expected_datasets)
    return True


_INPUT_PROTOCOL_FIELDS = (
    "boundary_radius_px",
    "control_shift_fraction",
    "control_min_iou",
    "control_max_iou",
    "control_max_attempts",
)


def _layerwise_input_signature(config: dict[str, Any], model_id: str, dataset_id: str) -> dict[str, Any]:
    """Input-construction settings that must match before a layerwise cache is reused.

    The per-row perturbation seed already ties a cache to the split seed and the
    object/perturbation identity; this signature additionally pins the crop
    geometry, the nominal perturbation parameters (the ``config["perturbations"]``
    block that determines the actual degraded pixels) and the input-affecting
    protocol values, none of which the per-row seed check can detect.  Caches
    written without a signature, or under different input settings, compare
    unequal and are recomputed.
    """

    input_cfg = config.get("input", {}) if isinstance(config.get("input"), dict) else {}
    protocol = config.get("protocol", {}) if isinstance(config.get("protocol"), dict) else {}
    perturbations = config.get("perturbations", {}) if isinstance(config.get("perturbations"), dict) else {}
    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "crop_protocol": input_cfg.get("protocol"),
        "size_px": input_cfg.get("size_px"),
        "padding": input_cfg.get("padding"),
        "obb_soft_mask_supersampling": input_cfg.get("obb_soft_mask_supersampling"),
        "perturbations": {key: perturbations[key] for key in sorted(perturbations)},
        "protocol": {key: protocol.get(key) for key in _INPUT_PROTOCOL_FIELDS},
        "split_seed": config.get("split", {}).get("seed"),
        "scope_perturbations": sorted(str(item) for item in config.get("scope", {}).get("perturbations", [])),
    }


def _cached_e06_rows_match(metadata: Sequence[Any], expected_samples: Sequence[Any]) -> bool:
    """Whether cached rows align row-for-row with the current sample order.

    Compares the ordered ``(object_id, perturbation)`` sequence and the sample
    count, so a cache built from a different object set or ordering cannot be
    silently reused even if the perturbation set and split seed agree.
    """

    if len(metadata) != len(expected_samples):
        return False

    def field(row: Any, name: str) -> Any:
        if isinstance(row, Mapping):
            return row.get(name)
        return getattr(row, name, None)

    for cached_row, expected in zip(metadata, expected_samples):
        if field(cached_row, "object_id") != field(expected, "object_id"):
            return False
        if field(cached_row, "perturbation") != field(expected, "perturbation"):
            return False
    return True


def _run_offline_e06(
    *,
    run_root: Path,
    project_root: Path,
    config: dict[str, Any],
    manifests: dict[str, Any],
    selected_by_dataset: dict[str, Any],
    model_id: str,
    asset: Any,
    runtime: Any,
    dataset_id: str,
) -> dict[str, Any]:
    selected_layers = list(config.get("robustness", {}).get("selected_layers", []))
    if not selected_layers:
        raise OfflineRunnerError("E06 requires robustness.selected_layers")
    if dataset_id not in selected_by_dataset:
        raise OfflineRunnerError(f"E06 dataset is not present in the manifest: {dataset_id}")
    readout_enabled = bool(config.get("robustness", {}).get("layerwise_readout", True))
    epsilon = float(config["protocol"]["near_zero_norm_epsilon"])
    expected_samples = make_feature_samples({dataset_id: selected_by_dataset[dataset_id]}, config)
    cached = _load_cached_e06_inputs(
        run_root=run_root,
        config=config,
        model_id=model_id,
        dataset_id=dataset_id,
        selected_layers=selected_layers,
        readout_enabled=readout_enabled,
        expected_samples=expected_samples,
    )
    fresh_artifact = False
    readout_arrays: dict[str, dict[str, np.ndarray]] | None = None
    if cached is not None:
        layer_inputs, samples, readout_arrays = cached
        logger.info(f"[E06] {model_id} {dataset_id}: 复用已缓存逐层特征，不加载编码器")
    else:
        fresh_artifact = True
        samples = expected_samples
        object_index = {item.object_id: item for manifest in manifests.values() for item in manifest.objects}
        clear_values: dict[str, list[np.ndarray]] = {layer: [] for layer in selected_layers}
        degraded_values: dict[str, list[np.ndarray]] = {layer: [] for layer in selected_layers}
        raw_values: dict[str, list[np.ndarray]] = {layer: [] for layer in selected_layers}
        layer_metadata: dict[str, dict[str, Any]] = {}
        with load_adapter(
            model_id,
            asset,
            runtime=runtime,
            registry_root=project_root / "registry",
            device="auto",
        ) as adapter:
            context_batches = iter_context_pair_batches_parallel(
                samples,
                manifests,
                object_index,
                config,
                batch_size=int(runtime.batch_size),
                cpu_workers=int(os.environ.get("CFRA_CPU_WORKERS", "0")) or None,
            )
            from concurrent.futures import ThreadPoolExecutor

            def _next_or_none() -> list[dict[str, Any]] | None:
                return next(context_batches, None)

            buffered = _next_or_none()
            with ThreadPoolExecutor(max_workers=1) as prefetcher:
                while buffered is not None:
                    future = prefetcher.submit(_next_or_none)
                    clear = adapter.encode_layerwise(
                        [context["true_clear_degraded"] for context in buffered],
                        selected_layers,
                        output_device="cpu",
                    )
                    degraded = adapter.encode_layerwise(
                        [context["true_degraded_degraded"] for context in buffered],
                        selected_layers,
                        output_device="cpu",
                    )
                    if tuple(clear.values) != tuple(selected_layers) or tuple(degraded.values) != tuple(selected_layers):
                        raise OfflineRunnerError(f"E06 layer output mismatch for model {model_id}")
                    raw = None
                    if readout_enabled:
                        raw = adapter.encode_layerwise(
                            [context["raw"] for context in buffered],
                            selected_layers,
                            output_device="cpu",
                        )
                        if tuple(raw.values) != tuple(selected_layers):
                            raise OfflineRunnerError(f"E06 raw layer output mismatch for model {model_id}")
                    for layer in selected_layers:
                        clear_values[layer].append(_numpy_encoded(clear.values[layer]))
                        degraded_values[layer].append(_numpy_encoded(degraded.values[layer]))
                        if raw is not None:
                            raw_values[layer].append(_numpy_encoded(raw.values[layer]))
                        layer_metadata[layer] = dict(clear.metadata[layer])
                    buffered = future.result()

        layer_inputs = {
            layer: {
                "features": {
                    "clear": np.concatenate(clear_values[layer], axis=0),
                    "degraded": np.concatenate(degraded_values[layer], axis=0),
                },
                "metadata": samples,
                "representation": layer_metadata[layer],
                "model_id": model_id,
            }
            for layer in selected_layers
        }
        if readout_enabled:
            readout_arrays = {
                layer: {
                    "raw": np.concatenate(raw_values[layer], axis=0),
                    "clear": layer_inputs[layer]["features"]["clear"],
                    "degraded": layer_inputs[layer]["features"]["degraded"],
                }
                for layer in selected_layers
            }

    result = run_e06(
        {"layers": layer_inputs, "metadata": samples, "model_id": model_id},
        selected_layers,
        epsilon=epsilon,
    )
    if readout_enabled and readout_arrays is not None:
        result["readout_intervention"] = run_e06_readout(
            readout_arrays,
            samples,
            selected_layers,
            epsilon=epsilon,
        )
    result.update(
        {
            "status": "completed",
            "execution_status": "completed",
            "configured_enabled": True,
            "interface_called": True,
            "layer_output_status": "completed",
            "input_source": "current_run_layerwise_features" if fresh_artifact else "cached_layerwise_features",
        }
    )
    artifact_root = run_root / "intermediates" / "features"
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / f"layerwise_{model_id}_{dataset_id}.npz"
    if fresh_artifact:
        artifact_arrays = {
            f"{layer}__clear": layer_inputs[layer]["features"]["clear"] for layer in selected_layers
        }
        artifact_arrays.update(
            {f"{layer}__degraded": layer_inputs[layer]["features"]["degraded"] for layer in selected_layers}
        )
        if readout_enabled:
            artifact_arrays.update(
                {f"{layer}__raw": np.concatenate(raw_values[layer], axis=0) for layer in selected_layers}
            )
        np.savez_compressed(artifact_path, **artifact_arrays)
        artifact_manifest = {
            "schema_version": 1,
            "artifact_kind": "layerwise_paired_features",
            "model_id": model_id,
            "selected_layers": selected_layers,
            "layer_metadata": _json_safe(layer_metadata),
            "sample_count": len(samples),
            "row_metadata": [_json_safe(item) for item in (vars(sample) if hasattr(sample, "__dict__") else sample for sample in samples)],
            "array_path": artifact_path.relative_to(run_root).as_posix(),
            "input_signature": _layerwise_input_signature(config, model_id, dataset_id),
        }
        artifact_manifest["dataset_id"] = dataset_id
        (artifact_root / f"layerwise_{model_id}_{dataset_id}.json").write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        write_manifest_json(run_root, f"manifest/layerwise_{model_id}_{dataset_id}.json", artifact_manifest)
    result["dataset_id"] = dataset_id
    _write_json_file(run_root / "metrics" / f"E06_{model_id}_{dataset_id}.json", result)
    write_manifest_json(
        run_root,
        f"manifest/E06_{model_id}_{dataset_id}.json",
        {
            "stage": "E06",
            "model_id": model_id,
            "dataset_id": dataset_id,
            "status": "completed",
            "input_source": "current_run_layerwise_features",
        },
    )
    return result


def _text_class_names(class_registry: Mapping[str, Any], dataset_id: str) -> dict[str, str]:
    classes = class_registry.get(dataset_id)
    if not isinstance(classes, Mapping) or not classes:
        raise OfflineRunnerError(f"E08 text baseline has no class registry for dataset {dataset_id}")
    result: dict[str, str] = {}
    for class_id, info in classes.items():
        name = getattr(info, "name", None)
        if name is None and isinstance(info, Mapping):
            name = info.get("name")
        if not isinstance(name, str) or not name.strip():
            raise OfflineRunnerError(
                f"E08 text baseline has an invalid class name for {dataset_id}:{class_id}"
            )
        result[f"{dataset_id}:{int(class_id)}"] = name.strip()
    return result


def _compute_text_baseline(
    *,
    table: Any,
    text_model: str,
    context: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
    seed: int,
    statistics_config: Mapping[str, Any],
    perturbations: list[str],
    datasets: list[str],
) -> list[dict[str, Any]]:
    text_results: list[dict[str, Any]] = []
    with load_shared_text_encoder(
        text_model,
        context["model_assets"][text_model],
        registry_root=project_root / "registry",
        runtime=context["runtime"],
        device="auto",
    ) as text_encoder:
        templates = tuple(
            config.get("text_baseline_protocol", {}).get(
                "prompt_templates",
                ("a satellite image of a {class_name}.",),
            )
        )
        variants = (("raw", False, False), ("centered", True, True))
        for dataset_id in datasets:
            class_names = _text_class_names(context["class_registry"], dataset_id)
            for perturbation in perturbations:
                candidate_mask, query_mask = _retrieval_masks(
                    table,
                    dataset_id=dataset_id,
                    perturbation=perturbation,
                )
                for space_name, centered, center_images in variants:
                    baseline = SharedTextPrototypeBaseline(
                        text_encoder,
                        class_names,
                        templates=templates,
                        centered=centered,
                        center_image_features=center_images,
                    )
                    text_result = run_e08(
                        table,
                        dataset_id=dataset_id,
                        perturbation=perturbation,
                        candidate_mask=candidate_mask,
                        query_mask=query_mask,
                        methods=("text_baseline",),
                        text_baseline=baseline,
                        k=int(config.get("retrieval", {}).get("k", 10)),
                        epsilon=float(config["protocol"]["near_zero_norm_epsilon"]),
                        bootstrap_replicates=int(statistics_config.get("bootstrap_replicates", 0)),
                        bootstrap_seed=seed,
                        bootstrap_unit="source_image",
                    )
                    text_result.update(
                        {
                            "status": "completed",
                            "execution_status": "completed",
                            "configured_enabled": True,
                            "interface_called": True,
                            "text_model_id": text_model,
                            "text_shared_space_id": text_encoder.shared_space_id,
                            "prompt_template_count": len(templates),
                            "text_baseline_space": space_name,
                        }
                    )
                    text_results.append(text_result)
    return text_results


def run_full_offline(
    *,
    run_root: Path | str,
    project_root: Path | str,
    config: dict[str, Any],
    config_path: Path | str,
    completed: list[str],
    run_e04_5: bool,
    export_only_handoff: bool = False,
) -> None:
    """Execute every pending stage from raw inputs, writing a checkpoint per stage."""

    root = Path(run_root).expanduser().resolve(strict=True)
    project = Path(project_root).expanduser().resolve(strict=True)
    experiments = [str(item) for item in config["scope"]["experiment_ids"]]
    models = [str(item) for item in config["scope"]["models"]]
    datasets = [str(item) for item in config["scope"]["datasets"]]
    perturbations = _enabled_perturbations(config)
    seed = int(config["split"]["seed"])
    statistics_config = config.get("statistics", {})
    total = len(experiments)
    step = 0

    manifests: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    selected: dict[str, Any] | None = None

    def resume_or_run(stage_id: str, call: Any) -> Any:
        nonlocal step
        step += 1
        if _stage_done(completed, stage_id):
            logger.info(f"[Step{step:02d}/{total:02d}] [{stage_id}] 跳过（已在 checkpoint/产物中完成）")
            return None
        started = time.time()
        log_stage_start(logger, stage_id, step, total)
        payload = call()
        log_stage_complete(logger, stage_id, time.time() - started)
        add_completed_stage(root, stage_id)
        return payload

    if not _stage_done(completed, "E00"):
        manifests, context = resume_or_run(
            "E00",
            lambda: build_e00(
                project_root=project,
                run_root=root,
                config=config,
                config_path=config_path,
            ),
        )
        if manifests is None:
            manifests = load_current_manifests(root, config)
            context = _rebuild_context(project, config)
    else:
        manifests = load_current_manifests(root, config)
        context = _rebuild_context(project, config)

    if not _stage_done(completed, "E01"):
        selected = resume_or_run(
            "E01",
            lambda: build_e01(run_root=root, manifests=manifests, config=config),
        )
        if selected is None:
            selected = load_e01_selected(root, config)
    else:
        selected = load_e01_selected(root, config)

    if not _stage_done(completed, "E02"):
        resume_or_run(
            "E02",
            lambda: build_e02(
                run_root=root,
                manifests=manifests,
                selected_by_dataset=selected,
                config=config,
            ),
        )

    needs_features = bool({"E03", "E04", "E04-1", "E04-2", "E04-3", "E04-4", "E04-5", "E05", "E07", "E08"}.intersection(experiments))
    if needs_features:
        for model_id in models:
            stage_id = f"features:{model_id}:nominal"
            if _stage_done(completed, stage_id):
                continue
            step += 1
            started = time.time()
            log_stage_start(logger, stage_id, step, total, f"{model_id} 成对特征提取")
            extract_feature_table(
                project_root=project,
                run_root=root,
                config=config,
                manifests=manifests,
                selected_by_dataset=selected,
                model_id=model_id,
                asset=context["model_assets"][model_id],
                runtime=context["runtime"],
            )
            log_stage_complete(
                logger,
                stage_id,
                time.time() - started,
                f"{model_id} 特征提取完成",
            )
            add_completed_stage(root, stage_id)
            import gc

            gc.collect()

    results: dict[str, Any] = {"models": {model_id: {} for model_id in models}}

    core_stage_ids = [
        stage
        for stage in ("E03", "E04", "E04-1", "E04-2", "E04-3", "E04-4", "E04-5", "E05", "E07")
        if stage in experiments
    ]
    # Model-level resume: each (stage, model) cell is checkpointed independently
    # so an interrupted run resumes from the nearest completed model instead of
    # restarting the whole stage.  Legacy whole-stage checkpoints (e.g. "E03")
    # are honoured as all-models-complete.
    def cell_done(cell_id: str, stage_id: str, model_id: str) -> bool:
        if _stage_done(completed, f"{cell_id}:{model_id}"):
            return True
        if _stage_done(completed, stage_id):
            return _legacy_stage_covers_model(stage_id, model_id)
        return False

    def load_cell_result(cell_id: str, model_id: str) -> dict[str, Any] | None:
        cached = load_stage_result(root, f"{cell_id}_{model_id}")
        if cached is not None:
            return cached
        cached = load_stage_result(root, cell_id)
        if cached is not None:
            return cached.get("models", {}).get(model_id)
        return None

    first_pending: bool = True
    for model_id in models:
        table: Any | None = None

        def get_table(_model_id: str = model_id) -> Any:
            nonlocal table
            if table is None and f"features:{_model_id}:nominal" in completed:
                table = load_feature_table(root, _model_id)
            return table

        for stage_id in core_stage_ids:
            if stage_id == "E07":
                e07_all = os.environ.get("CFRA_E07_ALL_MODELS", "1") != "0"
                allowed = set(models) if e07_all else set(config.get("scope", {}).get("primary_models", models))
                if model_id not in allowed:
                    continue
            if stage_id == "E04-5" and not run_e04_5:
                continue
            cell_id = stage_id
            if cell_done(cell_id, stage_id, model_id):
                if export_only_handoff:
                    # 51 reads each stage directly; do not rebuild an in-memory
                    # multi-GB aggregate merely to skip the legacy writer.
                    continue
                cached = load_cell_result(cell_id, model_id)
                if cached is None and _stage_done(completed, stage_id):
                    continue
                results["models"].setdefault(model_id, {})[stage_id] = cached if cached is not None else {}
                continue
            if first_pending:
                step += 1
                log_stage_start(logger, stage_id, step, total, f"{model_id} (模型级续跑)")
                first_pending = False
            table = get_table()
            if table is None:
                logger.warning(f"  [{stage_id}] {model_id} 无特征表，跳过")
                continue
            model_started = time.time()
            logger.info(f"  [{stage_id}] {model_id} 开始")
            if stage_id == "E04-5":
                # Stream one variant table at a time: the generator releases each
                # table after its rows are computed, so only the nominal table and
                # the current variant stay resident instead of the whole grid.
                # Patch-budget variants also reuse the nominal rows for every
                # perturbation they do not change (bit-identical, no recompute).
                def iter_variant_tables(
                    _nominal_table: Any = table,
                    _model_id: str = model_id,
                ) -> Iterator[tuple[str, Any]]:
                    yield "nominal", _nominal_table
                    for variant_name, strength_variant, overrides in _sensitivity_variants(config):
                        if variant_name == "nominal":
                            continue
                        variant_stage = f"features:{_model_id}:{variant_name}"
                        if _stage_done(completed, variant_stage):
                            variant_table = load_feature_table(
                                root, _model_id, strength_variant=variant_name
                            )
                        else:
                            variant_table = extract_feature_table(
                                project_root=project,
                                run_root=root,
                                config=config,
                                manifests=manifests,
                                selected_by_dataset=selected,
                                model_id=_model_id,
                                asset=context["model_assets"][_model_id],
                                runtime=context["runtime"],
                                strength_variant=variant_name,
                                perturbation_overrides=overrides,
                            )
                            add_completed_stage(root, variant_stage)
                        yield variant_name, variant_table

                payload = _core_stage(
                    "E04-5",
                    iter_variant_tables(),
                    datasets=datasets,
                    perturbations=perturbations,
                    config=config,
                    seed=seed,
                    run_root=root,
                )
            else:
                payload = _core_stage(
                    stage_id,
                    table,
                    datasets=datasets,
                    perturbations=perturbations,
                    config=config,
                    seed=seed,
                    run_root=root,
                )
            if not export_only_handoff:
                results["models"].setdefault(model_id, {})[stage_id] = payload
            write_stage_result(root, f"{cell_id}_{model_id}", {"stage": cell_id, "model_id": model_id, "results": payload})
            add_completed_stage(root, f"{cell_id}:{model_id}")
            logger.info(
                f"  [{stage_id}] {model_id} 完成 (耗时 {time.time() - model_started:.1f}s): "
                + (f"{len(payload)} 条记录" if isinstance(payload, (list, dict)) else "完成")
            )
        if table is not None:
            del table
            table = None
            import gc

            gc.collect()

    if "E04-5" in experiments and not run_e04_5:
        step += 1
        logger.info(f"[Step{step:02d}/{total:02d}] [E04-5] 跳过：E04-5 网格开关为关闭")
        skipped = {
            "stage": "E04-5",
            "status": "skipped",
            "reason": "E04-5 perturbation-strength grid is disabled",
            "configured_enabled": False,
            "evidence_eligible": False,
        }
        for model_id in models:
            results["models"].setdefault(model_id, {})["E04-5"] = skipped
        write_stage_result(root, "E04-5", {"stage": "E04-5", "models": {m: skipped for m in models}})
        if not _stage_done(completed, "E04-5"):
            add_completed_stage(root, "E04-5")

    if "E06" in experiments:
        selected_layers = list(config.get("robustness", {}).get("selected_layers", []))
        for model_id in models:
            stage_id = f"E06:{model_id}"
            if _stage_done(completed, stage_id):
                continue
            e06_model_started = time.time()
            layer_results = []
            if selected_layers:
                for dataset_id in datasets:
                    layer_results.append(
                        _run_offline_e06(
                            run_root=root,
                            project_root=project,
                            config=config,
                            manifests=manifests,
                            selected_by_dataset=selected,
                            model_id=model_id,
                            asset=context["model_assets"][model_id],
                            runtime=context["runtime"],
                            dataset_id=dataset_id,
                        )
                    )
                write_stage_result(
                    root,
                    "E06",
                    _json_safe({"stage": "E06", "model_id": model_id, "results": layer_results}),
                )
                logger.info(f"[E06] {model_id} 完成 (耗时 {time.time() - e06_model_started:.1f}s)")
                add_completed_stage(root, stage_id)
            else:
                logger.warning(f"[E06] {model_id} 无 selected_layers，跳过层提取")

    if "E08" in experiments:
        retrieval_config = config.get("retrieval", {})
        methods = tuple(retrieval_config.get("methods", ("raw_prototype", "cfp_direction", "cav_like", "random")))
        k = int(retrieval_config.get("k", 10))
        for model_id in models:
            stage_id = f"E08:{model_id}"
            if _stage_done(completed, stage_id):
                continue
            table = load_feature_table(root, model_id)
            e08_model_started = time.time()
            cell_results: list[dict[str, Any]] = []
            for dataset_id in datasets:
                for perturbation in perturbations:
                    cell_path = root / "metrics" / f"E08_{model_id}_{dataset_id}_{perturbation}.json"
                    if cell_path.is_file():
                        cell = _read_json_file(cell_path)
                        cell_results.append(cell)
                        logger.info(
                            f"[E08] {model_id} {dataset_id}/{perturbation} 已完成，跳过 "
                            f"(共 {len(cell_results)} 格)"
                        )
                        continue
                    candidate_mask, query_mask = _retrieval_masks(
                        table,
                        dataset_id=dataset_id,
                        perturbation=perturbation,
                    )
                    retrieval_result = run_e08(
                        table,
                        dataset_id=dataset_id,
                        perturbation=perturbation,
                        candidate_mask=candidate_mask,
                        query_mask=query_mask,
                        methods=methods,
                        k=k,
                        epsilon=float(config["protocol"]["near_zero_norm_epsilon"]),
                        bootstrap_replicates=int(statistics_config.get("bootstrap_replicates", 0)),
                        bootstrap_seed=seed,
                        bootstrap_unit="source_image",
                    )
                    retrieval_result.update(
                        {
                            "status": "completed",
                            "execution_status": "completed",
                            "configured_enabled": True,
                            "interface_called": True,
                            "input_source": "current_run_feature_table",
                        }
                    )
                    _write_json_file(cell_path, _json_safe(retrieval_result))
                    cell_results.append(retrieval_result)
                    logger.info(
                        f"[E08] {model_id} {dataset_id}/{perturbation} 已保存 "
                        f"(共 {len(cell_results)}/{len(datasets) * len(perturbations)} 格)"
                    )
            model_e08 = {
                "stage": "E08",
                "status": "completed",
                "execution_status": "completed",
                "configured_enabled": True,
                "interface_called": True,
                "results": cell_results,
            }
            _write_json_file(root / "metrics" / f"E08_{model_id}.json", model_e08)
            write_manifest_json(
                root,
                f"manifest/E08_{model_id}.json",
                {
                    "stage": "E08",
                    "model_id": model_id,
                    "status": "completed",
                    "input_source": "current_run_feature_table",
                },
            )
            logger.info(
                f"[E08] {model_id} 完成 (耗时 {time.time() - e08_model_started:.1f}s): "
                f"{len(cell_results)} 个数据集×扰动组合"
            )
            add_completed_stage(root, stage_id)
            del table
            table = None
            import gc

            gc.collect()

        text_models = [
            text_model
            for text_model in config.get("text_baseline_protocol", {}).get("models", [])
            if text_model in context["model_assets"]
        ]
        for text_model in text_models:
            stage_id = f"E08text:{text_model}"
            if _stage_done(completed, stage_id):
                continue
            table = None
            if f"features:{text_model}:nominal" in completed:
                from cfra.pipeline.features import FeatureStageError

                try:
                    table = load_feature_table(root, text_model)
                except (OSError, ValueError, FeatureStageError, json.JSONDecodeError):
                    table = None
            if table is None:
                if text_model in models:
                    continue
                table = extract_feature_table(
                    project_root=project,
                    run_root=root,
                    config=config,
                    manifests=manifests,
                    selected_by_dataset=selected,
                    model_id=text_model,
                    asset=context["model_assets"][text_model],
                    runtime=context["runtime"],
                )
            text_started = time.time()
            text_results = _compute_text_baseline(
                table=table,
                text_model=text_model,
                context=context,
                config=config,
                project_root=project,
                seed=seed,
                statistics_config=statistics_config,
                perturbations=perturbations,
                datasets=datasets,
            )
            _write_json_file(root / "metrics" / f"E08_text_{text_model}.json", {"results": text_results})
            write_manifest_json(
                root,
                f"manifest/E08_text_{text_model}.json",
                {
                    "stage": "E08",
                    "model_id": text_model,
                    "status": "completed",
                    "baseline_type": "official_shared_text",
                },
            )
            logger.info(
                f"[E08] text_baseline={text_model} 完成 (耗时 {time.time() - text_started:.1f}s): "
                f"{len(text_results)} 条"
            )
            add_completed_stage(root, stage_id)

    if export_only_handoff:
        from cfra.export_handoff import write_export_handoff

        handoff = write_export_handoff(root, config)
        if handoff["missing"]:
            raise OfflineRunnerError(f"51 export handoff has {len(handoff['missing'])} missing artifacts; see manifest/export_handoff.json")
        _reload_offline_run(root, config)
        (root / "STATUS").write_text("completed\n", encoding="ascii")
        logger.info("[handoff] 独立阶段产物已就绪；综合归档和旧报告已跳过")
        logger.info("[handoff] 下一步：python scripts/51_export_paper_data.py")
        logger.info(f"离线运行完成：{root}")
        return

    step += 1
    started = time.time()
    log_stage_start(logger, "results", step, total, "写出机器可读结果与统计")
    write_core_results(root, results)
    write_core_statistics(root, results, statistics_config=statistics_config)
    log_stage_complete(logger, "results", time.time() - started)
    add_completed_stage(root, "results")

    from cfra.reporting import ReportingError, generate_report

    # Report generation materialises every metrics record in memory.  A full
    # offline run can persist several GB of metrics, which OOMs the report step
    # (and the process) long after the scientific results are already written.
    # The report is best-effort lineage, so oversized runs skip it by default.
    metrics_root = root / "metrics"
    metrics_bytes = (
        sum(path.stat().st_size for path in metrics_root.rglob("*") if path.is_file())
        if metrics_root.exists()
        else 0
    )
    report_limit = int(os.environ.get("CFRA_REPORT_MAX_BYTES", str(2_000_000_000)))
    if os.environ.get("CFRA_SKIP_REPORT", "0") == "1":
        logger.info("[report] 已通过 CFRA_SKIP_REPORT=1 跳过报告生成")
    elif metrics_bytes > report_limit:
        logger.warning(
            f"[report] 跳过报告生成：metrics 目录 {metrics_bytes / 1e9:.2f}GB 超过 "
            f"CFRA_REPORT_MAX_BYTES={report_limit / 1e9:.2f}GB；报告为 best-effort 血缘摘要，"
            "不影响已写入的 metrics/statistics 结果。设置 CFRA_SKIP_REPORT=0 与更大的 "
            "CFRA_REPORT_MAX_BYTES 可强制执行。"
        )
    else:
        try:
            generate_report(root)
        except (ReportingError, OSError, ValueError, KeyError, MemoryError) as exc:
            logger.warning(f"[report] 报告生成失败（不影响结果）：{type(exc).__name__}: {exc}")

    _reload_offline_run(root, config)
    logger.info("=" * 72)
    logger.info(f"离线运行完成：{root}")
    logger.info("=" * 72)


def _rebuild_context(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    from cfra.configuration import ConfigurationError, load_local_config

    local = load_local_config(project_root / "config" / "local.toml")
    model_ids = [
        *config["scope"].get("models", []),
        *config["scope"].get("representation_variants", []),
    ]
    model_assets = resolve_model_assets(
        local, model_ids, registry_root=project_root / "registry"
    )
    runtime = parse_model_runtime_config(config, model_ids)
    try:
        class_registry = load_class_registry(project_root / "registry")
    except (OSError, ValueError, KeyError, ConfigurationError):
        class_registry = {}
    return {
        "local": local,
        "class_registry": class_registry,
        "model_assets": model_assets,
        "runtime": runtime,
    }


__all__ = [
    "OfflineRunnerError",
    "build_offline_config",
    "infer_completed_stages",
    "open_offline_run",
    "parameters_signature",
    "run_full_offline",
]
