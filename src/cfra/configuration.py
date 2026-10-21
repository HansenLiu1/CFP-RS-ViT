"""TOML loading, inheritance, hashing, and protocol validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any


RESUME_POLICY_HASH_CHAIN = "stage_checkpoint_hash_chain"
E04_5_CONTROL_SECTION = "experiment_controls.E04-5"


def resolve_e04_5_policy(config: dict[str, Any]) -> tuple[bool, str]:
    """Resolve whether the E04-5 grid must execute for the current run.

    The wrapper entry scripts are the single authority: ``CFRA_RUN_E04_5=1``
    forces execution, ``CFRA_RUN_E04_5=0`` forces a skip, and an unset variable
    falls back to ``[experiment_controls.E04-5].enabled`` in the configuration.
    Returns ``(run_grid, skip_reason)``.
    """

    controls = config.get("experiment_controls", {})
    if not isinstance(controls, dict):
        controls = {}
    e04_5 = controls.get("E04-5", {})
    if not isinstance(e04_5, dict):
        e04_5 = {}
    config_enabled = bool(e04_5.get("enabled", False))
    default_reason = str(
        e04_5.get(
            "skip_reason",
            "E04-5 perturbation-strength grid is disabled in this run; it does not affect primary conclusions",
        )
    )
    env_value = os.environ.get("CFRA_RUN_E04_5")
    if env_value is not None:
        if env_value.strip() == "1":
            return True, ""
        return False, "CFRA_RUN_E04_5=0 requested an explicit skip"
    if config_enabled:
        return True, ""
    return False, default_reason


class ConfigurationError(ValueError):
    """Raised when a configuration violates a project invariant."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_experiment_config(path: Path | str, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a TOML config and recursively merge a relative ``extends`` file."""

    config_path = Path(path).expanduser().resolve(strict=True)
    if config_path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, config_path))
        raise ConfigurationError(f"Configuration inheritance cycle: {chain}")

    current = _read_toml(config_path)
    parent_name = current.pop("extends", None)
    if parent_name is None:
        return current
    if not isinstance(parent_name, str) or not parent_name.strip():
        raise ConfigurationError(f"Invalid extends value in {config_path}")

    parent_path = (config_path.parent / parent_name).resolve(strict=True)
    parent = load_experiment_config(parent_path, (*_stack, config_path))
    return _deep_merge(parent, current)


def load_local_config(path: Path | str) -> dict[str, Any]:
    """Load machine-local paths without inheritance."""

    config_path = Path(path).expanduser().resolve(strict=True)
    return _read_toml(config_path)


def config_sha256(config: dict[str, Any]) -> str:
    """Hash the resolved config, independent of TOML formatting."""

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_for_seed(
    config: dict[str, Any],
    seed: int,
    *,
    required_experiments: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve the immutable per-seed formal configuration projection.

    The seed projection is shared by group planning and formal auditing so the
    recorded seed hash cannot depend on which entry point produced it.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("Formal split seed must be an integer")
    resolved = deepcopy(config)
    split = resolved.setdefault("split", {})
    if not isinstance(split, dict):
        raise ConfigurationError("split must be a TOML table")
    split["seed"] = seed
    split["split_seeds"] = [seed]
    robustness = resolved.get("robustness")
    if isinstance(robustness, dict):
        robustness["split_seeds"] = [seed]
    statistics = resolved.get("statistics")
    if isinstance(statistics, dict):
        statistics["bootstrap_seed"] = seed
    if required_experiments is not None:
        scope = resolved.setdefault("scope", {})
        if not isinstance(scope, dict):
            raise ConfigurationError("scope must be a TOML table")
        scope["experiment_ids"] = list(required_experiments)
    return resolved


def _external_path_tables(local_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        paths = local_config["paths"]
        dataset_paths = paths["datasets"]
        model_paths = paths["models"]
    except (KeyError, TypeError) as exc:
        raise ConfigurationError("local.toml must define [paths.datasets] and [paths.models]") from exc

    if not isinstance(dataset_paths, dict) or not isinstance(model_paths, dict):
        raise ConfigurationError("Dataset and model path sections must be TOML tables")
    return dataset_paths, model_paths


def _configured_paths_for_ids(
    values: dict[str, Any],
    identifiers: list[str] | tuple[str, ...],
    field: str,
) -> dict[str, Path]:
    requested = list(dict.fromkeys(identifiers))
    missing = sorted(set(requested) - values.keys())
    if missing:
        raise ConfigurationError(f"{field} lacks required mappings: {', '.join(missing)}")

    result: dict[str, Path] = {}
    for identifier in requested:
        value = values[identifier]
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"{field}.{identifier} must be a non-empty string")
        if "path/to" in value.replace("\\", "/").lower():
            raise ConfigurationError(f"{field}.{identifier} still contains a placeholder path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ConfigurationError(f"{field}.{identifier} must be an absolute path")
        result[identifier] = path
    return result


def configured_dataset_paths(
    local_config: dict[str, Any], dataset_ids: list[str] | tuple[str, ...]
) -> dict[str, Path]:
    """Return explicit machine-local paths for the requested datasets."""

    dataset_paths, _ = _external_path_tables(local_config)
    return _configured_paths_for_ids(dataset_paths, dataset_ids, "paths.datasets")


def configured_model_paths(
    local_config: dict[str, Any], model_ids: list[str] | tuple[str, ...]
) -> dict[str, Path]:
    """Return explicit machine-local checkpoint paths for the requested model assets."""

    _, model_paths = _external_path_tables(local_config)
    return _configured_paths_for_ids(model_paths, model_ids, "paths.models")


def configured_external_paths(local_config: dict[str, Any]) -> list[Path]:
    """Extract every explicitly configured dataset and model path."""

    dataset_paths, model_paths = _external_path_tables(local_config)
    datasets = _configured_paths_for_ids(dataset_paths, tuple(dataset_paths), "paths.datasets")
    models = _configured_paths_for_ids(model_paths, tuple(model_paths), "paths.models")
    return [*datasets.values(), *models.values()]


def validate_execution_policy(config: dict[str, Any]) -> None:
    """Enforce the no-resume, no-cross-run-cache formal execution policy."""

    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise ConfigurationError("Missing [execution] section")

    required_false = (
        "allow_cross_run_cache",
        "overwrite_existing_run",
        "persist_sample_features_across_runs",
    )
    violations = [key for key in required_false if execution.get(key) is not False]
    if violations:
        raise ConfigurationError("Clean-room execution fields must be false: " + ", ".join(violations))

    # Stage-level resume is permitted only under the registered hash-chain policy.
    # A resumed stage sequence stays auditable because every completed stage is
    # bound to the resolved config hash and the code version in checkpoint.json;
    # cross-run derived caches remain forbidden by the fields above.
    allow_resume = execution.get("allow_resume")
    if allow_resume not in (True, False):
        raise ConfigurationError("execution.allow_resume must be a boolean")
    if allow_resume is True and execution.get("resume_policy") != RESUME_POLICY_HASH_CHAIN:
        raise ConfigurationError(
            "execution.allow_resume=true requires "
            f"execution.resume_policy={RESUME_POLICY_HASH_CHAIN!r}"
        )

    split = config.get("split", {})
    if split.get("unit") != "source_image":
        raise ConfigurationError("Formal split unit must be source_image")

    intervention = config.get("intervention", {})
    removal_alphas = intervention.get("removal_alphas", [])
    if not removal_alphas or min(removal_alphas) < 0 or max(removal_alphas) > 1:
        raise ConfigurationError("Removal alphas must be non-empty and contained in [0, 1]")
    if intervention.get("primary_alpha") != 1.0:
        raise ConfigurationError("The primary removal alpha must be 1.0")


def validate_formal_freeze(config: dict[str, Any]) -> None:
    """Refuse an unfrozen formal configuration."""

    run = config.get("run", {})
    if run.get("evidence_tier") != "formal":
        return
    if run.get("frozen") is not True:
        raise ConfigurationError("Formal configuration is not frozen; external method review is required first")
    if config.get("release", {}).get("status") != "frozen":
        raise ConfigurationError("Formal evidence release is not frozen")
    model_runtime = config.get("model_runtime", {})
    if not isinstance(model_runtime, dict) or model_runtime.get("formal_approved") is not True:
        raise ConfigurationError("Formal model runtime has not been approved")
    if model_runtime.get("precision") not in {"fp32", "fp16", "bf16"}:
        raise ConfigurationError("Formal model runtime must use an explicit precision")


def _read_registry(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _index_unique(rows: list[dict[str, str]], key: str, source: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        value = row.get(key, "")
        if not value:
            raise ConfigurationError(f"Missing {key} in {source}")
        if value in indexed:
            duplicates.append(value)
        indexed[value] = row
    if duplicates:
        raise ConfigurationError(f"Duplicate {key} values in {source}: {', '.join(sorted(set(duplicates)))}")
    return indexed


def _as_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ConfigurationError(f"{field} must be a list of non-empty strings")
    return value


def validate_registry_contract(config: dict[str, Any], registry_root: Path | str) -> None:
    """Validate model-family identity and experiment/model applicability."""

    root = Path(registry_root).expanduser().resolve(strict=True)
    models = _index_unique(_read_registry(root / "models.csv"), "model_id", "models.csv")
    experiments = _index_unique(_read_registry(root / "experiments.csv"), "experiment_id", "experiments.csv")

    for model_id, model in models.items():
        parent_id = model.get("analysis_variant_of", "")
        if not parent_id:
            continue
        parent = models.get(parent_id)
        if parent is None:
            raise ConfigurationError(f"Model variant {model_id} has unknown parent {parent_id}")
        if model.get("independence_group") != parent.get("independence_group"):
            raise ConfigurationError(f"Model variant {model_id} must share independence_group with {parent_id}")

    scope = config.get("scope", {})
    if not isinstance(scope, dict):
        raise ConfigurationError("Missing [scope] section")

    configured_models: set[str] = set()
    for field in ("models", "representation_variants", "primary_models", "generalization_models"):
        values = scope.get(field, [])
        if values:
            configured_models.update(_as_string_list(values, f"scope.{field}"))
    unknown_models = sorted(configured_models - models.keys())
    if unknown_models:
        raise ConfigurationError("Config references unregistered models: " + ", ".join(unknown_models))

    for model_id in scope.get("primary_models", []):
        if models[model_id].get("role") != "primary_vlm":
            raise ConfigurationError(f"Primary model {model_id} is not registered as primary_vlm")
    for model_id in scope.get("generalization_models", []):
        model = models[model_id]
        if model.get("pretraining_modality") != "image_only" or model.get("analysis_variant_of"):
            raise ConfigurationError(f"Generalization model {model_id} must be an independent image_only model")

    gate = config.get("model_family_gate", {})
    required_models = _as_string_list(
        gate.get("required_image_only_models", []), "model_family_gate.required_image_only_models"
    )
    required_families = set(
        _as_string_list(
            gate.get("required_image_only_objective_families", []),
            "model_family_gate.required_image_only_objective_families",
        )
    )
    observed_families: set[str] = set()
    for model_id in required_models:
        model = models.get(model_id)
        if model is None:
            raise ConfigurationError(f"Model-family gate references unregistered model: {model_id}")
        if model.get("pretraining_modality") != "image_only" or model.get("analysis_variant_of"):
            raise ConfigurationError(f"Model-family gate model {model_id} is not an independent image_only model")
        observed_families.add(model.get("objective_family", ""))
    if observed_families != required_families:
        raise ConfigurationError(
            "Model-family gate objective mismatch: "
            f"observed={sorted(observed_families)}, required={sorted(required_families)}"
        )

    experiment_ids = _as_string_list(scope.get("experiment_ids", []), "scope.experiment_ids")
    unknown_experiments = sorted(set(experiment_ids) - experiments.keys())
    if unknown_experiments:
        raise ConfigurationError("Config references unregistered experiments: " + ", ".join(unknown_experiments))

    if config.get("run", {}).get("evidence_tier") == "formal":
        _validate_release_contract(config, experiments, experiment_ids)

    experiment_models = config.get("experiment_models", {})
    if not isinstance(experiment_models, dict):
        raise ConfigurationError("[experiment_models] must be a TOML table")
    for experiment_id, configured in experiment_models.items():
        if experiment_id not in experiments:
            raise ConfigurationError(f"experiment_models contains unknown experiment: {experiment_id}")
        configured_set = set(_as_string_list(configured, f"experiment_models.{experiment_id}"))
        outside_scope = configured_set - configured_models
        if outside_scope:
            raise ConfigurationError(
                f"Experiment {experiment_id} references models outside scope: {', '.join(sorted(outside_scope))}"
            )
        unknown = configured_set - models.keys()
        if unknown:
            raise ConfigurationError(
                f"Experiment {experiment_id} references unregistered models: {', '.join(sorted(unknown))}"
            )
        registered_text = experiments[experiment_id].get("models", "")
        if registered_text not in {"all", "none"}:
            registered_set = {item for item in registered_text.split(";") if item}
            if config.get("run", {}).get("evidence_tier") == "formal" and configured_set != registered_set:
                raise ConfigurationError(
                    f"Experiment {experiment_id} model mismatch: "
                    f"config={sorted(configured_set)}, registry={sorted(registered_set)}"
                )

    if config.get("run", {}).get("evidence_tier") == "formal":
        required_mappings = {
            experiment_id
            for experiment_id in experiment_ids
            if experiments[experiment_id].get("models", "") not in {"all", "none"}
        }
        missing_mappings = sorted(required_mappings - experiment_models.keys())
        if missing_mappings:
            raise ConfigurationError(
                "Formal config lacks experiment model mappings: " + ", ".join(missing_mappings)
            )

    text_baselines = config.get("text_baseline_models", {})
    if not isinstance(text_baselines, dict):
        raise ConfigurationError("[text_baseline_models] must be a TOML table")
    for experiment_id, configured in text_baselines.items():
        if experiment_id not in experiments:
            raise ConfigurationError(f"text_baseline_models contains unknown experiment: {experiment_id}")
        for model_id in _as_string_list(configured, f"text_baseline_models.{experiment_id}"):
            if model_id not in configured_models:
                raise ConfigurationError(f"Text baseline model {model_id} is outside configured scope")
            model = models.get(model_id)
            if model is None:
                raise ConfigurationError(f"Text baseline references unregistered model: {model_id}")
            if model.get("text_shared_space") != "yes":
                raise ConfigurationError(f"Text baseline model {model_id} is not in an official shared text space")

    if config.get("run", {}).get("evidence_tier") == "formal" and config.get("run", {}).get("frozen") is True:
        protocol_version = config.get("project", {}).get("protocol_version")
        for model_id in configured_models:
            model = models[model_id]
            if model.get("status") != "checkpoint_verified":
                raise ConfigurationError(f"Formal model checkpoint is not verified: {model_id}")
            if (
                not model.get("checkpoint_revision")
                or not model.get("checkpoint_sha256")
                or not model.get("loader_config_sha256")
            ):
                raise ConfigurationError(f"Formal model checkpoint identity is incomplete: {model_id}")
        for experiment_id in experiment_ids:
            experiment = experiments[experiment_id]
            if experiment.get("protocol_state") != "frozen":
                raise ConfigurationError(f"Formal experiment protocol is not frozen: {experiment_id}")
            if experiment.get("protocol_version") != protocol_version:
                raise ConfigurationError(
                    f"Protocol version mismatch for {experiment_id}: "
                    f"{experiment.get('protocol_version')} != {protocol_version}"
                )
            if not experiment.get("frozen_at") or not experiment.get("approved_by"):
                raise ConfigurationError(f"Frozen protocol lacks approval metadata: {experiment_id}")


def _validate_release_contract(
    config: dict[str, Any],
    experiments: dict[str, dict[str, str]],
    experiment_ids: list[str],
) -> None:
    release = config.get("release")
    run_groups = config.get("run_groups")
    if not isinstance(release, dict) or not isinstance(run_groups, dict):
        raise ConfigurationError("Formal config must define [release] and [run_groups.*]")

    group_order = _as_string_list(release.get("run_groups", []), "release.run_groups")
    if len(group_order) != len(set(group_order)):
        raise ConfigurationError("release.run_groups contains duplicates")
    missing_groups = sorted(set(group_order) - run_groups.keys())
    extra_groups = sorted(run_groups.keys() - set(group_order))
    if missing_groups or extra_groups:
        raise ConfigurationError(
            f"Release/run-group mismatch: missing={missing_groups}, extra={extra_groups}"
        )
    if release.get("aggregation_policy") != "frozen_manifest_only":
        raise ConfigurationError("Formal release aggregation_policy must be frozen_manifest_only")

    selected_experiments = set(experiment_ids)
    qualification = {"E00", "E01", "E02"}
    group_experiments: dict[str, set[str]] = {}
    dependencies: dict[str, set[str]] = {}
    experiment_to_groups: dict[str, set[str]] = {experiment_id: set() for experiment_id in experiment_ids}

    for group_name in group_order:
        group = run_groups[group_name]
        if not isinstance(group, dict):
            raise ConfigurationError(f"run_groups.{group_name} must be a TOML table")
        configured_experiments = set(
            _as_string_list(group.get("experiments", []), f"run_groups.{group_name}.experiments")
        )
        unknown_experiments = configured_experiments - experiments.keys()
        if unknown_experiments:
            raise ConfigurationError(
                f"Run group {group_name} references unknown experiments: {sorted(unknown_experiments)}"
            )
        if not qualification.issubset(configured_experiments):
            raise ConfigurationError(f"Run group {group_name} must independently execute E00-E02")
        if group.get("from_raw_inputs") is not True:
            raise ConfigurationError(f"Run group {group_name} must start from raw inputs")
        if group.get("allow_cross_group_derived_inputs") is not False:
            raise ConfigurationError(f"Run group {group_name} must reject cross-group derived inputs")

        configured_dependencies = set(
            _as_string_list(
                group.get("scientific_dependencies", []),
                f"run_groups.{group_name}.scientific_dependencies",
            )
        )
        if group_name in configured_dependencies:
            raise ConfigurationError(f"Run group {group_name} cannot depend on itself")
        unknown_dependencies = configured_dependencies - set(group_order)
        if unknown_dependencies:
            raise ConfigurationError(
                f"Run group {group_name} has unknown dependencies: {sorted(unknown_dependencies)}"
            )

        group_experiments[group_name] = configured_experiments
        dependencies[group_name] = configured_dependencies
        for experiment_id in configured_experiments & selected_experiments:
            experiment_to_groups[experiment_id].add(group_name)

    _assert_acyclic_groups(dependencies)

    uncovered = sorted(experiment_id for experiment_id, groups in experiment_to_groups.items() if not groups)
    if uncovered:
        raise ConfigurationError("Release does not cover experiments: " + ", ".join(uncovered))

    for experiment_id, configured_groups in experiment_to_groups.items():
        registered_groups = {
            item for item in experiments[experiment_id].get("run_groups", "").split(";") if item
        }
        if configured_groups != registered_groups:
            raise ConfigurationError(
                f"Experiment {experiment_id} run-group mismatch: "
                f"config={sorted(configured_groups)}, registry={sorted(registered_groups)}"
            )

    for group_name, configured_experiments in group_experiments.items():
        available = set(configured_experiments)
        for dependency in _transitive_dependencies(group_name, dependencies):
            available.update(group_experiments[dependency])
        for experiment_id in configured_experiments:
            prerequisites = {
                item
                for item in experiments[experiment_id].get("scientific_prerequisites", "").split(";")
                if item and item != "none"
            }
            missing = prerequisites - available
            if missing:
                raise ConfigurationError(
                    f"Run group {group_name} cannot satisfy prerequisites for {experiment_id}: {sorted(missing)}"
                )


def _assert_acyclic_groups(dependencies: dict[str, set[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(group_name: str) -> None:
        if group_name in visiting:
            raise ConfigurationError(f"Run-group dependency cycle includes {group_name}")
        if group_name in visited:
            return
        visiting.add(group_name)
        for dependency in dependencies[group_name]:
            visit(dependency)
        visiting.remove(group_name)
        visited.add(group_name)

    for group_name in dependencies:
        visit(group_name)


def _transitive_dependencies(group_name: str, dependencies: dict[str, set[str]]) -> set[str]:
    result: set[str] = set()
    pending = list(dependencies[group_name])
    while pending:
        dependency = pending.pop()
        if dependency in result:
            continue
        result.add(dependency)
        pending.extend(dependencies[dependency])
    return result
