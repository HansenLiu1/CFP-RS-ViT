"""Resolve configured sensitivity grids into concrete feature variants.

This module used to live in ``scripts/03_run_core_development.py`` and was
loaded dynamically by the offline runner.  Keeping it in the package makes the
public pipeline self-contained and removes the dependency on a development
entry point.
"""

from __future__ import annotations

from typing import Any


def enabled_perturbations(config: dict[str, Any]) -> list[str]:
    """Return the perturbations enabled by the resolved configuration."""

    return [
        name
        for name in config["scope"]["perturbations"]
        if bool(config.get("perturbations", {}).get(name, {}).get("enabled", True))
    ]


def _short_number(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def resolve_sensitivity_variants(
    config: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any] | None]]:
    """Resolve configured strength and patch budget/side grids.

    The returned tuples are ``(variant_name, strength_variant, overrides)``.
    ``overrides`` is ``None`` for a plain strength variant and a nested
    ``patch_mask`` mapping for a patch-grid variant.
    """

    protocol = config.get("protocol", {})
    sensitivity = config.get("sensitivity", {})
    protocol_grid = protocol.get("e04_5", {}) if isinstance(protocol.get("e04_5", {}), dict) else {}
    if not isinstance(sensitivity, dict):
        sensitivity = {}
    variants = list(
        sensitivity.get(
            "strength_variants",
            protocol_grid.get(
                "strength_variants",
                protocol.get("perturbation_strength_variants", ["low", "nominal", "high"]),
            ),
        )
    )
    if "nominal" not in variants:
        variants.insert(0, "nominal")
    resolved: list[tuple[str, str, dict[str, Any] | None]] = [
        (str(variant), str(variant), None) for variant in variants
    ]

    if "patch_mask" in enabled_perturbations(config):
        patch_config = config.get("perturbations", {}).get("patch_mask", {})
        patch_strengths = config.get("perturbation_strengths", {}).get("patch_mask", {})
        patch_budgets = sensitivity.get(
            "patch_budget_grid",
            sensitivity.get(
                "patch_budgets",
                protocol_grid.get(
                    "patch_budget_grid",
                    protocol.get("patch_budget_grid", patch_config.get("budget_grid", [])),
                ),
            ),
        )
        patch_sides = sensitivity.get(
            "patch_side_fraction_grid",
            sensitivity.get(
                "patch_side_grid",
                protocol_grid.get(
                    "patch_side_grid",
                    protocol.get(
                        "patch_side_fraction_grid",
                        patch_config.get("side_fraction_grid", []),
                    ),
                ),
            ),
        )
        if patch_budgets and not patch_sides and len(patch_strengths) > 1:
            patch_budgets = [value for key, value in patch_strengths.items() if key != "nominal"]
        if patch_budgets and patch_sides:
            for budget in patch_budgets:
                for side in patch_sides:
                    name = f"patch_budget_{_short_number(budget)}_side_{_short_number(side)}"
                    resolved.append(
                        (
                            name,
                            "nominal",
                            {
                                "patch_mask": {
                                    "patch_fraction": float(budget),
                                    "patch_side_fraction": float(side),
                                }
                            },
                        )
                    )

    unique: dict[str, tuple[str, str, dict[str, Any] | None]] = {}
    for item in resolved:
        unique[item[0]] = item
    return list(unique.values())
