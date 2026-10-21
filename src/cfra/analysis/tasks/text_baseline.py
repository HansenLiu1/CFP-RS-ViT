"""Shared-space text prototype baseline (S12).

This is the public CPU/GPU analysis entry point for the E08 text baseline.  It
uses the persisted ``raw`` clean-ROI features, registry class names and the
official text tower of RemoteCLIP, SigLIP2 or the CLIP shared space.  The CLIP
shared space can be reconstructed from the stored 768-d CLIP features with the
checkpoint's bias-free ``visual_projection`` when the run did not persist a
separate ``clip_shared`` feature file.

The task needs ``.[models]``, a valid ``config/local.toml`` and the model
weights.  It is intentionally not exercised by the default CI job, which has no
torch or text assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cfra.analysis.common import DATASETS, PERTURBATIONS, RunLayout
from cfra.configuration import load_local_config
from cfra.data.index import load_class_registry
from cfra.models import load_adapter, resolve_model_assets
from cfra.models.text import (
    DEFAULT_TEXT_TEMPLATES,
    SharedTextPrototypeBaseline,
    load_shared_text_encoder,
)
from cfra.offline import _retrieval_masks, _text_class_names
from cfra.pipeline.retrieval import run_e08

TEXT_MODELS = ("remoteclip", "siglip2", "clip_shared")
SPACES = ("raw", "centered")
TEMPLATE_LABELS = ("t1", "t2", "t3", "t4")
METRICS = ("macro_ap", "ndcg_at_k", "recall_at_k", "precision_at_k")
SHARED_SPACE = {
    "remoteclip": "openclip:ViT-B-32:projected-512",
    "siglip2": "transformers:SigLIP2-base-patch16-224:text-768",
    "clip_shared": "transformers:CLIP-ViT-B-16:projected-512",
}


class SingleTemplateBaseline(SharedTextPrototypeBaseline):
    """E08 scorer with one prompt template instead of the four-template mean."""

    def __init__(self, encoder, class_names, template: str, *, centered: bool, center_image_features: bool):
        super().__init__(
            encoder,
            class_names,
            templates=DEFAULT_TEXT_TEMPLATES,
            centered=centered,
            center_image_features=center_image_features,
        )
        self.template = template

    def fit(self, features, labels, metadata):  # type: ignore[override]
        fitted = dict(super().fit(features, labels, metadata))
        keys = list(fitted["class_keys"])
        names = [self.class_names[key] for key in keys]
        prompts = [self.template.format(class_name=name) for name in names]
        encoded = self.encoder.encode(prompts, normalize=True, output_device="cpu")
        values = encoded.values.detach().cpu().numpy() if hasattr(encoded.values, "detach") else encoded.values
        values = np.asarray(values, dtype=np.float64)
        fitted["prototypes"] = values
        fitted["text_center"] = values.mean(axis=0) if values.shape[0] else None
        return fitted


def _runtime(layout: RunLayout, seed: int, model: str):
    path = layout.run_dir(seed) / "manifest" / "run.json"
    if not path.is_file():
        return None
    from cfra.models.runtime import parse_model_runtime_config

    config = json.loads(path.read_text(encoding="utf-8"))["resolved_config"]
    return parse_model_runtime_config(config, [model])


def _raw_table(layout: RunLayout, seed: int, model: str) -> SimpleNamespace:
    path = layout.feature_path(seed, model)
    with np.load(path, allow_pickle=False) as payload:
        raw = np.asarray(payload["raw"])
        fields = {
            "dataset_id": np.asarray(payload["dataset_id"]),
            "object_id": np.asarray(payload["object_id"]),
            "source_image_id": np.asarray(payload["source_image_id"]),
            "class_id": np.asarray(payload["class_id"]),
            "class_name": np.asarray(payload["class_name"]),
            "coarse_name": np.asarray(payload["coarse_name"]),
            "role": np.asarray(payload["role"]),
            "perturbation": np.asarray(payload["perturbation"]),
        }
    metadata = tuple(
        SimpleNamespace(
            dataset_id=str(fields["dataset_id"][index]),
            object_id=str(fields["object_id"][index]),
            source_image_id=str(fields["source_image_id"][index]),
            class_id=int(fields["class_id"][index]),
            class_name=str(fields["class_name"][index]),
            coarse_name=str(fields["coarse_name"][index]),
            role=str(fields["role"][index]),
            perturbation=str(fields["perturbation"][index]),
        )
        for index in range(len(fields["dataset_id"]))
    )
    return SimpleNamespace(
        model_id=model,
        metadata=metadata,
        arrays={"raw": raw},
        representation={"shared_space_id": SHARED_SPACE.get(model, ""), "feature_key": "raw"},
    )


def _clip_projection(
    project_root: Path,
    layout: RunLayout,
    seed: int,
    explicit: Path | None,
) -> np.ndarray:
    if explicit is not None:
        import torch

        payload = torch.load(explicit, map_location="cpu", weights_only=True)
        return np.asarray(payload["weight"].to(torch.float32).numpy(), dtype=np.float64)
    # Derive the fixed projection from the registered CLIP checkpoint.  This
    # avoids requiring a separately cached 512x768 tensor.
    import torch

    local = load_local_config(project_root / "config" / "local.toml")
    assets = resolve_model_assets(local, ["clip"], registry_root=project_root / "registry")
    adapter = load_adapter(
        "clip",
        assets["clip"],
        runtime=_runtime(layout, seed, "clip"),
        registry_root=project_root / "registry",
        device="cpu",
    )
    try:
        weight = adapter.model.visual_projection.weight.detach().cpu().to(torch.float32).numpy()
    finally:
        adapter.close()
    return np.asarray(weight, dtype=np.float64)


def _project(features: np.ndarray, projection: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(features, dtype=np.float32) @ projection.T)


def _table_for(table: SimpleNamespace, shared_space: str, features: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=table.model_id,
        metadata=table.metadata,
        arrays={"raw": features},
        representation={"shared_space_id": shared_space, "feature_key": "raw"},
    )


def _score_cell(table: SimpleNamespace, baseline, dataset: str, perturbation: str, k: int) -> dict:
    candidate_mask, query_mask = _retrieval_masks(table, dataset_id=dataset, perturbation=perturbation)
    result = run_e08(
        table,
        dataset_id=dataset,
        perturbation=perturbation,
        candidate_mask=candidate_mask,
        query_mask=query_mask,
        methods=("text_baseline",),
        text_baseline=baseline,
        k=k,
        epsilon=1.0e-8,
        bootstrap_replicates=0,
    )
    method = result["methods"]["text_baseline"]
    return {
        "metrics": {name: float(method[name]) for name in METRICS},
        "query_count": int(result["query_count"]),
        "candidate_count": int(result["candidate_count"]),
        "fit_class_count": int(result["fit_class_count"]),
    }


def _baseline(encoder, class_names, template: str | None, space: str):
    centered = space == "centered"
    if template is None:
        return SharedTextPrototypeBaseline(
            encoder,
            class_names,
            templates=DEFAULT_TEXT_TEMPLATES,
            centered=centered,
            center_image_features=centered,
        )
    return SingleTemplateBaseline(
        encoder,
        class_names,
        template,
        centered=centered,
        center_image_features=centered,
    )


def run(
    layout: RunLayout,
    out: Path,
    *,
    project_root: Path,
    models: tuple[str, ...] = TEXT_MODELS,
    seeds: tuple[int, ...],
    datasets: tuple[str, ...] = DATASETS,
    perturbations: tuple[str, ...] = PERTURBATIONS,
    templates: bool = False,
    clip_projection: Path | None = None,
    k: int = 10,
    max_cells: int = 0,
) -> None:
    """Run the text baseline for each model/seed and write cell tables."""

    out.mkdir(parents=True, exist_ok=True)
    project_root = project_root.expanduser().resolve()
    registry = load_class_registry(project_root / "registry")
    local = load_local_config(project_root / "config" / "local.toml")
    all_rows: list[dict] = []
    for model in models:
        if model not in SHARED_SPACE:
            raise ValueError(f"text baseline supports {TEXT_MODELS}, got {model!r}")
        asset_model = model
        source_model = "clip" if model == "clip_shared" else model
        assets = resolve_model_assets(local, [asset_model], registry_root=project_root / "registry")
        projection = None
        if model == "clip_shared":
            projection = _clip_projection(project_root, layout, seeds[0], clip_projection)
        encoder = load_shared_text_encoder(
            asset_model,
            assets[asset_model],
            registry_root=project_root / "registry",
            runtime=_runtime(layout, seeds[0], asset_model),
            device="auto",
        )
        try:
            for seed in seeds:
                table = _raw_table(layout, seed, source_model)
                features = table.arrays["raw"]
                provenance = "persisted_raw"
                if projection is not None:
                    features = _project(features, projection)
                    provenance = "reconstructed_visual_projection"
                prepared = _table_for(table, SHARED_SPACE[model], features)
                variants = [("ensemble", None, None)]
                if templates:
                    variants += [(label, template, template) for label, template in zip(TEMPLATE_LABELS, DEFAULT_TEXT_TEMPLATES)]
                pairs = [(dataset, perturbation) for dataset in datasets for perturbation in perturbations]
                if max_cells:
                    pairs = pairs[:max_cells]
                for dataset, perturbation in pairs:
                    class_names = _text_class_names(registry, dataset)
                    for space in SPACES:
                        for label, template, text in variants:
                            baseline = _baseline(encoder, class_names, template, space)
                            result = _score_cell(prepared, baseline, dataset, perturbation, k)
                            for metric, value in result["metrics"].items():
                                all_rows.append(
                                    dict(
                                        seed=seed,
                                        model=model,
                                        dataset=dataset,
                                        perturbation=perturbation,
                                        template=label,
                                        template_text=text or "four_template_unit_mean",
                                        space=space,
                                        metric=metric,
                                        value=value,
                                        query_count=result["query_count"],
                                        candidate_count=result["candidate_count"],
                                        fit_class_count=result["fit_class_count"],
                                        image_features=provenance,
                                        feature_dim=int(features.shape[1]),
                                    )
                                )
                print(f"  text-baseline {model} {seed}", flush=True)
        finally:
            encoder.close()
    frame = pd.DataFrame(all_rows)
    frame.to_csv(out / "text_baseline_cells.csv", index=False)
    if frame.empty:
        return
    summary = frame.groupby(
        ["model", "dataset", "perturbation", "template", "space", "metric"], as_index=False
    )["value"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "value_mean", "std": "value_sd", "count": "n_seeds"})
    summary.to_csv(out / "text_baseline_summary.csv", index=False)
    shift = summary.pivot_table(
        index=["model", "dataset", "perturbation"], columns=["space", "metric"], values="value_mean"
    ).reset_index()
    shift.to_csv(out / "text_baseline_space_delta.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--models", default=",".join(TEXT_MODELS))
    parser.add_argument("--seeds", default="")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--perturbations", default=",".join(PERTURBATIONS))
    parser.add_argument("--templates", action="store_true")
    parser.add_argument("--clip-projection", type=Path, default=None)
    parser.add_argument("--k", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = {}
    for item in args.runs.split(","):
        seed, name = item.split("=", 1)
        runs[int(seed)] = name
    layout = RunLayout(args.run_root, runs)
    run(
        layout,
        args.out,
        project_root=args.project_root,
        models=tuple(item for item in args.models.split(",") if item),
        seeds=tuple(int(item) for item in args.seeds.split(",") if item) or layout.seeds,
        datasets=tuple(item for item in args.datasets.split(",") if item),
        perturbations=tuple(item for item in args.perturbations.split(",") if item),
        templates=args.templates,
        clip_projection=args.clip_projection,
        k=args.k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
