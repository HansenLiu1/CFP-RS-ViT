"""Frozen model and representation metadata independent of runtime libraries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessSpec:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str
    antialias: bool
    source: str


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    asset_id: str
    backend: str
    asset_kind: str
    representation_id: str
    dimension: int
    pooling: str
    projection: str
    text_shared_space: bool
    checkpoint_native_input_size: int
    patch_size: int
    weight_filename: str | None
    loader_config_sha256: str | None
    supports_position_interpolation: bool
    global_semantics_source: str
    preprocess: PreprocessSpec
    shared_space_id: str | None = None


OPENAI_CLIP_PREPROCESS = PreprocessSpec(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
    interpolation="bicubic",
    antialias=True,
    source="OpenAI CLIP/OpenCLIP evaluation convention",
)
IMAGENET_PREPROCESS = PreprocessSpec(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    interpolation="bicubic",
    antialias=True,
    source="registered ImageNet evaluation convention; verify per checkpoint before formal freeze",
)
SIGLIP_PREPROCESS = PreprocessSpec(
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    interpolation="bicubic",
    antialias=True,
    source="checkpoint preprocessor_config.json",
)


MODEL_SPECS: dict[str, ModelSpec] = {
    "remoteclip": ModelSpec(
        model_id="remoteclip",
        asset_id="remoteclip",
        backend="open_clip",
        asset_kind="file",
        representation_id="remoteclip_projected_post_ln_cls",
        dimension=512,
        pooling="post_ln_cls",
        projection="official_visual_projection",
        text_shared_space=True,
        checkpoint_native_input_size=224,
        patch_size=32,
        weight_filename="RemoteCLIP-ViT-B-32.pt",
        loader_config_sha256="f4f248d7794283aa4a4d3a8cc2e518785158f2fb372b7294208393b6c95f18c7",
        supports_position_interpolation=True,
        global_semantics_source="OpenCLIP visual tower token pool followed by official visual projection",
        preprocess=OPENAI_CLIP_PREPROCESS,
        shared_space_id="openclip:ViT-B-32:projected-512",
    ),
    "clip": ModelSpec(
        model_id="clip",
        asset_id="clip",
        backend="transformers_clip",
        asset_kind="directory",
        representation_id="clip_post_ln_cls",
        dimension=768,
        pooling="post_ln_cls",
        projection="none",
        text_shared_space=False,
        checkpoint_native_input_size=224,
        patch_size=16,
        weight_filename="pytorch_model.bin",
        loader_config_sha256=None,
        supports_position_interpolation=True,
        global_semantics_source="CLIP vision transformer pooler_output before visual projection",
        preprocess=OPENAI_CLIP_PREPROCESS,
        shared_space_id=None,
    ),
    "clip_shared": ModelSpec(
        model_id="clip_shared",
        asset_id="clip",
        backend="transformers_clip",
        asset_kind="directory",
        representation_id="clip_visual_projection_output",
        dimension=512,
        pooling="post_ln_cls",
        projection="official_visual_projection",
        text_shared_space=True,
        checkpoint_native_input_size=224,
        patch_size=16,
        weight_filename="pytorch_model.bin",
        loader_config_sha256=None,
        supports_position_interpolation=True,
        global_semantics_source="CLIP post-LN CLS followed by official visual projection",
        preprocess=OPENAI_CLIP_PREPROCESS,
        shared_space_id="transformers:CLIP-ViT-B-16:projected-512",
    ),
    "siglip2": ModelSpec(
        model_id="siglip2",
        asset_id="siglip2",
        backend="transformers_siglip",
        asset_kind="directory",
        representation_id="siglip2_map_attention_pooler",
        dimension=768,
        pooling="map_attention",
        projection="official_pooler",
        text_shared_space=True,
        checkpoint_native_input_size=224,
        patch_size=16,
        weight_filename="model.safetensors",
        loader_config_sha256=None,
        supports_position_interpolation=True,
        global_semantics_source="SigLIP vision MAP/attention pooler output, not a CLS token",
        preprocess=SIGLIP_PREPROCESS,
        shared_space_id="transformers:SigLIP2-base-patch16-224:text-768",
    ),
    "dinov2": ModelSpec(
        model_id="dinov2",
        asset_id="dinov2",
        backend="transformers_dinov2",
        asset_kind="directory",
        representation_id="dinov2_post_ln_cls",
        dimension=768,
        pooling="post_ln_cls",
        projection="none",
        text_shared_space=False,
        checkpoint_native_input_size=518,
        patch_size=14,
        weight_filename="model.safetensors",
        loader_config_sha256=None,
        supports_position_interpolation=True,
        global_semantics_source="DINOv2 final LayerNorm CLS token",
        preprocess=IMAGENET_PREPROCESS,
    ),
    "mae": ModelSpec(
        model_id="mae",
        asset_id="mae",
        backend="transformers_vit_mae",
        asset_kind="directory",
        representation_id="mae_no_mask_post_ln_patch_mean",
        dimension=768,
        pooling="post_ln_patch_mean",
        projection="none",
        text_shared_space=False,
        checkpoint_native_input_size=224,
        patch_size=16,
        weight_filename="model.safetensors",
        loader_config_sha256=None,
        supports_position_interpolation=True,
        global_semantics_source="deterministic no-mask encoder patch-token mean after final LayerNorm",
        preprocess=IMAGENET_PREPROCESS,
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported model representation: {model_id!r}") from exc


def primary_model_ids() -> tuple[str, ...]:
    return "remoteclip", "clip", "siglip2", "dinov2", "mae"
