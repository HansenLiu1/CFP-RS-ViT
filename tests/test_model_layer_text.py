import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cfra.models import (
    DEFAULT_TEXT_TEMPLATES,
    SHARED_SPACE_IDS,
    EncodedTextFeatures,
    LayerwiseEncodedFeatures,
    ModelAdapterError,
    SharedTextEncoder,
    SharedTextPrototypeBaseline,
    get_model_spec,
)
from cfra.models.adapters import Dinov2Adapter, RemoteClipAdapter, _hidden_state_layerwise
from cfra.models.assets import ModelAsset
from cfra.models.preprocessing import PreprocessSpec
from cfra.models.runtime import ModelRuntimeConfig
from cfra.pipeline.features import FeatureStageError, extract_layerwise_feature_artifact
from cfra.pipeline.layerwise import LayerwiseError, load_layerwise_feature_artifact


def _asset(model_id: str, asset_id: str | None = None) -> ModelAsset:
    return ModelAsset(
        model_id=model_id,
        asset_id=asset_id or model_id,
        path=Path("checkpoint"),
        weight_path=Path("checkpoint"),
        loader_config_path=None,
        revision=None,
        checkpoint_sha256="0" * 64,
        loader_config_sha256="1" * 64,
        asset_fingerprint="2" * 64,
    )


def _runtime(model_id: str, size: int = 28) -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        offline_only=True,
        formal_approved=False,
        precision="fp32",
        batch_size=4,
        max_loaded_models=1,
        pin_memory=False,
        non_blocking=False,
        encoder_input_sizes={model_id: size},
    )


class _HiddenStateVision:
    def eval(self):
        return self

    def requires_grad_(self, _value):
        return self

    def to(self, _device):
        return self


class _SyntheticDinov2(Dinov2Adapter):
    def _forward_global(self, pixel_values):
        return pixel_values.mean(dim=(-2, -1))

    def _forward_layerwise(self, pixel_values, selected_layers):
        import torch

        batch = pixel_values.shape[0]
        hidden_states = (
            torch.arange(batch * 5 * 3, dtype=torch.float32).reshape(batch, 5, 3),
            torch.ones((batch, 5, 3)),
            torch.full((batch, 5, 3), 3.0),
        )
        return _hidden_state_layerwise(
            self,
            hidden_states,
            selected_layers,
            batch_size=batch,
            has_cls_token=True,
            internal_pooling="cls",
            final_tokens=hidden_states[-1],
            final_pooled=hidden_states[-1][:, 0, :],
            final_pooling="post_ln_cls",
            final_projection="none",
            final_shared_space=False,
            block_count=2,
        )


class LayerwiseVisionContractTests(unittest.TestCase):
    def _adapter(self) -> _SyntheticDinov2:
        import torch

        spec = replace(
            get_model_spec("dinov2"),
            dimension=3,
            checkpoint_native_input_size=28,
            patch_size=14,
            preprocess=PreprocessSpec(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                interpolation="bicubic",
                antialias=True,
                source="test",
            ),
        )
        return _SyntheticDinov2(
            spec=spec,
            asset=_asset("dinov2"),
            model=_HiddenStateVision(),
            device=torch.device("cpu"),
            runtime=_runtime("dinov2"),
        )

    def test_layerwise_shape_and_metadata_are_stable(self) -> None:
        import torch

        with self._adapter() as adapter:
            encoded = adapter.encode_layerwise(
                torch.zeros((2, 3, 28, 28), dtype=torch.uint8),
                ["patch_emb", "L01", "final"],
            )
        self.assertIsInstance(encoded, LayerwiseEncodedFeatures)
        self.assertEqual(tuple(encoded.values["patch_emb"].shape), (2, 3))
        self.assertEqual(tuple(encoded.values["L01"].shape), (2, 3))
        self.assertEqual(tuple(encoded.values["final"].shape), (2, 3))
        self.assertEqual(encoded.metadata["patch_emb"]["kind"], "patch_emb")
        self.assertEqual(encoded.metadata["patch_emb"]["pooling"], "patch_mean")
        self.assertEqual(encoded.metadata["L01"]["block"], 1)
        self.assertFalse(encoded.metadata["L01"]["text_shared_space"])
        self.assertIsNone(encoded.metadata["final"]["block"])
        self.assertIn("grid", encoded.metadata["final"])
        self.assertIn("interpolation", encoded.metadata["final"])

    def test_layerwise_rejects_duplicate_or_unknown_layers(self) -> None:
        import torch

        with self._adapter() as adapter:
            with self.assertRaises(ModelAdapterError):
                adapter.encode_layerwise(torch.zeros((1, 3, 28, 28), dtype=torch.uint8), ["L01", "L01"])
            with self.assertRaises(ModelAdapterError):
                adapter.encode_layerwise(torch.zeros((1, 3, 28, 28), dtype=torch.uint8), ["L99"])


class RemoteClipHookContractTests(unittest.TestCase):
    def test_remoteclip_requires_auditable_resblocks(self) -> None:
        adapter = object.__new__(RemoteClipAdapter)
        adapter.model = SimpleNamespace(transformer=SimpleNamespace())
        with patch("cfra.models.adapters._require_torch"):
            with self.assertRaisesRegex(ModelAdapterError, "resblocks"):
                adapter._forward_layerwise(object(), ["L01"])


class SharedTextContractTests(unittest.TestCase):
    def _encoder(self) -> SharedTextEncoder:
        import torch

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                self.kwargs = kwargs
                return {
                    "input_ids": torch.ones((len(texts), kwargs["max_length"]), dtype=torch.long),
                    "attention_mask": torch.ones((len(texts), kwargs["max_length"]), dtype=torch.long),
                }

        class TextModel:
            config = SimpleNamespace(projection_dim=4)

            def eval(self):
                return self

            def requires_grad_(self, _value):
                return self

            def to(self, _device):
                return self

            def get_text_features(self, input_ids, attention_mask=None):
                return input_ids[:, :4].to(torch.float32)

        return SharedTextEncoder(
            model_id="siglip2",
            model=TextModel(),
            tokenizer=Tokenizer(),
            shared_space_id=SHARED_SPACE_IDS["siglip2"],
            dimension=4,
            context_length=8,
            tokenizer_id="fake-tokenizer",
            asset=_asset("siglip2"),
            device=torch.device("cpu"),
            precision="fp32",
            batch_size=8,
        )

    def test_text_and_four_template_prototypes_are_normalized(self) -> None:
        import torch

        with self._encoder() as encoder:
            encoded = encoder.encode(["plane", "ship"])
            prototypes = encoder.encode_class_prototypes(["plane", "ship"])
        self.assertIsInstance(encoded, EncodedTextFeatures)
        self.assertEqual(tuple(encoded.values.shape), (2, 4))
        self.assertEqual(tuple(prototypes.values.shape), (2, 4))
        self.assertTrue(torch.allclose(encoded.values.norm(dim=1), torch.ones(2)))
        self.assertTrue(torch.allclose(prototypes.values.norm(dim=1), torch.ones(2)))
        self.assertEqual(prototypes.template_count, 4)
        self.assertEqual(prototypes.template_ensemble, "mean_unit_normalized_text_embeddings")

    def test_text_encoder_rejects_space_and_template_contract_mismatch(self) -> None:
        with self.assertRaises(ModelAdapterError):
            SharedTextEncoder(
                model_id="siglip2",
                model=object(),
                tokenizer=lambda *_args, **_kwargs: None,
                shared_space_id="wrong-space",
                dimension=4,
                context_length=8,
                tokenizer_id="fake",
                asset=_asset("siglip2"),
                device=SimpleNamespace(type="cpu"),
            )
        with self._encoder() as encoder:
            with self.assertRaises(ModelAdapterError):
                encoder.encode_class_prototypes(["plane"], DEFAULT_TEXT_TEMPLATES[:3])
            with self.assertRaises(ModelAdapterError):
                encoder.encode_class_prototypes(["plane"], ["{class_name}"] * 4)
            with self.assertRaises(ModelAdapterError):
                encoder.assert_shared_space("wrong-space")

    def test_non_shared_models_are_rejected_before_loading(self) -> None:
        from cfra.models import load_shared_text_encoder

        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ModelAdapterError):
            load_shared_text_encoder(
                "dinov2",
                _asset("dinov2"),
                registry_root=Path(directory),
            )

    def test_shared_text_prototype_baseline_uses_dataset_class_keys_without_gt(self) -> None:
        encoder = self._encoder()
        baseline = SharedTextPrototypeBaseline(
            encoder,
            {"synthetic:0": "plane", "synthetic:1": "ship"},
        )
        metadata = [
            {"dataset_id": "synthetic", "class_id": 0},
            {"dataset_id": "synthetic", "class_id": 1},
        ]
        with encoder:
            fitted = baseline.fit(np.ones((2, 4)), np.asarray([0, 1]), metadata)
            scores = baseline.score(
                np.ones((2, 4)),
                np.ones((2, 4)),
                metadata,
                [{"dataset_id": "synthetic", "class_id": 99}, {"dataset_id": "synthetic", "class_id": 99}],
                fitted,
            )
        self.assertEqual(scores.shape, (2, 2))
        self.assertEqual(fitted["class_keys"], ("synthetic:0", "synthetic:1"))


class LayerwiseArtifactTests(unittest.TestCase):
    def test_paired_artifact_is_current_run_local_and_round_trips(self) -> None:
        import torch

        class FakeAdapter:
            def encode_layerwise(self, images, selected_layers, *, output_device):
                count = len(images)
                scale = float(images[0])
                values = {layer: torch.full((count, 3), scale + index) for index, layer in enumerate(selected_layers)}
                metadata = {
                    layer: {
                        "kind": "block",
                        "block": index + 1,
                        "dim": 3,
                        "pooling": "cls",
                        "projection": "none",
                        "text_shared_space": False,
                        "token": {"count": 5, "includes_cls": True, "pooled": "cls"},
                        "grid": {"patch_count": 4},
                        "interpolation": {"enabled": False},
                    }
                    for index, layer in enumerate(selected_layers)
                }
                return LayerwiseEncodedFeatures(
                    values=values,
                    metadata=metadata,
                    model_id="fake",
                    representation_id="fake_layerwise",
                    checkpoint_sha256="a" * 64,
                    loader_config_sha256="b" * 64,
                    asset_fingerprint="c" * 64,
                    runtime_fingerprint="d" * 64,
                    inference_precision="fp32",
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "intermediates" / "features").mkdir(parents=True)
            (root / "manifest").mkdir()
            result = extract_layerwise_feature_artifact(
                adapter=FakeAdapter(),
                clear_images=[1, 1],
                degraded_images=[0, 0],
                selected_layers=["L01", "final"],
                run_root=root,
                row_metadata=[{"role": "confirmation", "source_image_id": "img-1"}, {"role": "confirmation", "source_image_id": "img-2"}],
            )
            table = load_layerwise_feature_artifact(root)
            self.assertTrue(result["npz"].is_file())
            self.assertEqual(table.model_id, "fake")
            self.assertEqual(table.layers["L01"]["features"]["clear"].shape, (2, 3))
            self.assertTrue(np.all(table.layers["L01"]["features"]["clear"] > table.layers["L01"]["features"]["degraded"]))

            with self.assertRaises(FeatureStageError):
                extract_layerwise_feature_artifact(
                    adapter=FakeAdapter(),
                    clear_images=[1],
                    degraded_images=[0],
                    selected_layers=["L01"],
                    run_root=root,
                    artifact_name="..\\outside",
                )
            with self.assertRaises(LayerwiseError):
                load_layerwise_feature_artifact(root, "..\\outside")


if __name__ == "__main__":
    unittest.main()
