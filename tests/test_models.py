import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cfra.models import (
    AdapterCache,
    ImageBatchError,
    ModelAsset,
    ModelAssetError,
    ModelRuntimeConfig,
    ModelRuntimeConfigurationError,
    VisionEncoderAdapter,
    get_model_spec,
    iter_encoded_batches,
    parse_model_runtime_config,
    primary_model_ids,
    stack_image_batch,
    validate_model_asset,
    verify_registered_model_asset,
)
from cfra.models.adapters import ClipAdapter, Dinov2Adapter, MaeAdapter, RemoteClipAdapter, SiglipAdapter
from cfra.models.specs import PreprocessSpec

try:
    import torch  # noqa: F401
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True

requires_torch = unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed; install .[models]")


class _MeanRgbModel:
    def __init__(self) -> None:
        import torch

        self.module = torch.nn.Identity()

    def eval(self):
        self.module.eval()
        return self

    def requires_grad_(self, value):
        self.module.requires_grad_(value)
        return self

    def to(self, device):
        self.module.to(device)
        return self


class _MeanRgbAdapter(VisionEncoderAdapter):
    def _forward_global(self, pixel_values):
        return pixel_values.mean(dim=(-2, -1))


class _ClosableAdapter:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    @property
    def is_closed(self) -> bool:
        return self.closed


class ModelSpecTests(unittest.TestCase):
    def test_primary_models_have_distinct_registered_global_readouts(self) -> None:
        specs = {model_id: get_model_spec(model_id) for model_id in primary_model_ids()}
        self.assertEqual(specs["remoteclip"].pooling, "post_ln_cls")
        self.assertEqual(specs["remoteclip"].projection, "official_visual_projection")
        self.assertEqual(specs["clip"].dimension, 768)
        self.assertFalse(specs["clip"].text_shared_space)
        self.assertEqual(specs["siglip2"].pooling, "map_attention")
        self.assertEqual(specs["dinov2"].checkpoint_native_input_size, 518)
        self.assertEqual(specs["mae"].pooling, "post_ln_patch_mean")

    def test_clip_shared_reuses_checkpoint_but_not_representation(self) -> None:
        clip = get_model_spec("clip")
        shared = get_model_spec("clip_shared")
        self.assertEqual(clip.asset_id, shared.asset_id)
        self.assertNotEqual(clip.representation_id, shared.representation_id)
        self.assertEqual(clip.dimension, 768)
        self.assertEqual(shared.dimension, 512)
        self.assertFalse(clip.text_shared_space)
        self.assertTrue(shared.text_shared_space)

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_model_spec("unknown")

    def test_runtime_requires_explicit_patch_aligned_input_sizes(self) -> None:
        config = {
            "model_runtime": {
                "offline_only": True,
                "formal_approved": False,
                "precision": "fp32",
                "batch_size": 4,
                "max_loaded_models": 1,
                "pin_memory": True,
                "non_blocking": True,
                "encoder_input_sizes": {"clip": 224},
            }
        }
        runtime = parse_model_runtime_config(config, ["clip"])
        self.assertEqual(runtime.encoder_input_sizes, {"clip": 224})
        self.assertEqual(len(runtime.fingerprint), 64)
        config["model_runtime"]["encoder_input_sizes"]["clip"] = 225
        with self.assertRaises(ModelRuntimeConfigurationError):
            parse_model_runtime_config(config, ["clip"])


class ModelAssetTests(unittest.TestCase):
    def test_validates_explicit_file_and_directory_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "RemoteCLIP-ViT-B-32.pt"
            remote.write_bytes(b"checkpoint")
            validated_remote = validate_model_asset("remoteclip", remote)
            self.assertEqual(validated_remote.path, remote.resolve())

            revision = "a" * 40
            clip = root / revision
            clip.mkdir()
            (clip / "config.json").write_text("{}\n", encoding="ascii")
            (clip / "pytorch_model.bin").write_bytes(b"checkpoint")
            validated_clip = validate_model_asset("clip", clip)
            self.assertEqual(validated_clip.revision, revision)
            self.assertEqual(len(validated_clip.checkpoint_sha256), 64)
            self.assertEqual(len(validated_clip.loader_config_sha256), 64)
            self.assertEqual(len(validated_clip.asset_fingerprint), 64)

    def test_rejects_checkpoint_file_that_escapes_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / ("a" * 40)
            outside = root / "outside"
            snapshot.mkdir()
            outside.mkdir()
            (snapshot / "config.json").write_text("{}\n", encoding="ascii")
            target = outside / "pytorch_model.bin"
            target.write_bytes(b"checkpoint")
            try:
                (snapshot / "pytorch_model.bin").symlink_to(target)
            except OSError:
                self.skipTest("Symbolic links are unavailable in this environment")
            with self.assertRaises(ModelAssetError):
                validate_model_asset("clip", snapshot)

    def test_final_registry_verification_rebuilds_asset_identity(self) -> None:
        import csv

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision = "a" * 40
            snapshot = root / revision
            registry = root / "registry"
            snapshot.mkdir()
            registry.mkdir()
            (snapshot / "config.json").write_text("{}\n", encoding="ascii")
            (snapshot / "pytorch_model.bin").write_bytes(b"checkpoint")
            canonical = validate_model_asset("clip", snapshot)
            with (registry / "models.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "model_id",
                        "checkpoint_revision",
                        "checkpoint_sha256",
                        "loader_config_sha256",
                        "status",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "model_id": "clip",
                        "checkpoint_revision": revision,
                        "checkpoint_sha256": canonical.checkpoint_sha256,
                        "loader_config_sha256": canonical.loader_config_sha256,
                        "status": "checkpoint_verified",
                    }
                )
            forged = replace(canonical, checkpoint_sha256="f" * 64)
            with self.assertRaises(ModelAssetError):
                verify_registered_model_asset(forged, "clip", registry)

    def test_rejects_ambiguous_or_incomplete_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ModelAssetError):
                validate_model_asset("remoteclip", root)
            (root / "config.json").write_text("{}\n", encoding="ascii")
            with self.assertRaises(ModelAssetError):
                validate_model_asset("clip", root)


@requires_torch
class ImagePreprocessingTests(unittest.TestCase):
    def test_stacks_uint8_rgb_images(self) -> None:
        import torch

        image = torch.zeros((3, 8, 8), dtype=torch.uint8)
        batch = stack_image_batch([image, image + 1])
        self.assertEqual(tuple(batch.shape), (2, 3, 8, 8))
        self.assertEqual(batch.dtype, torch.uint8)

    def test_rejects_non_square_or_out_of_range_images(self) -> None:
        import torch

        with self.assertRaises(ImageBatchError):
            stack_image_batch(torch.zeros((1, 3, 8, 9), dtype=torch.uint8))
        with self.assertRaises(ImageBatchError):
            stack_image_batch(torch.full((1, 3, 8, 8), 1.1, dtype=torch.float32))


@requires_torch
class AdapterTests(unittest.TestCase):
    def _adapter(self):
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
        runtime = ModelRuntimeConfig(
            offline_only=True,
            formal_approved=False,
            precision="fp32",
            batch_size=2,
            max_loaded_models=1,
            pin_memory=False,
            non_blocking=False,
            encoder_input_sizes={"dinov2": 28},
        )
        return _MeanRgbAdapter(
            spec=spec,
            asset=ModelAsset(
                model_id="dinov2",
                asset_id="dinov2",
                path=Path("checkpoint"),
                weight_path=Path("checkpoint"),
                loader_config_path=None,
                revision=None,
                checkpoint_sha256="0" * 64,
                loader_config_sha256="1" * 64,
                asset_fingerprint="2" * 64,
            ),
            model=_MeanRgbModel(),
            device=torch.device("cpu"),
            runtime=runtime,
        )

    def test_encode_returns_metadata_bound_unnormalized_features(self) -> None:
        import torch

        with self._adapter() as adapter:
            encoded = adapter.encode(torch.full((2, 3, 28, 28), 255, dtype=torch.uint8))
        self.assertEqual(tuple(encoded.values.shape), (2, 3))
        self.assertTrue(torch.equal(encoded.values, torch.ones((2, 3))))
        self.assertEqual(encoded.representation_id, "dinov2_post_ln_cls")
        self.assertFalse(encoded.l2_normalized)
        self.assertEqual(encoded.checkpoint_sha256, "0" * 64)
        self.assertEqual(encoded.runtime_fingerprint, adapter.runtime.fingerprint)
        self.assertEqual(encoded.inference_precision, "fp32")

    def test_encode_does_not_modify_float_input(self) -> None:
        import torch

        images = torch.rand((2, 3, 28, 28), dtype=torch.float32)
        original = images.clone()
        with self._adapter() as adapter:
            adapter.encode(images)
        self.assertTrue(torch.equal(images, original))

    def test_iterates_without_retaining_feature_cache(self) -> None:
        import torch

        with self._adapter() as adapter:
            batches = list(
                iter_encoded_batches(
                    adapter,
                    torch.zeros((5, 3, 28, 28), dtype=torch.uint8),
                )
            )
        self.assertEqual([batch.values.shape[0] for batch in batches], [2, 2, 1])

    def test_auto_device_resolves_to_single_gpu_or_cpu(self) -> None:
        """The project targets one accelerator; auto must never enumerate devices."""

        from cfra.models.adapters import resolve_device

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.device_count", return_value=1
        ):
            resolved = resolve_device("auto")
        self.assertEqual(resolved.type, "cuda")
        self.assertIn(resolved.index, (None, 0))

        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto").type, "cpu")

    def test_unavailable_cuda_index_is_rejected(self) -> None:
        from cfra.models.adapters import ModelAdapterError, resolve_device

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.device_count", return_value=1
        ), self.assertRaises(ModelAdapterError):
            resolve_device("cuda:1")

    def test_adapter_cache_only_reuses_loaded_models_and_evicts_lru(self) -> None:
        first = _ClosableAdapter()
        second = _ClosableAdapter()
        assets = {
            "clip": ModelAsset(
                "clip",
                "clip",
                Path("clip"),
                Path("clip"),
                None,
                None,
                "0" * 64,
                "1" * 64,
                "2" * 64,
            ),
            "dinov2": ModelAsset(
                "dinov2",
                "dinov2",
                Path("dinov2"),
                Path("dinov2"),
                None,
                None,
                "3" * 64,
                "4" * 64,
                "5" * 64,
            ),
        }
        runtime = ModelRuntimeConfig(
            offline_only=True,
            formal_approved=False,
            precision="fp32",
            batch_size=2,
            max_loaded_models=1,
            pin_memory=False,
            non_blocking=False,
            encoder_input_sizes={"clip": 224, "dinov2": 224},
        )
        with patch("cfra.models.adapters.load_adapter", side_effect=[first, second]) as loader:
            registry_root = Path(__file__).resolve().parents[1] / "registry"
            cache = AdapterCache(runtime, registry_root)
            self.assertIs(cache.get("clip", assets["clip"]), first)
            self.assertIs(cache.get("clip", assets["clip"]), first)
            self.assertIs(cache.get("dinov2", assets["dinov2"]), second)
            self.assertTrue(first.closed)
            self.assertFalse(second.closed)
            cache.clear()
            self.assertTrue(second.closed)
        self.assertEqual(loader.call_count, 2)


@requires_torch
class ModelReadoutTests(unittest.TestCase):
    def test_remoteclip_uses_projected_visual_tower_output(self) -> None:
        import torch

        adapter = object.__new__(RemoteClipAdapter)
        adapter.model = lambda pixel_values: pixel_values.mean(dim=(-2, -1))[:, :2]
        values = adapter._forward_global(torch.ones((2, 3, 4, 4)))
        self.assertTrue(torch.equal(values, torch.ones((2, 2))))

    def test_clip_primary_and_shared_spaces_split_at_visual_projection(self) -> None:
        import torch

        pooled = torch.tensor([[1.0, 2.0, 3.0]])

        class Vision:
            def __init__(self) -> None:
                self.interpolate = None

            def __call__(self, **kwargs):
                self.interpolate = kwargs["interpolate_pos_encoding"]
                return SimpleNamespace(pooler_output=pooled)

        vision = Vision()
        model = SimpleNamespace(vision_model=vision, visual_projection=lambda value: value[:, :2] + 10.0)

        primary = object.__new__(ClipAdapter)
        primary.model = model
        primary.spec = get_model_spec("clip")
        primary.encoder_input_size = 224
        self.assertTrue(torch.equal(primary._forward_global(torch.empty(1)), pooled))
        self.assertFalse(vision.interpolate)

        shared = object.__new__(ClipAdapter)
        shared.model = model
        shared.spec = get_model_spec("clip_shared")
        shared.encoder_input_size = 448
        self.assertTrue(torch.equal(shared._forward_global(torch.empty(1)), torch.tensor([[11.0, 12.0]])))
        self.assertTrue(vision.interpolate)

    def test_siglip2_uses_map_pooler_not_patch_tokens(self) -> None:
        import torch

        pooled = torch.tensor([[7.0, 8.0]])
        tokens = torch.full((1, 4, 2), -1.0)
        adapter = object.__new__(SiglipAdapter)
        adapter.model = lambda **kwargs: SimpleNamespace(pooler_output=pooled, last_hidden_state=tokens)
        adapter.spec = get_model_spec("siglip2")
        adapter.encoder_input_size = 224
        self.assertTrue(torch.equal(adapter._forward_global(torch.empty(1)), pooled))

    def test_dinov2_uses_final_cls_pooler(self) -> None:
        import torch

        pooled = torch.tensor([[3.0, 4.0]])
        tokens = torch.full((1, 5, 2), -2.0)
        adapter = object.__new__(Dinov2Adapter)
        adapter.model = lambda **kwargs: SimpleNamespace(pooler_output=pooled, last_hidden_state=tokens)
        adapter.spec = get_model_spec("dinov2")
        adapter.encoder_input_size = 518
        self.assertTrue(torch.equal(adapter._forward_global(torch.empty(1)), pooled))

    def test_mae_excludes_cls_and_preserves_identity_patch_order(self) -> None:
        import torch

        class MaeModel:
            def __init__(self) -> None:
                self.noise = None

            def __call__(self, **kwargs):
                self.noise = kwargs["noise"]
                batch = kwargs["pixel_values"].shape[0]
                cls = torch.full((batch, 1, 2), 100.0)
                patches = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [5.0, 7.0], [7.0, 9.0]]]).expand(
                    batch, -1, -1
                )
                identity = torch.arange(4).unsqueeze(0).expand(batch, -1)
                return SimpleNamespace(
                    last_hidden_state=torch.cat((cls, patches), dim=1),
                    mask=torch.zeros((batch, 4)),
                    ids_restore=identity,
                )

        model = MaeModel()
        adapter = object.__new__(MaeAdapter)
        adapter.model = model
        adapter.spec = replace(get_model_spec("mae"), patch_size=2, checkpoint_native_input_size=4)
        adapter.encoder_input_size = 4
        values = adapter._forward_global(torch.empty((2, 3, 4, 4)))
        self.assertTrue(torch.equal(values, torch.tensor([[4.0, 6.0], [4.0, 6.0]])))
        self.assertTrue(torch.equal(model.noise, torch.arange(4, dtype=torch.float32).expand(2, -1)))


if __name__ == "__main__":
    unittest.main()
