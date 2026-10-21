import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from cfra.data import (
    ROLE_NAMES,
    ClassInfo,
    DatasetIndexError,
    DatasetManifest,
    ImageRecord,
    ObjectRecord,
    assign_source_image_roles,
    build_dataset_manifest,
    image_path,
    write_dataset_manifest,
)
from cfra.data.index import manifest_exclusion_stats, write_schema_manifests
from cfra.pipeline.stages import StageError, _audit_geometry_control_feasibility, build_e01, build_e02
from cfra.protocols import (
    ProtocolError,
    apply_perturbation,
    build_paired_inputs,
    crop_object,
    geometry_control_feasibility,
    object_mask,
    pair_pixel_invariants,
)
from cfra.runs import RunInitializationError, create_run, write_manifest_json


def _object(
    source_image_id: str,
    *,
    object_index: int = 0,
    content_group_id: str | None = None,
    class_id: int = 0,
    image_relpath: str = "train/Images/image.png",
    width: int = 32,
    height: int = 32,
) -> ObjectRecord:
    return ObjectRecord(
        dataset_id="synthetic",
        split="train",
        source_image_id=source_image_id,
        content_group_id=content_group_id or f"content:{source_image_id}",
        object_id=f"{source_image_id}:{object_index:06d}",
        image_relpath=image_relpath,
        label_relpath="train/Labels/image.txt",
        class_id=class_id,
        class_name=f"class-{class_id}",
        coarse_name="coarse",
        polygon_raw=(8.0, 8.0, 24.0, 8.0, 24.0, 24.0, 8.0, 24.0),
        polygon_visible=(8.0, 8.0, 24.0, 8.0, 24.0, 24.0, 8.0, 24.0),
        image_width=width,
        image_height=height,
        difficulty=None,
        raw_coordinate_out_of_bounds=False,
        visible_area_px2=256.0,
        visible_bbox_width_px=16.0,
        visible_bbox_height_px=16.0,
        visible_side_long_px=16.0,
        visible_side_short_px=16.0,
        center_x_px=16.0,
        center_y_px=16.0,
        orientation_rad=0.0,
        eligible=True,
    )


def _manifest(objects: tuple[ObjectRecord, ...], root: Path | None = None) -> DatasetManifest:
    root = root or Path(tempfile.gettempdir())
    images = tuple(
        ImageRecord(
            dataset_id="synthetic",
            split="train",
            source_image_id=source_image_id,
            content_group_id=content_group_id,
            image_relpath="train/Images/image.png",
            label_relpath="train/Labels/image.txt",
            width=32,
            height=32,
            image_sha256="0" * 64,
            label_sha256="1" * 64,
            status="ok",
        )
        for source_image_id, content_group_id in sorted(
            {(item.source_image_id, item.content_group_id) for item in objects}
        )
    )
    return DatasetManifest(
        dataset_id="synthetic",
        root=str(root),
        class_registry_sha256="2" * 64,
        image_count=len(images),
        labeled_image_count=len(images),
        object_count=len(objects),
        eligible_object_count=sum(item.eligible for item in objects),
        images=images,
        objects=objects,
    )


class RoleProtocolTests(unittest.TestCase):
    def test_role_image_and_dataset_limits_are_hard(self) -> None:
        objects = tuple(_object(f"synthetic:train:{index}") for index in range(6))
        manifest = _manifest(objects)
        selected, report = assign_source_image_roles(
            manifest,
            fractions=[0.25, 0.25, 0.25, 0.25],
            seed=1729,
            minimum_images_per_class={role: 0 for role in ROLE_NAMES},
            maximum_objects_per_class={role: 1 for role in ROLE_NAMES},
            max_objects_per_image_per_class=1,
            maximum_source_images_per_role=1,
            maximum_source_images_per_dataset=1,
        )

        self.assertLessEqual(sum(report["role_image_counts"].values()), 1)
        self.assertTrue(all(count <= 1 for count in report["role_image_counts"].values()))
        self.assertGreater(report["unassigned_unit_count"], 0)
        self.assertTrue(report["source_image_disjoint"])
        self.assertTrue(report["content_group_disjoint"])
        selected_by_role = {
            role: {item.source_image_id for item in selected if item.role == role}
            for role in ROLE_NAMES
        }
        self.assertEqual(sum(len(values) for values in selected_by_role.values()), len({item.source_image_id for item in selected}))
        self.assertIn("role_object_counts", report)
        self.assertIn("role_class_breakdown", report)
        self.assertIn("role_feasibility", report)

    def test_content_duplicate_is_atomic_and_not_split_between_roles(self) -> None:
        objects = tuple(
            _object(
                f"synthetic:train:{index}",
                content_group_id="content:duplicate",
            )
            for index in range(4)
        )
        selected, report = assign_source_image_roles(
            _manifest(objects),
            fractions=[0.25, 0.25, 0.25, 0.25],
            seed=11,
            minimum_images_per_class={role: 0 for role in ROLE_NAMES},
            maximum_objects_per_class={role: 10 for role in ROLE_NAMES},
            max_objects_per_image_per_class=2,
            maximum_source_images_per_role=1,
            maximum_source_images_per_dataset=4,
        )
        self.assertTrue(report["content_group_disjoint"])
        roles = {item.role for item in selected if item.content_group_id == "content:duplicate"}
        self.assertLessEqual(len(roles), 1)
        self.assertTrue(all(report["role_image_counts"][role] <= 1 for role in ROLE_NAMES))

    def test_e01_report_has_no_double_source_image_prefix(self) -> None:
        objects = tuple(_object(f"synthetic:train:{index}") for index in range(4))
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            run = create_run(
                project_root=project,
                run_id="synthetic-e01",
                evidence_tier="development",
                resolved_config={},
                resolved_config_sha256="0" * 64,
            )
            raw_root = Path(directory) / "raw"
            (raw_root / "train" / "Images").mkdir(parents=True)
            (raw_root / "train" / "Labels").mkdir(parents=True)
            Image.new("RGB", (32, 32), (1, 2, 3)).save(raw_root / "train" / "Images" / "image.png")
            (raw_root / "train" / "Labels" / "image.txt").write_text("", encoding="ascii")
            config = {
                "run": {"evidence_tier": "development"},
                "split": {"fractions": [0.25, 0.25, 0.25, 0.25], "seed": 7},
                "sampling": {
                    "minimum_direction_fit_images_per_class": 0,
                    "minimum_readout_fit_images_per_class": 0,
                    "minimum_validation_images_per_class": 0,
                    "minimum_confirmation_images_per_class": 0,
                    "direction_fit_objects_per_class": 1,
                    "readout_fit_objects_per_class": 1,
                    "validation_objects_per_class": 1,
                    "confirmation_objects_per_class": 1,
                    "max_objects_per_image_per_class": 1,
                },
                "scope": {
                    "max_source_images_per_role": 1,
                    "max_source_images_per_dataset": 4,
                },
            }
            build_e01(run_root=run, manifests={"synthetic": _manifest(objects, raw_root)}, config=config)
            payload = json.loads((run / "manifest" / "e01.json").read_text(encoding="utf-8"))
            image_ids = [value for values in payload["all_role_source_image_sets"].values() for value in values]
            self.assertTrue(image_ids)
            self.assertTrue(all("synthetic:synthetic:" not in image_id for image_id in image_ids))

    def test_e01_formal_records_infeasible_classes_without_failing(self) -> None:
        objects = tuple(
            _object(f"synthetic:train:{index}", class_id=index % 3)
            for index in range(12)
        )
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            run = create_run(
                project_root=project,
                run_id="synthetic-e01-formal",
                evidence_tier="formal",
                resolved_config={},
                resolved_config_sha256="0" * 64,
            )
            raw_root = Path(directory) / "raw"
            (raw_root / "train" / "Images").mkdir(parents=True)
            (raw_root / "train" / "Labels").mkdir(parents=True)
            Image.new("RGB", (32, 32), (1, 2, 3)).save(raw_root / "train" / "Images" / "image.png")
            (raw_root / "train" / "Labels" / "image.txt").write_text("", encoding="ascii")
            config = {
                "run": {"evidence_tier": "formal"},
                "split": {"fractions": [0.25, 0.25, 0.25, 0.25], "seed": 7},
                "sampling": {
                    "minimum_direction_fit_images_per_class": 5,
                    "minimum_readout_fit_images_per_class": 5,
                    "minimum_validation_images_per_class": 5,
                    "minimum_confirmation_images_per_class": 5,
                    "direction_fit_objects_per_class": 10,
                    "readout_fit_objects_per_class": 10,
                    "validation_objects_per_class": 10,
                    "confirmation_objects_per_class": 10,
                    "max_objects_per_image_per_class": 1,
                },
                "scope": {
                    "max_source_images_per_role": 2,
                    "max_source_images_per_dataset": 12,
                },
            }
            selected = build_e01(run_root=run, manifests={"synthetic": _manifest(objects, raw_root)}, config=config)
            excluded = json.loads(
                (run / "manifest" / "e01_excluded_classes.json").read_text(encoding="utf-8")
            )
            self.assertIn("synthetic", excluded)
            self.assertTrue(excluded["synthetic"]["class_ids"])
            self.assertEqual(
                excluded["synthetic"]["reason"],
                "insufficient_role_class_capacity_under_image_disjoint_split",
            )
            payload = json.loads((run / "manifest" / "e01.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["all_role_class_requirements_feasible"])
            self.assertIn("excluded_classes", payload)
            self.assertIn("scope", excluded["synthetic"])


class PathAndManifestTests(unittest.TestCase):
    def test_geometry_control_infeasible_objects_are_preexcluded_and_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            image_path_value = dataset_root / "train" / "Images" / "image.png"
            image_path_value.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), (10, 20, 30)).save(image_path_value)
            item = _object("synthetic:train:0", width=32, height=32)
            manifest = _manifest((item,), dataset_root)
            updated, audit = _audit_geometry_control_feasibility(
                manifest,
                config={
                    "controls": {"random_geometry_matched_obb": True},
                    "input": {"size_px": 16, "obb_soft_mask_supersampling": 1},
                    "protocol": {"control_min_iou": 0.0, "control_max_iou": 0.05},
                },
            )

            self.assertEqual(audit["policy"], "pre_exclude_geometry_control_infeasible")
            self.assertEqual(audit["excluded_objects"], 1)
            self.assertFalse(updated.objects[0].eligible)
            self.assertEqual(updated.objects[0].exclusion_reason, "geometry_control_infeasible")
            stats = manifest_exclusion_stats(updated)
            self.assertEqual(stats["objects"]["by_category"]["geometry_control_infeasible"], 1)
            self.assertEqual(stats["objects"]["total_count"], 1)
            self.assertEqual(stats["objects"]["excluded_count"], 1)
            self.assertEqual(stats["objects"]["affected_source_image_count"], 1)

    def test_relative_paths_are_resolved_before_containment_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            outside = root / "outside.png"
            Image.new("RGB", (2, 2), (1, 2, 3)).save(outside)
            record = _object("synthetic:train:0", image_relpath="../outside.png", width=2, height=2)
            manifest = _manifest((record,), dataset_root)
            with self.assertRaises(DatasetIndexError):
                image_path(manifest, record)

            project = root / "project"
            project.mkdir()
            run = create_run(
                project_root=project,
                run_id="path-test",
                evidence_tier="development",
                resolved_config={},
                resolved_config_sha256="0" * 64,
            )
            with self.assertRaises(RunInitializationError):
                write_manifest_json(run, "../escape.json", {"bad": True})

    def test_e00_schema_and_exclusion_categories_are_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            image_dir = dataset_root / "train" / "Images"
            label_dir = dataset_root / "train" / "Labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            cases = (
                ("small", (1, 1, 4, 1, 4, 4, 1, 4), (10, 10)),
                ("large", (0, 0, 9, 0, 9, 9, 0, 9), (10, 10)),
                ("huge", (0, 0, 10, 0, 10, 10, 0, 10), (11, 11)),
            )
            for stem, points, size in cases:
                Image.new("RGB", size, (20, 30, 40)).save(image_dir / f"{stem}.png")
                coordinates = " ".join(str(value) for value in points)
                (label_dir / f"{stem}.txt").write_text(f"0 {coordinates}\n", encoding="ascii")

            registry = {"synthetic": {0: ClassInfo(0, "vehicle", "coarse")}}
            manifest = build_dataset_manifest(
                "synthetic",
                dataset_root,
                class_registry=registry,
                class_registry_sha256="3" * 64,
                minimum_side_px=5.0,
                maximum_side_px=8.0,
                maximum_image_pixels=100,
                maximum_difficulty=1,
            )
            stats = manifest_exclusion_stats(manifest)
            self.assertEqual(stats["objects"]["by_category"]["minimum_side"], 1)
            self.assertEqual(stats["objects"]["by_category"]["oversized_crop"], 1)
            self.assertEqual(stats["images"]["by_category"]["image_pixel_limit"], 1)
            self.assertEqual(stats["by_dataset_class"]["synthetic"]["0"]["class_name"], "vehicle")

            output = root / "output"
            write_dataset_manifest(manifest, output)
            write_schema_manifests({"synthetic": manifest}, output)
            for name in ("source_files.csv", "images.csv", "objects.csv", "splits.csv", "models.csv"):
                self.assertTrue((output / name).is_file(), name)
            summary = json.loads((output / "synthetic_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["exclusion_stats"]["objects"]["by_category"]["minimum_side"], 1)
            with (output / "source_files.csv").open(encoding="utf-8", newline="") as handle:
                source_rows = list(csv.DictReader(handle))
            self.assertEqual(len(source_rows), 6)

    def test_manifest_schema_rejects_escaping_label_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            dataset_root = root / "dataset"
            image_path_value = dataset_root / "image.png"
            dataset_root.mkdir()
            Image.new("RGB", (4, 4), (1, 1, 1)).save(image_path_value)
            item = _object("synthetic:train:0", image_relpath="image.png", width=4, height=4)
            image = ImageRecord(
                dataset_id="synthetic",
                split="train",
                source_image_id=item.source_image_id,
                content_group_id=item.content_group_id,
                image_relpath="image.png",
                label_relpath="../outside.txt",
                width=4,
                height=4,
                image_sha256="0" * 64,
                label_sha256=None,
                status="ok",
            )
            manifest = replace(_manifest((item,), dataset_root), images=(image,))
            with self.assertRaises(DatasetIndexError):
                write_schema_manifests({"synthetic": manifest}, root / "manifest")


class PixelProtocolTests(unittest.TestCase):
    def test_object_mask_matches_crop_mask(self) -> None:
        item = _object("synthetic:train:0", width=16, height=16)
        image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        crop = crop_object(image, item, size_px=8, mask_supersampling=2)
        np.testing.assert_array_equal(crop.mask, object_mask(item, size_px=8, mask_supersampling=2))

    def test_geometry_control_feasibility_is_deterministic_and_reports_policy(self) -> None:
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[12:20, 12:20] = 1.0
        result = geometry_control_feasibility(mask, max_iou=0.05)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["policy"], "boundary_grid_3x3")
        self.assertEqual(result, geometry_control_feasibility(mask, max_iou=0.05))

    def test_geometry_control_feasibility_rejects_full_crop(self) -> None:
        mask = np.ones((8, 8), dtype=np.float32)
        result = geometry_control_feasibility(mask, max_iou=0.05)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["best_iou"], 1.0)

    def test_changed_mask_is_actual_quantized_difference_and_overlap_is_explicit(self) -> None:
        constant = np.full((16, 16, 3), 10, dtype=np.uint8)
        result = apply_perturbation(
            constant,
            "patch_mask",
            rng=np.random.default_rng(1),
            patch_fraction=0.5,
            patch_side_fraction=0.25,
            allow_overlap=False,
        )
        self.assertEqual(int(result.changed_mask.sum()), 0)
        self.assertEqual(result.parameters["actual_changed_pixels"], 0)
        self.assertEqual(result.parameters["changed_mask_semantics"], "actual_uint8_rgb_difference_any_channel")
        self.assertFalse(result.parameters["allow_overlap"])
        self.assertEqual(result.parameters["overlap_pixels"], 0)

        varied = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        overlapping = apply_perturbation(
            varied,
            "patch_mask",
            rng=np.random.default_rng(2),
            patch_fraction=1.0,
            patch_side_fraction=0.5,
            allow_overlap=True,
        )
        self.assertTrue(overlapping.parameters["allow_overlap"])
        self.assertEqual(
            overlapping.parameters["overlap_pixels"],
            overlapping.parameters["nominal_placement_pixels"] - overlapping.parameters["actual_pixels"],
        )

    def test_pixel_tolerance_and_changed_mask_audit_fields(self) -> None:
        image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        object_record = _object("synthetic:train:0", width=16, height=16)
        crop = crop_object(image, object_record, size_px=8, mask_supersampling=2)
        perturbation = apply_perturbation(image=crop.image, kind="blur", rng=np.random.default_rng(3), sigma_px=1.0)
        invariants = pair_pixel_invariants(build_paired_inputs(crop, perturbation), tolerance=0.25)
        self.assertEqual(invariants["pixel_tolerance"], 0.25)
        self.assertEqual(invariants["pixel_tolerance_semantics"], "absolute_atol_only_rtol_zero")
        self.assertTrue(invariants["changed_mask_consistent_with_uint8_difference"])
        with self.assertRaises(ProtocolError):
            pair_pixel_invariants(build_paired_inputs(crop, perturbation), tolerance=-1.0)


class E02GateTests(unittest.TestCase):
    def test_development_control_failure_writes_auditable_bundle_and_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            image_path_value = dataset_root / "train" / "Images" / "image.png"
            image_path_value.parent.mkdir(parents=True)
            Image.new("RGB", (16, 16), (10, 20, 30)).save(image_path_value)
            item = _object(
                "synthetic:train:0",
                image_relpath="train/Images/image.png",
                width=16,
                height=16,
            )
            manifest = _manifest((item,), dataset_root)
            project = root / "project"
            project.mkdir()
            run = create_run(
                project_root=project,
                run_id="development-e02-failure",
                evidence_tier="development",
                resolved_config={},
                resolved_config_sha256="0" * 64,
            )
            controls = {name: False for name in ("shifted_obb", "background_only", "boundary_only", "interior_eroded", "mask_eroded", "mask_dilated")}
            controls["random_geometry_matched_obb"] = True
            config = {
                "run": {"evidence_tier": "development"},
                "input": {"size_px": 8, "obb_soft_mask_supersampling": 1},
                "scope": {"perturbations": ["blur"]},
                "perturbations": {
                    "blur": {"enabled": True, "sigma_px": 1.0},
                    "patch_mask": {"allow_overlap": False},
                },
                "protocol": {
                    "pixel_tolerance": 0.0,
                    "boundary_radius_px": 1,
                    "control_shift_fraction": 0.75,
                    "control_max_iou": 0.05,
                    "control_max_attempts": 0,
                    "interaction_background_states": ["clean", "degraded"],
                },
                "controls": controls,
                "split": {"seed": 1729},
            }
            with self.assertRaises(StageError) as context:
                build_e02(
                    run_root=run,
                    manifests={"synthetic": manifest},
                    selected_by_dataset={"synthetic": (item,)},
                    config=config,
                )
            error = context.exception
            self.assertIsNotNone(error.failure_bundle)
            bundle_path = run / "intermediates" / "e02_pixel_protocol" / "failure_bundle.json"
            self.assertTrue(bundle_path.is_file())
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["evidence_tier"], "development")
            self.assertGreater(bundle["control_failure_count"], 0)
            report = json.loads((run / "manifest" / "e02.json").read_text(encoding="utf-8"))
            self.assertTrue(report["model_experiments_blocked"])
            self.assertFalse(report["all_registered_controls_available"])


if __name__ == "__main__":
    unittest.main()
