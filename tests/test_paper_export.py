import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cfra.paper_export import PaperExport
from cfra.paper_export_cache import classwise, mapping_permutation


class PaperExportTests(unittest.TestCase):
    def test_seed_and_strength_missing_cells_are_explicit(self):
        from cfra.paper_export_grid import expand_missing_grid
        with tempfile.TemporaryDirectory() as tmp:
            out=PaperExport([],Path(tmp)/'out')
            ctx=dict(dataset='dota',model='clip',perturbation='patch_mask',split_seed=1729,
                     run_id='one',stage='E04-4',control='true_obb',operator='euclidean_mean_subtraction',normalization='unit')
            for metric in ['own_argmax_rate','mean_diagonal_margin','macro_ovr_auc']:
                out.emit('specificity_full',ctx,metric,.8)
            expand_missing_grid(out)
            self.assertEqual(len(out.tables['seed_stability']),6)
            self.assertEqual(len(out.tables['strength_sweep']),27)
            self.assertTrue(all(r['estimate'] is None for r in out.tables['strength_sweep']))
            self.assertTrue(all(r['split_seed']!=1729 for r in out.tables['seed_stability']))
            out.units.close()

    def test_class_mapping_null_and_source_class_means(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=PaperExport([],Path(tmp)/'out',bootstrap_replicates=8,permutation_replicates=99)
            context=dict(run_id='test',stage='specificity_recomputed',dataset='x',model='x',layer='final',
                         perturbation='blur',operator='raw',normalization='unit',split_seed=1729)
            scores=np.tile(np.eye(4),(3,1)); labels=np.tile(np.arange(4),3)
            images=np.asarray([f'im{i}' for i in range(12)])
            for metric in ['own_argmax_rate','mean_diagonal_margin','macro_ovr_auc']:
                out.emit('specificity_full',context,metric,1.,1.,1.)
            classwise(out,context,scores,labels,images,np.arange(4))
            mapping_permutation(out,context,scores,labels,images,np.arange(4))
            self.assertEqual(len(out.tables['specificity_classwise']),16)
            for row in out.master[:3]:
                self.assertGreater(row['p_value'],0)
                self.assertLess(row['p_value'],.15)
            out.units.close()

    def test_baseline_identity_and_nonfinite_interval_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); artifact=root/'stage.json'; artifact.write_text('{}')
            out=PaperExport([],root/'out',recompute_layers=False,recompute_final=False)
            out.config={'split':{'seed':1729}}; out.models={}
            out.ingest(root,'E04-1',dict(dataset_id='dota',model_id='clip',baseline='clear_degraded_cav',
                       macro_ovr_auc=.8,macro_ovr_auc_ci=[float('nan'),float('nan')]),artifact,'baseline_comparison')
            self.assertEqual(out.master[0]['operator'],'clear_degraded_cav')
            self.assertEqual(out.master[0]['statistics_status'],'ci_missing')
            self.assertIsNone(out.master[0]['p_value'])
            out.units.close()


class FusionAndLegacySplitTests(unittest.TestCase):
    """Cache-only analyses that reconstruct the paper-export experiments."""

    DATASET = "dota"
    PERTURBATION = "blur"
    ROLE_COUNTS = (("direction_fit", 6), ("readout_fit", 6), ("validation", 6), ("confirmation", 8))

    def _table(self):
        from cfra.pipeline.features import FeatureSample, FeatureTable
        rng = np.random.default_rng(7)
        samples = []
        raw_rows = []
        for class_id in (0, 1):
            for role, count in self.ROLE_COUNTS:
                for index in range(count):
                    samples.append(FeatureSample(
                        dataset_id=self.DATASET,
                        object_id=f"c{class_id}-{role}-{index}",
                        source_image_id=f"img-{class_id}-{role}-{index // 2}",
                        class_id=class_id,
                        class_name=f"class{class_id}",
                        coarse_name=f"group{class_id}",
                        role=role,
                        perturbation=self.PERTURBATION,
                        seed=1729,
                        geometry=(float(np.log1p(100 + index)), 0.1, 0.0, 1.0),
                    ))
                    raw_rows.append(rng.normal(size=8) + 1.5 * np.eye(8)[class_id])
        metadata = tuple(samples)
        raw = np.stack(raw_rows)
        signals = np.eye(8)[np.asarray([sample.class_id for sample in metadata])] * 0.8
        arrays = {
            "raw": raw,
            "true_clear_degraded": raw + signals,
            "true_degraded_degraded": raw.copy(),
        }
        return FeatureTable(model_id="clip", metadata=metadata, arrays=arrays, representation={})

    def _fixtures(self, table):
        from cfra.pipeline.core import _fit_operator_and_directions
        labels = np.asarray([sample.class_id for sample in table.metadata], dtype=np.int64)
        images = np.asarray([sample.source_image_id for sample in table.metadata], dtype="U")
        ds = np.asarray([sample.dataset_id for sample in table.metadata])
        perts = np.asarray([sample.perturbation for sample in table.metadata])
        roles = np.asarray([sample.role for sample in table.metadata])
        delta = table.arrays["true_clear_degraded"] - table.arrays["true_degraded_degraded"]
        fit = np.flatnonzero((ds == self.DATASET) & (perts == self.PERTURBATION) & (roles == "direction_fit"))
        _, directions, classes = _fit_operator_and_directions(
            delta[fit] / np.linalg.norm(delta[fit], axis=1, keepdims=True), labels[fit],
            "euclidean_mean_subtraction", background_fit_values=None, pca_rank=5, epsilon=1e-8)
        return labels, images, ds, perts, roles, directions, classes

    def _export(self, root):
        from cfra.paper_export import PaperExport
        from cfra.pipeline.core import _statistics_settings
        out = PaperExport([], root / "out", bootstrap_replicates=8, permutation_replicates=19,
                          recompute_layers=False, recompute_final=False)
        out.config = {"split": {"seed": 1729, "fractions": [0.25, 0.25, 0.20, 0.30]},
                      "scope": {"datasets": [self.DATASET], "models": ["clip"],
                                "perturbations": [self.PERTURBATION]}}
        out.models = {}
        settings = _statistics_settings({"bootstrap_replicates": 8, "bootstrap_seed": 1729,
                                         "confidence_level": 0.95, "cluster_unit": "source_image",
                                         "permutation_replicates": 19})
        return out, settings

    def test_fusion_emits_standalone_and_paired_rows(self):
        from cfra.paper_export_cache import fusion
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "stage.json"
            artifact.write_text("{}")
            table = self._table()
            labels, images, ds, perts, roles, directions, classes = self._fixtures(table)
            out, settings = self._export(root)
            fusion(out, root, artifact, table, "clip", self.DATASET, self.PERTURBATION,
                   directions, classes, labels, images, ds, perts, roles, settings)
            table_rows = out.tables["prototype_cfp_complementarity"]
            standalone = [row for row in table_rows if row.get("metric") == "macro_ovr_auc"
                          and row.get("operator") in ("raw_prototype", "cfp_cmc", "prototype_cfp_fusion")]
            self.assertEqual(len(standalone), 3)
            for row in standalone:
                self.assertTrue(np.isfinite(row["estimate"]))
                self.assertTrue(np.isfinite(row["CI_low"]) and np.isfinite(row["CI_high"]))
            gains = [row for row in table_rows if row.get("metric") == "fusion_minus_prototype_macro_ovr_auc"]
            self.assertEqual(len(gains), 1)
            self.assertTrue(np.isfinite(gains[0]["estimate"]))
            out.units.close()

    def test_legacy_resplit_reports_overlap_and_emits_rows(self):
        from collections import defaultdict
        from cfra.paper_export_cache import _legacy_role_plan, legacy_resplit
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "stage.json"
            artifact.write_text("{}")
            table = self._table()
            plan = _legacy_role_plan(table.metadata, self.DATASET, [0.25, 0.25, 0.20, 0.30], 1729)
            self.assertEqual(len(plan), len({sample.object_id for sample in table.metadata}))
            self.assertEqual(set(plan.values()),
                             {"direction_fit", "readout_fit", "validation", "confirmation"})
            role_by_image = defaultdict(set)
            for sample in table.metadata:
                role_by_image[sample.source_image_id].add(plan[sample.object_id])
            self.assertTrue(any(len(image_roles) > 1 for image_roles in role_by_image.values()))
            out, settings = self._export(root)
            canonical = {"macro_ovr_auc": 0.5, "own_argmax_rate": 0.5, "mean_diagonal_margin": 0.0}
            legacy_resplit(out, root, artifact, table, "clip", self.DATASET, self.PERTURBATION,
                           plan, canonical, settings, 2)
            table_rows = out.tables["legacy_split"]
            primary = [row for row in table_rows if row.get("metric") == "macro_ovr_auc"
                       and row.get("analysis") == "legacy_instance_resplit"]
            self.assertEqual(len(primary), 1)
            self.assertTrue(np.isfinite(primary[0]["estimate"]))
            self.assertGreaterEqual(primary[0]["shared_source_images"], 0)
            deltas = [row for row in table_rows if str(row.get("metric", "")).startswith("canonical_minus_legacy_")]
            self.assertGreater(len(deltas), 0)
            cosines = [row for row in table_rows if row.get("metric") == "pair_cosine"]
            self.assertEqual(len(cosines), 2)
            out.units.close()


class LayerwiseReadoutExportTests(unittest.TestCase):
    """The layerwise centered-space readout table must be exported faithfully."""

    def _export(self, root):
        out = PaperExport([], root / "out", recompute_layers=False, recompute_final=False)
        out.config = {"split": {"seed": 1729},
                      "scope": {"models": ["clip"], "datasets": ["dota"], "perturbations": ["blur"]},
                      "perturbations": {"blur": {}}}
        out.models = {}
        return out

    def _payload(self):
        alpha_block = {
            "own@1": {"linear_accuracy_drop": 0.05, "linear_margin_drop": 0.10,
                      "prototype_accuracy_drop": 0.03, "prototype_margin_drop": 0.07},
            "own@0.25": {"linear_accuracy_drop": 0.01},
            "wrong_raw@1": {"linear_accuracy_drop": 0.02, "linear_margin_drop": 0.04},
        }
        layers = {
            layer: {"by_perturbation": {"blur": {"confirmation_objects": 15 - index,
                                                 "alpha": dict(alpha_block)}}}
            for index, layer in enumerate(("patch_emb", "L03", "L06", "final"))
        }
        # The block trend only covers transformer blocks, so ``patch_emb`` and
        # ``final`` must not appear here.
        summary = {"own_minus_wrong_linear_margin_drop_by_layer":
                   {"own_minus_wrong_linear_margin_drop": {"L03": -0.01, "L06": 0.04}}}
        return {"model_id": "clip", "dataset_id": "dota",
                "readout_intervention": {"selected_layers": ["patch_emb", "L03", "L06", "final"],
                                         "layers": layers, "summary": summary}}

    def test_exports_readout_metrics_and_trend(self):
        from cfra.paper_export_cache import export_layerwise_readout
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics"
            metrics.mkdir(parents=True)
            (metrics / "E06_clip_dota.json").write_text(json.dumps(self._payload()), encoding="utf-8")
            out = self._export(root)
            export_layerwise_readout(out, root)
            rows = out.tables["layerwise_readout"]
            drops = [r for r in rows if r["metric"] == "linear_margin_drop"
                     and r["layer"] == "L06" and r["condition"] == "own"]
            self.assertEqual(len(drops), 1)
            self.assertAlmostEqual(drops[0]["estimate"], 0.10)
            self.assertEqual(drops[0]["space"], "centered")
            self.assertEqual(drops[0]["status"], "observed")
            self.assertEqual(drops[0]["valid_n_objects"], 13)
            self.assertEqual(str(drops[0]["alpha"]), "1")
            # wrong_raw is exported with its own condition.
            wrong_raw = [r for r in rows if r["condition"] == "wrong_raw"]
            self.assertTrue(wrong_raw)
            self.assertTrue(all(r["status"] == "observed" for r in wrong_raw))
            # Endpoints and transformer blocks are separated: every requested
            # layer has drop rows, but only block layers carry the trend metric.
            layers_with_drops = {r["layer"] for r in rows if r["metric"] == "linear_margin_drop"}
            self.assertEqual(layers_with_drops, {"patch_emb", "L03", "L06", "final"})
            trend_layers = {r["layer"] for r in rows
                            if r["metric"] == "own_minus_wrong_linear_margin_drop"}
            self.assertEqual(trend_layers, {"L03", "L06"})
            # No row is emitted twice for the same identity.
            identities = [
                (r["layer"], r["perturbation"], r["condition"], str(r["alpha"]), r["metric"])
                for r in rows if r["metric"] in ("linear_margin_drop", "linear_accuracy_drop")
            ]
            self.assertEqual(len(identities), len(set(identities)))
            out.units.close()

    def test_absent_payload_is_explicit_missing(self):
        from cfra.paper_export_cache import export_layerwise_readout
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics"
            metrics.mkdir(parents=True)
            (metrics / "E06_clip_dota.json").write_text(
                json.dumps({"model_id": "clip", "dataset_id": "dota"}), encoding="utf-8"
            )
            out = self._export(root)
            export_layerwise_readout(out, root)
            rows = out.tables["layerwise_readout"]
            self.assertTrue(rows)
            self.assertTrue(all(r["status"] == "missing" for r in rows))
            out.units.close()


class StabilityExportTests(unittest.TestCase):
    """``stability`` must advance labels/images through both ``unit`` masks.

    ``unit`` returns a compressed matrix plus a mask over its input; the
    centered-space branch previously indexed the full-length label/image arrays
    with the compressed mask, raising an IndexError whenever any delta was
    invalid (zero norm), e.g. patch_mask.
    """

    def test_invalid_rows_do_not_break_centered_space(self):
        from cfra.paper_export_cache import stability
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "stage.json"
            artifact.write_text("{}")
            out = PaperExport([], root / "out", bootstrap_replicates=8)
            out.config = {"split": {"seed": 1729}}
            out.models = {}
            rng = np.random.default_rng(0)
            values = rng.normal(size=(12, 6))
            values[[2, 5, 9]] = 0.0  # zero-norm deltas -> filtered by unit()
            labels = np.asarray([0, 1] * 6, dtype=np.int64)
            images = np.asarray([f"im{i % 6}" for i in range(12)], dtype="U")
            stability(out, root, artifact, "clip", "dota", "patch_mask", values, labels, images)
            rows = out.tables["common_mode_stability"]
            spaces = {row["space"] for row in rows}
            self.assertIn("raw", spaces)
            self.assertIn("cmc", spaces)
            self.assertTrue(any(row["space"] == "cmc" and row["status"] == "observed" for row in rows))
            out.units.close()


class CheckpointResumeTests(unittest.TestCase):
    """Committed work units let an interrupted export resume exactly once."""

    def _export(self, output):
        return PaperExport([], output, progress=lambda message: None)

    def test_units_persist_replay_and_do_not_duplicate(self):
        import gzip
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            root = Path(tmp) / "run"
            root.mkdir()
            context = {"dataset": "dota", "model": "clip", "split_seed": 1729,
                       "run_id": "run", "stage": "demo"}

            first = self._export(out_dir)
            first.begin_unit(root, ("demo",))
            first.emit("specificity_full", context, "own_argmax_rate", 0.5)
            first.units.write('{"unit": 1}\n')
            first.end_unit()
            self.assertTrue(first.unit_done(root, ("demo",)))
            first.units.close()

            second = self._export(out_dir)
            self.assertTrue(second.unit_done(root, ("demo",)))
            self.assertEqual(len(second.tables["specificity_full"]), 1)
            self.assertEqual(len(second.master), 1)
            second.begin_unit(root, ("demo2",))
            second.emit("specificity_full", context, "own_argmax_rate", 0.7)
            second.end_unit()
            second.units.close()

            third = self._export(out_dir)
            self.assertEqual(len(third.master), 2)
            estimates = sorted(row["estimate"] for row in third.tables["specificity_full"])
            self.assertEqual(estimates, [0.5, 0.7])
            third._assemble_source_units()
            with gzip.open(out_dir / "source_units.jsonl.gz", "rt", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            self.assertEqual(lines, ['{"unit": 1}'])
            third.units.close()

    def test_uncommitted_unit_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            root = Path(tmp) / "run"
            root.mkdir()
            export = self._export(out_dir)
            export.begin_unit(root, ("crash",))
            export.emit("specificity_full", {"run_id": "run", "stage": "demo"},
                        "own_argmax_rate", 0.5)
            # no end_unit: simulate an interruption before the commit point
            export.abort_unit()
            export.units.close()
            resumed = self._export(out_dir)
            self.assertFalse(resumed.unit_done(root, ("crash",)))
            self.assertEqual(len(resumed.master), 0)
            resumed.units.close()


class AlignmentLayerwiseTests(unittest.TestCase):
    """``alignment`` must accept the layerwise table's mapping metadata rows."""

    def test_alignment_accepts_layerwise_dict_metadata(self):
        from types import SimpleNamespace
        from cfra.paper_export_cache import alignment
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "layerwise.npz"
            artifact.write_text("{}")
            out = PaperExport([], root / "out", bootstrap_replicates=8,
                              progress=lambda message: None)
            out.config = {"split": {"seed": 1729}, "protocol": {"primary_fit_size": 2}}
            out.models = {}
            rng = np.random.default_rng(0)
            metadata = []
            for pert in ("blur", "noise"):
                for class_id in (0, 1):
                    for fit in range(2):
                        metadata.append({"object_id": f"fit-{pert}-{class_id}-{fit}",
                                         "class_id": class_id, "role": "direction_fit",
                                         "perturbation": pert, "source_image_id": f"fit-{class_id}-{fit}"})
                    for image in (f"conf-{class_id}-a", f"conf-{class_id}-b"):
                        metadata.append({"object_id": f"conf-{pert}-{class_id}-{image}",
                                         "class_id": class_id, "role": "confirmation",
                                         "perturbation": pert, "source_image_id": image})
            delta = rng.normal(size=(len(metadata), 6)) + np.eye(6)[[m["class_id"] for m in metadata]] * 1.5
            roles = np.asarray([m["role"] for m in metadata])
            labels = np.asarray([m["class_id"] for m in metadata], dtype=np.int64)
            perts = np.asarray([m["perturbation"] for m in metadata])
            images = np.asarray([m["source_image_id"] for m in metadata], dtype="U")
            table = SimpleNamespace(metadata=metadata)
            alignment(out, root, artifact, table, "clip", "dota", "L06",
                      delta, roles, labels, perts, images)
            rows = out.tables["layerwise_alignment"]
            self.assertTrue(rows)
            self.assertTrue(all(row["status"] == "observed" for row in rows))
            self.assertEqual({row["space"] for row in rows}, {"raw", "cmc"})
            out.units.close()
