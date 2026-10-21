import csv
import json
import tempfile
import unittest
from pathlib import Path

from cfra.configuration import config_sha256
from cfra.provenance import sha256_json
from cfra.reporting import ReleaseError, audit_run, create_release, generate_report, validate_release
from cfra.runs import create_run, update_run_status


class ReportingTests(unittest.TestCase):
    def _make_run(self, project: Path, *, evidence_tier: str = "formal") -> tuple[Path, dict[str, object]]:
        config: dict[str, object] = {
            "run": {"evidence_tier": evidence_tier, "frozen": True},
            "release": {
                "status": "frozen",
                "aggregation_policy": "frozen_manifest_only",
                "run_groups": ["core"],
            },
            "scope": {"experiment_ids": ["E00", "E01", "E02", "E03", "E04"]},
            "statistics": {"cluster_unit": "source_image", "confidence_level": 0.95},
        }
        run = create_run(
            project_root=project,
            run_id=f"{evidence_tier}-synthetic",
            evidence_tier=evidence_tier,
            resolved_config=config,
            resolved_config_sha256=config_sha256(config),
        )
        (run / "manifest" / "e00.json").write_text(
            json.dumps({"stage": "E00", "config_sha256": config_sha256(config), "datasets": {"synthetic": {"root": "raw"}}}) + "\n",
            encoding="utf-8",
        )
        (run / "manifest" / "e01.json").write_text(
            json.dumps({
                "stage": "E01",
                "source_image_disjoint": True,
                "content_group_disjoint": True,
                "role_overlap": {},
                "role_overlap_count": 0,
                "content_group_overlap": {},
                "content_group_overlap_count": 0,
            })
            + "\n",
            encoding="utf-8",
        )
        (run / "manifest" / "e02.json").write_text(
            json.dumps({
                "stage": "E02",
                "all_pixel_invariants_pass": True,
                "all_registered_controls_available": True,
                "failure_count": 0,
                "control_failure_count": 0,
                "row_count": 2,
            })
            + "\n",
            encoding="utf-8",
        )
        (run / "manifest" / "source_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_manifest": "synthetic-source",
                    "code": {"git_commit": "synthetic", "dirty_status": "clean"},
                    "environment": {"python_version": "test", "platform": "test"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run / "metrics" / "results.json").write_text(
            json.dumps(
                {
                    "E03": [{"metric": "macro_ovr_auc", "estimate": 0.75, "status": "completed"}],
                    "E04": [{"metric": "own_argmax_rate", "estimate": 0.5, "status": "completed"}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run / "statistics" / "results.json").write_text(
            json.dumps({"source_image_count": 2, "confidence_level": 0.95, "ci_low": 0.1, "ci_high": 0.9}) + "\n",
            encoding="utf-8",
        )
        update_run_status(run, "completed")
        return run, config

    def test_report_lineage_and_audit_are_generated_from_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            run, _ = self._make_run(project)
            result = generate_report(run)
            self.assertEqual(result["record_count"], 2)
            self.assertTrue((run / "tables" / "summary.csv").is_file())
            self.assertTrue((run / "figures" / "metric_source.csv").is_file())
            self.assertTrue((run / "figures" / "workflow.svg").read_text(encoding="utf-8").startswith("<?xml"))
            with (run / "tables" / "summary.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["experiment_id"] for row in rows}, {"E03", "E04"})
            self.assertTrue(all(row["source_sha256"] and row["lineage_sha256"] for row in rows))

            audit = audit_run(run, required_experiments=["E00", "E01", "E02", "E03", "E04"])
            self.assertTrue(audit["passed"], audit)
            self.assertEqual((run / "STATUS").read_text(encoding="ascii").strip(), "audited")

    def test_development_run_is_rejected_by_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            run, config = self._make_run(project, evidence_tier="development")
            generate_report(run)
            audit = audit_run(run, required_experiments=["E00", "E01", "E02", "E03", "E04"])
            self.assertTrue(audit["passed"], audit)
            groups = {
                "pre_registered_at": "2026-08-18T00:00:00+00:00",
                "run_groups": [{"group_id": "core", "run_id": run.name, "experiments": ["E00", "E01", "E02", "E03", "E04"]}],
            }
            groups["manifest_hash"] = sha256_json(groups)
            with self.assertRaises(ReleaseError):
                create_release(project, "development-release", run_group_manifest=groups, config=config)

    def test_release_validates_audited_formal_run_and_detects_lineage_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            run, config = self._make_run(project)
            generate_report(run)
            audit = audit_run(run, required_experiments=["E00", "E01", "E02", "E03", "E04"])
            self.assertTrue(audit["passed"], audit)
            groups = {
                "pre_registered_at": "2026-08-18T00:00:00+00:00",
                "run_groups": [{"group_id": "core", "run_id": run.name, "experiments": ["E00", "E01", "E02", "E03", "E04"]}],
            }
            groups["manifest_hash"] = sha256_json(groups)
            release_path = create_release(project, "synthetic-release", run_group_manifest=groups, config=config)
            validation = validate_release(release_path, project_root=project, config=config)
            self.assertTrue(validation["valid"])

            source = run / "metrics" / "results.json"
            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ReleaseError):
                validate_release(release_path, project_root=project, config=config)

    def test_audit_refuses_skipped_interface_only_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            run, _ = self._make_run(project)
            (run / "metrics" / "results.json").write_text(
                json.dumps(
                    {
                        "E06": {
                            "status": "skipped",
                            "configured_enabled": True,
                            "interface_called": True,
                            "layer_output_status": "skipped",
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            generate_report(run)
            audit = audit_run(run, required_experiments=["E00", "E01", "E02", "E03", "E04", "E06"])
            self.assertFalse(audit["passed"])
            self.assertIn("E06", " ".join(item["details"] for item in audit["checks"] if not item["passed"]))

    def test_frozen_group_manifest_requires_all_seed_and_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            config = {
                "run": {"evidence_tier": "formal", "frozen": True},
                "release": {
                    "status": "frozen",
                    "aggregation_policy": "frozen_manifest_only",
                    "run_groups": ["core", "retrieval"],
                },
                "run_groups": {
                    "core": {"experiments": ["E00", "E01", "E02"], "split_seeds": [7, 11], "scientific_dependencies": []},
                    "retrieval": {"experiments": ["E00", "E01", "E02", "E08"], "split_seeds": [7, 11], "scientific_dependencies": ["core"]},
                },
                "scope": {"experiment_ids": ["E00", "E01", "E02", "E08"]},
            }
            group_manifest = {
                "schema_version": 1,
                "manifest_type": "formal_run_group",
                "group_id": "retrieval",
                "status": "audited",
                "audit_status": "audited",
                "config_sha256": config_sha256(config),
                "experiments": ["E00", "E01", "E02", "E08"],
                "dependencies": ["core"],
                "pre_registered_at": "2026-08-18T00:00:00+00:00",
                "seeds": [
                    {"seed": 7, "run_id": "formal-retrieval-seed-7", "run_path": "artifacts/runs/formal-retrieval-seed-7", "config_sha256": config_sha256(config), "status": "audited", "audit_status": "audited"}
                ],
            }
            group_manifest["manifest_hash"] = sha256_json(group_manifest)
            with self.assertRaises(ReleaseError):
                create_release(project, "missing-seed-group", run_group_manifest=group_manifest, config=config)

    def test_frozen_group_manifest_rejects_config_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            run, run_config = self._make_run(project)
            run_manifest_path = run / "manifest" / "run.json"
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            run_manifest["seed"] = 7
            run_manifest_path.write_text(json.dumps(run_manifest) + "\n", encoding="utf-8")
            generate_report(run)
            audit = audit_run(run, required_experiments=["E00", "E01", "E02", "E03", "E04"])
            self.assertTrue(audit["passed"], audit)
            config = dict(run_config)
            config["release"] = dict(run_config["release"])  # type: ignore[index]
            config["release"]["run_groups"] = ["core"]  # type: ignore[index]
            config["run_groups"] = {
                "core": {
                    "experiments": ["E00", "E01", "E02", "E03", "E04"],
                    "split_seeds": [7],
                    "scientific_dependencies": [],
                }
            }
            group_manifest = {
                "schema_version": 1,
                "manifest_type": "formal_run_group",
                "group_id": "core",
                "status": "audited",
                "audit_status": "audited",
                "config_sha256": "0" * 64,
                "experiments": ["E00", "E01", "E02", "E03", "E04"],
                "dependencies": [],
                "pre_registered_at": "2026-08-18T00:00:00+00:00",
                "seeds": [],
            }
            group_manifest["seeds"] = [
                {
                    "seed": 7,
                    "run_id": run.name,
                    "run_path": run.relative_to(project).as_posix(),
                    "config_sha256": config_sha256(run_config),
                    "status": "audited",
                    "audit_status": "audited",
                }
            ]
            group_manifest["manifest_hash"] = sha256_json(group_manifest)
            with self.assertRaisesRegex(ReleaseError, "config hash"):
                create_release(project, "group-hash-mismatch", run_group_manifest=group_manifest, config=config)


if __name__ == "__main__":
    unittest.main()
