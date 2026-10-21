import json
import tempfile
import unittest
from pathlib import Path

from cfra.runs import RunInitializationError, create_run


class RunTests(unittest.TestCase):
    def test_run_is_created_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            (project / "artifacts").mkdir(parents=True)
            config = {"execution": {"allow_resume": False}}
            run = create_run(
                project_root=project,
                run_id="development-test",
                evidence_tier="development",
                resolved_config=config,
                resolved_config_sha256="abc123",
            )

            manifest = json.loads((run / "manifest" / "run.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["allow_resume"])
            self.assertEqual((run / "STATUS").read_text(encoding="ascii").strip(), "running")

            with self.assertRaises(RunInitializationError):
                create_run(
                    project_root=project,
                    run_id="development-test",
                    evidence_tier="development",
                    resolved_config=config,
                    resolved_config_sha256="abc123",
                )

    def test_run_id_cannot_escape_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            for run_id in ("../escape", "a/b", "", " has-space"):
                with self.subTest(run_id=run_id), self.assertRaises(RunInitializationError):
                    create_run(
                        project_root=project,
                        run_id=run_id,
                        evidence_tier="development",
                        resolved_config={},
                        resolved_config_sha256="abc123",
                    )


if __name__ == "__main__":
    unittest.main()
