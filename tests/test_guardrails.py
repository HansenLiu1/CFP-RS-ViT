import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfra.guardrails import CleanRoomViolation, assert_clean_room_paths


class GuardrailTests(unittest.TestCase):
    def test_rejects_configured_legacy_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "retired" / "project"
            candidate = legacy / "outputs" / "x.csv"
            with mock.patch.dict(os.environ, {"CFRA_LEGACY_ROOTS": str(legacy)}):
                with self.assertRaises(CleanRoomViolation):
                    assert_clean_room_paths([candidate], root)

    def test_allows_unrelated_raw_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_data = root / "ExternalData" / "DOTA"
            assert_clean_room_paths([raw_data], root)

    def test_does_not_reject_a_prefix_only_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "retired"
            independent = root / "retired-independent"
            with mock.patch.dict(os.environ, {"CFRA_LEGACY_ROOTS": str(legacy)}):
                assert_clean_room_paths([independent], root)


if __name__ == "__main__":
    unittest.main()
