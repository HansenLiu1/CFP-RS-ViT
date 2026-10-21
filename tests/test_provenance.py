import hashlib
import tempfile
import unittest
from pathlib import Path

from cfra.provenance import sha256_file


class ProvenanceTests(unittest.TestCase):
    def test_sha256_file_streams_consistently(self) -> None:
        payload = b"CFRA checkpoint identity\n" * 17
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(sha256_file(path), expected)
            self.assertEqual(sha256_file(path, chunk_size=7), expected)

    def test_sha256_file_rejects_invalid_chunk_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"payload")
            for value in (0, -1, True):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    sha256_file(path, chunk_size=value)


if __name__ == "__main__":
    unittest.main()
