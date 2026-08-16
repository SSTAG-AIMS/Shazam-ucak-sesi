from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_sealed_consensus_manifest import build_manifest
from catalog_review import read_jsonl


class SealedConsensusManifestTests(unittest.TestCase):
    def test_manifest_is_portable_existing_and_source_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sealed.jsonl"
            report = build_manifest(output, per_class_limit=5)
            rows = read_jsonl(output)
            self.assertEqual(report["selected"], len(rows))
            self.assertEqual(len(rows), len({row["source_group"] for row in rows}))
            self.assertTrue(all(Path(row["path"]).is_file() for row in rows))
            self.assertTrue(all(row["benchmark_role"] == "SEALED_TEST_ONLY" for row in rows))
            self.assertLessEqual(max(report["distribution"].values()), 5)


if __name__ == "__main__":
    unittest.main()
