from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from enrich_aircraft_review_queue import enrich


class EnrichAircraftQueueTests(unittest.TestCase):
    def test_manifest_hint_is_joined_by_source_uri(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "a.wav"
            audio.touch()
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["path", "label", "icao_type", "hex_id", "session", "split", "source_doi", "license"],
                )
                writer.writeheader()
                writer.writerow({
                    "path": str(audio), "label": "AIRBUS_A320", "icao_type": "A320",
                    "hex_id": "ABC", "session": "s", "split": "train",
                    "source_doi": "10.test/x", "license": "TEST_ONLY",
                })
            source_uri = f"doi:10.test/x#{audio.name}"
            queue = root / "queue.jsonl"
            queue.write_text(json.dumps({"source_uri": source_uri}) + "\n", encoding="utf-8")
            output = root / "out.jsonl"
            report = enrich(queue, output, manifest)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["missing_manifest_match"], 0)
            self.assertEqual(row["manifest_category_hint"], "AIRCRAFT")
            self.assertEqual(row["manifest_subtype_hint"], "AIRBUS_A320")


if __name__ == "__main__":
    unittest.main()
