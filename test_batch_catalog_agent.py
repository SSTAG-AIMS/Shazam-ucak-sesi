from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from batch_catalog_agent import (
    build_review_queue,
    load_aircraft_candidates,
    select_one_existing_per_subtype,
)
from catalog_review import read_jsonl


class BatchCatalogAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.audio = self.root / "aircraft.wav"
        sr = 22050
        t = np.arange(sr * 2) / sr
        wavfile.write(self.audio, sr, (0.2 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16))
        self.manifest = self.root / "manifest.csv"
        fields = ["path", "label", "icao_type", "hex_id", "session", "split", "source_doi", "license"]
        with self.manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            base = {
                "path": str(self.audio), "label": "AIRBUS_A320", "icao_type": "A320",
                "hex_id": "ABC123", "session": "session-1",
                "source_doi": "10.example/test", "license": "TEST_ONLY",
            }
            writer.writerow({**base, "split": "train"})
            writer.writerow({**base, "path": str(self.root / "test.wav"), "split": "test"})

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_candidate_loading_excludes_test(self):
        candidates = load_aircraft_candidates(self.manifest)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["dataset_split"], "train")

    def test_batch_run_is_resumable_without_duplicates(self):
        output = self.root / "queue.jsonl"
        candidates = load_aircraft_candidates(self.manifest)
        first = build_review_queue(candidates, output)
        second = build_review_queue(candidates, output)
        self.assertEqual(first["stats"]["added"], 1)
        self.assertEqual(second["stats"]["skipped_existing"], 1)
        self.assertEqual(len(read_jsonl(output)), 1)

    def test_output_preserves_split_and_airframe_provenance(self):
        output = self.root / "queue.jsonl"
        build_review_queue(load_aircraft_candidates(self.manifest), output)
        record = read_jsonl(output)[0]
        self.assertEqual(record["dataset_split"], "train")
        self.assertEqual(record["physical_airframe_id"], "ABC123")
        self.assertEqual(record["catalog_role"], "FINGERPRINT_CANDIDATE")
        self.assertEqual(record["manifest_category_hint"], "AIRCRAFT")
        self.assertEqual(record["manifest_subtype_hint"], "AIRBUS_A320")

    def test_stratified_selector_keeps_one_existing_file_per_subtype(self):
        candidates = load_aircraft_candidates(self.manifest)
        duplicate = dict(candidates[0])
        duplicate["source_uri"] = "local:duplicate"
        selected = select_one_existing_per_subtype([*candidates, duplicate])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["subtype_hint"], "AIRBUS_A320")


if __name__ == "__main__":
    unittest.main()
