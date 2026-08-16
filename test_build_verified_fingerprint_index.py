from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from build_verified_fingerprint_index import build_verified_index, select_index_records
from catalog_review import append_decision
from category_fingerprint_v1 import CategoryFingerprintDatabaseV1
from dataset_catalog import sha256_file


class VerifiedFingerprintIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.audio = self.root / "bus.wav"
        sr = 22050
        t = np.arange(sr * 4) / sr
        # Several tones create enough stable spectral landmarks for exact-match testing.
        samples = sum(0.12 * np.sin(2 * np.pi * hz * t) for hz in (170, 310, 620, 930))
        wavfile.write(self.audio, sr, (samples * 32767).astype(np.int16))
        self.decisions = self.root / "decisions.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def record(self, **changes):
        record = {
            "asset_id": "asset-bus", "audio_path": str(self.audio),
            "sha256": sha256_file(self.audio), "category": "TRAFFIC", "subtype": "BUS",
            "review_status": "APPROVED", "taxonomy_version": "1.0.0",
            "source_recording_id": "src-bus", "source_uri": "local:bus",
            "license": "TEST_ONLY", "dataset_split": "train",
            "catalog_role": "FINGERPRINT_CANDIDATE",
        }
        record.update(changes)
        return record

    def test_only_approved_train_candidate_is_selected(self):
        append_decision(self.record(), self.decisions)
        append_decision(self.record(asset_id="test", source_uri="local:test", dataset_split="test"), self.decisions)
        append_decision(self.record(asset_id="rejected", source_uri="local:reject", review_status="REJECTED"), self.decisions)
        selected, excluded = select_index_records(self.decisions)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(excluded), 2)

    def test_changed_file_hash_is_rejected(self):
        append_decision(self.record(sha256="b" * 64), self.decisions)
        selected, excluded = select_index_records(self.decisions)
        self.assertEqual(selected, [])
        self.assertIn("SHA-256", excluded[0]["reason"])

    def test_build_and_exact_match(self):
        append_decision(self.record(), self.decisions)
        database_path = self.root / "verified.sqlite3"
        report = build_verified_index(self.decisions, database_path)
        self.assertEqual(report["indexed_count"], 1)
        database = CategoryFingerprintDatabaseV1(database_path)
        match = database.match_file(self.audio)
        self.assertIsNotNone(match)
        self.assertTrue(match.accepted)
        self.assertEqual((match.category, match.subtype), ("TRAFFIC", "BUS"))

    def test_empty_approval_set_does_not_replace_existing_database(self):
        database_path = self.root / "existing.sqlite3"
        database_path.write_bytes(b"keep-me")
        append_decision(self.record(review_status="REJECTED"), self.decisions)
        with self.assertRaises(ValueError):
            build_verified_index(self.decisions, database_path)
        self.assertEqual(database_path.read_bytes(), b"keep-me")


if __name__ == "__main__":
    unittest.main()
