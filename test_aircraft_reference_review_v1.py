import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from aircraft_reference_intake_v1 import stage_reference
from aircraft_reference_review_v1 import (
    ReferenceReviewError, append_review_decision, build_human_verified_index,
    append_uncertain_decision, pending_intakes, select_approved_references,
)


class AircraftReferenceReviewTests(unittest.TestCase):
    def _stage(self, root: Path, name: str = "sample.wav") -> dict:
        audio = root / name
        sf.write(audio, np.sin(np.linspace(0, 1800, 66150)).astype(np.float32) * .25, 22050)
        return stage_reference(
            audio, aircraft_type="Airbus A321", icao_type="A21N", physical_airframe_id="ABC123",
            source_uri=f"https://example.test/{name}", license_name="CC BY 4.0",
            inbox=root / "inbox", queue=root / "queue.jsonl",
        )

    def test_pending_and_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); record = self._stage(root)
            self.assertEqual(len(pending_intakes(root / "queue.jsonl", root / "decisions.jsonl")), 1)
            append_review_decision(record, approved=True, reviewer="Test User", decisions_path=root / "decisions.jsonl")
            self.assertEqual(pending_intakes(root / "queue.jsonl", root / "decisions.jsonl"), [])
            accepted, excluded = select_approved_references(root / "decisions.jsonl")
            self.assertEqual(len(accepted), 1); self.assertFalse(excluded)
            self.assertEqual(len(list((root / "KABUL_EDILEN").rglob("*.wav"))), 1)
            self.assertTrue((root / "kabul_edilenler.jsonl").is_file())

    def test_quarantine_needs_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); record = self._stage(root); record["intake_status"] = "QUARANTINE"
            with self.assertRaises(ReferenceReviewError):
                append_review_decision(record, approved=True, reviewer="Test", decisions_path=root / "d.jsonl")

    def test_uncertain_is_separate_and_never_indexable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); record = self._stage(root); decisions = root / "decisions.jsonl"
            decision = append_uncertain_decision(record, reviewer="Test User", decisions_path=decisions)
            self.assertEqual(decision["review_status"], "UNCERTAIN")
            self.assertIn("EMIN_OLUNAMAYANLAR", decision["decision_artifact_path"])
            accepted, excluded = select_approved_references(decisions)
            self.assertFalse(accepted); self.assertEqual(len(excluded), 1)
            self.assertTrue((root / "emin_olunamayanlar.jsonl").is_file())

    def test_builds_isolated_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); record = self._stage(root)
            decisions = root / "decisions.jsonl"
            append_review_decision(record, approved=True, reviewer="Test", decisions_path=decisions)
            report = build_human_verified_index(decisions, root / "verified.sqlite3")
            self.assertEqual(report["indexed_count"], 1)
            self.assertTrue((root / "verified.sqlite3").is_file())
            self.assertIn("KABUL_EDILEN", report["indexed"][0]["source_path"])


if __name__ == "__main__":
    unittest.main()
