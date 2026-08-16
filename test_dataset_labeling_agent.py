from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from dataset_catalog import ReviewStatus, Taxonomy, fingerprint_eligibility
from dataset_labeling_agent import LabelEvidence, LabelingAgent


class FakeProvider:
    def __init__(self, evidence: LabelEvidence):
        self.evidence = evidence

    def analyze(self, audio_path: Path) -> LabelEvidence:
        return self.evidence


class DatasetLabelingAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audio_path = Path(self.temp_dir.name) / "sample.wav"
        sr = 22050
        time = np.arange(sr * 2) / sr
        samples = (0.25 * np.sin(2 * np.pi * 440 * time) * 32767).astype(np.int16)
        wavfile.write(self.audio_path, sr, samples)

    def tearDown(self):
        self.temp_dir.cleanup()

    def process(self, agent: LabelingAgent):
        return agent.process(
            self.audio_path,
            source_recording_id="source-1",
            source_uri="local:test",
            license_name="TEST_ONLY",
        )

    def test_agent_never_self_approves(self):
        evidence = LabelEvidence("AIRCRAFT", "AIRBUS_A320", 0.99, 0.99, "shazam", True)
        record = self.process(LabelingAgent(evidence_provider=FakeProvider(evidence)))
        self.assertEqual(record["review_status"], ReviewStatus.PENDING_REVIEW.value)
        self.assertEqual(record["agent_action"], "SEND_TO_HUMAN_REVIEW")
        self.assertEqual(record["category_suggested_confidence"], 0.99)
        self.assertEqual(record["subtype_suggested_confidence"], 0.99)
        self.assertEqual(fingerprint_eligibility(record)[0], False)

    def test_invalid_subtype_becomes_category_unknown(self):
        evidence = LabelEvidence("AIRCRAFT", "DOG", 0.8, 0.7, "model")
        record = self.process(LabelingAgent(evidence_provider=FakeProvider(evidence)))
        self.assertEqual(record["subtype"], "UNKNOWN_AIRCRAFT")

    def test_silent_audio_is_quarantined(self):
        wavfile.write(self.audio_path, 22050, np.zeros(44100, dtype=np.int16))
        record = self.process(LabelingAgent())
        self.assertEqual(record["agent_action"], "QUARANTINE")
        self.assertIn("TOO_SILENT", record["quality_issues"])
        self.assertEqual(record["review_status"], ReviewStatus.PENDING_REVIEW.value)

    def test_trace_contains_react_phases(self):
        record = self.process(LabelingAgent())
        self.assertEqual(
            [step["phase"] for step in record["react_trace"]],
            ["OBSERVE", "REASON", "ACT"],
        )


if __name__ == "__main__":
    unittest.main()
