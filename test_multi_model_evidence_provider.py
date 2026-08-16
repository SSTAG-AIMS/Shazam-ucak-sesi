from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset_labeling_agent import LabelEvidence
from multi_model_evidence_provider import MultiModelEvidenceProvider


class FakeSystem:
    eff_model = cnn_model = ml_model = beats_model = object()

    def __init__(self, outputs):
        self.outputs = outputs

    def analyze_for_gui(self, path, *, model_pref, identify_subtype):
        return {"summary": self.outputs[model_pref]}


class FakeClap:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_file(self, path, categories):
        return self.probabilities


class FakeSubtype:
    def analyze(self, path):
        return LabelEvidence("AIRCRAFT", "AIRBUS_A320", 0.9, 0.8, "shazam", True)


class MultiModelEvidenceProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "audio.wav"
        self.path.touch()

    def tearDown(self):
        self.temp.cleanup()

    def provider(self, efficientnet, clap):
        outputs = {
            "efficientnet": efficientnet,
            "cnn": {"WIND": 70, "AIRCRAFT": 30},
            "svm": {"AIRCRAFT": 90, "WIND": 10},
            "beats": {"LOGISTICS": 100},
        }
        return MultiModelEvidenceProvider(
            system=FakeSystem(outputs),
            clap=FakeClap(clap),
            subtype_provider=FakeSubtype(),
        )

    def test_corroborated_aircraft_is_marked_accepted_for_review(self):
        evidence = self.provider(
            {"AIRCRAFT": 80, "OTHER": 20},
            {"AIRCRAFT": 0.7, "OTHER": 0.3},
        ).analyze(self.path)
        self.assertEqual(evidence.category, "AIRCRAFT")
        self.assertTrue(evidence.details["consensus_accepted"])
        self.assertEqual(evidence.subtype, "AIRBUS_A320")
        self.assertIn("verified_fingerprint", evidence.details["corroborators"])
        self.assertTrue(evidence.details["human_approval_required"])

    def test_disagreement_does_not_create_false_consensus(self):
        evidence = self.provider(
            {"TRAFFIC": 70, "OTHER": 30},
            {"AIRCRAFT": 0.8, "TRAFFIC": 0.2},
        ).analyze(self.path)
        self.assertEqual(evidence.category, "TRAFFIC")
        self.assertFalse(evidence.details["consensus_accepted"])
        self.assertIsNone(evidence.subtype)


if __name__ == "__main__":
    unittest.main()
