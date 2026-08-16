from __future__ import annotations

import unittest

from multi_model_consensus import ModelVote, WeightedConsensus


def vote(name, label, confidence=0.9, weight=1.0):
    remainder = (1.0 - confidence) / 2
    alternatives = [value for value in ("AIRCRAFT", "TRAFFIC", "OTHER") if value != label]
    return ModelVote(
        name,
        {label: confidence, alternatives[0]: remainder, alternatives[1]: remainder},
        weight=weight,
    )


class MultiModelConsensusTests(unittest.TestCase):
    def setUp(self):
        self.engine = WeightedConsensus()

    def test_four_of_five_high_confidence_is_strong(self):
        votes = [vote(f"m{i}", "AIRCRAFT") for i in range(4)] + [vote("m5", "TRAFFIC")]
        result = self.engine.decide(votes)
        self.assertTrue(result.accepted)
        self.assertEqual(result.label, "AIRCRAFT")
        self.assertEqual(result.strength, "STRONG")

    def test_three_of_five_can_be_normal_not_automatic_truth(self):
        votes = [vote(f"a{i}", "TRAFFIC", 0.95) for i in range(3)]
        votes += [vote("o1", "OTHER", 0.8), vote("o2", "AIRCRAFT", 0.8)]
        result = self.engine.decide(votes)
        self.assertTrue(result.accepted)
        self.assertEqual(result.strength, "NORMAL")
        self.assertAlmostEqual(result.agreement_ratio, 0.6)

    def test_conflicting_models_return_unknown(self):
        votes = [
            vote("m1", "AIRCRAFT"), vote("m2", "AIRCRAFT"),
            vote("m3", "TRAFFIC"), vote("m4", "TRAFFIC"), vote("m5", "OTHER"),
        ]
        result = self.engine.decide(votes)
        self.assertFalse(result.accepted)
        self.assertEqual(result.label, "UNKNOWN_CATEGORY")

    def test_low_confidence_models_abstain(self):
        votes = [
            ModelVote("m1", {"AIRCRAFT": 0.3, "TRAFFIC": 0.35, "OTHER": 0.35}),
            vote("m2", "AIRCRAFT"),
            vote("m3", "AIRCRAFT"),
        ]
        result = self.engine.decide(votes)
        self.assertFalse(result.accepted)
        self.assertIn("m1", result.abstaining_models)
        self.assertIn("INSUFFICIENT_MODELS", result.reasons)

    def test_low_quality_model_can_receive_lower_weight(self):
        votes = [
            vote("efficientnet", "AIRCRAFT", weight=1.0),
            vote("cnn", "AIRCRAFT", weight=1.0),
            vote("svm", "AIRCRAFT", weight=1.0),
            vote("weak_model", "OTHER", weight=0.1),
        ]
        result = self.engine.decide(votes)
        self.assertTrue(result.accepted)
        self.assertEqual(result.label, "AIRCRAFT")


if __name__ == "__main__":
    unittest.main()
