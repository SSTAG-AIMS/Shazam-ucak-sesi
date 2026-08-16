from __future__ import annotations

import unittest

from evaluate_multi_model_consensus import probabilities_from_result


class EvaluateConsensusTests(unittest.TestCase):
    def test_uses_frame_probability_average_when_available(self):
        result = {
            "frame_probs": [[0.8, 0.2], [0.6, 0.4]],
            "class_names": ["AIRCRAFT", "OTHER"],
            "summary": {"AIRCRAFT": 100.0},
        }
        probabilities = probabilities_from_result(result)
        self.assertAlmostEqual(probabilities["AIRCRAFT"], 0.7)
        self.assertAlmostEqual(probabilities["OTHER"], 0.3)

    def test_falls_back_to_window_vote_distribution(self):
        probabilities = probabilities_from_result(
            {"frame_probs": [], "class_names": [], "summary": {"TRAFFIC": 75, "OTHER": 25}}
        )
        self.assertEqual(probabilities, {"TRAFFIC": 0.75, "OTHER": 0.25})


if __name__ == "__main__":
    unittest.main()
