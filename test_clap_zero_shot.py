import unittest

from clap_zero_shot import aggregate_prompt_probabilities, category_prompt_batch


class ClapZeroShotHelpersTest(unittest.TestCase):
    def test_prompt_batch_keeps_category_mapping(self):
        labels, prompts = category_prompt_batch(["AIRCRAFT", "TRAFFIC"])
        self.assertEqual(len(labels), len(prompts))
        self.assertEqual(labels.count("AIRCRAFT"), 2)
        self.assertEqual(labels.count("TRAFFIC"), 2)

    def test_prompt_probabilities_are_aggregated_and_normalized(self):
        result = aggregate_prompt_probabilities(
            ["AIRCRAFT", "AIRCRAFT", "TRAFFIC", "TRAFFIC"],
            [0.4, 0.2, 0.1, 0.1],
        )
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertGreater(result["AIRCRAFT"], result["TRAFFIC"])


if __name__ == "__main__":
    unittest.main()
