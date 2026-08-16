import unittest

from review_comparison import extract_model_predictions, general_result


class ReviewComparisonTests(unittest.TestCase):
    def test_extracts_all_model_rows_in_display_order(self):
        record = {
            "react_trace": [{"phase": "REASON", "evidence": {"details": {"models": {
                "clap": {"predicted": "OTHER", "confidence": 0.6, "role": "CORROBORATOR"},
                "efficientnet": {"predicted": "AIRCRAFT", "confidence": 0.8, "role": "PRIMARY"},
            }}}}]
        }
        rows = extract_model_predictions(record)
        self.assertEqual([row["key"] for row in rows], ["efficientnet", "clap"])
        self.assertEqual(rows[0]["predicted"], "AIRCRAFT")

    def test_general_result_does_not_confuse_model_and_human_approval(self):
        result = general_result({
            "category": "AIRCRAFT", "subtype": "AIRBUS_A320",
            "model_consensus_accepted": True,
        })
        self.assertEqual(result["state"], "MODEL UZLAŞMASI KABUL EDİLDİ")


if __name__ == "__main__":
    unittest.main()
