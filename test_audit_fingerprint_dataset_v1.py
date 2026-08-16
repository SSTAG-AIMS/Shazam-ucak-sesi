import unittest

from audit_fingerprint_dataset_v1 import build_report


class FingerprintDatasetAuditTests(unittest.TestCase):
    def test_report_keeps_candidates_separate_from_approved_references(self):
        report = build_report()
        aircraft = report["categories"]["AIRCRAFT"]
        self.assertGreaterEqual(aircraft["candidate_audio_count"], 100)
        self.assertGreaterEqual(aircraft["candidate_subtype_count"], 40)
        self.assertFalse(aircraft["direct_index_ready"])
        self.assertIn("isolated_demo_approved_count", aircraft)

    def test_generic_split_sources_do_not_leak(self):
        report = build_report()
        for category in ("TRAFFIC", "OTHER"):
            self.assertEqual(report["categories"][category]["cross_split_source_leakage_count"], 0)


if __name__ == "__main__":
    unittest.main()
