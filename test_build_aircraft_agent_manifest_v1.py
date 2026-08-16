import unittest

from build_aircraft_agent_manifest_v1 import assign_splits


class AircraftAgentManifestTests(unittest.TestCase):
    def test_splits_are_physical_airframe_disjoint(self):
        rows = [
            {"folder": "TYPE_A", "hex_id": code, "output_file": f"{code}.wav"}
            for code in ("A", "B", "C", "D")
        ]
        manifest, coverage = assign_splits(rows)
        by_split = {
            split: {row["physical_airframe_id"] for row in manifest if row["split"] == split}
            for split in ("train", "validation", "test")
        }
        self.assertTrue(by_split["train"].isdisjoint(by_split["validation"]))
        self.assertTrue(by_split["train"].isdisjoint(by_split["test"]))
        self.assertEqual(coverage[0]["agent_training_status"], "PILOT_READY")

    def test_rare_type_is_reference_only(self):
        rows = [
            {"folder": "TYPE_A", "hex_id": code, "output_file": f"{code}.wav"}
            for code in ("A", "B")
        ]
        manifest, coverage = assign_splits(rows)
        self.assertEqual({row["split"] for row in manifest}, {"reference_only"})
        self.assertEqual(coverage[0]["agent_training_status"], "MORE_DATA_REQUIRED")


if __name__ == "__main__":
    unittest.main()
