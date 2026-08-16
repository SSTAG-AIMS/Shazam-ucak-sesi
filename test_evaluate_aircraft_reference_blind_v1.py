import unittest

from evaluate_aircraft_reference_blind_v1 import split_by_physical_airframe


class BlindAircraftSplitTests(unittest.TestCase):
    def test_heldout_airframe_never_appears_in_training(self):
        rows = [
            {"folder": "A", "hex_id": code, "source_file": f"{code}.wav", "output_file": f"{code}.wav"}
            for code in ("1", "2", "3")
        ]
        train, test, insufficient = split_by_physical_airframe(rows)
        self.assertFalse(insufficient)
        self.assertEqual(len(test), 1)
        heldout = test[0]["hex_id"]
        self.assertNotIn(heldout, {row["hex_id"] for row in train})

    def test_rare_type_is_not_claimed_as_blind_test(self):
        rows = [
            {"folder": "A", "hex_id": code, "source_file": f"{code}.wav", "output_file": f"{code}.wav"}
            for code in ("1", "2")
        ]
        train, test, insufficient = split_by_physical_airframe(rows)
        self.assertEqual(len(train), 2)
        self.assertFalse(test)
        self.assertEqual(insufficient[0]["physical_airframes"], 2)


if __name__ == "__main__":
    unittest.main()
