import unittest
from unittest.mock import Mock

from category_fingerprint_v1 import CategoryFingerprintDatabaseV1


class CategoryFingerprintV1Tests(unittest.TestCase):
    def test_accepted_encoded_label_is_split(self):
        raw = Mock(
            aircraft_type="TRAFFIC::BUS",
            reference_name="bus_reference",
            matched_hashes=20,
            aligned_hashes=15,
            query_hashes=30,
            confidence=0.5,
            accepted=True,
        )
        database = CategoryFingerprintDatabaseV1("unused.sqlite3")
        database._database = Mock()
        database._database.match_file.return_value = raw
        database._database.list_references.return_value = [
            ("TRAFFIC::BUS", "bus_reference", 100)
        ]

        match = database.match_file("query.wav")

        self.assertEqual(match.category, "TRAFFIC")
        self.assertEqual(match.subtype, "BUS")
        self.assertTrue(match.accepted)
        self.assertEqual(match.as_subtype_dict()["method"], "shazam_v1")

    def test_rejected_match_is_reported_as_unknown(self):
        raw = Mock(
            aircraft_type="UNKNOWN_AIRCRAFT",
            reference_name="dog_reference",
            matched_hashes=2,
            aligned_hashes=1,
            query_hashes=100,
            confidence=0.01,
            accepted=False,
        )
        database = CategoryFingerprintDatabaseV1("unused.sqlite3")
        database._database = Mock()
        database._database.match_file.return_value = raw
        database._database.list_references.return_value = [
            ("OTHER::DOG", "dog_reference", 100)
        ]

        match = database.match_file("query.wav")

        self.assertFalse(match.accepted)
        self.assertEqual(match.as_subtype_dict()["subtype"], "UNKNOWN_OTHER")


if __name__ == "__main__":
    unittest.main()
