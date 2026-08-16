import unittest

from build_aircraft_subtype_catalog_v1 import build_catalog


class AircraftSubtypeCatalogTests(unittest.TestCase):
    def test_separates_fingerprint_availability_from_generalization(self):
        rows = []
        for code in ("A", "B", "C"):
            rows.append({"folder": "READY", "hex_id": code, "aircraft_type": "R1", "manufacturer": "M", "model": "X"})
        rows.append({"folder": "REFERENCE_ONLY", "hex_id": "D", "aircraft_type": "R2", "manufacturer": "M", "model": "Y"})
        catalog = build_catalog(rows); by_label = {row["label"]: row for row in catalog["types"]}
        self.assertEqual(catalog["fingerprint_reference_ready"], 2)
        self.assertEqual(catalog["generalization_ready"], 1)
        self.assertEqual(by_label["REFERENCE_ONLY"]["agent_status"], "MORE_AIRFRAMES_REQUIRED")
        self.assertEqual(by_label["REFERENCE_ONLY"]["missing_airframes_for_agent"], 2)


if __name__ == "__main__": unittest.main()
