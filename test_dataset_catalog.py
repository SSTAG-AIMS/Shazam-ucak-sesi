from __future__ import annotations

import unittest

from dataset_catalog import (
    CatalogValidationError,
    ReviewStatus,
    Taxonomy,
    fingerprint_eligibility,
    validate_catalog_record,
)


def valid_record(**changes):
    record = {
        "asset_id": "asset-001",
        "audio_path": "Self_Data/example.wav",
        "sha256": "a" * 64,
        "category": "aircraft",
        "subtype": "airbus-a320",
        "review_status": ReviewStatus.APPROVED.value,
        "taxonomy_version": "1.0.0",
        "source_recording_id": "source-001",
        "source_uri": "doi:10.example/test",
        "license": "CC BY 4.0",
    }
    record.update(changes)
    return record


class DatasetCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = Taxonomy.load()

    def test_taxonomy_has_exactly_two_levels(self):
        self.assertIn("AIRCRAFT", self.taxonomy.categories)
        self.assertIn("AIRBUS_A320", self.taxonomy.categories["AIRCRAFT"])
        self.assertNotIn("V2527_A5", self.taxonomy.categories["AIRCRAFT"])

    def test_record_labels_are_normalized(self):
        result = validate_catalog_record(valid_record(), self.taxonomy)
        self.assertEqual(result["category"], "AIRCRAFT")
        self.assertEqual(result["subtype"], "AIRBUS_A320")

    def test_cross_category_subtype_is_rejected(self):
        with self.assertRaises(CatalogValidationError):
            validate_catalog_record(valid_record(subtype="DOG"), self.taxonomy)

    def test_only_approved_known_subtype_is_indexable(self):
        self.assertEqual(fingerprint_eligibility(valid_record(), self.taxonomy)[0], True)
        self.assertEqual(
            fingerprint_eligibility(
                valid_record(review_status=ReviewStatus.PENDING_REVIEW.value), self.taxonomy
            )[0],
            False,
        )
        self.assertEqual(
            fingerprint_eligibility(
                valid_record(subtype="UNKNOWN_AIRCRAFT"), self.taxonomy
            )[0],
            False,
        )
        self.assertEqual(
            fingerprint_eligibility(
                valid_record(agent_action="QUARANTINE", quality_issues=["TOO_SILENT"]),
                self.taxonomy,
            )[0],
            False,
        )

    def test_invalid_hash_is_rejected(self):
        with self.assertRaises(CatalogValidationError):
            validate_catalog_record(valid_record(sha256="not-a-hash"), self.taxonomy)


if __name__ == "__main__":
    unittest.main()
