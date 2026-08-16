from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from catalog_review import (
    append_decision,
    create_human_decision,
    pending_records,
)
from dataset_catalog import CatalogValidationError, ReviewStatus


def pending_record(**changes):
    record = {
        "asset_id": "asset-1",
        "audio_path": "sample.wav",
        "sha256": "a" * 64,
        "category": "AIRCRAFT",
        "subtype": "AIRBUS_A320",
        "review_status": "PENDING_REVIEW",
        "taxonomy_version": "1.0.0",
        "source_recording_id": "source-1",
        "source_uri": "local:test",
        "license": "TEST_ONLY",
    }
    record.update(changes)
    return record


class CatalogReviewTests(unittest.TestCase):
    def test_human_can_approve_known_subtype(self):
        decision = create_human_decision(
            pending_record(), reviewer="Reviewer One", status="APPROVED",
            category="AIRCRAFT", subtype="AIRBUS_A320"
        )
        self.assertEqual(decision["review_status"], "APPROVED")
        self.assertEqual(decision["reviewer"], "Reviewer One")
        self.assertIn("decision_id", decision)

    def test_unknown_subtype_cannot_be_approved(self):
        with self.assertRaises(CatalogValidationError):
            create_human_decision(
                pending_record(), reviewer="Reviewer", status="APPROVED",
                category="AIRCRAFT", subtype="UNKNOWN_AIRCRAFT"
            )

    def test_rejection_allows_unknown_subtype(self):
        decision = create_human_decision(
            pending_record(), reviewer="Reviewer", status="REJECTED",
            category="AIRCRAFT", subtype="UNKNOWN_AIRCRAFT", note="Belirsiz kayıt"
        )
        self.assertEqual(decision["review_status"], "REJECTED")

    def test_quarantined_record_cannot_be_approved(self):
        with self.assertRaises(CatalogValidationError):
            create_human_decision(
                pending_record(agent_action="QUARANTINE", quality_issues=["TOO_SILENT"]),
                reviewer="Reviewer",
                status="APPROVED",
                category="AIRCRAFT",
                subtype="AIRBUS_A320",
            )

    def test_reviewer_is_required(self):
        with self.assertRaises(CatalogValidationError):
            create_human_decision(
                pending_record(), reviewer="", status="APPROVED",
                category="AIRCRAFT", subtype="AIRBUS_A320"
            )

    def test_decided_asset_leaves_pending_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.jsonl"
            decisions = Path(directory) / "decisions.jsonl"
            append_decision(pending_record(), queue)
            self.assertEqual(len(pending_records(queue, decisions)), 1)
            decision = create_human_decision(
                pending_record(), reviewer="Reviewer", status=ReviewStatus.REJECTED,
                category="AIRCRAFT", subtype="AIRBUS_A320"
            )
            append_decision(decision, decisions)
            self.assertEqual(pending_records(queue, decisions), [])


if __name__ == "__main__":
    unittest.main()
