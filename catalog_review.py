"""Append-only human review decisions for the verified audio catalog."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dataset_catalog import (
    CatalogValidationError,
    ReviewStatus,
    Taxonomy,
    fingerprint_eligibility,
    validate_catalog_record,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CatalogValidationError(
                    f"{path}:{line_number} geçerli JSON değil"
                ) from exc
    return records


def latest_decisions(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        asset_id = str(record.get("asset_id") or "")
        if asset_id:
            latest[asset_id] = record
    return latest


def pending_records(queue_path: Path, decisions_path: Path) -> list[dict[str, Any]]:
    queue = read_jsonl(queue_path)
    decided = latest_decisions(read_jsonl(decisions_path))
    return [record for record in queue if str(record.get("asset_id")) not in decided]


def create_human_decision(
    record: dict[str, Any],
    *,
    reviewer: str,
    status: ReviewStatus | str,
    category: str,
    subtype: str,
    note: str = "",
    taxonomy: Taxonomy | None = None,
) -> dict[str, Any]:
    taxonomy = taxonomy or Taxonomy.load()
    reviewer = reviewer.strip()
    if not reviewer:
        raise CatalogValidationError("İnceleyen kişi boş olamaz")

    if not isinstance(status, ReviewStatus):
        status = ReviewStatus(str(status).strip().upper())
    if status is ReviewStatus.PENDING_REVIEW:
        raise CatalogValidationError("İnsan kararı APPROVED veya REJECTED olmalıdır")

    decision = dict(record)
    decision.update(
        {
            "category": category,
            "subtype": subtype,
            "review_status": status.value,
            "reviewer": reviewer,
            "review_note": note.strip(),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "decision_id": str(uuid.uuid4()),
            "supersedes_status": record.get("review_status"),
        }
    )
    decision = validate_catalog_record(decision, taxonomy)

    if status is ReviewStatus.APPROVED:
        eligible, reason = fingerprint_eligibility(decision, taxonomy)
        if not eligible:
            raise CatalogValidationError(f"Kayıt onaylanamaz: {reason}")
    return decision


def append_decision(decision: dict[str, Any], decisions_path: Path) -> None:
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
