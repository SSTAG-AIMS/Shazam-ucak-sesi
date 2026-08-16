"""Human review and isolated promotion for aircraft reference intake records."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aircraft_fingerprint import AircraftFingerprintDatabase
from aircraft_reference_intake_v1 import DEFAULT_QUEUE, read_jsonl
from dataset_catalog import normalize_label, sha256_file


ROOT = Path(__file__).resolve().parent
DEFAULT_DECISIONS = ROOT / "cache" / "aircraft_reference_intake_decisions_v1.jsonl"
DEFAULT_DATABASE = ROOT / "models" / "aircraft_human_verified_v1.sqlite3"


class ReferenceReviewError(ValueError):
    pass


def latest_decisions(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["intake_id"]): row for row in read_jsonl(path) if row.get("intake_id")}


def pending_intakes(queue_path: Path = DEFAULT_QUEUE, decisions_path: Path = DEFAULT_DECISIONS) -> list[dict]:
    decided = latest_decisions(decisions_path)
    return [row for row in read_jsonl(queue_path) if str(row.get("intake_id")) not in decided]


def append_review_decision(
    intake: dict,
    *,
    approved: bool,
    reviewer: str,
    subtype: str | None = None,
    note: str = "",
    decisions_path: Path = DEFAULT_DECISIONS,
    allow_quarantine_override: bool = False,
) -> dict:
    reviewer = reviewer.strip()
    if not reviewer:
        raise ReferenceReviewError("İnceleyen kişinin adı zorunludur")
    if not intake.get("intake_id"):
        raise ReferenceReviewError("Geçersiz intake kaydı")
    if approved and intake.get("intake_status") == "QUARANTINE" and not allow_quarantine_override:
        raise ReferenceReviewError("Karantinadaki kayıt normal onayla kabul edilemez")
    final_subtype = normalize_label(subtype or str(intake.get("proposed_subtype", "")))
    if approved and not final_subtype:
        raise ReferenceReviewError("Onaylanan uçak alt türü zorunludur")
    decision = {
        **intake,
        "review_status": "APPROVED" if approved else "REJECTED",
        "human_approved": bool(approved),
        "approved_subtype": final_subtype,
        "reviewer": reviewer,
        "review_note": note.strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "quarantine_override": bool(allow_quarantine_override),
        "fingerprint_indexed": False,
    }
    workspace = decisions_path.parent
    bucket_name = "KABUL_EDILEN" if approved else "REDDEDILEN"
    bucket = workspace / bucket_name / final_subtype
    bucket.mkdir(parents=True, exist_ok=True)
    source_audio = Path(str(intake["audio_path"]))
    artifact = bucket / f"{str(intake['intake_id'])[:8]}_{source_audio.name}"
    if source_audio.is_file() and not artifact.exists():
        shutil.copy2(source_audio, artifact)
    decision["decision_artifact_path"] = str(artifact.resolve())
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
    audit_path = workspace / ("kabul_edilenler.jsonl" if approved else "reddedilenler.jsonl")
    with audit_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
    return decision


def append_uncertain_decision(
    intake: dict,
    *,
    reviewer: str,
    subtype: str | None = None,
    note: str = "",
    decisions_path: Path = DEFAULT_DECISIONS,
) -> dict:
    """Record an inconclusive review without indexing it into Shazam."""
    reviewer = reviewer.strip()
    if not reviewer:
        raise ReferenceReviewError("İnceleyen kişinin adı zorunludur")
    if not intake.get("intake_id"):
        raise ReferenceReviewError("Geçersiz intake kaydı")
    final_subtype = normalize_label(subtype or str(intake.get("proposed_subtype", "")))
    decision = {
        **intake,
        "review_status": "UNCERTAIN", "human_approved": False,
        "approved_subtype": final_subtype, "reviewer": reviewer,
        "review_note": note.strip(), "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "quarantine_override": False, "fingerprint_indexed": False,
    }
    workspace = decisions_path.parent
    bucket = workspace / "EMIN_OLUNAMAYANLAR" / (final_subtype or "UNKNOWN")
    bucket.mkdir(parents=True, exist_ok=True)
    source_audio = Path(str(intake["audio_path"]))
    artifact = bucket / f"{str(intake['intake_id'])[:8]}_{source_audio.name}"
    if source_audio.is_file() and not artifact.exists():
        shutil.copy2(source_audio, artifact)
    decision["decision_artifact_path"] = str(artifact.resolve())
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    for audit_path in (decisions_path, workspace / "emin_olunamayanlar.jsonl"):
        with audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
    return decision


def select_approved_references(decisions_path: Path = DEFAULT_DECISIONS) -> tuple[list[dict], list[dict]]:
    approved: list[dict] = []
    excluded: list[dict] = []
    hashes: set[str] = set()
    for intake_id, row in latest_decisions(decisions_path).items():
        reason = ""
        path = Path(str(row.get("decision_artifact_path") or row.get("audio_path", "")))
        if row.get("review_status") != "APPROVED" or not row.get("human_approved"):
            reason = "İnsan tarafından onaylanmadı"
        elif not path.is_file():
            reason = "Ses dosyası bulunamadı"
        elif sha256_file(path) != row.get("sha256"):
            reason = "Dosya SHA-256 değeri değişti"
        elif row.get("sha256") in hashes:
            reason = "Aynı sesin mükerrer kaydı"
        elif not row.get("approved_subtype"):
            reason = "Onaylı alt tür yok"
        if reason:
            excluded.append({"intake_id": intake_id, "reason": reason})
        else:
            hashes.add(str(row["sha256"]))
            approved.append({**row, "index_audio_path": str(path.resolve())})
    return approved, excluded


def build_human_verified_index(
    decisions_path: Path = DEFAULT_DECISIONS,
    database_path: Path = DEFAULT_DATABASE,
) -> dict:
    records, excluded = select_approved_references(decisions_path)
    if not records:
        raise ReferenceReviewError("İndekslenecek insan onaylı referans yok; mevcut indeks korunuyor")
    database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = database_path.with_suffix(database_path.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    db = AircraftFingerprintDatabase(temporary)
    db.reset()
    indexed = []
    try:
        for row in records:
            count = db.add_reference(row["index_audio_path"], row["approved_subtype"])
            indexed.append({
                "intake_id": row["intake_id"], "subtype": row["approved_subtype"],
                "source_path": row["index_audio_path"], "hash_count": count,
            })
        db.compact()
        os.replace(temporary, database_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    report = {
        "database": str(database_path),
        "indexed_count": len(indexed),
        "excluded_count": len(excluded),
        "indexed": indexed,
        "excluded": excluded,
    }
    database_path.with_suffix(database_path.suffix + ".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="İnsan onaylı uçak referanslarından izole Shazam indeksi üret")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    print(json.dumps(build_human_verified_index(args.decisions, args.database), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
