"""Build an isolated Shazam index from human-approved catalog decisions only."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from catalog_review import latest_decisions, read_jsonl
from category_fingerprint_v1 import CategoryFingerprintDatabaseV1
from dataset_catalog import Taxonomy, fingerprint_eligibility, sha256_file


ROOT = Path(__file__).resolve().parent
DEFAULT_DECISIONS = ROOT / "cache" / "catalog_review_decisions_v1.jsonl"
DEFAULT_DATABASE = ROOT / "models" / "verified_catalog_fingerprints_v1.sqlite3"


def select_index_records(
    decisions_path: Path,
    taxonomy: Taxonomy | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    taxonomy = taxonomy or Taxonomy.load()
    latest = latest_decisions(read_jsonl(decisions_path))
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    seen_hashes: set[str] = set()

    for asset_id, record in latest.items():
        eligible, reason = fingerprint_eligibility(record, taxonomy)
        if not eligible:
            excluded.append({"asset_id": asset_id, "reason": reason})
            continue
        if str(record.get("dataset_split", "train")).lower() != "train":
            excluded.append({"asset_id": asset_id, "reason": "Benchmark/validation kaydı indekslenemez"})
            continue
        if record.get("catalog_role", "FINGERPRINT_CANDIDATE") != "FINGERPRINT_CANDIDATE":
            excluded.append({"asset_id": asset_id, "reason": "Katalog rolü indekslemeye uygun değil"})
            continue
        audio_path = Path(str(record["audio_path"]))
        if not audio_path.is_file():
            excluded.append({"asset_id": asset_id, "reason": "Ses dosyası bulunamadı"})
            continue
        actual_hash = sha256_file(audio_path)
        if actual_hash != record["sha256"]:
            excluded.append({"asset_id": asset_id, "reason": "Dosya SHA-256 değeri değişmiş"})
            continue
        if actual_hash in seen_hashes:
            excluded.append({"asset_id": asset_id, "reason": "Aynı sesin mükerrer SHA-256 kaydı"})
            continue
        seen_hashes.add(actual_hash)
        accepted.append(record)
    return accepted, excluded


def build_verified_index(
    decisions_path: Path,
    database_path: Path,
    *,
    taxonomy: Taxonomy | None = None,
) -> dict[str, Any]:
    taxonomy = taxonomy or Taxonomy.load()
    records, excluded = select_index_records(decisions_path, taxonomy)
    if not records:
        raise ValueError("İndekslenecek insan onaylı kayıt bulunamadı; mevcut veritabanına dokunulmadı")

    database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_suffix(database_path.suffix + ".building")
    if temporary_path.exists():
        temporary_path.unlink()

    database = CategoryFingerprintDatabaseV1(temporary_path)
    database.reset()
    indexed = []
    try:
        for record in records:
            hash_count = database.add_reference(
                record["audio_path"], record["category"], record["subtype"]
            )
            indexed.append(
                {
                    "asset_id": record["asset_id"],
                    "category": record["category"],
                    "subtype": record["subtype"],
                    "hash_count": hash_count,
                }
            )
        database.compact()
        os.replace(temporary_path, database_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    distribution = Counter((row["category"], row["subtype"]) for row in indexed)
    report = {
        "taxonomy_version": taxonomy.version,
        "database": str(database_path),
        "indexed_count": len(indexed),
        "excluded_count": len(excluded),
        "distribution": {
            f"{category}::{subtype}": count
            for (category, subtype), count in sorted(distribution.items())
        },
        "indexed": indexed,
        "excluded": excluded,
    }
    report_path = database_path.with_suffix(database_path.suffix + ".manifest.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Onaylı kayıtlardan izole Shazam indeksi oluştur")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    report = build_verified_index(args.decisions, args.database)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
