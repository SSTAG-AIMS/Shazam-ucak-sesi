"""Build the full non-aircraft Shazam catalogue without test leakage."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from category_fingerprint_v1 import CategoryFingerprintDatabaseV1


ROOT = Path(__file__).resolve().parent
BASE_MANIFEST = ROOT / "cache" / "category_subtypes_350.csv"
ESC_META = ROOT / "downloads" / "ESC-50" / "ESC-50-master" / "meta" / "esc50.csv"
ESC_AUDIO = ESC_META.parent.parent / "audio"
LOGISTICS = ROOT / "Self_Data" / "LOGISTICS"
OUTPUT_MANIFEST = ROOT / "cache" / "category_shazam_extended_v1.csv"
DATABASE = ROOT / "models" / "category_fingerprints_full_v1.sqlite3"
REPORT = ROOT / "outputs" / "category_shazam_extended_v1_report.json"

ESC_LABELS = {
    "AMBIENT": {
        "clock_tick", "door_wood_creaks", "keyboard_typing",
        "washing_machine", "vacuum_cleaner",
    },
    "SPEECH": {"clapping", "laughing", "crying_baby", "crowd", "footsteps"},
    "WIND": {"wind", "rain", "thunderstorm", "sea_waves"},
}


def _base_rows() -> list[dict[str, str]]:
    with BASE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if row["split"].strip().lower() != "train":
                continue
            rows.append({
                "path": row["path"],
                "category": row["category"].strip().upper(),
                "subtype": row["subtype"].strip().upper(),
                "split": "train",
                "source_group": row["source_group"],
                "source_dataset": row["source_dataset"],
                "license": row["license"],
            })
    return rows


def _esc_rows() -> list[dict[str, str]]:
    category_by_subtype = {
        subtype: category
        for category, subtypes in ESC_LABELS.items()
        for subtype in subtypes
    }
    rows = []
    with ESC_META.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            subtype = row["category"]
            category = category_by_subtype.get(subtype)
            if category is None:
                continue
            fold = int(row["fold"])
            split = "train" if fold <= 3 else "validation" if fold == 4 else "test"
            rows.append({
                "path": str((ESC_AUDIO / row["filename"]).resolve()),
                "category": category,
                "subtype": subtype.upper(),
                "split": split,
                "source_group": row["src_file"],
                "source_dataset": "ESC-50",
                "license": "CC BY-NC 3.0",
            })
    return rows


def _logistics_rows() -> list[dict[str, str]]:
    # build_logistics_manifest.py writes chunks in sorted parent-file order.
    parents = [
        ("derrickmckinnon-bus-at-bus-stop-241622", "BUS_STOP", 8, "train"),
        ("freesound_community-diesel-tractor-2-55064", "DIESEL_TRACTOR", 15, "train"),
        ("soundreality-tractor-work-306473", "TRACTOR_WORK", 12, "test"),
    ]
    rows = []
    index = 1
    for source_group, subtype, count, split in parents:
        for _ in range(count):
            rows.append({
                "path": str((LOGISTICS / "chunks" / f"logistics_{index:04d}.wav").resolve()),
                "category": "LOGISTICS",
                "subtype": subtype,
                "split": split,
                "source_group": source_group,
                "source_dataset": "Freesound source recordings",
                "license": "See original source attribution in filename/source page",
            })
            index += 1
    return rows


def build() -> dict:
    rows = _base_rows() + _esc_rows() + _logistics_rows()
    missing = [row["path"] for row in rows if not Path(row["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest dosyası bulunamadı: {missing[:3]}")

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    training = [row for row in rows if row["split"] == "train"]
    temporary = DATABASE.with_suffix(".sqlite3.building")
    if temporary.exists():
        temporary.unlink()
    database = CategoryFingerprintDatabaseV1(temporary)
    database.reset()
    hash_counts: Counter[str] = Counter()
    clip_counts: Counter[str] = Counter()
    for index, row in enumerate(training, 1):
        count = database.add_reference(row["path"], row["category"], row["subtype"])
        hash_counts[row["category"]] += count
        clip_counts[row["category"]] += 1
        if index % 100 == 0 or index == len(training):
            print(f"[Category Shazam] {index}/{len(training)}", flush=True)
    database.compact()

    with closing(sqlite3.connect(temporary)) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        orphan_count = connection.execute(
            "SELECT COUNT(*) FROM fingerprints f LEFT JOIN tracks t "
            "ON t.id=f.track_id WHERE t.id IS NULL"
        ).fetchone()[0]
    if integrity != "ok" or orphan_count:
        raise RuntimeError(f"SQLite doğrulaması başarısız: {integrity}, orphan={orphan_count}")
    os.replace(temporary, DATABASE)

    report = {
        "database": str(DATABASE),
        "manifest": str(OUTPUT_MANIFEST),
        "training_clips": dict(sorted(clip_counts.items())),
        "fingerprints": dict(sorted(hash_counts.items())),
        "total_training_clips": sum(clip_counts.values()),
        "total_fingerprints": sum(hash_counts.values()),
        "reserved_validation_clips": sum(row["split"] == "validation" for row in rows),
        "reserved_test_clips": sum(row["split"] == "test" for row in rows),
        "sqlite_integrity": integrity,
        "orphan_fingerprints": orphan_count,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
