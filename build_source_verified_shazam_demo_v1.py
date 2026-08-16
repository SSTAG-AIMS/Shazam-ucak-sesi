"""Build a leakage-aware, source-labelled Shazam demonstration catalog.

This is not a human-labelled benchmark. Labels come from the source dataset
metadata (ADS-B/ICAO for aircraft, dataset folder labels for generic sounds).
The generated SQLite database is isolated from production and from the human-
approved database. Independent test source IDs are never indexed.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from category_fingerprint_v1 import CategoryFingerprintDatabaseV1
from dataset_catalog import sha256_file


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Test_Folder" / "SHAZAM_SOURCE_VERIFIED_V1"
DATABASE = OUTPUT / "source_verified_demo.sqlite3"


def _balanced(rows: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["subtype"]).upper()].append(row)
    selected: list[dict[str, Any]] = []
    offset = 0
    keys = sorted(groups)
    while len(selected) < limit:
        added = False
        for key in keys:
            if offset < len(groups[key]) and len(selected) < limit:
                selected.append(groups[key][offset]); added = True
        if not added:
            break
        offset += 1
    return selected


def _aircraft_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_manifest = ROOT / "Self_Data" / "AIRCRAFT_100_REFERENCE_V1" / "references_manifest.json"
    test_manifest = ROOT / "Test_Folder" / "AIRCRAFT_REFERENCE_LAB_V1" / "test_manifest.json"
    source_rows = json.loads(source_manifest.read_text(encoding="utf-8"))
    test_payload = json.loads(test_manifest.read_text(encoding="utf-8"))
    test_ids = {
        str(row["physical_airframe_id"]).upper()
        for row in test_payload.get("records", [])
    }
    accepted = []
    for row in source_rows:
        path = (ROOT / row["output_file"]).resolve()
        if not path.is_file() or str(row.get("hex_id", "")).upper() in test_ids:
            continue
        accepted.append({
            "path": str(path), "category": "AIRCRAFT", "subtype": str(row["folder"]).upper(),
            "source_recording_id": str(row["hex_id"]).upper(),
            "source_uri": f"https://doi.org/{row['source_doi']}", "license": row["license"],
            "label_origin": "ADS-B/ICAO metadata", "split": "accepted_reference",
        })
    tests = []
    for row in test_payload.get("records", []):
        path = Path(row["audio_path"]).resolve()
        if path.is_file():
            tests.append({
                "path": str(path), "category": "AIRCRAFT", "subtype": row["aircraft_type"],
                "source_recording_id": str(row["physical_airframe_id"]).upper(),
                "source_uri": row["source_uri"], "license": row["license"],
                "label_origin": "ADS-B/ICAO metadata", "split": "independent_test",
            })
    return _balanced(accepted, 120), tests


def _generic_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = ROOT / "cache" / "category_subtypes_350.csv"
    accepted_pool: list[dict[str, Any]] = []
    test_pool: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            path = Path(row["path"]).resolve()
            if not path.is_file():
                continue
            converted = {
                "path": str(path), "category": row["category"].upper(),
                "subtype": row["subtype"].upper(), "source_recording_id": row["source_group"],
                "source_uri": f"https://www.kaggle.com/datasets/{row['source_dataset']}",
                "license": row["license"], "label_origin": "source dataset folder label",
                "split": "accepted_reference" if row["split"].lower() == "train" else "independent_test",
            }
            if row["split"].lower() == "train":
                accepted_pool.append(converted)
            elif row["split"].lower() == "test":
                test_pool.append(converted)
    accepted: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for category in ("TRAFFIC", "OTHER"):
        accepted.extend(_balanced((row for row in accepted_pool if row["category"] == category), 120))
        tests.extend(_balanced((row for row in test_pool if row["category"] == category), 30))
    return accepted, tests


def _materialize(row: dict[str, Any], root: Path) -> dict[str, Any]:
    source = Path(row["path"])
    destination = root / row["category"] / row["subtype"] / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return {
        **row,
        "source_path": str(source),
        "audio_path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "verification_status": "SOURCE_LABEL_VERIFIED_NOT_HUMAN_REVIEWED",
    }


def build() -> dict[str, Any]:
    aircraft_accepted, aircraft_tests = _aircraft_rows()
    generic_accepted, generic_tests = _generic_rows()
    accepted_rows = [
        _materialize(row, OUTPUT / "ACCEPTED_SOURCE_LABELS")
        for row in aircraft_accepted + generic_accepted
    ]
    test_rows = [
        _materialize(row, OUTPUT / "INDEPENDENT_TEST")
        for row in aircraft_tests + generic_tests
    ]
    accepted_hashes = {row["sha256"] for row in accepted_rows}
    accepted_source_ids = {(row["category"], row["source_recording_id"]) for row in accepted_rows}
    leakage = [
        row for row in test_rows
        if row["sha256"] in accepted_hashes
        or (row["category"], row["source_recording_id"]) in accepted_source_ids
    ]
    if leakage:
        raise RuntimeError(f"Accepted/test leakage detected: {len(leakage)} records")

    for filename, rows in (
        ("accepted_source_labels.jsonl", accepted_rows),
        ("independent_test.jsonl", test_rows),
    ):
        (OUTPUT / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    temporary = DATABASE.with_suffix(".sqlite3.building")
    if temporary.exists():
        temporary.unlink()
    db = CategoryFingerprintDatabaseV1(temporary)
    db.reset(); indexed = []
    try:
        for row in accepted_rows:
            count = db.add_reference(row["audio_path"], row["category"], row["subtype"])
            indexed.append({
                "category": row["category"], "subtype": row["subtype"],
                "audio_path": row["audio_path"], "hash_count": count,
                "verification_status": row["verification_status"],
            })
        db.compact(); os.replace(temporary, DATABASE)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    category_counts: dict[str, int] = defaultdict(int)
    subtype_sets: dict[str, set[str]] = defaultdict(set)
    for row in accepted_rows:
        category_counts[row["category"]] += 1; subtype_sets[row["category"]].add(row["subtype"])
    report = {
        "database": str(DATABASE.resolve()),
        "catalog_kind": "SOURCE_VERIFIED_DEMO_NOT_HUMAN_LABELLED",
        "accepted_count": len(accepted_rows), "independent_test_count": len(test_rows),
        "leakage_count": 0,
        "coverage": {
            category: {"audio_count": count, "subtype_count": len(subtype_sets[category])}
            for category, count in sorted(category_counts.items())
        },
        "indexed": indexed,
        "missing_categories": ["AMBIENT", "SPEECH", "WIND"],
    }
    (OUTPUT / "index_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "SUNUM_NOTU.txt").write_text(
        "Bu katalog insan etiketli değildir. Etiketler kaynak veri metadatasından gelir.\n"
        "ACCEPTED_SOURCE_LABELS Shazam indeksine alınmıştır; INDEPENDENT_TEST indekslenmemiştir.\n"
        "index_manifest.json ve SQLite dosyası indeksleme kanıtıdır.\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
