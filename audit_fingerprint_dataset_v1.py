"""Audit dataset coverage without approving or indexing any audio.

The report deliberately separates raw candidate coverage from human-approved
fingerprint references.  File count alone must never be presented as proof of
correct labels or Shazam readiness.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "fingerprint_dataset_coverage_v1.json"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _existing_audio(path_value: str) -> bool:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES


def _latest_approved(path: Path, id_field: str = "intake_id") -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get(id_field):
            latest[str(row[id_field])] = row
    return [row for row in latest.values() if row.get("review_status") == "APPROVED"]


def _aircraft_candidates() -> dict[str, Any]:
    manifest_path = ROOT / "Self_Data" / "AIRCRAFT_100_REFERENCE_V1" / "references_manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else []
    valid = [row for row in rows if _existing_audio(str(row.get("output_file", "")))]
    by_subtype = Counter(str(row.get("folder", "UNKNOWN_AIRCRAFT")).upper() for row in valid)
    source_ids = {str(row.get("hex_id", "")).strip().upper() for row in valid if row.get("hex_id")}
    licenses = Counter(str(row.get("license", "MISSING")).strip() or "MISSING" for row in valid)
    missing_provenance = sum(
        not row.get("hex_id") or not row.get("source_doi") or not row.get("license") for row in valid
    )
    production_approved = _latest_approved(ROOT / "cache" / "aircraft_reference_intake_decisions_v1.jsonl")
    demo_approved = _latest_approved(
        ROOT / "Test_Folder" / "AIRCRAFT_REFERENCE_LAB_V1" / "workspace" / "intake_decisions.jsonl"
    )
    return {
        "candidate_audio_count": len(valid),
        "candidate_subtype_count": len(by_subtype),
        "unique_source_recording_count": len(source_ids),
        "distribution": dict(sorted(by_subtype.items())),
        "licenses": dict(sorted(licenses.items())),
        "missing_provenance_count": missing_provenance,
        "human_approved_count": len(production_approved),
        "isolated_demo_approved_count": len(demo_approved),
        "direct_index_ready": False,
        "status": "Candidate pool only; each record still requires the review workflow.",
    }


def _generic_candidates() -> dict[str, dict[str, Any]]:
    manifest_path = ROOT / "cache" / "category_subtypes_350.csv"
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if _existing_audio(row.get("path", "")):
                    grouped[row.get("category", "UNKNOWN").strip().upper()].append(row)

    output: dict[str, dict[str, Any]] = {}
    for category, rows in grouped.items():
        subtype_counts = Counter(row["subtype"].strip().upper() for row in rows)
        split_counts = Counter(row.get("split", "MISSING").strip().lower() for row in rows)
        source_to_splits: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            source_to_splits[row.get("source_group", "MISSING")].add(row.get("split", "MISSING").lower())
        leaking_sources = sorted(source for source, splits in source_to_splits.items() if len(splits) > 1)
        output[category] = {
            "candidate_audio_count": len(rows),
            "candidate_subtype_count": len(subtype_counts),
            "distribution": dict(sorted(subtype_counts.items())),
            "split_distribution": dict(sorted(split_counts.items())),
            "cross_split_source_leakage_count": len(leaking_sources),
            "cross_split_source_leakage_examples": leaking_sources[:10],
            "human_approved_count": 0,
            "direct_index_ready": False,
            "status": "Downloaded labels are candidates; human review and license audit are required.",
        }
    return output


def build_report() -> dict[str, Any]:
    taxonomy_path = ROOT / "dataset_taxonomy_v1.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    categories = {
        category: {
            "taxonomy_known_subtype_count": len(
                [value for value in definition["subtypes"] if not value.startswith("UNKNOWN_")]
            ),
            "target_subtype_count": 50,
            "target_total_audio_count": 100,
            "candidate_audio_count": 0,
            "candidate_subtype_count": 0,
            "human_approved_count": 0,
            "direct_index_ready": False,
            "status": "No audited candidate source connected.",
        }
        for category, definition in taxonomy["categories"].items()
    }
    categories["AIRCRAFT"].update(_aircraft_candidates())
    for category, data in _generic_candidates().items():
        if category in categories:
            categories[category].update(data)

    for data in categories.values():
        data["candidate_scale_target_met"] = (
            data.get("candidate_subtype_count", 0) >= data["target_subtype_count"]
            and data.get("candidate_audio_count", 0) >= data["target_total_audio_count"]
        )
        data["approved_scale_target_met"] = (
            data.get("candidate_subtype_count", 0) >= data["target_subtype_count"]
            and data.get("human_approved_count", 0) >= data["target_total_audio_count"]
        )

    return {
        "report_version": "1.0",
        "meaning_of_target": "50 known subtypes and 100 total audio files per main category",
        "scientific_rules": [
            "Candidate files are not accepted references.",
            "Only human-approved, sourced, licensed and hash-verified records may enter Shazam.",
            "An indexed recording may not be reused as an independent generalization test.",
            "Fingerprint identity tests and unseen-recording classifier tests must be reported separately.",
        ],
        "categories": categories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fingerprint dataset coverage")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
