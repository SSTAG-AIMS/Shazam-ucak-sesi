"""Leakage-resistant Shazam evaluation for the expanded aircraft catalog."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from aircraft_fingerprint import AircraftFingerprintDatabase


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "Self_Data" / "AIRCRAFT_100_REFERENCE_V1" / "references_manifest.json"
DEFAULT_DATABASE = ROOT / "models" / "aircraft_100_blind_train_v1.sqlite3"
DEFAULT_REPORT = ROOT / "outputs" / "aircraft_100_blind_report_v1.json"


def split_by_physical_airframe(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Hold out one physical aircraft only when two other airframes remain."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["folder"])].append(row)

    train: list[dict] = []
    test: list[dict] = []
    insufficient: list[dict] = []
    for label, items in sorted(grouped.items()):
        by_airframe: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            by_airframe[str(item.get("hex_id") or item["source_file"])].append(item)
        airframes = sorted(by_airframe)
        if len(airframes) < 3:
            insufficient.append({
                "label": label,
                "record_count": len(items),
                "physical_airframes": len(airframes),
                "reason": "EN_AZ_3_FARKLI_FIZIKSEL_UCAK_GEREKLI",
            })
            train.extend(items)
            continue
        heldout_airframe = airframes[-1]
        heldout_rows = sorted(by_airframe[heldout_airframe], key=lambda row: row["output_file"])
        test.append(heldout_rows[0])
        train.extend(item for item in items if str(item.get("hex_id")) != heldout_airframe)
    return train, test, insufficient


def evaluate(manifest: Path, database: Path, report: Path) -> dict:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    train, test, insufficient = split_by_physical_airframe(rows)
    matcher = AircraftFingerprintDatabase(database)
    matcher.reset()
    for row in train:
        matcher.add_reference(ROOT / row["output_file"], str(row["folder"]))
    matcher.compact()
    reference_label_by_name = {
        Path(str(row["output_file"])).stem: str(row["folder"]) for row in train
    }

    results = []
    for row in test:
        match = matcher.match_file(ROOT / row["output_file"])
        payload = match.as_dict() if match else {
            "aircraft_type": "UNKNOWN_AIRCRAFT", "reference_name": None,
            "matched_hashes": 0, "aligned_hashes": 0, "query_hashes": 0,
            "confidence": 0.0, "accepted": False,
        }
        payload.update({
            "expected": row["folder"],
            "candidate_aircraft_type": reference_label_by_name.get(
                str(payload.get("reference_name") or ""), "UNKNOWN_AIRCRAFT"
            ),
            "query_path": str((ROOT / row["output_file"]).resolve()),
            "query_airframe": row.get("hex_id"),
            "correct": bool(payload["accepted"] and payload["aircraft_type"] == row["folder"]),
        })
        payload["candidate_correct_before_threshold"] = (
            payload["candidate_aircraft_type"] == row["folder"]
        )
        results.append(payload)

    correct = sum(bool(item["correct"]) for item in results)
    accepted = sum(bool(item["accepted"]) for item in results)
    payload = {
        "protocol": "PHYSICAL_AIRFRAME_DISJOINT_ONE_QUERY_PER_TYPE",
        "source_manifest": str(manifest.resolve()),
        "database": str(database.resolve()),
        "train_references": len(train),
        "tested_types": len(test),
        "accepted_matches": accepted,
        "correct_matches": correct,
        "accuracy": correct / len(test) if test else 0.0,
        "coverage": accepted / len(test) if test else 0.0,
        "results": results,
        "insufficient_reference_types": insufficient,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = evaluate(args.manifest, args.database, args.report)
    print(json.dumps({key: result[key] for key in (
        "train_references", "tested_types", "accepted_matches", "correct_matches",
        "accuracy", "coverage",
    )}, ensure_ascii=False, indent=2))
    print(f"Rapor: {args.report}")


if __name__ == "__main__":
    main()
