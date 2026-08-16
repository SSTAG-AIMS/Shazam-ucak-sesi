"""Calibrate Category Shazam v1 thresholds on the full held-out split."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from category_fingerprint_v1 import CategoryFingerprintDatabaseV1


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "cache" / "category_subtypes_350.csv"
DATABASE = ROOT / "models" / "category_fingerprints_v1.sqlite3"
OUTPUT_CSV = ROOT / "outputs" / "category_shazam_v1_raw_matches.csv"
OUTPUT_JSON = ROOT / "outputs" / "category_shazam_v1_calibration.json"

CONFIDENCE_THRESHOLDS = (0.03, 0.05, 0.07, 0.10, 0.15, 0.20)
ALIGNED_THRESHOLDS = (5, 8, 12, 16, 24)


def main() -> None:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split", "").strip().lower() == "test"
            and Path(row["path"]).is_file()
        ]

    database = CategoryFingerprintDatabaseV1(
        DATABASE, min_aligned_hashes=1, min_confidence=0.0
    )
    matches: list[dict] = []
    for index, row in enumerate(rows, 1):
        expected_category = row["category"].strip().upper()
        expected_subtype = row["subtype"].strip().upper()
        match = database.match_file(row["path"])
        matches.append(
            {
                "category": expected_category,
                "expected_subtype": expected_subtype,
                "matched_category": match.category if match else "",
                "matched_subtype": match.subtype if match else "",
                "aligned_hashes": match.aligned_hashes if match else 0,
                "query_hashes": match.query_hashes if match else 0,
                "confidence": match.confidence if match else 0.0,
                "label_correct": bool(
                    match
                    and match.category == expected_category
                    and match.subtype == expected_subtype
                ),
                "audio_path": row["path"],
            }
        )
        if index % 50 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] parmak izi ölçüldü")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matches[0]))
        writer.writeheader()
        writer.writerows(matches)

    candidates: list[dict] = []
    for aligned_threshold in ALIGNED_THRESHOLDS:
        for confidence_threshold in CONFIDENCE_THRESHOLDS:
            accepted = [
                row
                for row in matches
                if int(row["aligned_hashes"]) >= aligned_threshold
                and float(row["confidence"]) >= confidence_threshold
            ]
            correct = sum(bool(row["label_correct"]) for row in accepted)
            false = len(accepted) - correct
            candidates.append(
                {
                    "min_aligned_hashes": aligned_threshold,
                    "min_confidence": confidence_threshold,
                    "accepted": len(accepted),
                    "correct_accepts": correct,
                    "false_accepts": false,
                    "precision": correct / len(accepted) if accepted else 1.0,
                    "coverage": correct / len(matches),
                }
            )

    zero_false = [row for row in candidates if row["false_accepts"] == 0]
    recommended = max(
        zero_false or candidates,
        key=lambda row: (row["correct_accepts"], row["precision"]),
    )
    output = {
        "test_files": len(matches),
        "recommended": recommended,
        "candidates": candidates,
    }
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print(json.dumps(output["recommended"], ensure_ascii=False, indent=2))
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
