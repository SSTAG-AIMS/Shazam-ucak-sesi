"""Evaluate experimental category fingerprints and BEATs fallback."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from aircraft_fingerprint import load_audio
from category_fingerprint_v1 import CategoryFingerprintDatabaseV1
from noise_detector_category_fp_v1 import CategoryFingerprintV1System


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "cache" / "category_subtypes_350.csv"
DATABASE = ROOT / "models" / "category_fingerprints_v1.sqlite3"
OUTPUT_CSV = ROOT / "outputs" / "category_shazam_v1_evaluation.csv"
OUTPUT_JSON = ROOT / "outputs" / "category_shazam_v1_summary.json"


def load_manifest() -> list[dict]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_cases(rows: list[dict], split: str, per_subtype: int) -> list[dict]:
    selected: list[dict] = []
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row.get("split", "").strip().lower() != split:
            continue
        key = (row["category"].strip().upper(), row["subtype"].strip().upper())
        if counts[key] >= per_subtype:
            continue
        if not Path(row["path"]).is_file():
            continue
        selected.append(row)
        counts[key] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-per-subtype", type=int, default=3)
    args = parser.parse_args()

    rows = load_manifest()
    references = select_cases(rows, "train", 1)
    unseen = select_cases(rows, "test", args.test_per_subtype)
    fingerprint_db = CategoryFingerprintDatabaseV1(DATABASE)
    system = CategoryFingerprintV1System(output_dir=str(ROOT / "outputs_gui"))

    report_rows: list[dict] = []

    for row in references:
        expected_category = row["category"].strip().upper()
        expected_subtype = row["subtype"].strip().upper()
        match = fingerprint_db.match_file(row["path"])
        predicted = match.subtype if match and match.accepted else "REJECTED"
        report_rows.append(
            {
                "test_kind": "indexed_reference",
                "category": expected_category,
                "expected_subtype": expected_subtype,
                "predicted_subtype": predicted,
                "method": "shazam_v1",
                "accepted": bool(match and match.accepted),
                "confidence": match.confidence if match else 0.0,
                "correct": bool(match and match.accepted and predicted == expected_subtype),
                "audio_path": row["path"],
            }
        )

    for index, row in enumerate(unseen, 1):
        expected_category = row["category"].strip().upper()
        expected_subtype = row["subtype"].strip().upper()
        fingerprint_match = fingerprint_db.match_file(row["path"])
        fingerprint_accepted = bool(
            fingerprint_match
            and fingerprint_match.accepted
            and fingerprint_match.category == expected_category
        )
        if fingerprint_accepted:
            predicted = fingerprint_match.subtype
            method = "shazam_v1"
            confidence = fingerprint_match.confidence
        else:
            samples = load_audio(row["path"])
            beats_match = system._infer_category_subtype(samples, expected_category)
            predicted = beats_match.get("predicted_subtype", "UNKNOWN")
            method = "beats_multi_window_vote"
            confidence = float(beats_match.get("confidence", 0.0))

        report_rows.append(
            {
                "test_kind": "held_out_test",
                "category": expected_category,
                "expected_subtype": expected_subtype,
                "predicted_subtype": predicted,
                "method": method,
                "accepted": fingerprint_accepted,
                "confidence": confidence,
                "correct": predicted == expected_subtype,
                "audio_path": row["path"],
            }
        )
        print(
            f"[{index}/{len(unseen)}] {expected_category}/{expected_subtype}"
            f" -> {predicted} ({method}, %{confidence * 100:.1f})"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(report_rows[0])
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    summary: dict = {}
    for test_kind in ("indexed_reference", "held_out_test"):
        group = [row for row in report_rows if row["test_kind"] == test_kind]
        correct = sum(bool(row["correct"]) for row in group)
        summary[test_kind] = {
            "total": len(group),
            "correct": correct,
            "accuracy": correct / len(group) if group else 0.0,
            "methods": dict(Counter(row["method"] for row in group)),
        }

    by_subtype: dict[str, dict] = {}
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in report_rows:
        if row["test_kind"] == "held_out_test":
            grouped[f"{row['category']}/{row['expected_subtype']}"].append(row)
    for label, group in sorted(grouped.items()):
        correct = sum(bool(row["correct"]) for row in group)
        by_subtype[label] = {
            "total": len(group),
            "correct": correct,
            "accuracy": correct / len(group),
        }
    summary["held_out_by_subtype"] = by_subtype

    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
