"""End-to-end evaluation of the real Category Shazam v1 GUI pipeline."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from noise_detector_category_fp_v1 import CategoryFingerprintV1System


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "cache" / "category_subtypes_350.csv"
OUTPUT_CSV = ROOT / "outputs" / "category_shazam_v1_end_to_end.csv"
OUTPUT_JSON = ROOT / "outputs" / "category_shazam_v1_end_to_end_summary.json"
ERROR_CSV = ROOT / "outputs" / "other_v2_hard_examples.csv"

FIELDS = [
    "category",
    "expected_subtype",
    "predicted_category",
    "predicted_subtype",
    "subtype_method",
    "subtype_confidence",
    "main_correct",
    "subtype_correct",
    "end_to_end_correct",
    "aircraft_false_positive",
    "audio_path",
]


def load_cases() -> list[dict]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("split", "").strip().lower() == "test"
            and Path(row["path"]).is_file()
        ]


def load_completed() -> tuple[list[dict], set[str]]:
    if not OUTPUT_CSV.is_file():
        return [], set()
    with OUTPUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, {row["audio_path"] for row in rows}


def append_result(row: dict) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = OUTPUT_CSV.is_file() and OUTPUT_CSV.stat().st_size > 0
    with OUTPUT_CSV.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def truth(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def build_summary(rows: list[dict]) -> dict:
    total = len(rows)
    main_correct = sum(truth(row["main_correct"]) for row in rows)
    subtype_correct = sum(truth(row["subtype_correct"]) for row in rows)
    end_correct = sum(truth(row["end_to_end_correct"]) for row in rows)
    aircraft_fp = sum(truth(row["aircraft_false_positive"]) for row in rows)

    by_subtype = {}
    groups: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"{row['category']}/{row['expected_subtype']}"].append(row)
    for label, group in sorted(groups.items()):
        main = sum(truth(row["main_correct"]) for row in group)
        end = sum(truth(row["end_to_end_correct"]) for row in group)
        by_subtype[label] = {
            "total": len(group),
            "main_correct": main,
            "main_accuracy": main / len(group),
            "end_to_end_correct": end,
            "end_to_end_accuracy": end / len(group),
        }

    return {
        "total": total,
        "main_category": {
            "correct": main_correct,
            "accuracy": main_correct / total if total else 0.0,
        },
        "subtype_prediction": {
            "correct": subtype_correct,
            "accuracy": subtype_correct / total if total else 0.0,
        },
        "end_to_end": {
            "correct": end_correct,
            "accuracy": end_correct / total if total else 0.0,
        },
        "aircraft_false_positives": aircraft_fp,
        "methods": dict(Counter(row["subtype_method"] for row in rows)),
        "by_subtype": by_subtype,
    }


def main() -> None:
    cases = load_cases()
    completed_rows, completed_paths = load_completed()
    remaining = [row for row in cases if row["path"] not in completed_paths]
    print(
        f"Toplam={len(cases)} tamamlanan={len(completed_rows)} "
        f"kalan={len(remaining)}"
    )

    system = CategoryFingerprintV1System(output_dir=str(ROOT / "outputs_gui"))
    for index, case in enumerate(remaining, len(completed_rows) + 1):
        expected_category = case["category"].strip().upper()
        expected_subtype = case["subtype"].strip().upper()
        result = system.analyze_for_gui(case["path"], model_pref="auto")
        summary = result.get("summary") or {}
        predicted_category = max(summary, key=summary.get) if summary else "UNKNOWN"
        subtype = result.get("subtype_match") or {}
        predicted_subtype = subtype.get("subtype", "UNKNOWN")
        method = subtype.get("method", "none")
        confidence = float(subtype.get("confidence", 0.0))
        main_ok = predicted_category == expected_category
        subtype_ok = predicted_subtype == expected_subtype
        output = {
            "category": expected_category,
            "expected_subtype": expected_subtype,
            "predicted_category": predicted_category,
            "predicted_subtype": predicted_subtype,
            "subtype_method": method,
            "subtype_confidence": confidence,
            "main_correct": main_ok,
            "subtype_correct": subtype_ok,
            "end_to_end_correct": main_ok and subtype_ok,
            "aircraft_false_positive": predicted_category == "AIRCRAFT",
            "audio_path": case["path"],
        }
        append_result(output)
        completed_rows.append(output)
        print(
            f"[{index}/{len(cases)}] {expected_category}/{expected_subtype}"
            f" -> {predicted_category}/{predicted_subtype} ({method})"
        )

    evaluation = build_summary(completed_rows)
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)

    hard_examples = [
        row
        for row in completed_rows
        if row["category"] == "OTHER"
        and (
            not truth(row["main_correct"])
            or not truth(row["subtype_correct"])
        )
    ]
    with ERROR_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(hard_examples)

    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"OTHER v2 zor örnekleri: {ERROR_CSV}")


if __name__ == "__main__":
    main()
