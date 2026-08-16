"""Create a portable, source-disjoint benchmark for consensus calibration.

Only existing test-split assets are used. One clip per original source group is
kept so repeated windows from the same recording cannot inflate the score.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_catalog import sha256_file


ROOT = Path(__file__).resolve().parent
AIRCRAFT_MANIFEST = ROOT / "cache" / "aircraft_type_clips_350.csv"
CATEGORY_MANIFEST = ROOT / "cache" / "category_subtypes_350.csv"
DEFAULT_OUTPUT = ROOT / "cache" / "sealed_consensus_benchmark_v1.jsonl"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def collect_candidates(
    aircraft_manifest: Path = AIRCRAFT_MANIFEST,
    category_manifest: Path = CATEGORY_MANIFEST,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in _read_csv(aircraft_manifest):
        if row.get("split") != "test" or str(row.get("augmented", "")).lower() == "true":
            continue
        candidates.append(
            {
                "path": str(Path(row["path"]).resolve()),
                "label": "AIRCRAFT",
                "subtype": row["label"],
                "source_group": f"aircraft::{row['source_file']}",
                "source_dataset": "AeroSonicDB",
                "benchmark_role": "SEALED_TEST_ONLY",
            }
        )
    for row in _read_csv(category_manifest):
        if row.get("split") != "test" or str(row.get("augmented", "")).lower() == "true":
            continue
        if row.get("category") not in {"TRAFFIC", "OTHER"}:
            continue
        candidates.append(
            {
                "path": str(Path(row["path"]).resolve()),
                "label": row["category"],
                "subtype": row["subtype"],
                "source_group": f"{row['source_dataset']}::{row['source_group']}",
                "source_dataset": row["source_dataset"],
                "benchmark_role": "SEALED_TEST_ONLY",
            }
        )
    return candidates


def build_manifest(
    output_path: Path,
    *,
    per_class_limit: int | None = None,
) -> dict[str, Any]:
    candidates = collect_candidates()
    selected: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    class_counts: Counter[str] = Counter()
    missing = 0

    for row in sorted(candidates, key=lambda item: (item["label"], item["source_group"], item["path"])):
        if row["source_group"] in seen_groups:
            continue
        if per_class_limit is not None and class_counts[row["label"]] >= per_class_limit:
            continue
        path = Path(row["path"])
        if not path.is_file():
            missing += 1
            continue
        row = dict(row)
        row["sha256"] = sha256_file(path)
        selected.append(row)
        seen_groups.add(row["source_group"])
        class_counts[row["label"]] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "output": str(output_path),
        "selected": len(selected),
        "missing": missing,
        "distribution": dict(sorted(class_counts.items())),
        "unique_source_groups": len(seen_groups),
        "per_class_limit": per_class_limit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sızıntısız çoklu-model test manifesti oluştur")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class-limit", type=int)
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.output, per_class_limit=args.per_class_limit), indent=2))


if __name__ == "__main__":
    main()
