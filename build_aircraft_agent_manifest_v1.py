"""Build an airframe-disjoint manifest for the expanded aircraft agent."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "Self_Data" / "AIRCRAFT_100_REFERENCE_V1" / "references_manifest.json"
DEFAULT_OUTPUT = ROOT / "cache" / "aircraft_agent_manifest_v1.csv"
DEFAULT_COVERAGE = ROOT / "cache" / "aircraft_agent_coverage_v1.json"


def assign_splits(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["folder"])].append(dict(row))
    output: list[dict] = []
    coverage: list[dict] = []
    for label, items in sorted(grouped.items()):
        airframes = sorted({str(item.get("hex_id") or "") for item in items})
        eligible = len(airframes) >= 3
        validation_airframe = airframes[-2] if eligible else None
        test_airframe = airframes[-1] if eligible else None
        for item in items:
            airframe = str(item.get("hex_id") or "")
            split = (
                "test" if airframe == test_airframe else
                "validation" if airframe == validation_airframe else
                "train" if eligible else "reference_only"
            )
            output.append({
                "path": str((ROOT / item["output_file"]).resolve()),
                "label": label,
                "icao_type": item.get("aircraft_type", ""),
                "physical_airframe_id": airframe,
                "session_id": item.get("session", ""),
                "split": split,
                "source_doi": item.get("source_doi", ""),
                "license": item.get("license", ""),
                "reference_strength": item.get("reference_strength", ""),
            })
        coverage.append({
            "label": label,
            "records": len(items),
            "physical_airframes": len(airframes),
            "agent_training_status": "PILOT_READY" if eligible else "MORE_DATA_REQUIRED",
        })
    return output, coverage


def build(source: Path, output: Path, coverage_path: Path) -> tuple[list[dict], list[dict]]:
    rows = json.loads(source.read_text(encoding="utf-8"))
    manifest, coverage = assign_splits(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    coverage_path.write_text(
        json.dumps({
            "target_aircraft_types": 100,
            "catalogued_types": len(coverage),
            "pilot_ready_types": sum(item["agent_training_status"] == "PILOT_READY" for item in coverage),
            "more_data_required_types": sum(item["agent_training_status"] == "MORE_DATA_REQUIRED" for item in coverage),
            "types": coverage,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    args = parser.parse_args()
    manifest, coverage = build(args.source, args.output, args.coverage)
    print(f"Manifest kayıtları: {len(manifest)}")
    print(f"Kataloglanan tip: {len(coverage)} / 100")
    print(f"Pilot eğitime hazır: {sum(x['agent_training_status'] == 'PILOT_READY' for x in coverage)}")
    print(f"Ek veri gerekli: {sum(x['agent_training_status'] == 'MORE_DATA_REQUIRED' for x in coverage)}")


if __name__ == "__main__":
    main()
