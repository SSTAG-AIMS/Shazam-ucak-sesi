"""Build the honest two-tier aircraft subtype catalogue.

Tier 1 (fingerprint catalogue) needs a source-verified recording and can only
recognise that registered recording. Tier 2 (generalising agent) needs at
least three distinct physical airframes so train/validation/test can remain
airframe-disjoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "Self_Data" / "AIRCRAFT_100_REFERENCE_V1" / "references_manifest.json"
DEFAULT_JSON = ROOT / "models" / "aircraft_subtype_catalog_v1.json"
DEFAULT_CSV = ROOT / "outputs" / "aircraft_subtype_catalog_v1.csv"
MIN_AGENT_AIRFRAMES = 3


def build_catalog(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["folder"])].append(row)
    types = []
    for label, items in sorted(grouped.items()):
        airframes = sorted({str(row.get("hex_id", "")).strip() for row in items if row.get("hex_id")})
        first = items[0]
        agent_ready = len(airframes) >= MIN_AGENT_AIRFRAMES
        types.append({
            "label": label,
            "icao_type": str(first.get("aircraft_type", "")),
            "manufacturer": str(first.get("manufacturer", "")),
            "model": str(first.get("model", "")),
            "verified_recordings": len(items),
            "physical_airframes": len(airframes),
            "fingerprint_catalog_status": "REFERENCE_AVAILABLE",
            "agent_status": "GENERALIZATION_READY" if agent_ready else "MORE_AIRFRAMES_REQUIRED",
            "missing_airframes_for_agent": max(0, MIN_AGENT_AIRFRAMES - len(airframes)),
            "license": str(first.get("license", "")),
            "source_doi": str(first.get("source_doi", "")),
        })
    return {
        "schema_version": 1,
        "target_subtypes": 100,
        "catalogued_subtypes": len(types),
        "fingerprint_reference_ready": sum(x["fingerprint_catalog_status"] == "REFERENCE_AVAILABLE" for x in types),
        "generalization_ready": sum(x["agent_status"] == "GENERALIZATION_READY" for x in types),
        "important_note": "Fingerprint availability does not imply generalisation to unseen aircraft.",
        "types": types,
    }


def write_catalog(source: Path = DEFAULT_SOURCE, json_path: Path = DEFAULT_JSON, csv_path: Path = DEFAULT_CSV) -> dict:
    rows = json.loads(source.read_text(encoding="utf-8"))
    catalog = build_catalog(rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(catalog["types"][0]))
        writer.writeheader(); writer.writerows(catalog["types"])
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON); parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(); result = write_catalog(args.source, args.json, args.csv)
    print(f"Katalog alt türü: {result['catalogued_subtypes']}")
    print(f"Shazam referansına uygun: {result['fingerprint_reference_ready']}")
    print(f"Genelleme modeline uygun: {result['generalization_ready']}")
    print(f"JSON: {args.json.resolve()}\nCSV: {args.csv.resolve()}")


if __name__ == "__main__": main()
