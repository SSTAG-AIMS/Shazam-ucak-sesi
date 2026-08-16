"""Join aircraft manifest hints into an existing agent review queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch_catalog_agent import DEFAULT_AIRCRAFT_MANIFEST, load_aircraft_candidates
from catalog_review import read_jsonl


def enrich(queue_path: Path, output_path: Path, manifest_path: Path) -> dict:
    rows = read_jsonl(queue_path)
    candidates = load_aircraft_candidates(manifest_path, ("train", "validation", "test"))
    by_source = {str(candidate["source_uri"]): candidate for candidate in candidates}
    enriched = []
    missing = 0
    for row in rows:
        candidate = by_source.get(str(row.get("source_uri")))
        updated = dict(row)
        if candidate is None:
            missing += 1
        else:
            updated["manifest_category_hint"] = candidate.get("category_hint")
            updated["manifest_subtype_hint"] = candidate.get("subtype_hint")
        enriched.append(updated)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in enriched:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"records": len(enriched), "missing_manifest_match": missing, "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Kuyruğa kaynak manifest etiketlerini ekle")
    parser.add_argument("queue", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_AIRCRAFT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(enrich(args.queue, args.output, args.manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
