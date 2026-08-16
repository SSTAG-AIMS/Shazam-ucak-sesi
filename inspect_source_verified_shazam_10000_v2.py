"""Print presentation-ready evidence for the isolated 10,000-track catalog."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "Test_Folder" / "SHAZAM_10000_SOURCE_VERIFIED_V2"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inspect() -> dict:
    accepted = _jsonl(CATALOG / "accepted_source_labels.jsonl")
    tests = _jsonl(CATALOG / "independent_test.jsonl")
    database = CATALOG / "source_verified_10000.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        tracks = connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        fingerprints = connection.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]
        sqlite_categories = dict(
            connection.execute(
                "SELECT substr(aircraft_type,1,instr(aircraft_type,'::')-1), COUNT(*) "
                "FROM tracks GROUP BY 1 ORDER BY 1"
            )
        )
    categories = Counter(row["category"] for row in accepted)
    subtypes: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        subtypes[row["category"]].add(row["subtype"])
    accepted_hashes = {row["sha256"] for row in accepted}
    test_hashes = {row["sha256"] for row in tests}
    accepted_ids = {row["source_recording_id"] for row in accepted}
    test_ids = {row["source_recording_id"] for row in tests}
    return {
        "accepted_audio": len(accepted),
        "independent_test_audio": len(tests),
        "main_category_count": len(categories),
        "subtype_count": len({(row["category"], row["subtype"]) for row in accepted}),
        "coverage": {
            category: {"audio": count, "subtypes": len(subtypes[category])}
            for category, count in sorted(categories.items())
        },
        "datasets": dict(sorted(Counter(row["dataset"] for row in accepted).items())),
        "licenses": dict(sorted(Counter(row["license"] for row in accepted).items())),
        "sqlite_tracks": tracks,
        "sqlite_fingerprints": fingerprints,
        "sqlite_category_counts": sqlite_categories,
        "sha256_leakage": len(accepted_hashes & test_hashes),
        "source_id_leakage": len(accepted_ids & test_ids),
        "proof_database": str(database.resolve()),
        "proof_manifest": str((CATALOG / "accepted_source_labels.jsonl").resolve()),
    }


if __name__ == "__main__":
    print(json.dumps(inspect(), ensure_ascii=False, indent=2))
