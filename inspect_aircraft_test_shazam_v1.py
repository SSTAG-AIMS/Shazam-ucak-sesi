"""Direct SQLite proof that approved aircraft audio was indexed by Shazam."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from prepare_aircraft_reference_lab_v1 import LAB


DEFAULT_DATABASE = LAB / "workspace" / "aircraft_test_fingerprints.sqlite3"


def inspect_database(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        return {"database": str(path), "exists": False, "track_count": 0, "fingerprint_count": 0, "tracks": []}
    connection = sqlite3.connect(path)
    try:
        tracks = [
            {"aircraft_type": row[0], "reference_name": row[1], "source_path": row[2], "hash_count": int(row[3])}
            for row in connection.execute(
                "SELECT aircraft_type, reference_name, source_path, hash_count FROM tracks ORDER BY aircraft_type, reference_name"
            )
        ]
        fingerprints = int(connection.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0])
    finally:
        connection.close()
    return {
        "database": str(path), "exists": True, "track_count": len(tracks),
        "fingerprint_count": fingerprints, "tracks": tracks,
        "all_sources_are_accepted": all("KABUL_EDILEN" in row["source_path"] for row in tracks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Shazam SQLite içeriğini doğrudan doğrula")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args(); print(json.dumps(inspect_database(args.database), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
