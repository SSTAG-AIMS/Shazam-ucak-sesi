"""Match one recording against the aircraft fingerprint database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aircraft_fingerprint import AircraftFingerprintDatabase


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = PROJECT_ROOT / "models" / "aircraft_fingerprints.sqlite3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bir ses kaydındaki uçak türünü eşleştirir.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    result = AircraftFingerprintDatabase(args.database).match_file(args.audio)
    if result is None:
        raise SystemExit("Eşleşme bulunamadı veya parmak izi veritabanı hazır değil.")
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
