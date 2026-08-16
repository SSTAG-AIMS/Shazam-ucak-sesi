"""Build or inspect the local aircraft fingerprint database."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aircraft_fingerprint import AircraftFingerprintDatabase


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "Self_Data" / "AIRCRAFT_TYPES"
DEFAULT_DATABASE = PROJECT_ROOT / "models" / "aircraft_fingerprints.sqlite3"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uçak referans seslerinden Shazam tarzı parmak izi veritabanı oluşturur."
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="İstenirse yalnızca CSV manifestindeki kayıtları indeksler.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="--manifest kullanıldığında indekslenecek split (varsayılan: train).",
    )
    parser.add_argument("--list", action="store_true", help="Kayıtlı referansları listeler.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Yalnızca üretilmiş parmak izi tablolarını temizleyip baştan indeksler.",
    )
    args = parser.parse_args()
    database = AircraftFingerprintDatabase(args.database)
    if args.list:
        rows = list(database.list_references())
        if not rows:
            print("Veritabanında referans yok.")
            return
        for aircraft_type, name, hash_count in rows:
            print(f"{aircraft_type:24} {name:40} {hash_count:6} hash")
        return
    if args.rebuild:
        database.reset()
    if args.manifest:
        with args.manifest.open("r", encoding="utf-8", newline="") as stream:
            rows = [
                row for row in csv.DictReader(stream)
                if row.get("split") == args.split
            ]
        indexed = {}
        for row in rows:
            path = Path(row["path"])
            indexed[str(path)] = database.add_reference(path, row["label"])
    else:
        indexed = database.index_reference_tree(args.reference_dir)
    if not indexed:
        raise SystemExit(
            "Ses dosyası bulunamadı. Örnek yapı: "
            f"{args.reference_dir}\\AIRBUS_A320\\ornek.wav"
        )
    print(f"\n{len(indexed)} referans indekslendi: {args.database}")
    for path, count in indexed.items():
        print(f"  {count:6} hash  {path}")
    database.compact()


if __name__ == "__main__":
    main()
