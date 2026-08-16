"""Build the isolated experimental category fingerprint index."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from category_fingerprint_v1 import CategoryFingerprintDatabaseV1


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "cache" / "category_subtypes_350.csv"
DEFAULT_DATABASE = ROOT / "models" / "category_fingerprints_full_v1.sqlite3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--max-per-subtype",
        type=int,
        default=None,
        help="Alt tür başına üst sınır; verilmezse seçilen split'in tamamı.",
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    database = CategoryFingerprintDatabaseV1(args.database)
    if args.rebuild:
        database.reset()

    counts: Counter[tuple[str, str]] = Counter()
    indexed = 0
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            category = row["category"].strip().upper()
            subtype = row["subtype"].strip().upper()
            if row.get("split", "").strip().lower() != args.split.lower():
                continue
            key = (category, subtype)
            if (
                args.max_per_subtype is not None
                and counts[key] >= args.max_per_subtype
            ):
                continue
            path = Path(row["path"])
            if not path.is_file():
                continue
            hash_count = database.add_reference(path, category, subtype)
            counts[key] += 1
            indexed += 1
            print(f"[{indexed}] {category}/{subtype}: {path.name} ({hash_count} hash)")

    database.compact()
    print(f"Tamamlandı: {indexed} referans, {len(counts)} alt tür")
    print(f"Veritabanı: {args.database}")


if __name__ == "__main__":
    main()
