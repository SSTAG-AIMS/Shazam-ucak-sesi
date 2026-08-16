"""Create a small, session-disjoint aircraft test pack from the held-out split."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "cache" / "aircraft_type_manifest.csv"
DEFAULT_OUTPUT = ROOT / "Test_Data" / "AIRCRAFT_TYPES_UNSEEN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-type", type=int, default=3)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8", newline="") as stream:
        test_rows = [
            row for row in csv.DictReader(stream) if row["split"] == "test"
        ]

    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        by_type[row["label"]].append(row)

    selected: list[dict[str, str]] = []
    for label, rows in sorted(by_type.items()):
        # Farklı oturumları önceleyerek mümkün olduğunca çeşitli örnek seç.
        seen_sessions: set[str] = set()
        ordered = sorted(rows, key=lambda row: (row["session"], row["path"]))
        for row in ordered:
            if row["session"] not in seen_sessions:
                selected.append(row)
                seen_sessions.add(row["session"])
                if len([item for item in selected if item["label"] == label]) >= args.per_type:
                    break
        if len([item for item in selected if item["label"] == label]) < args.per_type:
            for row in ordered:
                if row not in selected:
                    selected.append(row)
                    if len([item for item in selected if item["label"] == label]) >= args.per_type:
                        break

    copied_manifest = []
    for row in selected:
        source = Path(row["path"])
        type_dir = args.output / row["label"]
        type_dir.mkdir(parents=True, exist_ok=True)
        destination = type_dir / source.name
        shutil.copy2(source, destination)
        copied_manifest.append(
            {
                "expected_type": row["label"],
                "file": str(destination.relative_to(ROOT)),
                "session": row["session"],
                "source_doi": row["source_doi"],
                "license": row["license"],
                "shazam_indexed": False,
            }
        )

    (args.output / "test_manifest.json").write_text(
        json.dumps(copied_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "README.txt").write_text(
        "Bu klasördeki dosyalar held-out TEST kayıtlarıdır.\n"
        "Shazam veritabanına eklenmemiştir; tür tahmini BEATs ile sınanır.\n"
        "Dosyayı GUI'de açın, Analiz Et'e basın ve UÇAK TÜRÜ sonucunu\n"
        "klasör adıyla karşılaştırın.\n",
        encoding="utf-8",
    )
    print(f"{len(copied_manifest)} test dosyası hazırlandı: {args.output}")
    for label in sorted(by_type):
        count = sum(item["expected_type"] == label for item in copied_manifest)
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
