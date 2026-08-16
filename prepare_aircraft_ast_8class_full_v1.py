"""Build a source-faithful 8-class AST dataset from the local AeroSonicDB archive."""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Self_Data" / "AeroSonicDB_source"
SOURCE_ZIP = SOURCE / "audio.zip"
SOURCE_META = SOURCE / "sample_meta.csv"
OUTPUT = ROOT / "Self_Data" / "AIRCRAFT_AST_8CLASS_FULL_V1"
OUTPUT_ZIP = ROOT / "Self_Data" / "AIRCRAFT_AST_8CLASS_FULL_V1.zip"

TYPE_TO_LABEL = {
    "A320": "AIRBUS_A320",
    "B738": "BOEING_737_800",
    "DH8C": "DASH_8_300",
    "DA42": "DIAMOND_DA42",
    "E190": "EMBRAER_E190",
    "F100": "FOKKER_100",
    "PC12": "PILATUS_PC12",
    "SF34": "SAAB_340",
}


def main() -> None:
    if not SOURCE_ZIP.is_file() or not SOURCE_META.is_file():
        raise FileNotFoundError("AeroSonicDB audio.zip veya sample_meta.csv bulunamadı")

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    metadata = pd.read_csv(SOURCE_META)
    selected = metadata[
        metadata["class"].eq(1) & metadata["typedesig"].isin(TYPE_TO_LABEL)
    ].copy()
    selected["label"] = selected["typedesig"].map(TYPE_TO_LABEL)

    manifest_rows: list[dict] = []
    with zipfile.ZipFile(SOURCE_ZIP) as source_archive:
        by_basename = {
            Path(name).name: name
            for name in source_archive.namelist()
            if name.lower().endswith(".wav")
        }
        for row in selected.to_dict("records"):
            filename = str(row["filename"])
            archive_name = by_basename.get(filename)
            if archive_name is None:
                raise FileNotFoundError(f"Arşivde ses bulunamadı: {filename}")

            label = str(row["label"])
            destination = OUTPUT / label / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source_archive.open(archive_name) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

            manifest_rows.append({
                "relative_path": destination.relative_to(OUTPUT).as_posix(),
                "label": label,
                "typedesig": str(row["typedesig"]),
                "model": str(row["model"]),
                "airframe_id": str(row["hex_id"]).upper(),
                "registration": str(row["reg"]),
                "source_filename": filename,
                "source_dataset": "AeroSonicDB YPAD-0523 v1.1.2",
                "source_doi": "10.5281/zenodo.10215080",
                "license": "CC BY-NC 4.0",
            })

    manifest = OUTPUT / "dataset_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {}
    for label in sorted(set(row["label"] for row in manifest_rows)):
        rows = [row for row in manifest_rows if row["label"] == label]
        summary[label] = {
            "recordings": len(rows),
            "independent_airframes": len({row["airframe_id"] for row in rows}),
        }

    (OUTPUT / "dataset_summary.json").write_text(
        json.dumps({
            "recordings": len(manifest_rows),
            "classes": len(summary),
            "class_summary": summary,
            "source_doi": "10.5281/zenodo.10215080",
            "license": "CC BY-NC 4.0",
            "note": "Original source recordings; no synthetic duplication.",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "README.txt").write_text(
        "AIRCRAFT AST 8-CLASS FULL V1\n\n"
        "AeroSonicDB YPAD-0523 v1.1.2 kaynak kayıtlarından hazırlanmıştır.\n"
        "DOI: 10.5281/zenodo.10215080\n"
        "Lisans: CC BY-NC 4.0\n"
        "Aynı fiziksel uçak kimliği train/validation/test arasında bölünmemelidir.\n"
        "Kayıtlar yapay olarak çoğaltılmamış orijinal WAV dosyalarıdır.\n",
        encoding="utf-8",
    )

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    shutil.make_archive(str(OUTPUT_ZIP.with_suffix("")), "zip", OUTPUT.parent, OUTPUT.name)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"TOTAL={len(manifest_rows)}")
    print(f"ZIP={OUTPUT_ZIP}")


if __name__ == "__main__":
    main()
