"""Build a source-traceable 41-type aircraft dataset from local AeroSonicDB files.

The script never invents or duplicates recordings.  It extracts only rows whose
class is Aircraft and whose ICAO type designator is known, then stores the audio
under one directory per type together with a leakage-safe metadata manifest.
"""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Self_Data" / "AeroSonicDB_source"
SAMPLE_CSV = SOURCE / "sample_meta.csv"
AUDIO_ZIP = SOURCE / "audio.zip"
OUTPUT = ROOT / "Self_Data" / "AIRCRAFT_AST_41CLASS_V1"
OUTPUT_ZIP = ROOT / "Self_Data" / "AIRCRAFT_AST_41CLASS_V1.zip"
DOI = "10.5281/zenodo.10215080"
LICENSE = "CC BY-NC 4.0"


def normalise_label(type_designator: str, model: str) -> str:
    type_code = type_designator.strip().upper()
    if not type_code:
        raise ValueError("ICAO type designator is empty")
    # The learning target is the ICAO type designator.  Manufacturer model
    # strings contain variants (for example several B738 submodels) and must
    # not accidentally create extra classes.
    return f"ICAO_{type_code}"


def main() -> None:
    if not SAMPLE_CSV.exists() or not AUDIO_ZIP.exists():
        raise FileNotFoundError("AeroSonicDB sample_meta.csv or audio.zip is missing")

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    with SAMPLE_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))

    aircraft_rows = [
        row for row in source_rows
        if row.get("class", "").strip() == "1"
        and row.get("typedesig", "").strip()
    ]
    archive_map: dict[str, str] = {}
    with zipfile.ZipFile(AUDIO_ZIP) as archive:
        for member in archive.namelist():
            if member.lower().endswith(".wav"):
                archive_map[Path(member).name.lower()] = member

        manifest: list[dict[str, object]] = []
        missing: list[str] = []
        for row in aircraft_rows:
            filename = Path(row["filename"]).name
            member = archive_map.get(filename.lower())
            if member is None:
                missing.append(filename)
                continue

            label = normalise_label(row["typedesig"], row.get("model", ""))
            destination = OUTPUT / label / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

            manifest.append({
                "path": destination.relative_to(OUTPUT).as_posix(),
                "label": label,
                "icao_type": row["typedesig"].strip().upper(),
                "model": row.get("model", "").strip(),
                "physical_airframe_id": row.get("hex_id", "").strip().upper(),
                "session_id": row.get("session", "").strip(),
                "source_split": row.get("train-test", "").strip(),
                "source_doi": DOI,
                "license": LICENSE,
            })

    if missing:
        raise RuntimeError(f"{len(missing)} WAV files were missing; first: {missing[:5]}")

    labels = sorted({str(row["label"]) for row in manifest})
    counts = Counter(str(row["label"]) for row in manifest)
    airframes: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        airframes[str(row["label"])].add(str(row["physical_airframe_id"]))

    with (OUTPUT / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    coverage = {
        "dataset": "AeroSonicDB YPAD-0523 v1.1.2",
        "source_doi": DOI,
        "license": LICENSE,
        "recordings": len(manifest),
        "classes": len(labels),
        "physical_airframes": len({str(row["physical_airframe_id"]) for row in manifest}),
        "warning": "Classes with fewer than 3 physical airframes cannot have a valid independent train/validation/test evaluation.",
        "types": [
            {
                "label": label,
                "recordings": counts[label],
                "physical_airframes": len(airframes[label]),
                "status": "INDEPENDENT_TEST_READY" if len(airframes[label]) >= 3 else "MORE_DATA_REQUIRED",
            }
            for label in labels
        ],
    }
    (OUTPUT / "coverage_report.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "README.txt").write_text(
        "AIRCRAFT AST 41 CLASS V1\n"
        f"Source: AeroSonicDB YPAD-0523 v1.1.2, DOI {DOI}\n"
        f"License: {LICENSE}\n"
        "No synthetic or duplicated source recordings are included.\n"
        "manifest.csv contains physical-airframe IDs for leakage-safe splitting.\n",
        encoding="utf-8",
    )
    trainer = ROOT / "kaggle_train_aircraft_ast_41_v1.py"
    if trainer.exists():
        shutil.copy2(trainer, OUTPUT / trainer.name)

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    shutil.make_archive(str(OUTPUT_ZIP.with_suffix("")), "zip", OUTPUT.parent, OUTPUT.name)

    print(f"Dataset: {OUTPUT}")
    print(f"ZIP: {OUTPUT_ZIP}")
    print(f"Recordings: {len(manifest)} | classes: {len(labels)}")
    for item in coverage["types"]:
        print(f"{item['label']:<35} recordings={item['recordings']:>3} airframes={item['physical_airframes']:>3} {item['status']}")


if __name__ == "__main__":
    main()
