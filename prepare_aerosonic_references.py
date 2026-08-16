"""Prepare a balanced aircraft-type reference library from AeroSonicDB.

The script reads audio directly from the official ZIP archive, crops each
aircraft event using the dataset's strong annotations, resamples it and writes
the selected files into ``Self_Data/AIRCRAFT_TYPES/<TYPE>/``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "Self_Data" / "AeroSonicDB_source"
DEFAULT_OUTPUT = PROJECT_ROOT / "Self_Data" / "AIRCRAFT_TYPES"
TYPE_TO_FOLDER = {
    "A320": "AIRBUS_A320",
    "B738": "BOEING_737_800",
    "E190": "EMBRAER_E190",
    "PC12": "PILATUS_PC12",
    "F100": "FOKKER_100",
    "SF34": "SAAB_340",
    "DH8C": "DASH_8_300",
    "DA42": "DIAMOND_DA42",
}


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def build_type_mapping(rows: list[dict[str, str]], all_types: bool) -> dict[str, str]:
    """Build stable labels while preserving the eight already trained names."""
    mapping = dict(TYPE_TO_FOLDER)
    if not all_types:
        return mapping
    for row in rows:
        code = str(row.get("typedesig") or "").strip().upper()
        if not code or code in mapping:
            continue
        model = _safe_label(str(row.get("model") or code))
        mapping[code] = f"ICAO_{code}_{model}" if model != code else f"ICAO_{code}"
    return mapping


def _number(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        row for row in rows
        if row.get("class") == "1" and row.get("typedesig")
    ]


def balanced_selection(
    rows: list[dict[str, str]], per_type: int, type_mapping: dict[str, str]
) -> list[dict[str, str]]:
    """Prefer different physical aircraft, then fill with distinct sessions."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["typedesig"]].append(row)

    selected: list[dict[str, str]] = []
    for type_code in type_mapping:
        candidates = sorted(
            grouped[type_code],
            key=lambda row: (
                row.get("hex_id", ""),
                row.get("session", ""),
                row.get("filename", ""),
            ),
        )
        chosen: list[dict[str, str]] = []
        used_aircraft: set[str] = set()
        used_sessions: set[str] = set()

        for row in candidates:
            aircraft = row.get("hex_id", "")
            session = row.get("session", "")
            if aircraft and aircraft not in used_aircraft:
                chosen.append(row)
                used_aircraft.add(aircraft)
                used_sessions.add(session)
            if len(chosen) >= per_type:
                break

        if len(chosen) < per_type:
            chosen_files = {row["filename"] for row in chosen}
            for row in candidates:
                if row["filename"] in chosen_files:
                    continue
                session = row.get("session", "")
                if session and session in used_sessions:
                    continue
                chosen.append(row)
                used_sessions.add(session)
                if len(chosen) >= per_type:
                    break

        # Some rare types have fewer distinct aircraft/sessions than the
        # requested quota. Fill the remaining slots with other recordings,
        # while still preventing the same file from being selected twice.
        if len(chosen) < per_type:
            chosen_files = {row["filename"] for row in chosen}
            for row in candidates:
                if row["filename"] in chosen_files:
                    continue
                chosen.append(row)
                chosen_files.add(row["filename"])
                if len(chosen) >= per_type:
                    break

        selected.extend(chosen[:per_type])
    return selected


def zip_member_lookup(archive: zipfile.ZipFile) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for member in archive.namelist():
        if member.lower().endswith(".wav"):
            lookup[Path(member).name] = member
    return lookup


def crop_event(audio_bytes: bytes, row: dict[str, str], target_sr: int) -> np.ndarray:
    samples, source_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = np.mean(samples, axis=1)
    offset = max(0.0, _number(row.get("offset", "0")))
    duration = _number(row.get("duration", "0"), len(mono) / source_sr)
    start = min(len(mono), int(round(offset * source_sr)))
    end = min(len(mono), int(round((offset + duration) * source_sr)))
    event = mono[start:end]
    if event.size == 0:
        raise ValueError(f"Boş event aralığı: {row['filename']}")
    if source_sr != target_sr:
        event = librosa.resample(event, orig_sr=source_sr, target_sr=target_sr)
    peak = float(np.max(np.abs(event)))
    if peak > 1.0:
        event = event / peak
    return np.asarray(event, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument(
        "--all-types", action="store_true",
        help="Arşivdeki tüm ICAO tiplerini hazırlar (varsayılan: eğitilmiş 8 tip).",
    )
    args = parser.parse_args()
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    metadata_path = args.source_dir / "sample_meta.csv"
    archive_path = args.source_dir / "audio.zip"
    if not metadata_path.exists() or not archive_path.exists():
        raise SystemExit(
            f"Eksik kaynak. Beklenen dosyalar: {metadata_path} ve {archive_path}"
        )

    metadata = load_metadata(metadata_path)
    type_mapping = build_type_mapping(metadata, args.all_types)
    metadata = [row for row in metadata if row.get("typedesig") in type_mapping]
    rows = balanced_selection(metadata, args.per_type, type_mapping)
    manifest: list[dict] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = zip_member_lookup(archive)
        for row in rows:
            filename = row["filename"]
            member = members.get(filename)
            if member is None:
                print(f"[ATLA] Arşivde bulunamadı: {filename}")
                continue
            folder = type_mapping[row["typedesig"]]
            destination_dir = args.output_dir / folder
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / filename
            event = crop_event(archive.read(member), row, args.sample_rate)
            sf.write(destination, event, args.sample_rate, subtype="PCM_16")
            manifest.append(
                {
                    "output_file": str(destination.relative_to(PROJECT_ROOT)),
                    "source_file": filename,
                    "aircraft_type": row["typedesig"],
                    "folder": folder,
                    "hex_id": row.get("hex_id", ""),
                    "registration": row.get("reg", ""),
                    "manufacturer": row.get("manu", ""),
                    "model": row.get("model", ""),
                    "engine_model": row.get("engmodel", ""),
                    "session": row.get("session", ""),
                    "offset_seconds": _number(row.get("offset", "0")),
                    "duration_seconds": len(event) / args.sample_rate,
                    "license": "CC BY-NC 4.0",
                    "source_doi": "10.5281/zenodo.10215080",
                    "reference_strength": "STRONG" if sum(
                        1 for candidate in rows
                        if candidate.get("typedesig") == row.get("typedesig")
                    ) >= 3 else "WEAK_RARE_TYPE",
                }
            )
            print(f"[OK] {folder:18} {filename}")

    manifest_path = args.output_dir / "references_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    counts: dict[str, int] = defaultdict(int)
    for item in manifest:
        counts[item["folder"]] += 1
    print(f"\nManifest: {manifest_path}")
    for folder in type_mapping.values():
        print(f"  {folder:18} {counts[folder]} kayıt")


if __name__ == "__main__":
    main()
