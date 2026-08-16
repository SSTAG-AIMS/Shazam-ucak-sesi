"""Build an isolated 3,000-clip aircraft Shazam reference catalogue.

Every clip is a non-overlapping five-second excerpt of an AeroSonicDB recording
whose aircraft identity was supplied by ADS-B metadata.  A clip is not presented
as an independent flight recording: parent recording, physical airframe and time
range remain in the manifest.  Airframes reserved by the blind lab are excluded.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf

from aircraft_fingerprint import FingerprintConfig, fingerprint_samples, load_audio
from dataset_catalog import sha256_file


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Test_Folder" / "SHAZAM_AIRCRAFT_3000_V1"
CLIPS = OUTPUT / "ACCEPTED_REFERENCE_CLIPS"
DATABASE = ROOT / "models" / "aircraft_fingerprints_3000.sqlite3"
MANIFEST = OUTPUT / "accepted_aircraft_3000.jsonl"
REPORT = OUTPUT / "index_manifest.json"
TARGET = 3_000
CLIP_SECONDS = 5.0


def _normalise(value: str) -> str:
    return str(value).strip().upper().replace("-", "_").replace(" ", "_")


def _load_source_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in (
        "Self_Data/AIRCRAFT_TYPES/references_manifest.json",
        "Self_Data/AIRCRAFT_100_REFERENCE_V1/references_manifest.json",
    ):
        rows.extend(json.loads((ROOT / relative).read_text(encoding="utf-8")))
    # The same source recording exists in both curated views; keep it once.
    unique = {str(row["source_file"]): row for row in rows}
    blind = json.loads(
        (ROOT / "Test_Folder/AIRCRAFT_REFERENCE_LAB_V1/test_manifest.json").read_text(encoding="utf-8")
    )
    reserved_airframes = {
        str(row["physical_airframe_id"]).strip().upper() for row in blind["records"]
    }
    accepted = []
    for row in unique.values():
        path = (ROOT / row["output_file"]).resolve()
        if not path.is_file() or str(row.get("hex_id", "")).upper() in reserved_airframes:
            continue
        accepted.append({**row, "_path": path})
    return accepted


def _candidate_segments(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        info = sf.info(str(row["_path"]))
        segment_count = int((info.frames / info.samplerate) // CLIP_SECONDS)
        for index in range(segment_count):
            start = index * CLIP_SECONDS
            candidates.append({
                **row,
                "subtype": _normalise(row["folder"]),
                "segment_index": index,
                "segment_start_seconds": start,
                "segment_end_seconds": start + CLIP_SECONDS,
            })
    return candidates


def _balanced(rows: Iterable[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["subtype"]].append(row)
    for values in groups.values():
        values.sort(key=lambda row: (row["source_file"], row["segment_index"]))
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < target:
        added = False
        for subtype in sorted(groups):
            if offset < len(groups[subtype]) and len(selected) < target:
                selected.append(groups[subtype][offset]); added = True
        if not added:
            break
        offset += 1
    if len(selected) < target:
        raise RuntimeError(f"Yeterli doğrulanmış uçak segmenti yok: {len(selected)}/{target}")
    return selected


def _write_clip(row: dict[str, Any]) -> dict[str, Any]:
    source: Path = row["_path"]
    info = sf.info(str(source))
    start_frame = int(round(row["segment_start_seconds"] * info.samplerate))
    frame_count = int(round(CLIP_SECONDS * info.samplerate))
    samples, sample_rate = sf.read(
        str(source), start=start_frame, frames=frame_count, dtype="float32", always_2d=False
    )
    subtype = row["subtype"]
    digest = hashlib.sha256(
        f"{source.name}|{row['segment_start_seconds']:.3f}".encode("utf-8")
    ).hexdigest()[:12]
    destination = CLIPS / subtype / f"{source.stem}__{row['segment_index']:02d}_{digest}.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), samples, sample_rate, subtype="PCM_16")
    public = {key: value for key, value in row.items() if not key.startswith("_")}
    return {
        **public,
        "category": "AIRCRAFT",
        "audio_path": str(destination.resolve()),
        "parent_source_path": str(source),
        "source_recording_id": f"AEROSONIC:{row['source_file']}",
        "physical_airframe_id": str(row.get("hex_id", "")).upper(),
        "source_uri": f"https://doi.org/{row['source_doi']}",
        "verification_status": "ADS_B_SOURCE_LABEL_VERIFIED_NOT_PROJECT_HUMAN_REVIEWED",
        "sha256": sha256_file(destination),
    }


def _fingerprint(row: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, int]]]:
    config = FingerprintConfig()
    hashes = fingerprint_samples(load_audio(row["audio_path"], config.sample_rate), config)
    if not hashes:
        raise ValueError(f"Parmak izi üretilemedi: {row['audio_path']}")
    return row, hashes


def _build_database(rows: list[dict[str, Any]], target: Path) -> int:
    temporary = target.with_suffix(".sqlite3.building")
    if temporary.exists():
        temporary.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    total_hashes = 0
    with closing(sqlite3.connect(temporary)) as connection, connection:
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA journal_mode=OFF")
        connection.executescript(
            """
            CREATE TABLE tracks (
                id INTEGER PRIMARY KEY,
                aircraft_type TEXT NOT NULL,
                reference_name TEXT NOT NULL,
                source_path TEXT NOT NULL UNIQUE,
                hash_count INTEGER NOT NULL
            );
            CREATE TABLE fingerprints (
                hash TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                anchor_time INTEGER NOT NULL,
                FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
            );
            """
        )
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="aircraft-3000") as executor:
            for number, (row, hashes) in enumerate(executor.map(_fingerprint, rows, buffersize=8), 1):
                cursor = connection.execute(
                    "INSERT INTO tracks(aircraft_type,reference_name,source_path,hash_count) VALUES(?,?,?,?)",
                    (row["subtype"], Path(row["audio_path"]).stem, row["audio_path"], len(hashes)),
                )
                track_id = int(cursor.lastrowid)
                connection.executemany(
                    "INSERT INTO fingerprints(hash,track_id,anchor_time) VALUES(?,?,?)",
                    ((digest, track_id, anchor) for digest, anchor in hashes),
                )
                total_hashes += len(hashes)
                if number % 100 == 0:
                    print(f"[Aircraft Shazam 3000] {number}/{len(rows)}", flush=True)
        connection.execute("CREATE INDEX idx_fingerprints_hash ON fingerprints(hash)")
        connection.commit(); connection.execute("VACUUM")
    os.replace(temporary, target)
    return total_hashes


def build() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = _balanced(_candidate_segments(_load_source_rows()), TARGET)
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="aircraft-clip") as executor:
        rows = list(executor.map(_write_clip, selected, buffersize=8))
    wanted = {Path(row["audio_path"]).resolve() for row in rows}
    stale = 0
    if CLIPS.exists():
        for path in CLIPS.rglob("*.wav"):
            if path.resolve() not in wanted:
                path.unlink(); stale += 1
    with MANIFEST.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    fingerprint_count = _build_database(rows, DATABASE)
    counts = Counter(row["subtype"] for row in rows)
    report = {
        "catalog_kind": "ADS_B_SOURCE_VERIFIED_AIRCRAFT_SEGMENTS",
        "accepted_clip_count": len(rows),
        "parent_recording_count": len({row["source_recording_id"] for row in rows}),
        "physical_airframe_count": len({row["physical_airframe_id"] for row in rows}),
        "aircraft_subtype_count": len(counts),
        "clip_seconds": CLIP_SECONDS,
        "fingerprint_count": fingerprint_count,
        "database": str(DATABASE.resolve()),
        "manifest": str(MANIFEST.resolve()),
        "stale_clips_removed": stale,
        "subtype_counts": dict(sorted(counts.items())),
        "important_note": "3000 clips are excerpts of verified parent recordings, not 3000 independent flights.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "README_SUNUM.txt").write_text(
        "Bu katalog 3000 adet çakışmayan 5 saniyelik uçak referans klibi içerir.\n"
        "Klipler AeroSonicDB'nin ADS-B etiketli kaynak kayıtlarından üretilmiştir.\n"
        "3000 bağımsız uçuş değildir; kaynak kayıt, fiziksel uçak ve zaman aralığı manifestte kanıtlanır.\n"
        "İzole testte ayrılan fiziksel uçaklar kataloğa alınmamıştır.\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
