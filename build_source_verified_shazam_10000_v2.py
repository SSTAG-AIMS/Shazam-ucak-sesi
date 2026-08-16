"""Build a 10,000-track, source-labelled Shazam demonstration catalog.

The catalog is intentionally isolated from the production and human-approved
indexes.  It combines only records whose labels and licences are present in
their source metadata.  Source-labelled does *not* mean project-human-reviewed.
Independent tests are excluded by source recording id and exact SHA-256.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from aircraft_fingerprint import FingerprintConfig, fingerprint_samples, load_audio
from dataset_catalog import sha256_file


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Test_Folder" / "SHAZAM_10000_SOURCE_VERIFIED_V2"
DATABASE = OUTPUT / "source_verified_10000.sqlite3"
TARGET_ACCEPTED = 10_000

ESC_ROOT = ROOT / "downloads" / "ESC-50" / "ESC-50-master"
FSD_ROOT = ROOT / "downloads" / "FSD50K"

ALLOWED_FSD_LICENSES = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
    "http://creativecommons.org/licenses/by-nc/3.0/": "CC-BY-NC-3.0",
}

AIRCRAFT_LABELS = {"aircraft", "fixed-wing_aircraft_and_airplane"}
TRAFFIC_LABELS = {
    "accelerating_and_revving_and_vroom", "bicycle", "bicycle_bell",
    "boat_and_water_vehicle", "bus", "car", "car_passing_by", "engine",
    "engine_starting", "idling", "motor_vehicle_(road)", "motorcycle",
    "race_car_and_auto_racing", "rail_transport", "siren", "skateboard",
    "subway_and_metro_and_underground", "traffic_noise_and_roadway_noise",
    "train", "truck", "vehicle", "vehicle_horn_and_car_horn_and_honking",
}
SPEECH_LABELS = {
    "chatter", "child_speech_and_kid_speaking", "conversation",
    "female_singing", "female_speech_and_woman_speaking", "human_voice",
    "male_singing", "male_speech_and_man_speaking", "screaming", "shout",
    "singing", "speech", "speech_synthesizer", "whispering", "yell",
}
WIND_LABELS = {"wind"}
AMBIENT_LABELS = {
    "cricket", "drip", "fire", "frog", "insect", "liquid", "ocean",
    "rain", "raindrop", "stream", "thunder", "thunderstorm", "trickle_and_dribble",
    "water", "waves_and_surf", "wild_animals", "wind_chime",
}

ESC_AIRCRAFT = {"airplane", "helicopter"}
ESC_TRAFFIC = {"car_horn", "engine", "siren", "train"}
ESC_SPEECH = {
    "breathing", "coughing", "crying_baby", "laughing", "sneezing", "snoring",
}
ESC_WIND = {"wind"}
ESC_AMBIENT = {
    "chirping_birds", "crackling_fire", "crickets", "frog", "insects", "rain",
    "sea_waves", "thunderstorm", "water_drops",
}


def _normalise(value: str) -> str:
    return str(value).strip().upper().replace("-", "_").replace(" ", "_")


def _category_for(labels: Iterable[str]) -> tuple[str, str]:
    values = [str(label).strip() for label in labels if str(label).strip()]
    lowered = {value.lower(): value for value in values}
    for category, candidates in (
        ("AIRCRAFT", AIRCRAFT_LABELS),
        ("TRAFFIC", TRAFFIC_LABELS),
        ("SPEECH", SPEECH_LABELS),
        ("WIND", WIND_LABELS),
        ("AMBIENT", AMBIENT_LABELS),
    ):
        matches = sorted(candidates & lowered.keys())
        if matches:
            return category, _normalise(lowered[matches[0]])
    return "OTHER", _normalise(sorted(values, key=str.casefold)[0])


def _esc_category(label: str) -> str:
    if label in ESC_AIRCRAFT:
        return "AIRCRAFT"
    if label in ESC_TRAFFIC:
        return "TRAFFIC"
    if label in ESC_SPEECH:
        return "SPEECH"
    if label in ESC_WIND:
        return "WIND"
    if label in ESC_AMBIENT:
        return "AMBIENT"
    return "OTHER"


def _stable_test(source_id: str) -> bool:
    """Reserve roughly ten percent without depending on local file order."""
    return int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) % 10 == 0


def _aircraft_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_manifest = ROOT / "Self_Data" / "AIRCRAFT_100_REFERENCE_V1" / "references_manifest.json"
    test_manifest = ROOT / "Test_Folder" / "AIRCRAFT_REFERENCE_LAB_V1" / "test_manifest.json"
    source_rows = json.loads(source_manifest.read_text(encoding="utf-8"))
    test_payload = json.loads(test_manifest.read_text(encoding="utf-8"))
    test_ids = {str(row["physical_airframe_id"]).upper() for row in test_payload["records"]}
    accepted: list[dict[str, Any]] = []
    for row in source_rows:
        path = (ROOT / row["output_file"]).resolve()
        source_id = str(row.get("hex_id", "")).upper()
        if path.is_file() and source_id not in test_ids:
            accepted.append({
                "path": str(path), "category": "AIRCRAFT", "subtype": _normalise(row["folder"]),
                "source_recording_id": f"AEROSONIC:{source_id}",
                "source_uri": f"https://doi.org/{row['source_doi']}", "license": row["license"],
                "attribution": "AeroSonicDB / ADS-B-ICAO metadata",
                "source_labels": [_normalise(row["folder"])], "dataset": "AeroSonicDB",
            })
    tests: list[dict[str, Any]] = []
    for row in test_payload["records"]:
        path = Path(row["audio_path"]).resolve()
        if path.is_file():
            tests.append({
                "path": str(path), "category": "AIRCRAFT", "subtype": _normalise(row["aircraft_type"]),
                "source_recording_id": f"AEROSONIC:{str(row['physical_airframe_id']).upper()}",
                "source_uri": row["source_uri"], "license": row["license"],
                "attribution": "AeroSonicDB / ADS-B-ICAO metadata",
                "source_labels": [_normalise(row["aircraft_type"])], "dataset": "AeroSonicDB",
            })
    return accepted, tests


def _esc_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = ESC_ROOT / "meta" / "esc50.csv"
    if not metadata.is_file():
        raise FileNotFoundError(f"ESC-50 metadata bulunamadı: {metadata}")
    records: list[tuple[int, dict[str, Any]]] = []
    with metadata.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            path = (ESC_ROOT / "audio" / row["filename"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            label = row["category"].strip().lower()
            record = {
                "path": str(path), "category": _esc_category(label), "subtype": _normalise(label),
                "source_recording_id": f"ESC50:{row['src_file']}",
                "source_uri": "https://doi.org/10.7910/DVN/YDEPUT",
                "license": "CC-BY-NC-3.0 (dataset; ESC-10 is CC-BY-3.0)",
                "attribution": "ESC-50, Karol J. Piczak; original Freesound credits in ESC-50 LICENSE",
                "source_labels": [_normalise(label)], "dataset": "ESC-50", "source_fold": int(row["fold"]),
            }
            records.append((int(row["fold"]), record))
    # ESC folds are clip-level. Two original Freesound sources occur in more
    # than one fold, so reserve every derivative when any derivative is fold 5.
    test_source_ids = {
        record["source_recording_id"] for fold, record in records if fold == 5
    }
    accepted = [record for _, record in records if record["source_recording_id"] not in test_source_ids]
    tests = [record for _, record in records if record["source_recording_id"] in test_source_ids]
    return accepted, tests


def _find_fsd_audio() -> Path:
    candidates = [path for path in FSD_ROOT.rglob("*.wav") if path.stem.isdigit()]
    if len(candidates) < 10_000:
        raise FileNotFoundError(
            f"FSD50K eval sesleri henüz hazır değil ({len(candidates)}/10231 WAV). "
            "Önce resmi çok parçalı arşivi çıkarın."
        )
    return candidates[0].parent


def _fsd_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audio_root = _find_fsd_audio()
    info_path = FSD_ROOT / "metadata" / "FSD50K.metadata" / "eval_clips_info_FSD50K.json"
    labels_path = FSD_ROOT / "metadata" / "FSD50K.metadata" / "collection" / "collection_eval.csv"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    accepted: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    with labels_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            clip_id = row["fname"]
            metadata = info[clip_id]
            license_uri = metadata["license"]
            if license_uri not in ALLOWED_FSD_LICENSES:
                continue
            labels = [value.strip() for value in row["labels"].split(",") if value.strip()]
            category, subtype = _category_for(labels)
            path = (audio_root / f"{clip_id}.wav").resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            source_id = f"FREESOUND:{clip_id}"
            record = {
                "path": str(path), "category": category, "subtype": subtype,
                "source_recording_id": source_id, "source_uri": f"https://freesound.org/s/{clip_id}/",
                "license": ALLOWED_FSD_LICENSES[license_uri],
                "license_uri": license_uri, "attribution": f"{metadata.get('uploader', '')}: {metadata.get('title', '')}",
                "source_labels": [_normalise(value) for value in labels], "dataset": "FSD50K-eval",
            }
            (tests if _stable_test(source_id) else accepted).append(record)
    return accepted, tests


def _balanced(rows: Iterable[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["subtype"])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: row["source_recording_id"])
    selected: list[dict[str, Any]] = []
    offset = 0
    keys = sorted(groups)
    while len(selected) < limit:
        added = False
        for key in keys:
            if offset < len(groups[key]) and len(selected) < limit:
                selected.append(groups[key][offset]); added = True
        if not added:
            break
        offset += 1
    selected_ids = {row["source_recording_id"] for row in selected}
    surplus = [row for values in groups.values() for row in values if row["source_recording_id"] not in selected_ids]
    return selected, surplus


def _unique_rows(
    rows: Iterable[dict[str, Any]],
    *,
    seen_source_ids: set[str] | None = None,
    seen_hashes: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], set[str]]:
    """Count one independent source and one exact waveform only once."""
    source_ids = set(seen_source_ids or ())
    hashes = set(seen_hashes or ())
    unique: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for row in rows:
        source_id = row["source_recording_id"]
        digest = sha256_file(Path(row["path"]))
        enriched = {**row, "_source_sha256": digest}
        if source_id in source_ids or digest in hashes:
            duplicates.append(enriched)
            continue
        source_ids.add(source_id); hashes.add(digest); unique.append(enriched)
    return unique, duplicates, source_ids, hashes


def _materialize(row: dict[str, Any], destination_root: Path, split: str) -> dict[str, Any]:
    source = Path(row["path"])
    destination = destination_root / row["category"] / row["subtype"] / f"{row['dataset']}_{source.name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    public_row = {key: value for key, value in row.items() if not key.startswith("_")}
    return {
        **public_row, "source_path": str(source), "audio_path": str(destination.resolve()),
        "sha256": row.get("_source_sha256") or sha256_file(destination), "split": split,
        "verification_status": "SOURCE_LABEL_VERIFIED_NOT_HUMAN_REVIEWED",
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prune_generated(root: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Remove stale hardlinks from a generated catalog view only."""
    resolved_root = root.resolve()
    output_root = OUTPUT.resolve()
    if resolved_root == output_root or output_root not in resolved_root.parents:
        raise RuntimeError(f"Güvensiz temizleme hedefi: {resolved_root}")
    wanted = {Path(row["audio_path"]).resolve() for row in rows}
    removed = 0
    if resolved_root.is_dir():
        for path in resolved_root.rglob("*"):
            if path.is_file() and path.resolve() not in wanted:
                path.unlink(); removed += 1
        for directory in sorted((path for path in resolved_root.rglob("*") if path.is_dir()), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass
    return removed


def _fingerprint_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, int]]]:
    config = FingerprintConfig()
    samples = load_audio(row["audio_path"], config.sample_rate)
    hashes = fingerprint_samples(samples, config)
    if not hashes:
        raise ValueError(f"Parmak izi üretilemedi: {row['audio_path']}")
    return row, hashes


def _is_fingerprintable(row: dict[str, Any]) -> bool:
    config = FingerprintConfig()
    try:
        return bool(fingerprint_samples(load_audio(row["path"], config.sample_rate), config))
    except Exception:
        return False


def _ensure_fingerprintable(
    selected: list[dict[str, Any]], reserves: list[dict[str, Any]], target: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace silent/unfingerprintable selections without inflating count."""
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="fingerprint-gate") as executor:
        selected_flags = list(executor.map(_is_fingerprintable, selected, buffersize=8))
        reserve_flags = list(executor.map(_is_fingerprintable, reserves, buffersize=8))
    accepted = [row for row, valid in zip(selected, selected_flags) if valid]
    rejected = [row for row, valid in zip(selected, selected_flags) if not valid]
    valid_reserves = [row for row, valid in zip(reserves, reserve_flags) if valid]
    rejected.extend(row for row, valid in zip(reserves, reserve_flags) if not valid)
    needed = target - len(accepted)
    if needed > len(valid_reserves):
        raise RuntimeError(f"Parmak izi üretilebilir 10.000 kayıt bulunamadı; eksik: {needed - len(valid_reserves)}")
    accepted.extend(valid_reserves[:needed])
    return accepted, valid_reserves[needed:], rejected


def _build_index(rows: list[dict[str, Any]], temporary: Path) -> int:
    """Compute four tracks at a time and commit one atomic SQLite build."""
    if temporary.exists():
        temporary.unlink()
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
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="fingerprint") as executor:
            # Python 3.14 buffersize prevents thousands of decoded clips from
            # being retained while the single SQLite writer catches up.
            for number, (row, hashes) in enumerate(
                executor.map(_fingerprint_row, rows, buffersize=8), 1
            ):
                encoded = f"{row['category']}::{row['subtype']}"
                cursor = connection.execute(
                    "INSERT INTO tracks(aircraft_type, reference_name, source_path, hash_count) VALUES(?,?,?,?)",
                    (encoded, Path(row["audio_path"]).stem, row["audio_path"], len(hashes)),
                )
                track_id = int(cursor.lastrowid)
                connection.executemany(
                    "INSERT INTO fingerprints(hash,track_id,anchor_time) VALUES(?,?,?)",
                    ((digest, track_id, anchor_time) for digest, anchor_time in hashes),
                )
                total_hashes += len(hashes)
                if number % 100 == 0:
                    print(f"[Shazam 10000] {number}/{len(rows)} indekslendi", flush=True)
        connection.execute("CREATE INDEX idx_fingerprints_hash ON fingerprints(hash)")
        connection.commit()
        connection.execute("VACUUM")
    return total_hashes


def build(*, build_index: bool = True) -> dict[str, Any]:
    aircraft_accepted, aircraft_tests = _aircraft_rows()
    esc_accepted, esc_tests = _esc_rows()
    fsd_accepted, fsd_tests = _fsd_rows()
    fixed, duplicate_fixed, accepted_ids, accepted_hashes = _unique_rows(aircraft_accepted + esc_accepted)
    unique_fsd, duplicate_fsd, accepted_ids, accepted_hashes = _unique_rows(
        fsd_accepted, seen_source_ids=accepted_ids, seen_hashes=accepted_hashes
    )
    if len(fixed) > TARGET_ACCEPTED:
        raise RuntimeError("Sabit kaynak havuzu hedefi aşıyor")
    selected_fsd, surplus_fsd = _balanced(unique_fsd, TARGET_ACCEPTED - len(fixed))
    accepted_source, surplus_fsd, rejected_quality = _ensure_fingerprintable(
        fixed + selected_fsd, surplus_fsd, TARGET_ACCEPTED
    )
    if len(accepted_source) != TARGET_ACCEPTED:
        raise RuntimeError(f"10.000 bağımsız kayıt bulunamadı: {len(accepted_source)}")
    # Same-source derivatives discarded above are neither independent
    # references nor valid blind tests.  Only genuine surplus sources enter test.
    test_candidates = aircraft_tests + esc_tests + fsd_tests + surplus_fsd
    test_source, duplicate_test_sources, _, _ = _unique_rows(
        test_candidates, seen_source_ids=accepted_ids, seen_hashes=accepted_hashes
    )

    accepted_rows = [_materialize(row, OUTPUT / "ACCEPTED_SOURCE_LABELS", "accepted_reference") for row in accepted_source]
    accepted_hashes = {row["sha256"] for row in accepted_rows}
    accepted_ids = {row["source_recording_id"] for row in accepted_rows}
    test_rows: list[dict[str, Any]] = []
    duplicate_tests: list[dict[str, Any]] = []
    for row in test_source:
        materialized = _materialize(row, OUTPUT / "INDEPENDENT_TEST", "independent_test")
        if materialized["sha256"] in accepted_hashes or materialized["source_recording_id"] in accepted_ids:
            duplicate_tests.append(materialized)
        else:
            test_rows.append(materialized)
    if duplicate_tests:
        raise RuntimeError(f"Accepted/test leakage detected: {len(duplicate_tests)} records")

    stale_removed = _prune_generated(OUTPUT / "ACCEPTED_SOURCE_LABELS", accepted_rows)
    stale_removed += _prune_generated(OUTPUT / "INDEPENDENT_TEST", test_rows)

    _write_jsonl(OUTPUT / "accepted_source_labels.jsonl", accepted_rows)
    _write_jsonl(OUTPUT / "independent_test.jsonl", test_rows)

    total_hashes = 0
    if build_index:
        temporary = DATABASE.with_suffix(".sqlite3.building")
        try:
            total_hashes = _build_index(accepted_rows, temporary)
            os.replace(temporary, DATABASE)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

    category_counts = Counter(row["category"] for row in accepted_rows)
    subtype_sets: dict[str, set[str]] = defaultdict(set)
    dataset_counts = Counter(row["dataset"] for row in accepted_rows)
    license_counts = Counter(row["license"] for row in accepted_rows)
    for row in accepted_rows:
        subtype_sets[row["category"]].add(row["subtype"])
    report = {
        "catalog_kind": "SOURCE_VERIFIED_NOT_PROJECT_HUMAN_REVIEWED",
        "target_count": TARGET_ACCEPTED, "accepted_count": len(accepted_rows),
        "independent_test_count": len(test_rows), "leakage_count": 0,
        "database": str(DATABASE.resolve()) if build_index else None,
        "database_built": build_index, "fingerprint_count": total_hashes,
        "coverage": {
            category: {"audio_count": count, "subtype_count": len(subtype_sets[category])}
            for category, count in sorted(category_counts.items())
        },
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "license_counts": dict(sorted(license_counts.items())),
        "deduplicated_source_or_hash_count": len(duplicate_fixed) + len(duplicate_fsd) + len(duplicate_test_sources),
        "unfingerprintable_excluded_count": len(rejected_quality),
        "stale_generated_files_removed": stale_removed,
        "excluded_license": "CC-Sampling+ (FSD50K: 403 clips)",
        "evidence": ["accepted_source_labels.jsonl", "independent_test.jsonl", DATABASE.name],
    }
    (OUTPUT / "index_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "README_SUNUM.txt").write_text(
        "Bu katalog 10.000 kaynak etiketli kayıttan oluşur. Proje içi insan onayı değildir.\n"
        "Etiket, kaynak, lisans, atıf, SHA-256 ve kaynak kayıt kimliği accepted_source_labels.jsonl içindedir.\n"
        "INDEPENDENT_TEST dosyaları SQLite indeksine alınmaz. Kabul/test arasında SHA-256 ve kaynak kimliği sızıntısı yoktur.\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests-only", action="store_true", help="SQLite indeksini üretmeden katalogları doğrula")
    args = parser.parse_args()
    print(json.dumps(build(build_index=not args.manifests_only), ensure_ascii=False, indent=2))
