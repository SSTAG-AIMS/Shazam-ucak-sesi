"""Add a moderate, source-audited FSD50K layer to Category Shazam."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict, deque
from contextlib import closing
from pathlib import Path

from category_fingerprint_v1 import CategoryFingerprintDatabaseV1


ROOT = Path(__file__).resolve().parent
FSD_ROOT = ROOT / "downloads" / "FSD50K"
GROUND_TRUTH = FSD_ROOT / "ground_truth" / "FSD50K.ground_truth" / "eval.csv"
AUDIO_ROOT = FSD_ROOT / "audio" / "FSD50K.eval_audio"
CLIP_INFO = FSD_ROOT / "metadata" / "FSD50K.metadata" / "eval_clips_info_FSD50K.json"
BASE_MANIFEST = ROOT / "cache" / "category_shazam_extended_v1.csv"
OUTPUT_MANIFEST = ROOT / "cache" / "category_shazam_ideal_v1.csv"
DATABASE = ROOT / "models" / "category_fingerprints_full_v1.sqlite3"
REPORT = ROOT / "outputs" / "category_shazam_ideal_v1_report.json"

TARGET_POOL = {"AMBIENT": 400, "SPEECH": 435, "WIND": 348, "LOGISTICS": 396}
LABEL_GROUPS = {
    "AMBIENT": (
        "Domestic_sounds_and_home_sounds", "Mechanical_fan", "Printer",
        "Computer_keyboard", "Clock", "Typing",
    ),
    "SPEECH": (
        "Speech", "Conversation", "Crowd", "Child_speech_and_kid_speaking",
        "Female_speech_and_woman_speaking", "Male_speech_and_man_speaking",
    ),
    "WIND": ("Wind", "Rain", "Thunderstorm", "Thunder", "Ocean", "Waves_and_surf"),
    "LOGISTICS": ("Engine", "Engine_starting", "Idling", "Power_tool", "Tools", "Drill", "Hammer"),
}
FORBIDDEN = {
    "Aircraft", "Fixed-wing_aircraft_and_airplane", "Animal", "Music",
    "Musical_instrument", "Singing",
}


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _select_balanced(
    rows: list[dict], category: str, limit: int, excluded: set[str]
) -> list[dict]:
    groups: dict[str, deque[dict]] = defaultdict(deque)
    priorities = LABEL_GROUPS[category]
    for row in rows:
        if row["fname"] in excluded:
            continue
        labels = set(row["labels"].split(","))
        if labels & FORBIDDEN:
            continue
        subtype = next((label for label in priorities if label in labels), None)
        if subtype is None:
            continue
        row = dict(row)
        row["selected_subtype"] = subtype
        groups[subtype].append(row)
    for subtype, values in groups.items():
        groups[subtype] = deque(sorted(values, key=lambda row: _stable_key(row["fname"])))

    selected = []
    while len(selected) < limit:
        added = False
        for subtype in priorities:
            if groups[subtype] and len(selected) < limit:
                selected.append(groups[subtype].popleft())
                added = True
        if not added:
            break
    return selected


def build() -> dict:
    with GROUND_TRUTH.open("r", encoding="utf-8", newline="") as handle:
        ground_truth = list(csv.DictReader(handle))
    clip_info = json.loads(CLIP_INFO.read_text(encoding="utf-8"))
    with BASE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))

    additions = []
    used_source_ids: set[str] = set()
    for category, limit in TARGET_POOL.items():
        selected = _select_balanced(
            ground_truth, category, limit, used_source_ids
        )
        if len(selected) < int(limit * 0.90):
            raise RuntimeError(f"{category}: yalnızca {len(selected)}/{limit} uygun kayıt")
        used_source_ids.update(row["fname"] for row in selected)
        train_end = int(round(len(selected) * 0.70))
        validation_end = train_end + int(round(len(selected) * 0.15))
        for index, row in enumerate(selected):
            split = "train" if index < train_end else "validation" if index < validation_end else "test"
            info = clip_info[str(row["fname"])]
            additions.append({
                "path": str((AUDIO_ROOT / f"{row['fname']}.wav").resolve()),
                "category": category,
                "subtype": row["selected_subtype"].upper(),
                "split": split,
                "source_group": f"FSD50K:{row['fname']}",
                "source_dataset": "FSD50K eval",
                "license": info.get("license", ""),
            })
    missing = [row["path"] for row in additions if not Path(row["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} FSD50K dosyası eksik")

    all_rows = manifest_rows + additions
    with OUTPUT_MANIFEST.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    temporary = DATABASE.with_suffix(".sqlite3.ideal_building")
    if temporary.exists():
        temporary.unlink()
    database = CategoryFingerprintDatabaseV1(temporary)
    database.reset()
    clip_counts: Counter[str] = Counter()
    hash_counts: Counter[str] = Counter()
    training = [row for row in all_rows if row["split"] == "train"]
    for index, row in enumerate(training, 1):
        count = database.add_reference(row["path"], row["category"], row["subtype"])
        if row["source_dataset"] == "FSD50K eval":
            clip_counts[row["category"]] += 1
            hash_counts[row["category"]] += count
        if index % 100 == 0 or index == len(training):
            print(f"[FSD50K Category Shazam] {index}/{len(training)}", flush=True)
    database.compact()

    with closing(sqlite3.connect(temporary)) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        total_tracks = connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        total_hashes = connection.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]
        orphan_count = connection.execute(
            "SELECT COUNT(*) FROM fingerprints f LEFT JOIN tracks t ON t.id=f.track_id "
            "WHERE t.id IS NULL"
        ).fetchone()[0]
    if integrity != "ok" or orphan_count:
        raise RuntimeError(f"SQLite doğrulaması başarısız: {integrity}, orphan={orphan_count}")
    os.replace(temporary, DATABASE)

    report = {
        "added_training_clips": dict(sorted(clip_counts.items())),
        "added_fingerprints": dict(sorted(hash_counts.items())),
        "reserved_validation_clips": dict(Counter(
            row["category"] for row in additions if row["split"] == "validation"
        )),
        "reserved_test_clips": dict(Counter(
            row["category"] for row in additions if row["split"] == "test"
        )),
        "catalog_tracks": total_tracks,
        "catalog_fingerprints": total_hashes,
        "sqlite_integrity": integrity,
        "orphan_fingerprints": orphan_count,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
