"""Build a compact, source-verified six-category audio package for Kaggle.

The source catalog remains untouched. Selected clips are converted to 16 kHz
mono FLAC because AST consumes 16 kHz mono audio and FLAC is materially smaller
than WAV without introducing lossy compression. The catalog's independent test
split is preserved and is never used for model selection.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf


SEED = 42
SAMPLE_RATE = 16_000
TRAIN_POOL_CAP_PER_CLASS = 600
TEST_CAP_PER_CLASS = 150
VALIDATION_FRACTION = 0.15
CATEGORIES = ["AIRCRAFT", "AMBIENT", "OTHER", "SPEECH", "TRAFFIC", "WIND"]

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Test_Folder" / "SHAZAM_10000_SOURCE_VERIFIED_V2"
OUTPUT = ROOT / "Self_Data" / "GENERAL_CATEGORY_6CLASS_V1"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_rank(row: dict) -> str:
    material = "|".join(
        [
            str(SEED),
            str(row.get("category", "")),
            str(row.get("source_recording_id", "")),
            str(row.get("sha256", "")),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def select_per_category(rows: list[dict], cap: int) -> list[dict]:
    selected = []
    frame = pd.DataFrame(rows)
    for category in CATEGORIES:
        current = frame[frame["category"] == category].copy()
        current["_rank"] = current.apply(lambda item: stable_rank(item.to_dict()), axis=1)
        selected.extend(current.sort_values("_rank").head(cap).drop(columns="_rank").to_dict("records"))
    return selected


def assign_training_splits(rows: list[dict]) -> list[dict]:
    output = []
    frame = pd.DataFrame(rows)
    for category in CATEGORIES:
        current = frame[frame["category"] == category].copy()
        current["_rank"] = current.apply(lambda item: stable_rank(item.to_dict()), axis=1)
        current = current.sort_values("_rank").drop(columns="_rank")
        validation_count = max(1, round(len(current) * VALIDATION_FRACTION))
        for index, row in enumerate(current.to_dict("records")):
            row["split"] = "validation" if index < validation_count else "train"
            output.append(row)
    return output


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value[:80] or "audio"


def convert(row: dict, split: str, serial: int) -> dict:
    source_path = Path(row["audio_path"])
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    category = str(row["category"])
    digest = str(row.get("sha256") or hashlib.sha256(source_path.read_bytes()).hexdigest())
    stem = safe_name(f"{row.get('dataset', 'dataset')}_{row.get('subtype', 'unknown')}_{digest[:16]}")
    destination = OUTPUT / "audio" / split / category / f"{stem}.flac"
    destination.parent.mkdir(parents=True, exist_ok=True)

    audio, _ = librosa.load(source_path, sr=SAMPLE_RATE, mono=True)
    sf.write(destination, audio, SAMPLE_RATE, format="FLAC", subtype="PCM_16")

    return {
        "path": destination.relative_to(OUTPUT).as_posix(),
        "label": category,
        "split": split,
        "source_recording_id": str(row.get("source_recording_id", "")),
        "sha256": digest,
        "subtype": str(row.get("subtype", "")),
        "dataset": str(row.get("dataset", "")),
        "source_uri": str(row.get("source_uri", "")),
        "license": str(row.get("license", "")),
        "attribution": str(row.get("attribution", "")),
        "original_audio_path": str(source_path),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(
            f"Hedef klasor bos degil: {OUTPUT}\n"
            "Mevcut paketi korumak icin islem durduruldu."
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)

    accepted = read_jsonl(SOURCE / "accepted_source_labels.jsonl")
    independent = read_jsonl(SOURCE / "independent_test.jsonl")
    accepted = [row for row in accepted if row.get("category") in CATEGORIES]
    independent = [row for row in independent if row.get("category") in CATEGORIES]

    training_pool = assign_training_splits(
        select_per_category(accepted, TRAIN_POOL_CAP_PER_CLASS)
    )
    independent_test = select_per_category(independent, TEST_CAP_PER_CLASS)
    for row in independent_test:
        row["split"] = "test"

    train_ids = {row.get("source_recording_id") for row in training_pool}
    test_ids = {row.get("source_recording_id") for row in independent_test}
    train_hashes = {row.get("sha256") for row in training_pool}
    test_hashes = {row.get("sha256") for row in independent_test}
    if train_ids & test_ids or train_hashes & test_hashes:
        raise RuntimeError("Kaynak veya SHA sizintisi bulundu; paket uretilmedi.")

    manifest_rows, failures = [], []
    combined = training_pool + independent_test
    for serial, row in enumerate(combined, 1):
        try:
            manifest_rows.append(convert(row, row["split"], serial))
        except Exception as exc:  # Preserve evidence instead of hiding failed sources.
            failures.append({"audio_path": row.get("audio_path"), "error": repr(exc)})
        if serial % 100 == 0 or serial == len(combined):
            print(f"Donusturme: {serial}/{len(combined)} | hata={len(failures)}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTPUT / "manifest.csv", index=False)
    (OUTPUT / "conversion_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "name": "GENERAL_CATEGORY_6CLASS_V1",
        "sample_rate": SAMPLE_RATE,
        "format": "FLAC PCM_16 lossless",
        "categories": CATEGORIES,
        "selection_seed": SEED,
        "train_pool_cap_per_class": TRAIN_POOL_CAP_PER_CLASS,
        "independent_test_cap_per_class": TEST_CAP_PER_CLASS,
        "validation_fraction": VALIDATION_FRACTION,
        "row_count": len(manifest),
        "conversion_failure_count": len(failures),
        "counts": {
            f"{label}/{split}": int(count)
            for (label, split), count in manifest.groupby(["label", "split"]).size().items()
        },
        "leakage": {
            "source_recording_id_overlap": 0,
            "sha256_overlap": 0,
        },
        "source_catalog": str(SOURCE),
    }
    (OUTPUT / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "README.txt").write_text(
        "GENERAL_CATEGORY_6CLASS_V1\n"
        "==========================\n"
        "Kaynak dogrulamali AIRCRAFT, AMBIENT, OTHER, SPEECH, TRAFFIC ve WIND sesleri.\n"
        "Train/validation kabul katalogundan; test ayri independent_test katalogundan gelir.\n"
        "Ayni source_recording_id veya SHA-256 train ve testte birlikte bulunmaz.\n"
        "Sesler AST girdisine uygun 16 kHz mono, kayipsiz FLAC olarak saklanir.\n"
        "Lisans ve atif bilgileri manifest.csv icinde korunur.\n",
        encoding="utf-8",
    )

    archive = ROOT / "Self_Data" / "GENERAL_CATEGORY_6CLASS_V1"
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Paket klasoru: {archive}")


if __name__ == "__main__":
    main()
