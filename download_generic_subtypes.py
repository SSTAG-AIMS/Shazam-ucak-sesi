"""Selectively download balanced subtype clips from Generic Audio Samples."""

from __future__ import annotations

import argparse
import csv
import io
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf
from remotezip import RemoteZip


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Self_Data" / "CATEGORY_SUBTYPES_350"
MANIFEST = ROOT / "cache" / "category_subtypes_350.csv"
DATASET = "lokeshbhaskarnr/generic-audio-samples"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
LICENSE = "MIT (Kaggle dataset page); preserve original dataset attribution"

REMOTE_TO_LABEL = {
    "DATASET/Vehicles/car/": ("TRAFFIC", "CAR"),
    "DATASET/Vehicles/bus/": ("TRAFFIC", "BUS"),
    "DATASET/Vehicles/truck/": ("TRAFFIC", "TRUCK"),
    "DATASET/Vehicles/bike/": ("TRAFFIC", "MOTORCYCLE"),
    "DATASET/Vehicles/train/": ("TRAFFIC", "TRAIN"),
    "DATASET/Vehicles/bicycle/": ("TRAFFIC", "BICYCLE"),
    "DATASET/Animals/cat/": ("OTHER", "CAT"),
    "DATASET/Animals/dog/": ("OTHER", "DOG"),
    "DATASET/Birds/crow/": ("OTHER", "CROW"),
    "DATASET/Birds/sparrow/": ("OTHER", "SPARROW"),
    "DATASET/Birds/parrot/": ("OTHER", "PARROT"),
    "DATASET/Birds/peacock/": ("OTHER", "PEACOCK"),
}


def source_group(name: str) -> str:
    """Keep fragments from one original recording in the same split."""
    stem = Path(name).stem
    return re.sub(r"_part_\d+$", "", stem, flags=re.IGNORECASE)


def choose_balanced(names: list[str], per_subtype: int) -> list[tuple[str, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in sorted(names):
        grouped[source_group(name)].append(name)
    split_targets = {
        "train": int(per_subtype * 0.70),
        "validation": int(per_subtype * 0.15),
    }
    split_targets["test"] = per_subtype - sum(split_targets.values())
    result: list[tuple[str, str]] = []
    group_items = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    split_counts = {split: 0 for split in split_targets}
    split_groups: dict[str, list[list[str]]] = defaultdict(list)
    for _, clips in group_items:
        split = max(
            split_targets,
            key=lambda key: split_targets[key] - split_counts[key],
        )
        split_groups[split].append(clips)
        split_counts[split] += len(clips)
    for split, target in split_targets.items():
        candidates = [
            name for group in split_groups[split] for name in group
        ]
        result.extend((name, split) for name in candidates[:target])
    return result


def augment_audio(audio_bytes: bytes, variant: int) -> bytes:
    samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sr != 22050:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=22050)
        sr = 22050
    rng = np.random.default_rng(900_000 + variant)
    mono = np.roll(mono, int(rng.integers(-600, 601)))
    mono = mono * rng.uniform(0.85, 1.10)
    mono += rng.normal(0.0, rng.uniform(0.0002, 0.0015), len(mono))
    output = io.BytesIO()
    sf.write(output, np.clip(mono, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--per-subtype", type=int, default=350)
    args = parser.parse_args()

    response = requests.get(DOWNLOAD_URL, stream=True, timeout=60)
    response.raise_for_status()
    signed_url = response.url
    response.close()
    rows = []
    with RemoteZip(signed_url, timeout=(10, 30)) as archive:
        all_names = archive.namelist()
    thread_state = threading.local()

    def download_one(task: tuple[str, Path]) -> Path:
        remote_name, destination = task
        if destination.exists():
            return destination
        temporary = destination.with_suffix(destination.suffix + ".part")
        last_error = None
        for attempt in range(1, 6):
            try:
                if not hasattr(thread_state, "archive"):
                    thread_state.archive = RemoteZip(
                        signed_url, timeout=(10, 30)
                    )
                payload = thread_state.archive.read(remote_name)
                temporary.write_bytes(payload)
                temporary.replace(destination)
                return destination
            except Exception as error:
                last_error = error
                archive = getattr(thread_state, "archive", None)
                if archive is not None:
                    try:
                        archive.close()
                    except Exception:
                        pass
                    del thread_state.archive
                if temporary.exists():
                    temporary.unlink()
                time.sleep(min(10, attempt * 2))
        raise RuntimeError(
            f"5 denemede indirilemedi: {remote_name}"
        ) from last_error

    with ThreadPoolExecutor(max_workers=12) as pool:
        for prefix, (category, subtype) in REMOTE_TO_LABEL.items():
            candidates = [
                name for name in all_names
                if name.startswith(prefix) and name.lower().endswith(".wav")
            ]
            selected = choose_balanced(candidates, args.per_subtype)
            subtype_rows = []
            tasks = []
            for remote_name, split in selected:
                destination_dir = args.output / category / subtype / split
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / Path(remote_name).name
                tasks.append((remote_name, destination))
            for index, ((remote_name, split), destination) in enumerate(
                zip(selected, pool.map(download_one, tasks)), 1
            ):
                row = {
                        "path": str(destination),
                        "category": category,
                        "subtype": subtype,
                        "split": split,
                        "source_group": source_group(remote_name),
                        "source_dataset": DATASET,
                        "source_remote_path": remote_name,
                        "license": LICENSE,
                        "augmented": False,
                    }
                rows.append(row)
                subtype_rows.append(row)
                if index % 50 == 0 or index == len(selected):
                    print(
                        f"{category}/{subtype}: {index}/{len(selected)}",
                        flush=True,
                    )

            targets = {
                "train": int(args.per_subtype * 0.70),
                "validation": int(args.per_subtype * 0.15),
            }
            targets["test"] = args.per_subtype - sum(targets.values())
            for split, target in targets.items():
                natural = [row for row in subtype_rows if row["split"] == split]
                if not natural:
                    raise RuntimeError(f"{category}/{subtype}/{split}: doğal klip yok")
                missing = target - len(natural)
                for variant in range(missing):
                    source = natural[variant % len(natural)]
                    source_path = Path(source["path"])
                    destination = (
                        source_path.parent
                        / f"{subtype.lower()}_aug_{variant:04d}.wav"
                    )
                    if not destination.exists():
                        destination.write_bytes(
                            augment_audio(source_path.read_bytes(), variant)
                        )
                    augmented_row = dict(source)
                    augmented_row.update(
                        {
                            "path": str(destination),
                            "source_group": source["source_group"],
                            "source_remote_path": source["source_remote_path"],
                            "augmented": True,
                        }
                    )
                    rows.append(augmented_row)
                    subtype_rows.append(augmented_row)
                if missing:
                    print(
                        f"{category}/{subtype}/{split}: {missing} artırılmış klip",
                        flush=True,
                    )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Toplam {len(rows)} klip | {args.manifest}")


if __name__ == "__main__":
    main()
