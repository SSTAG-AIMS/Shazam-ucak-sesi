"""Build an exactly balanced aircraft-type clip dataset without session leakage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from build_aircraft_type_manifest import choose_session_splits


ROOT = Path(__file__).resolve().parent
REFERENCES = ROOT / "Self_Data" / "AIRCRAFT_TYPES" / "references_manifest.json"
OUTPUT = ROOT / "Self_Data" / "AIRCRAFT_TYPE_CLIPS_350"
MANIFEST = ROOT / "cache" / "aircraft_type_clips_350.csv"
SR = 22050


def windows(samples: np.ndarray, clip_samples: int, hop_samples: int) -> list[np.ndarray]:
    if len(samples) < clip_samples:
        return [np.pad(samples, (0, clip_samples - len(samples)))]
    starts = list(range(0, len(samples) - clip_samples + 1, hop_samples))
    if starts[-1] != len(samples) - clip_samples:
        starts.append(len(samples) - clip_samples)
    return [samples[start:start + clip_samples].copy() for start in starts]


def augment(samples: np.ndarray, variant: int) -> np.ndarray:
    """Deterministic mild augmentation; used only when natural windows are insufficient."""
    rng = np.random.default_rng(100_000 + variant)
    gain = rng.uniform(0.82, 1.12)
    noise_scale = rng.uniform(0.0002, 0.0020)
    shifted = np.roll(samples, int(rng.integers(-700, 701)))
    result = shifted * gain + rng.normal(0.0, noise_scale, len(samples))
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, default=REFERENCES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--per-type", type=int, default=350)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--hop-seconds", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    references = json.loads(args.references.read_text(encoding="utf-8"))
    assignment = choose_session_splits(references, args.seed)
    clip_samples = int(SR * args.clip_seconds)
    hop_samples = int(SR * args.hop_seconds)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for reference in references:
        split = assignment[str(reference["session"])]
        audio, _ = librosa.load(ROOT / reference["output_file"], sr=SR, mono=True)
        for window_index, clip in enumerate(windows(audio, clip_samples, hop_samples)):
            grouped[(reference["folder"], split)].append(
                {
                    "audio": clip.astype(np.float32),
                    "source": reference,
                    "window_index": window_index,
                    "augmented": False,
                }
            )

    rng = np.random.default_rng(args.seed)
    ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
    targets = {
        "train": int(args.per_type * ratios["train"]),
        "validation": int(args.per_type * ratios["validation"]),
    }
    targets["test"] = args.per_type - targets["train"] - targets["validation"]
    output_rows: list[dict[str, str | int | bool]] = []

    for label in sorted({reference["folder"] for reference in references}):
        for split in ("train", "validation", "test"):
            candidates = grouped[(label, split)]
            rng.shuffle(candidates)
            target = targets[split]
            if not candidates:
                raise RuntimeError(f"{label}/{split} için doğal klip yok")
            selected = candidates[:target]
            natural_count = len(selected)
            while len(selected) < target:
                source = candidates[(len(selected) - natural_count) % len(candidates)]
                copy = dict(source)
                copy["audio"] = augment(source["audio"], len(output_rows) + len(selected))
                copy["augmented"] = True
                selected.append(copy)

            destination_dir = args.output / split / label
            destination_dir.mkdir(parents=True, exist_ok=True)
            for index, item in enumerate(selected):
                destination = destination_dir / f"{label}_{split}_{index:04d}.flac"
                sf.write(destination, item["audio"], SR, format="FLAC", subtype="PCM_16")
                source = item["source"]
                output_rows.append(
                    {
                        "path": str(destination),
                        "label": label,
                        "split": split,
                        "session": source["session"],
                        "source_file": source["source_file"],
                        "window_index": item["window_index"],
                        "augmented": item["augmented"],
                        "source_doi": source["source_doi"],
                        "license": source["license"],
                    }
                )
            print(
                f"{label:20} {split:10} {target:3} "
                f"(doğal={natural_count}, artırılmış={target-natural_count})",
                flush=True,
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\nToplam {len(output_rows)} klip | Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
