"""Create a leakage-resistant manifest for the future aircraft type classifier."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCES = (
    PROJECT_ROOT / "Self_Data" / "AIRCRAFT_TYPES" / "references_manifest.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "cache" / "aircraft_type_manifest.csv"
SPLITS = ("train", "validation", "test")
TARGET_RATIOS = np.array([0.70, 0.15, 0.15], dtype=np.float64)


def choose_session_splits(rows: list[dict], seed: int, attempts: int = 20000) -> dict[str, str]:
    """Assign complete recording sessions while approximating class ratios."""
    sessions = sorted({str(row["session"]) for row in rows})
    classes = sorted({str(row["folder"]) for row in rows})
    class_index = {label: index for index, label in enumerate(classes)}
    session_counts: dict[str, np.ndarray] = {
        session: np.zeros(len(classes), dtype=np.int64) for session in sessions
    }
    for row in rows:
        session_counts[str(row["session"])][class_index[str(row["folder"])]] += 1
    totals = np.sum(list(session_counts.values()), axis=0)

    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_assignment: dict[str, str] | None = None
    for _ in range(attempts):
        choices = rng.choice(3, size=len(sessions), p=TARGET_RATIOS)
        if len(set(int(value) for value in choices)) < 3:
            continue
        split_counts = np.zeros((3, len(classes)), dtype=np.int64)
        for session, split_index in zip(sessions, choices):
            split_counts[int(split_index)] += session_counts[session]
        if np.any(split_counts == 0):
            continue
        actual = split_counts / np.maximum(1, totals)
        score = float(np.sum((actual - TARGET_RATIOS[:, None]) ** 2))
        if score < best_score:
            best_score = score
            best_assignment = {
                session: SPLITS[int(split_index)]
                for session, split_index in zip(sessions, choices)
            }
    if best_assignment is None:
        raise RuntimeError("Tüm sınıfları içeren session-disjoint split bulunamadı")
    return best_assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = json.loads(args.references.read_text(encoding="utf-8"))
    assignment = choose_session_splits(rows, args.seed)
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "path": str(PROJECT_ROOT / row["output_file"]),
                "label": row["folder"],
                "icao_type": row["aircraft_type"],
                "hex_id": row["hex_id"],
                "session": row["session"],
                "split": assignment[str(row["session"])],
                "source_doi": row["source_doi"],
                "license": row["license"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    distribution: dict[str, Counter] = defaultdict(Counter)
    for row in output_rows:
        distribution[row["split"]][row["label"]] += 1
    print(f"Manifest: {args.output}")
    for split in SPLITS:
        print(f"\n{split.upper()}")
        for label in sorted({row["label"] for row in output_rows}):
            print(f"  {label:18} {distribution[split][label]:3}")
    train_counts = distribution["train"]
    maximum = max(train_counts.values())
    print("\nÖnerilen sınıf ağırlıkları (max_count / class_count):")
    for label, count in sorted(train_counts.items()):
        print(f"  {label:18} {maximum / count:.3f}")


if __name__ == "__main__":
    main()
