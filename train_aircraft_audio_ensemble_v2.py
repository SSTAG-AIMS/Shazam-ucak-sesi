"""Train BEATs, CLAP and fused aircraft subtype evidence heads."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from aircraft_audio_ensemble_v2 import AudioFoundationBackbones


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "cache" / "aircraft_agent_manifest_v1.csv"
CACHE = ROOT / "models" / "aircraft_audio_foundation_embeddings_v2.npz"
MODEL = ROOT / "models" / "aircraft_audio_ensemble_v2.joblib"
REPORT = ROOT / "outputs" / "aircraft_audio_ensemble_v2_report.json"


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["split"] != "reference_only"]


def embeddings(rows: list[dict], rebuild: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = np.asarray([str(Path(row["path"]).resolve()) for row in rows])
    if CACHE.is_file() and not rebuild:
        cached = np.load(CACHE, allow_pickle=False)
        if np.array_equal(cached["paths"], paths):
            print(f"Embedding cache kullanılıyor: {CACHE}")
            return cached["beats"], cached["clap"], cached["row_indices"]
    extractor = AudioFoundationBackbones(); beats_parts, clap_parts, row_indices = [], [], []
    for index, row in enumerate(rows):
        beats, clap = extractor.embed_file(Path(row["path"]))
        count = min(len(beats), len(clap)); beats_parts.append(beats[:count]); clap_parts.append(clap[:count])
        row_indices.extend([index] * count); print(f"Foundation embedding: {index + 1}/{len(rows)}", flush=True)
    beats_all = np.vstack(beats_parts); clap_all = np.vstack(clap_parts); indices = np.asarray(row_indices)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, paths=paths, beats=beats_all, clap=clap_all, row_indices=indices)
    return beats_all, clap_all, indices


def model_set() -> dict:
    return {
        "BEATs-SVM": make_pipeline(StandardScaler(), SVC(C=4, probability=True, class_weight="balanced", random_state=42)),
        "CLAP-SVM": make_pipeline(StandardScaler(), SVC(C=4, probability=True, class_weight="balanced", random_state=42)),
        "BEATs+CLAP Fusion": make_pipeline(StandardScaler(), LogisticRegression(C=2, max_iter=5000, class_weight="balanced", random_state=42)),
    }


def evaluate(models: dict, rows: list[dict], row_indices: np.ndarray, features: dict, split: str) -> dict:
    grouped = defaultdict(list)
    for embedding_index, row_index in enumerate(row_indices):
        if rows[int(row_index)]["split"] == split: grouped[int(row_index)].append(embedding_index)
    truth, channel_predictions, ensemble = [], {name: [] for name in models}, []
    for row_index, embedding_indices in grouped.items():
        truth.append(rows[row_index]["label"]); votes = []
        for name, model in models.items():
            probability = np.mean(model.predict_proba(features[name][embedding_indices]), axis=0)
            label = str(model.classes_[int(np.argmax(probability))]); channel_predictions[name].append(label); votes.append(label)
        counts = {label: votes.count(label) for label in set(votes)}; best = max(counts.values())
        winners = [label for label, count in counts.items() if count == best]
        ensemble.append(winners[0] if best >= 2 and len(winners) == 1 else "UNKNOWN_AIRCRAFT")
    accepted = [i for i, label in enumerate(ensemble) if label != "UNKNOWN_AIRCRAFT"]
    return {
        "files": len(truth),
        "channels": {name: {"accuracy": float(accuracy_score(truth, values)), "macro_f1": float(f1_score(truth, values, average="macro", zero_division=0))} for name, values in channel_predictions.items()},
        "ensemble": {
            "accuracy": float(accuracy_score(truth, ensemble)), "macro_f1": float(f1_score(truth, ensemble, average="macro", zero_division=0)),
            "accepted": len(accepted), "coverage": len(accepted) / max(1, len(truth)),
            "selective_accuracy": sum(truth[i] == ensemble[i] for i in accepted) / max(1, len(accepted)),
        },
        "items": [
            {
                "path": rows[row_index]["path"],
                "expected": expected,
                "predicted": predicted,
                "channels": {
                    name: channel_predictions[name][item_index]
                    for name in models
                },
            }
            for item_index, (row_index, expected, predicted)
            in enumerate(zip(grouped, truth, ensemble))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--rebuild-cache", action="store_true"); args = parser.parse_args()
    rows = load_rows(MANIFEST); beats, clap, row_indices = embeddings(rows, args.rebuild_cache)
    labels = np.asarray([rows[int(index)]["label"] for index in row_indices]); train_mask = np.asarray([rows[int(index)]["split"] == "train" for index in row_indices])
    feature_sets = {"BEATs-SVM": beats, "CLAP-SVM": clap, "BEATs+CLAP Fusion": np.concatenate([beats, clap], axis=1)}
    models = model_set()
    for name, model in models.items(): print(f"Eğitim: {name}"); model.fit(feature_sets[name][train_mask], labels[train_mask])
    reports = {split: evaluate(models, rows, row_indices, feature_sets, split) for split in ("validation", "test")}
    MODEL.parent.mkdir(parents=True, exist_ok=True); joblib.dump({
        "models": models, "classes": sorted(set(labels[train_mask])),
        "metadata": {"manifest": str(MANIFEST.resolve()), "backbones": ["Microsoft BEATs", "LAION CLAP"], "airframe_disjoint": True},
    }, MODEL)
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: value["ensemble"] for split, value in reports.items()}, ensure_ascii=False, indent=2))
    print(f"Model: {MODEL}\nRapor: {REPORT}")


if __name__ == "__main__": main()
