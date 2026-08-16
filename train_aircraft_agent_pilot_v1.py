"""Train and evaluate the five-model aircraft agent pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from aircraft_agent_pilot_v1 import file_features


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "cache" / "aircraft_agent_manifest_v1.csv"
DEFAULT_MODEL = ROOT / "models" / "aircraft_agent_pilot_v1.joblib"
DEFAULT_REPORT = ROOT / "outputs" / "aircraft_agent_pilot_v1_report.json"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def dataset(rows: list[dict[str, str]], split: str) -> tuple[np.ndarray, np.ndarray]:
    vectors, labels = [], []
    for row in rows:
        if row["split"] != split:
            continue
        current = file_features(Path(row["path"]))
        vectors.extend(current)
        labels.extend([row["label"]] * len(current))
    return np.asarray(vectors, dtype=np.float32), np.asarray(labels)


def model_set(train_size: int) -> dict:
    return {
        "RBF-SVM": make_pipeline(StandardScaler(), SVC(C=3.0, probability=True, class_weight="balanced", random_state=42)),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(7, max(1, train_size)))),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    }


def file_level_predictions(models: dict, rows: list[dict[str, str]], split: str) -> tuple[list[str], dict[str, list[str]], list[str]]:
    truth, paths = [], []
    predictions = {name: [] for name in models}
    ensemble = []
    for row in rows:
        if row["split"] != split:
            continue
        features = file_features(Path(row["path"]))
        votes = []
        for name, model in models.items():
            probabilities = np.mean(model.predict_proba(features), axis=0)
            label = str(model.classes_[int(np.argmax(probabilities))])
            predictions[name].append(label)
            votes.append(label)
        counts = Counter(votes)
        label, count = counts.most_common(1)[0]
        ensemble.append(
            label
            if count >= 3 and sum(v == count for v in counts.values()) == 1
            else "UNKNOWN_AIRCRAFT"
        )
        truth.append(row["label"])
        paths.append(row["path"])
    return truth, predictions, ensemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    rows = load_rows(args.manifest)
    x_train, y_train = dataset(rows, "train")
    models = model_set(len(x_train))
    for name, model in models.items():
        print(f"[EĞİTİM] {name} | pencere={len(x_train)}")
        model.fit(x_train, y_train)

    reports = {}
    for split in ("validation", "test"):
        truth, predictions, ensemble = file_level_predictions(models, rows, split)
        reports[split] = {
            "files": len(truth),
            "models": {
                name: {
                    "accuracy": float(accuracy_score(truth, values)),
                    "macro_f1": float(f1_score(truth, values, average="macro", zero_division=0)),
                }
                for name, values in predictions.items()
            },
            "ensemble": {
                "accuracy": float(accuracy_score(truth, ensemble)),
                "macro_f1": float(f1_score(truth, ensemble, average="macro", zero_division=0)),
                "accepted": sum(value != "UNKNOWN_AIRCRAFT" for value in ensemble),
                "coverage": sum(value != "UNKNOWN_AIRCRAFT" for value in ensemble) / len(ensemble),
                "selective_accuracy": (
                    sum(expected == predicted for expected, predicted in zip(truth, ensemble))
                    / max(1, sum(value != "UNKNOWN_AIRCRAFT" for value in ensemble))
                ),
            },
            "items": [
                {"path": row["path"], "expected": expected, "predicted": predicted}
                for row, expected, predicted in zip(
                    [row for row in rows if row["split"] == split], truth, ensemble
                )
            ],
        }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": models,
        "classes": sorted(set(y_train)),
        "metadata": {"manifest": str(args.manifest.resolve()), "physical_airframe_disjoint": True},
    }, args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: value["ensemble"] for split, value in reports.items()}, indent=2))
    print(f"Model: {args.model}\nRapor: {args.report}")


if __name__ == "__main__":
    main()
