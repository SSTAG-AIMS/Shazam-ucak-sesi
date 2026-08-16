"""Train a binary aircraft/non-aircraft safety gate on hard negatives."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from noise_detector import (
    _ML_CLIP,
    _ML_HOP_SEC,
    _ML_SR,
    extract_features_ml,
)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "Test_Folder"
MODEL_PATH = ROOT / "models" / "aircraft_guard.pkl"
REPORT_PATH = ROOT / "outputs" / "aircraft_guard_report.json"
CACHE_PATH = ROOT / "cache" / "aircraft_guard_features.npz"


def file_features(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=_ML_SR, mono=True)
    window = int(_ML_CLIP * _ML_SR)
    hop = int(_ML_HOP_SEC * _ML_SR)
    if len(audio) <= window:
        audio = np.pad(audio, (0, window - len(audio)))
        chunks = [audio]
    else:
        starts = list(range(0, len(audio) - window + 1, hop))
        if starts[-1] != len(audio) - window:
            starts.append(len(audio) - window)
        chunks = [audio[start : start + window] for start in starts]
    return np.mean([extract_features_ml(chunk) for chunk in chunks], axis=0)


def dataset_paths() -> tuple[list[Path], np.ndarray, np.ndarray]:
    paths: list[Path] = []
    labels: list[int] = []
    splits: list[str] = []
    for split in ("train", "test"):
        for label_name, label in (("negative", 0), ("aircraft", 1)):
            folder = DATA_ROOT / f"{label_name}-{split}"
            for path in sorted(folder.glob("*.wav")):
                paths.append(path)
                labels.append(label)
                splits.append(split)
    return paths, np.asarray(labels), np.asarray(splits)


def load_or_extract(paths: list[Path]) -> np.ndarray:
    path_strings = np.asarray([str(path) for path in paths])
    if CACHE_PATH.exists():
        cached = np.load(CACHE_PATH, allow_pickle=False)
        if np.array_equal(cached["paths"], path_strings):
            return cached["features"].astype(np.float32)

    features = []
    for index, path in enumerate(paths, start=1):
        features.append(file_features(path))
        print(f"Özellik: {index}/{len(paths)} {path.name}", flush=True)
    matrix = np.asarray(features, dtype=np.float32)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, paths=path_strings, features=matrix)
    return matrix


def main() -> None:
    paths, labels, splits = dataset_paths()
    features = load_or_extract(paths)
    train_mask = splits == "train"
    test_mask = splits == "test"

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                SVC(
                    C=2.0,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(features[train_mask], labels[train_mask])
    probabilities = model.predict_proba(features[test_mask])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    report = classification_report(
        labels[test_mask],
        predictions,
        target_names=["negative", "aircraft"],
        output_dict=True,
        zero_division=0,
    )
    result = {
        "model": str(MODEL_PATH),
        "threshold": 0.5,
        "train_files": int(train_mask.sum()),
        "test_files": int(test_mask.sum()),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            labels[test_mask], predictions, labels=[0, 1]
        ).tolist(),
        "negative_max_probability": float(
            probabilities[labels[test_mask] == 0].max()
        ),
        "aircraft_min_probability": float(
            probabilities[labels[test_mask] == 1].min()
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "threshold": 0.5,
            "feature_dim": int(features.shape[1]),
        },
        MODEL_PATH,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
