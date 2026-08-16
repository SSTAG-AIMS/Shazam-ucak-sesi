"""Five-model pilot ensemble for aircraft subtype evidence."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import librosa
import numpy as np


SAMPLE_RATE = 22050
WINDOW_SECONDS = 5.0


def feature_vector(samples: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < sr:
        samples = np.pad(samples, (0, sr - samples.size))
    mel = librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=64, n_fft=2048, hop_length=512)
    log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=30)
    delta = librosa.feature.delta(mfcc)
    centroid = librosa.feature.spectral_centroid(y=samples, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(samples)
    rms = librosa.feature.rms(y=samples)
    blocks = [mfcc, delta, centroid, bandwidth, rolloff, zcr, rms]
    return np.concatenate([np.mean(block, axis=1) for block in blocks] + [
        np.std(block, axis=1) for block in blocks
    ]).astype(np.float32)


def file_features(path: Path, window_seconds: float = WINDOW_SECONDS) -> np.ndarray:
    samples, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    size = int(window_seconds * SAMPLE_RATE)
    vectors = []
    for start in range(0, max(1, len(samples)), size):
        window = samples[start:start + size]
        if len(window) < SAMPLE_RATE:
            continue
        rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)))
        if rms < 1e-5:
            continue
        vectors.append(feature_vector(window))
    if not vectors:
        vectors.append(feature_vector(samples))
    return np.vstack(vectors)


class AircraftAgentPilotV1:
    def __init__(self, bundle_path: Path | str) -> None:
        bundle = joblib.load(bundle_path)
        self.models = bundle["models"]
        self.classes = list(bundle["classes"])
        self.metadata = bundle.get("metadata", {})

    def predict_file(self, path: Path | str) -> dict:
        features = file_features(Path(path))
        rows = []
        votes = []
        for name, model in self.models.items():
            probabilities = np.mean(model.predict_proba(features), axis=0)
            index = int(np.argmax(probabilities))
            label = str(model.classes_[index])
            votes.append(label)
            rows.append({
                "model": name,
                "predicted": label,
                "confidence": float(probabilities[index]),
            })
        counts = Counter(votes)
        label, count = counts.most_common(1)[0]
        tied = sum(value == count for value in counts.values()) > 1
        return {
            "models": rows,
            "general_result": label if not tied else "UNKNOWN_AIRCRAFT",
            "votes": count,
            "total_models": len(votes),
            "consensus_accepted": bool(not tied and count >= 3),
        }
