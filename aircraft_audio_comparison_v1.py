"""Explainable comparison aids for the aircraft human-review lab.

These scores are not calibrated probabilities or identity predictions. Shazam
fingerprints are deliberately not used at this stage.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


TARGET_SR = 22050


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    samples, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return np.asarray(samples, dtype=np.float32), int(sr)


def log_mel_spectrogram(samples: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    if not samples.size:
        return np.zeros((64, 1), dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_fft=2048, hop_length=512, n_mels=64,
        fmin=30, fmax=min(10000, sr // 2), power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def _feature_vector(samples: np.ndarray, sr: int) -> np.ndarray:
    mel = log_mel_spectrogram(samples, sr)
    mel01 = np.clip((mel + 80.0) / 80.0, 0.0, 1.0)
    centroid = librosa.feature.spectral_centroid(y=samples, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(samples)
    return np.nan_to_num(np.concatenate([
        mel01.mean(axis=1), mel01.std(axis=1),
        np.asarray([
            centroid.mean() / (sr / 2), centroid.std() / (sr / 2),
            bandwidth.mean() / (sr / 2), bandwidth.std() / (sr / 2),
            rolloff.mean() / (sr / 2), rolloff.std() / (sr / 2),
            zcr.mean(), zcr.std(),
        ], dtype=np.float32),
    ]).astype(np.float32))


def _cosine_percent(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return round(float(np.clip(np.dot(left, right) / denominator, 0, 1) * 100), 1)


def compare_audio(query_path: Path, reference_path: Path) -> dict:
    query, query_sr = load_audio(query_path)
    reference, reference_sr = load_audio(reference_path)
    query_mel = log_mel_spectrogram(query, query_sr)
    reference_mel = log_mel_spectrogram(reference, reference_sr)
    query_profile = np.clip((query_mel.mean(axis=1) + 80.0) / 80.0, 0, 1)
    ref_profile = np.clip((reference_mel.mean(axis=1) + 80.0) / 80.0, 0, 1)
    spectral = _cosine_percent(query_profile, ref_profile)
    feature = _cosine_percent(
        _feature_vector(query, query_sr), _feature_vector(reference, reference_sr)
    )
    return {
        "spectral_similarity": spectral,
        "feature_similarity": feature,
        "combined_similarity": round((spectral + feature) / 2.0, 1),
        "disclaimer": "Yardımcı benzerliktir; doğruluk veya kimlik olasılığı değildir.",
    }
