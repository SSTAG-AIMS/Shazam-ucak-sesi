"""Aircraft subtype ensemble built on pretrained audio foundation models."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "models" / "aircraft_audio_ensemble_v2.joblib"
BEATS_CHECKPOINT = Path(r"C:\models\BEATs_iter3_plus_AS2M.pt")
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_CACHE = ROOT / "models" / "hf_cache"
WINDOW_SECONDS = 5.0


def audio_windows(path: Path, sample_rate: int, seconds: float = WINDOW_SECONDS) -> list[np.ndarray]:
    samples, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    size = int(sample_rate * seconds)
    if len(samples) <= size:
        return [np.pad(samples, (0, max(0, size - len(samples))))[:size].astype(np.float32)]
    starts = sorted({0, max(0, (len(samples) - size) // 2), max(0, len(samples) - size)})
    return [np.asarray(samples[start:start + size], dtype=np.float32) for start in starts]


class AudioFoundationBackbones:
    """Lazy BEATs and CLAP embedding extractors shared across predictions."""

    def __init__(self) -> None:
        self.beats = None; self.clap = None; self.clap_processor = None

    def load_beats(self) -> None:
        if self.beats is not None: return
        from BEATs import BEATs, BEATsConfig
        checkpoint = torch.load(BEATS_CHECKPOINT, map_location="cpu", weights_only=False)
        model = BEATs(BEATsConfig(checkpoint["cfg"])); model.load_state_dict(checkpoint["model"])
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad = False
        self.beats = model

    def load_clap(self) -> None:
        if self.clap is not None: return
        from transformers import ClapModel, ClapProcessor
        self.clap_processor = ClapProcessor.from_pretrained(
            CLAP_MODEL_ID, cache_dir=str(CLAP_CACHE), local_files_only=True,
        )
        self.clap = ClapModel.from_pretrained(
            CLAP_MODEL_ID, cache_dir=str(CLAP_CACHE), local_files_only=True,
        ); self.clap.eval()

    @torch.inference_mode()
    def beats_file(self, path: Path) -> np.ndarray:
        self.load_beats(); windows = audio_windows(path, 16_000)
        tensor = torch.from_numpy(np.stack(windows)); padding = torch.zeros(tensor.shape, dtype=torch.bool)
        features, _ = self.beats.extract_features(tensor, padding_mask=padding)
        return features.mean(dim=1).cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def clap_file(self, path: Path) -> np.ndarray:
        self.load_clap(); windows = audio_windows(path, 48_000)
        inputs = self.clap_processor(audio=windows, sampling_rate=48_000, return_tensors="pt")
        return self.clap.get_audio_features(**inputs).cpu().numpy().astype(np.float32)

    def embed_file(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        return self.beats_file(path), self.clap_file(path)


def averaged_prediction(model, features: np.ndarray) -> tuple[str, float, dict[str, float]]:
    probabilities = np.mean(model.predict_proba(features), axis=0)
    index = int(np.argmax(probabilities)); label = str(model.classes_[index])
    return label, float(probabilities[index]), {
        str(name): float(value) for name, value in zip(model.classes_, probabilities)
    }


class AircraftAudioEnsembleV2:
    def __init__(self, bundle_path: Path | str = DEFAULT_BUNDLE, *, backbones=None) -> None:
        bundle = joblib.load(bundle_path)
        self.models = bundle["models"]; self.classes = list(bundle["classes"])
        self.metadata = bundle.get("metadata", {}); self.backbones = backbones or AudioFoundationBackbones()

    def predict_file(self, path: Path | str) -> dict:
        beats, clap = self.backbones.embed_file(Path(path))
        channel_features = {
            "BEATs-SVM": beats,
            "CLAP-SVM": clap,
            "BEATs+CLAP Fusion": np.concatenate([beats, clap], axis=1),
        }
        rows, votes = [], []
        for name, features in channel_features.items():
            label, confidence, probabilities = averaged_prediction(self.models[name], features)
            rows.append({"model": name, "predicted": label, "confidence": confidence, "probabilities": probabilities})
            votes.append(label)
        counts = Counter(votes); label, count = counts.most_common(1)[0]
        accepted = count >= 2
        return {
            "models": rows, "general_result": label if accepted else "UNKNOWN_AIRCRAFT",
            "votes": count, "total_models": 3, "consensus_accepted": accepted,
            "method": "AUDIO_FOUNDATION_ENSEMBLE_V2",
        }
