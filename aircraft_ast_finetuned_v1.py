"""Inference adapter for the experimental Colab AST fine-tune.

The model is audit-only: it is visible to the reviewer but cannot alter the
current production decision policy until its independent score improves.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from window_voting import (
    aggregate_window_probabilities,
    select_audio_windows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models" / "aircraft_ast_finetuned_v1"


class AircraftASTFineTunedV1:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._extractor = None
        self._model = None

    @property
    def available(self) -> bool:
        return all(
            (self.model_dir / name).is_file()
            for name in ("config.json", "model.safetensors", "preprocessor_config.json")
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.available:
            raise FileNotFoundError(f"Deneysel AST modeli eksik: {self.model_dir}")
        import torch
        from transformers import ASTForAudioClassification, AutoFeatureExtractor

        self._extractor = AutoFeatureExtractor.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        self._model = ASTForAudioClassification.from_pretrained(
            str(self.model_dir), local_files_only=True
        ).eval()
        self._torch = torch

    @staticmethod
    def _windows(audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        windows, _ = select_audio_windows(
            audio,
            sample_rate * 10,
            hop_samples=sample_rate * 5,
            max_windows=5,
        )
        return list(windows)

    def predict_file(self, path: Path) -> dict:
        self._load()
        audio, sample_rate = librosa.load(str(path), sr=16_000, mono=True)
        probabilities = []
        with self._torch.inference_mode():
            for window in self._windows(audio.astype(np.float32), sample_rate):
                inputs = self._extractor(window, sampling_rate=sample_rate, return_tensors="pt")
                logits = self._model(**inputs).logits
                probabilities.append(self._torch.softmax(logits, dim=-1).cpu().numpy()[0])

        classes = [
            str(self._model.config.id2label[index])
            for index in range(len(probabilities[0]))
        ]
        voting = aggregate_window_probabilities(
            np.stack(probabilities), classes
        )
        return {
            "model": "AST-FineTune-v1 (deneysel)",
            "predicted": voting["winner"],
            "confidence": voting["confidence"],
            "window_count": len(probabilities),
            "vote_share": voting["vote_share"],
            "vote_counts": voting["vote_counts"],
            "window_predictions": voting["window_predictions"],
            "window_confidences": voting["window_confidences"],
            "probabilities": voting["mean_probabilities"],
            "method": "ast_multi_window_vote",
            "audit_only": True,
        }
