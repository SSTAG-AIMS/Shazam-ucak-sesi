"""Optional CLAP zero-shot evidence for top-level airport-noise categories.

This module is deliberately independent from the trained project models.  It
compares an audio embedding with natural-language category descriptions and
returns probabilities; it never changes the primary classifier by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import librosa
import numpy as np


DEFAULT_MODEL_ID = "laion/clap-htsat-unfused"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "models" / "hf_cache"

CATEGORY_PROMPTS: Mapping[str, tuple[str, ...]] = {
    "AIRCRAFT": (
        "the sound of an airplane or aircraft engine",
        "an aircraft taking off, landing, or flying overhead",
    ),
    "AMBIENT": (
        "quiet indoor or outdoor ambient background noise",
        "general environmental ambience without a distinct source",
    ),
    "SPEECH": (
        "human speech, conversation, or a spoken announcement",
        "people talking",
    ),
    "TRAFFIC": (
        "road or rail traffic including cars, buses, trucks, motorcycles, or trains",
        "the sound of vehicles and transportation traffic",
    ),
    "WIND": (
        "the sound of wind, wind gusts, or wind hitting a microphone",
        "strong outdoor wind noise",
    ),
    "OTHER": (
        "animal sounds including dogs, cats, and birds",
        "a distinct sound that is not aircraft, traffic, speech, wind, or ambience",
    ),
    "LOGISTICS": (
        "airport ground-support, cargo-handling, or logistics machinery",
        "industrial loading and ground-service equipment",
    ),
}


def category_prompt_batch(
    categories: Sequence[str],
) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    prompts: list[str] = []
    for raw_label in categories:
        label = str(raw_label).strip().upper()
        for prompt in CATEGORY_PROMPTS.get(label, (f"the sound category {label}",)):
            labels.append(label)
            prompts.append(prompt)
    return labels, prompts


def aggregate_prompt_probabilities(
    prompt_labels: Sequence[str], probabilities: Sequence[float]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for label, probability in zip(prompt_labels, probabilities):
        grouped.setdefault(str(label), []).append(max(0.0, float(probability)))
    scores = {label: float(np.mean(values)) for label, values in grouped.items()}
    total = sum(scores.values())
    return {label: value / total for label, value in scores.items()} if total else {}


@dataclass
class ClapZeroShotClassifier:
    model_id: str = DEFAULT_MODEL_ID
    cache_dir: Path = DEFAULT_CACHE_DIR
    sample_rate: int = 48_000

    def __post_init__(self) -> None:
        self._model = None
        self._processor = None

    def load(self, *, local_files_only: bool = True) -> None:
        from transformers import ClapModel, ClapProcessor

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._processor = ClapProcessor.from_pretrained(
            self.model_id,
            cache_dir=str(self.cache_dir),
            local_files_only=local_files_only,
        )
        self._model = ClapModel.from_pretrained(
            self.model_id,
            cache_dir=str(self.cache_dir),
            local_files_only=local_files_only,
        )
        self._model.eval()

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def predict_file(
        self, audio_path: str | Path, categories: Sequence[str]
    ) -> dict[str, float]:
        if not self.loaded:
            self.load(local_files_only=True)
        labels, prompts = category_prompt_batch(categories)
        samples, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return self.predict_samples(samples, labels, prompts)

    def predict_samples(
        self,
        samples: np.ndarray,
        prompt_labels: Sequence[str],
        prompts: Sequence[str],
    ) -> dict[str, float]:
        if not self.loaded:
            raise RuntimeError("CLAP modeli yüklenmedi")
        import torch

        inputs = self._processor(
            text=list(prompts),
            audio=[np.asarray(samples, dtype=np.float32)],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        with torch.inference_mode():
            output = self._model(**inputs)
            probabilities = output.logits_per_audio.softmax(dim=-1)[0].cpu().numpy()
        return aggregate_prompt_probabilities(prompt_labels, probabilities)
