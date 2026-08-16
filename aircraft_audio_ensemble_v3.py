"""Advanced aircraft subtype evidence using independent pretrained audio encoders."""

from __future__ import annotations

from collections import Counter
import gc
from pathlib import Path

import joblib
import numpy as np
import torch

from aircraft_audio_ensemble_v2 import AudioFoundationBackbones, audio_windows, averaged_prediction


ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "models" / "aircraft_audio_ensemble_v3.joblib"
HF_CACHE = ROOT / "models" / "hf_cache"
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
PANNS_CHECKPOINT = ROOT / "models" / "Cnn14_mAP_0.431.pth"


class AdvancedAudioBackbones:
    """BEATs, AST, PANNs and CLAP encoders, loaded only when first needed."""

    def __init__(self) -> None:
        self.base = AudioFoundationBackbones()
        self.ast = None
        self.ast_processor = None
        self.panns = None

    def load_ast(self) -> None:
        if self.ast is not None:
            return
        from transformers import ASTModel, AutoFeatureExtractor

        self.ast_processor = AutoFeatureExtractor.from_pretrained(
            AST_MODEL_ID, cache_dir=str(HF_CACHE), local_files_only=True,
        )
        self.ast = ASTModel.from_pretrained(
            AST_MODEL_ID, cache_dir=str(HF_CACHE), local_files_only=True,
        )
        self.ast.eval()
        for parameter in self.ast.parameters():
            parameter.requires_grad = False

    def load_panns(self) -> None:
        if self.panns is not None:
            return
        from panns_inference import AudioTagging

        self.panns = AudioTagging(checkpoint_path=str(PANNS_CHECKPOINT), device="cpu")

    @torch.inference_mode()
    def ast_file(self, path: Path) -> np.ndarray:
        self.load_ast()
        windows = audio_windows(path, 16_000)
        inputs = self.ast_processor(
            windows, sampling_rate=16_000, return_tensors="pt",
        )
        output = self.ast(**inputs)
        pooled = output.pooler_output if output.pooler_output is not None else output.last_hidden_state[:, 0]
        return pooled.cpu().numpy().astype(np.float32)

    def panns_file(self, path: Path) -> np.ndarray:
        self.load_panns()
        windows = np.stack(audio_windows(path, 32_000)).astype(np.float32)
        _, embedding = self.panns.inference(windows)
        return np.asarray(embedding, dtype=np.float32)

    def embed_file(self, path: Path) -> dict[str, np.ndarray]:
        # The four foundation encoders are large.  Keeping all of them resident
        # causes paging on ordinary laptops, so each encoder is released after
        # producing its evidence embedding.
        beats = self.base.beats_file(path)
        self.base.beats = None
        gc.collect()
        ast = self.ast_file(path)
        self.ast = None
        self.ast_processor = None
        gc.collect()
        panns = self.panns_file(path)
        self.panns = None
        gc.collect()
        clap = self.base.clap_file(path)
        self.base.clap = None
        self.base.clap_processor = None
        gc.collect()
        return {
            "BEATs-SVM": beats,
            "AST-SVM": ast,
            "PANNs-CNN14-SVM": panns,
            "CLAP-SVM": clap,
        }


class AircraftAudioEnsembleV3:
    """Five-channel council: four audio encoders plus learned embedding fusion."""

    def __init__(self, bundle_path: Path | str = DEFAULT_BUNDLE, *, backbones=None) -> None:
        bundle = joblib.load(bundle_path)
        self.models = bundle["models"]
        self.classes = list(bundle["classes"])
        self.metadata = bundle.get("metadata", {})
        self.backbones = backbones or AdvancedAudioBackbones()

    def predict_file(self, path: Path | str) -> dict:
        features = self.backbones.embed_file(Path(path))
        counts = {array.shape[0] for array in features.values()}
        if len(counts) != 1:
            count = min(counts)
            features = {name: values[:count] for name, values in features.items()}
        features["Multi-Embedding Fusion"] = np.concatenate(list(features.values()), axis=1)

        rows, votes = [], []
        for name, values in features.items():
            label, confidence, probabilities = averaged_prediction(self.models[name], values)
            rows.append({
                "model": name,
                "predicted": label,
                "confidence": confidence,
                "probabilities": probabilities,
                "role": "audio_foundation",
            })
            votes.append(label)
        label, count = Counter(votes).most_common(1)[0]
        tied = votes.count(label) == 1 or sum(value == count for value in Counter(votes).values()) > 1
        return {
            "models": rows,
            "general_result": label if not tied else "UNKNOWN_AIRCRAFT",
            "votes": count,
            "total_models": len(rows),
            "consensus_accepted": bool(not tied and count >= 3),
            "method": "ADVANCED_AUDIO_FOUNDATION_COUNCIL_V3",
        }
