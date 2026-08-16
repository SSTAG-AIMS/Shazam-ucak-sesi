"""Category Shazam v2: v1 plus validation-calibrated OTHER logits."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from noise_detector_category_fp_v1 import CategoryFingerprintV1System


class LogitBiasHead(nn.Module):
    def __init__(self, base_head: nn.Module, bias: list[float]):
        super().__init__()
        self.base_head = base_head
        self.register_buffer("logit_bias", torch.tensor(bias, dtype=torch.float32))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.base_head(embeddings) + self.logit_bias


class CategoryFingerprintV2System(CategoryFingerprintV1System):
    """Experimental v2 with calibrated CAT/CROW/PARROT decision balance."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        calibration_path = (
            Path(__file__).resolve().parent
            / "models"
            / "other_subtype_logit_bias_v2.json"
        )
        self.other_v2_calibration_loaded = False
        entry = self.category_subtype_models.get("OTHER")
        if entry is None or not calibration_path.is_file():
            print("[OTHER-v2] Kalibrasyon bulunamadı; v1 kullanılıyor")
            return
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if list(entry["classes"]) != list(calibration["classes"]):
                raise ValueError("Kalibrasyon sınıf sırası modelle uyuşmuyor")
            model_device = next(entry["model"].parameters()).device
            entry["model"] = LogitBiasHead(
                entry["model"], calibration["logit_bias"]
            ).to(model_device).eval()
            self.other_v2_calibration_loaded = True
            print(
                "[OTHER-v2] Validation logit kalibrasyonu yüklendi | "
                f"ValF1:{calibration['validation_f1_before']:.3f}"
                f"->{calibration['validation_f1_after']:.3f}"
            )
        except Exception as exc:
            print(f"[OTHER-v2] Kalibrasyon yükleme hatası: {exc}")

    def _infer_category_subtype(self, samples, category):
        result = super()._infer_category_subtype(samples, category)
        if (
            result is not None
            and category == "OTHER"
            and self.other_v2_calibration_loaded
        ):
            result["method"] = "beats_multi_window_vote_v2_calibrated"
        return result


class PresentationNoiseSystem(CategoryFingerprintV2System):
    """Clear teacher-demo flow: Shazam for aircraft, BEATs for other subtypes."""

    def __init__(self, *args, **kwargs):
        kwargs["enable_category_fingerprint"] = False
        super().__init__(*args, **kwargs)
