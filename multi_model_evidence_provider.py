"""Audit-oriented multi-model evidence provider for the labeling agent.

The provider does not use equal majority voting.  Pilot-qualified channels are
used as corroboration while every model output is retained for human review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dataset_catalog import Taxonomy
from dataset_labeling_agent import EvidenceProvider, LabelEvidence


MODEL_PREFERENCES = ("efficientnet", "cnn", "svm", "beats")


def probabilities_from_result(result: dict[str, Any]) -> dict[str, float]:
    frame_probs = result.get("frame_probs")
    raw_classes = result.get("class_names")
    classes = [str(value) for value in raw_classes] if raw_classes is not None else []
    if frame_probs is not None and len(frame_probs) and classes:
        average = np.mean(np.asarray(frame_probs, dtype=np.float64), axis=0)
        values = {label: max(0.0, float(value)) for label, value in zip(classes, average)}
    else:
        summary = result.get("summary") or {}
        values = {str(label): max(0.0, float(value) / 100.0) for label, value in summary.items()}
    total = sum(values.values())
    return {label: value / total for label, value in values.items()} if total else {}


class MultiModelEvidenceProvider:
    """Collect five model outputs and emit a conservative category suggestion.

    Current pilot policy:
    - EfficientNet is the primary semantic suggestion.
    - CLAP can corroborate every category.
    - SVM can corroborate AIRCRAFT only.
    - CNN and top-level BEATs are audit-only until recalibrated.
    """

    def __init__(
        self,
        *,
        system: Any | None = None,
        clap: Any | None = None,
        subtype_provider: EvidenceProvider | None = None,
        taxonomy: Taxonomy | None = None,
    ) -> None:
        if system is None:
            from noise_detector_category_fp_v2 import CategoryFingerprintV2System

            system = CategoryFingerprintV2System()
        if clap is None:
            from clap_zero_shot import ClapZeroShotClassifier

            clap = ClapZeroShotClassifier()
            clap.load(local_files_only=True)
        if subtype_provider is None:
            from dataset_labeling_agent import AirportModelEvidenceProvider

            subtype_provider = AirportModelEvidenceProvider(system=system)
        self.system = system
        self.clap = clap
        self.subtype_provider = subtype_provider
        self.taxonomy = taxonomy or Taxonomy.load()

    def _available(self, name: str) -> bool:
        attribute = {
            "efficientnet": "eff_model",
            "cnn": "cnn_model",
            "svm": "ml_model",
            "beats": "beats_model",
        }[name]
        return getattr(self.system, attribute, None) is not None

    def analyze(self, audio_path: Path) -> LabelEvidence:
        model_evidence: dict[str, dict[str, Any]] = {}
        for name in MODEL_PREFERENCES:
            if not self._available(name):
                model_evidence[name] = {"available": False}
                continue
            result = self.system.analyze_for_gui(
                str(audio_path), model_pref=name, identify_subtype=False
            )
            probabilities = probabilities_from_result(result)
            predicted = max(probabilities, key=probabilities.get) if probabilities else None
            model_evidence[name] = {
                "available": True,
                "predicted": predicted,
                "confidence": probabilities.get(predicted, 0.0) if predicted else 0.0,
                "probabilities": probabilities,
                "role": (
                    "PRIMARY"
                    if name == "efficientnet"
                    else "AIRCRAFT_CORROBORATOR"
                    if name == "svm"
                    else "AUDIT_ONLY"
                ),
            }

        clap_probabilities = self.clap.predict_file(
            audio_path, sorted(self.taxonomy.categories)
        )
        clap_label = max(clap_probabilities, key=clap_probabilities.get)
        model_evidence["clap"] = {
            "available": True,
            "predicted": clap_label,
            "confidence": clap_probabilities[clap_label],
            "probabilities": clap_probabilities,
            "role": "CORROBORATOR",
        }

        primary = model_evidence.get("efficientnet") or {}
        if not primary.get("predicted"):
            raise RuntimeError("Çoklu model agentı için EfficientNet kullanılamıyor")
        category = str(primary["predicted"])
        confidence = float(primary["confidence"])

        corroborators: list[str] = []
        if clap_label == category and clap_probabilities[clap_label] >= 0.45:
            corroborators.append("clap")
        svm = model_evidence.get("svm") or {}
        if (
            category == "AIRCRAFT"
            and svm.get("predicted") == "AIRCRAFT"
            and float(svm.get("confidence", 0.0)) >= 0.55
        ):
            corroborators.append("svm_aircraft")
        subtype_evidence = self.subtype_provider.analyze(audio_path)
        subtype = subtype_evidence.subtype if subtype_evidence.category == category else None
        subtype_confidence = (
            subtype_evidence.subtype_confidence
            if subtype_evidence.category == category
            else None
        )
        if subtype_evidence.fingerprint_accepted and subtype_evidence.category == category:
            corroborators.append("verified_fingerprint")

        # One corroborator requires a strong primary score.  Two or more
        # independent corroborators can support a mixed/long recording with a
        # lower averaged primary probability.  This is still only a review
        # suggestion and never human approval.
        consensus_accepted = (
            confidence >= 0.60 and len(corroborators) >= 1
        ) or (
            confidence >= 0.30 and len(corroborators) >= 2
        )
        return LabelEvidence(
            category=category,
            subtype=subtype,
            category_confidence=confidence,
            subtype_confidence=subtype_confidence,
            source="MULTI_MODEL_REVIEW_EVIDENCE_V1",
            fingerprint_accepted=subtype_evidence.fingerprint_accepted,
            details={
                "policy": "EFFICIENTNET_PRIMARY_CLAP_AND_AIRCRAFT_SVM_CORROBORATION",
                "consensus_accepted": consensus_accepted,
                "corroborators": corroborators,
                "models": model_evidence,
                "subtype_source": subtype_evidence.source,
                "human_approval_required": True,
            },
        )
