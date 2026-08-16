"""Human-in-the-loop ReAct agent for building the verified audio catalog.

The agent observes an audio asset, reasons over quality and optional inference
evidence, then routes it to human review or quarantine. It never emits APPROVED.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import librosa
import numpy as np

from dataset_catalog import ReviewStatus, Taxonomy, normalize_label, sha256_file


@dataclass(frozen=True)
class QualityObservation:
    sample_rate: int
    duration_seconds: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    finite: bool


@dataclass(frozen=True)
class LabelEvidence:
    category: str
    subtype: str | None
    category_confidence: float
    subtype_confidence: float | None
    source: str
    fingerprint_accepted: bool = False
    details: dict[str, Any] | None = None


class EvidenceProvider(Protocol):
    def analyze(self, audio_path: Path) -> LabelEvidence: ...


class AirportModelEvidenceProvider:
    """Lazy adapter for the project's current EfficientNet/Shazam/BEATs stack."""

    def __init__(self, system: Any | None = None) -> None:
        if system is None:
            from noise_detector_category_fp_v2 import CategoryFingerprintV2System

            system = CategoryFingerprintV2System()
        self.system = system

    def analyze(self, audio_path: Path) -> LabelEvidence:
        result = self.system.analyze_for_gui(str(audio_path), identify_subtype=True)
        summary = result.get("summary") or {}
        category = max(summary, key=summary.get) if summary else "OTHER"
        category_confidence = float(summary.get(category, 0.0)) / 100.0

        subtype_result = None
        if category == "AIRCRAFT":
            subtype_result = result.get("aircraft_match")
        else:
            subtype_result = result.get("subtype_match")

        subtype = None
        subtype_confidence = None
        source = str(result.get("model_used") or "project_model")
        fingerprint_accepted = False
        if subtype_result:
            subtype = subtype_result.get("predicted_type") or subtype_result.get("aircraft_type")
            subtype_confidence = _probability(subtype_result.get("confidence"))
            method = str(subtype_result.get("method") or "")
            fingerprint_accepted = bool(subtype_result.get("accepted")) and (
                "shazam" in method.lower() or "fingerprint" in method.lower()
            )
            source = method or source

        return LabelEvidence(
            category=category,
            subtype=subtype,
            category_confidence=category_confidence,
            subtype_confidence=subtype_confidence,
            source=source,
            fingerprint_accepted=fingerprint_accepted,
            details={"summary": summary},
        )


def _probability(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if result > 1.0:
        result /= 100.0
    return max(0.0, min(1.0, result))


def observe_audio(audio_path: Path, target_sample_rate: int = 22050) -> QualityObservation:
    samples, sample_rate = librosa.load(
        str(audio_path), sr=target_sample_rate, mono=True, dtype=np.float32
    )
    if samples.size == 0:
        raise ValueError("Ses dosyası boş")

    finite = bool(np.isfinite(samples).all())
    safe_samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    absolute = np.abs(safe_samples)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(safe_samples))))
    eps = np.finfo(np.float32).tiny
    return QualityObservation(
        sample_rate=int(sample_rate),
        duration_seconds=float(samples.size / sample_rate),
        peak_dbfs=float(20.0 * math.log10(max(peak, eps))),
        rms_dbfs=float(20.0 * math.log10(max(rms, eps))),
        clipping_ratio=float(np.mean(absolute >= 0.999)),
        silence_ratio=float(np.mean(absolute < 10 ** (-60.0 / 20.0))),
        finite=finite,
    )


def quality_issues(observation: QualityObservation) -> list[str]:
    issues = []
    if not observation.finite:
        issues.append("NON_FINITE_SAMPLES")
    if observation.duration_seconds < 1.0:
        issues.append("TOO_SHORT")
    if observation.rms_dbfs < -60.0:
        issues.append("TOO_SILENT")
    if observation.silence_ratio > 0.95:
        issues.append("EXCESSIVE_SILENCE")
    if observation.clipping_ratio > 0.05:
        issues.append("EXCESSIVE_CLIPPING")
    return issues


class LabelingAgent:
    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        evidence_provider: EvidenceProvider | None = None,
    ) -> None:
        self.taxonomy = taxonomy or Taxonomy.load()
        self.evidence_provider = evidence_provider

    def process(
        self,
        audio_path: Path,
        *,
        source_recording_id: str,
        source_uri: str,
        license_name: str,
        category_hint: str | None = None,
        subtype_hint: str | None = None,
    ) -> dict[str, Any]:
        audio_path = audio_path.resolve()
        observation = observe_audio(audio_path)
        issues = quality_issues(observation)
        trace: list[dict[str, Any]] = [
            {"phase": "OBSERVE", "quality": asdict(observation), "issues": issues}
        ]

        evidence = self.evidence_provider.analyze(audio_path) if self.evidence_provider else None
        category = normalize_label(
            evidence.category if evidence is not None else (category_hint or "OTHER")
        )
        if category not in self.taxonomy.categories:
            category = "OTHER"

        candidate_subtype = (
            evidence.subtype if evidence is not None and evidence.subtype else subtype_hint
        )
        subtype = self._safe_subtype(category, candidate_subtype)
        confidence = evidence.subtype_confidence if evidence is not None else None

        reason = {
            "phase": "REASON",
            "category": category,
            "suggested_subtype": subtype,
            "quality_passed": not issues,
            "evidence": asdict(evidence) if evidence is not None else None,
        }
        trace.append(reason)

        # An agent can route a record, but only a human can approve or reject it.
        action = "QUARANTINE" if issues else "SEND_TO_HUMAN_REVIEW"
        trace.append(
            {
                "phase": "ACT",
                "action": action,
                "reason": issues or ["HUMAN_APPROVAL_REQUIRED"],
            }
        )

        return {
            "asset_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_uri}#{source_recording_id}")),
            "audio_path": str(audio_path),
            "sha256": sha256_file(audio_path),
            "category": category,
            "subtype": subtype,
            "review_status": ReviewStatus.PENDING_REVIEW.value,
            "taxonomy_version": self.taxonomy.version,
            "source_recording_id": str(source_recording_id),
            "source_uri": str(source_uri),
            "license": str(license_name),
            "agent_action": action,
            "quality_issues": issues,
            "suggested_confidence": confidence,
            "category_suggested_confidence": (
                evidence.category_confidence if evidence is not None else None
            ),
            "subtype_suggested_confidence": confidence,
            "model_consensus_accepted": bool(
                evidence
                and evidence.details
                and evidence.details.get("consensus_accepted", False)
            ),
            "suggestion_source": evidence.source if evidence is not None else "metadata_hint",
            "fingerprint_accepted": bool(evidence and evidence.fingerprint_accepted),
            "react_trace": trace,
        }

    def _safe_subtype(self, category: str, candidate: str | None) -> str:
        if candidate:
            normalized = normalize_label(candidate)
            if normalized in self.taxonomy.categories[category]:
                return normalized
        return self.taxonomy.unknown_for(category)


def append_jsonl(record: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="İnsan onaylı veri kataloğu etiketleme agentı")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, default=Path("cache/catalog_review_queue_v1.jsonl"))
    parser.add_argument("--source-recording-id", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--license", dest="license_name", required=True)
    parser.add_argument("--category-hint")
    parser.add_argument("--subtype-hint")
    parser.add_argument("--with-models", action="store_true")
    parser.add_argument(
        "--with-multi-models",
        action="store_true",
        help="Beş model kanıtını topla; her durumda insan onayı ister",
    )
    args = parser.parse_args()

    if args.with_multi_models:
        from multi_model_evidence_provider import MultiModelEvidenceProvider

        provider = MultiModelEvidenceProvider()
    else:
        provider = AirportModelEvidenceProvider() if args.with_models else None
    agent = LabelingAgent(evidence_provider=provider)
    record = agent.process(
        args.audio,
        source_recording_id=args.source_recording_id,
        source_uri=args.source_uri,
        license_name=args.license_name,
        category_hint=args.category_hint,
        subtype_hint=args.subtype_hint,
    )
    append_jsonl(record, args.output)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
