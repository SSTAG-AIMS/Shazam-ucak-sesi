"""Transparent aircraft proposal for the human-reference laboratory.

Shazam is deliberately excluded.  Four pretrained audio encoders and a learned
fusion head provide the visible evidence.  The old handcrafted-feature council
is retained only as a hidden fallback when the two strongest advanced channels
do not agree.
"""

from __future__ import annotations

from pathlib import Path

from aircraft_agent_pilot_v1 import AircraftAgentPilotV1
from aircraft_audio_ensemble_v3 import AircraftAudioEnsembleV3
from aircraft_ast_finetuned_v4 import AircraftASTFineTunedV4


ROOT = Path(__file__).resolve().parent
PILOT_MODEL = ROOT / "models" / "aircraft_agent_pilot_v1.joblib"
FOUNDATION_MODEL = ROOT / "models" / "aircraft_audio_ensemble_v3.joblib"

_pilot_agent: AircraftAgentPilotV1 | None = None
_foundation_agent: AircraftAudioEnsembleV3 | None = None
_experimental_ast_agent: AircraftASTFineTunedV4 | None = None


def _pilot() -> AircraftAgentPilotV1:
    global _pilot_agent
    if _pilot_agent is None:
        _pilot_agent = AircraftAgentPilotV1(PILOT_MODEL)
    return _pilot_agent


def _foundation() -> AircraftAudioEnsembleV3:
    global _foundation_agent
    if _foundation_agent is None:
        _foundation_agent = AircraftAudioEnsembleV3(FOUNDATION_MODEL)
    return _foundation_agent


def _experimental_ast() -> AircraftASTFineTunedV4:
    global _experimental_ast_agent
    if _experimental_ast_agent is None:
        _experimental_ast_agent = AircraftASTFineTunedV4()
    return _experimental_ast_agent


def _row(result: dict, name: str) -> dict | None:
    return next((item for item in result.get("models", []) if item["model"] == name), None)


def predict_aircraft(path: Path) -> dict:
    path = path.resolve()
    if not PILOT_MODEL.is_file() or not FOUNDATION_MODEL.is_file():
        return {
            "predicted_subtype": "UNKNOWN_AIRCRAFT",
            "method": "KANIT_YOK",
            "decision_source": "MODEL_EKSIK",
            "confidence": 0.0,
            "accepted": False,
            "audio_corroborated": False,
            "audio_channels": [],
            "foundation_result": None,
            "evidence": ["Gelişmiş ses modeli veya güvenli yedek model eksik"],
        }

    legacy = _pilot().predict_file(path)
    legacy_label = str(legacy["general_result"])
    advanced = _foundation().predict_file(path)
    experimental_ast = None
    experimental_ast_error = None
    try:
        if _experimental_ast().available:
            experimental_ast = _experimental_ast().predict_file(path)
    except Exception as exc:
        experimental_ast_error = str(exc)

    visible_channels = list(advanced.get("models", []))
    if experimental_ast:
        visible_channels.append(experimental_ast)
    beats = _row(advanced, "BEATs-SVM")
    fusion = _row(advanced, "Multi-Embedding Fusion")

    advanced_agreement = bool(
        beats and fusion and str(beats["predicted"]) == str(fusion["predicted"])
    )
    if advanced_agreement:
        label = str(beats["predicted"])
        confidence = (float(beats["confidence"]) + float(fusion["confidence"])) / 2.0
        decision_source = "BEATs + Multi-Embedding Fusion uzlaşması"
    else:
        label = legacy_label
        confidence = legacy["votes"] / max(1, legacy["total_models"])
        decision_source = "Gelişmiş kanallar ayrıştı; güvenli yedek karar"

    evidence = ["── GELİŞMİŞ SES MODELLERİ (ANA KANIT) ──"]
    for item in visible_channels:
        marker = "✓" if str(item["predicted"]) == label else "·"
        evidence.append(
            f"{marker} {item['model']}: {item['predicted']} "
            f"(%{item['confidence'] * 100:.1f})"
        )
    evidence.extend([
        "",
        f"Karar politikası: {decision_source}",
        f"Yedek klasik kurul: {legacy_label} ({legacy['votes']}/{legacy['total_models']})",
        "Shazam kullanılmadı; kesin etiket insan onayıyla oluşur.",
    ])

    if experimental_ast:
        evidence.append(
            "AST Fine-Tune V4 bagimsiz testli denetim kanalidir; otomatik karari degistirmez."
        )
    elif experimental_ast_error:
        evidence.append(f"AST Fine-Tune V4 calistirilamadi: {experimental_ast_error}")

    return {
        "predicted_subtype": label,
        "method": "ADVANCED_AUDIO_REACT_COUNCIL_V3",
        "decision_policy": "BEATS_FUSION_AGREEMENT_ELSE_SAFE_FALLBACK",
        "decision_source": decision_source,
        "confidence": confidence,
        "accepted": bool(advanced_agreement or legacy["consensus_accepted"]),
        "audio_corroborated": bool(beats and str(beats["predicted"]) == label),
        "audio_channels": visible_channels,
        "foundation_result": advanced,
        "experimental_ast": experimental_ast,
        "legacy_fallback": {
            "predicted": legacy_label,
            "votes": legacy["votes"],
            "total_models": legacy["total_models"],
        },
        "evidence": evidence,
    }
