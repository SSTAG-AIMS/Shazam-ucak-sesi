"""Pure helpers for the human A/B audio review screen."""

from __future__ import annotations

from typing import Any


MODEL_ORDER = ("efficientnet", "cnn", "svm", "beats", "clap")
MODEL_NAMES = {
    "efficientnet": "EfficientNet",
    "cnn": "CNN",
    "svm": "SVM",
    "beats": "BEATs",
    "clap": "CLAP",
}
ROLE_NAMES = {
    "PRIMARY": "ana model",
    "CORROBORATOR": "doğrulayıcı",
    "AIRCRAFT_CORROBORATOR": "uçak doğrulayıcı",
    "AUDIT_ONLY": "denetim",
}


def extract_model_predictions(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every retained model prediction in a stable display order."""
    models: dict[str, Any] = {}
    for step in record.get("react_trace") or []:
        if step.get("phase") == "REASON":
            details = ((step.get("evidence") or {}).get("details") or {})
            models = details.get("models") or {}
            if models:
                break

    rows: list[dict[str, Any]] = []
    ordered = list(MODEL_ORDER) + sorted(set(models) - set(MODEL_ORDER))
    for key in ordered:
        item = models.get(key)
        if not item:
            continue
        available = bool(item.get("available", True))
        rows.append(
            {
                "key": key,
                "name": MODEL_NAMES.get(key, key.upper()),
                "available": available,
                "predicted": str(item.get("predicted") or "-") if available else "Kullanılamıyor",
                "confidence": float(item.get("confidence") or 0.0),
                "role": ROLE_NAMES.get(str(item.get("role") or ""), "destek modeli"),
            }
        )
    return rows


def general_result(record: dict[str, Any]) -> dict[str, str]:
    category = str(record.get("category") or "-")
    subtype = str(record.get("subtype") or "-")
    consensus = bool(record.get("model_consensus_accepted"))
    issues = record.get("quality_issues") or []
    if issues:
        state = "KARANTİNADA"
    elif consensus:
        state = "MODEL UZLAŞMASI KABUL EDİLDİ"
    else:
        state = "MODEL UZLAŞMASI YETERSİZ"
    return {"category": category, "subtype": subtype, "state": state}
