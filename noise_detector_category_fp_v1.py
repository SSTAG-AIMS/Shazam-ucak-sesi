"""Opt-in experimental detector with category-wide Shazam verification."""

from __future__ import annotations

from pathlib import Path

from category_fingerprint_v1 import CategoryFingerprintDatabaseV1
from noise_detector import AirportNoiseSystem


class CategoryFingerprintV1System(AirportNoiseSystem):
    """Production detector plus an isolated TRAFFIC/OTHER fingerprint layer."""

    def __init__(self, *args, enable_category_fingerprint: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_category_fingerprint = enable_category_fingerprint
        model_dir = Path(__file__).resolve().parent / "models"
        full_database = model_dir / "category_fingerprints_full_v1.sqlite3"
        legacy_database = model_dir / "category_fingerprints_v1.sqlite3"
        database_path = (
            full_database if full_database.is_file() else legacy_database
        )
        self.category_fingerprint_v1 = CategoryFingerprintDatabaseV1(database_path)
        if self.category_fingerprint_v1.exists:
            print(f"[CategoryFingerprintV1] Yüklendi: {database_path}")
        else:
            print("[CategoryFingerprintV1] İndeks bulunamadı; katman pasif")

    def analyze_for_gui(
        self,
        audio_path: str,
        model_pref: str = "auto",
        identify_subtype: bool = True,
    ) -> dict:
        result = super().analyze_for_gui(
            audio_path,
            model_pref=model_pref,
            identify_subtype=identify_subtype,
            defer_category_subtype=True,
        )
        if not identify_subtype:
            result["category_fingerprint_v1"] = None
            return result
        summary = result.get("summary") or {}
        dominant = max(summary, key=summary.get) if summary else None
        result["category_fingerprint_v1"] = None

        if dominant not in {
            "AMBIENT", "LOGISTICS", "OTHER", "SPEECH", "TRAFFIC", "WIND"
        }:
            return result

        # Gerçek çalışma sırası: önce parmak izi aranır. Yalnızca kabul edilen
        # bir eşleşme yoksa eğitilmiş alt tür modeli çalıştırılır.
        if self.enable_category_fingerprint and self.category_fingerprint_v1.exists:
            try:
                match = self.category_fingerprint_v1.match_file(audio_path)
            except Exception as exc:
                print(f"[CategoryFingerprintV1] Eşleştirme hatası: {exc}")
                match = None

            if match is not None:
                match_dict = match.as_subtype_dict()
                result["category_fingerprint_v1"] = match_dict
                if match.accepted and match.category == dominant:
                    result["subtype_match"] = match_dict
                    return result

        try:
            result["subtype_match"] = self._infer_category_subtype(
                result["samples"], dominant
            )
        except Exception as exc:
            print(f"[{dominant}-Subtype] Tahmin hatası: {exc}")
        return result
