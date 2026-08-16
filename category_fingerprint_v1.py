"""Experimental Shazam-style fingerprint layer for non-aircraft categories.

This module deliberately stores its index in a separate SQLite file and does
not modify the production aircraft fingerprint database.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aircraft_fingerprint import AircraftFingerprintDatabase


LABEL_SEPARATOR = "::"


@dataclass(frozen=True)
class CategoryFingerprintMatch:
    category: str
    subtype: str
    reference_name: str
    matched_hashes: int
    aligned_hashes: int
    query_hashes: int
    confidence: float
    accepted: bool

    def as_subtype_dict(self) -> dict:
        subtype = self.subtype if self.accepted else f"UNKNOWN_{self.category}"
        return {
            "category": self.category,
            "subtype": subtype,
            "predicted_subtype": self.subtype,
            "reference_name": self.reference_name,
            "matched_hashes": self.matched_hashes,
            "aligned_hashes": self.aligned_hashes,
            "query_hashes": self.query_hashes,
            "confidence": self.confidence,
            "accepted": self.accepted,
            "method": "shazam_v1",
            "n_windows": 1,
            "vote_counts": {self.subtype: 1},
        }


class CategoryFingerprintDatabaseV1:
    """Category-aware wrapper around the existing landmark hash engine."""

    def __init__(
        self,
        db_path: str | Path,
        min_aligned_hashes: int = 8,
        min_confidence: float = 0.05,
    ):
        self._database = AircraftFingerprintDatabase(
            db_path,
            min_aligned_hashes=min_aligned_hashes,
            min_confidence=min_confidence,
        )

    @property
    def exists(self) -> bool:
        return self._database.exists

    def reset(self) -> None:
        self._database.reset()

    def compact(self) -> None:
        self._database.compact()

    def add_reference(
        self,
        audio_path: str | Path,
        category: str,
        subtype: str,
    ) -> int:
        category = str(category).strip().upper()
        subtype = str(subtype).strip().upper()
        if not category or not subtype:
            raise ValueError("Kategori ve alt tür boş olamaz")
        return self._database.add_reference(
            audio_path, f"{category}{LABEL_SEPARATOR}{subtype}"
        )

    def match_file(self, audio_path: str | Path) -> CategoryFingerprintMatch | None:
        raw = self._database.match_file(audio_path)
        if raw is None:
            return None
        encoded = raw.reference_name
        # Rejected matches replace aircraft_type with UNKNOWN_AIRCRAFT, so read
        # the original stored label from the reference list by reference name.
        for stored_label, stored_name, _ in self._database.list_references():
            if stored_name == raw.reference_name:
                encoded = stored_label
                break
        if LABEL_SEPARATOR not in encoded:
            return None
        category, subtype = encoded.split(LABEL_SEPARATOR, 1)
        return CategoryFingerprintMatch(
            category=category,
            subtype=subtype,
            reference_name=raw.reference_name,
            matched_hashes=raw.matched_hashes,
            aligned_hashes=raw.aligned_hashes,
            query_hashes=raw.query_hashes,
            confidence=raw.confidence,
            accepted=raw.accepted,
        )

    def list_references(self):
        for encoded, reference_name, hash_count in self._database.list_references():
            if LABEL_SEPARATOR in encoded:
                category, subtype = encoded.split(LABEL_SEPARATOR, 1)
                yield category, subtype, reference_name, hash_count
