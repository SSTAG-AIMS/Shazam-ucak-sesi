"""Versioned two-level labels and quality gates for the reference catalog.

This module is deliberately independent from the GUI and inference pipeline.
It defines which records may enter the Shazam-style fingerprint index without
changing the currently working application.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "dataset_taxonomy_v1.json"


class ReviewStatus(str, Enum):
    """Human-review state of an audio asset."""

    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class CatalogValidationError(ValueError):
    """Raised when a catalog record violates the taxonomy or quality rules."""


@dataclass(frozen=True)
class Taxonomy:
    version: str
    categories: Mapping[str, frozenset[str]]
    unknown_subtypes: Mapping[str, str]

    @classmethod
    def load(cls, path: Path = DEFAULT_TAXONOMY_PATH) -> "Taxonomy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("maximum_depth") != 2:
            raise CatalogValidationError("Taksonomi yalnızca ana tür + alt tür içermelidir")

        categories: dict[str, frozenset[str]] = {}
        unknown_subtypes: dict[str, str] = {}
        for category, definition in payload["categories"].items():
            normalized_category = normalize_label(category)
            subtypes = frozenset(normalize_label(value) for value in definition["subtypes"])
            unknown = normalize_label(definition["unknown_subtype"])
            if unknown not in subtypes:
                raise CatalogValidationError(
                    f"{normalized_category}: unknown_subtype alt tür listesinde bulunmuyor"
                )
            categories[normalized_category] = subtypes
            unknown_subtypes[normalized_category] = unknown
        return cls(str(payload["taxonomy_version"]), categories, unknown_subtypes)

    def validate_label_pair(self, category: str, subtype: str) -> tuple[str, str]:
        category = normalize_label(category)
        subtype = normalize_label(subtype)
        if category not in self.categories:
            raise CatalogValidationError(f"Bilinmeyen ana tür: {category}")
        if subtype not in self.categories[category]:
            raise CatalogValidationError(f"{subtype}, {category} altında tanımlı değil")
        return category, subtype

    def unknown_for(self, category: str) -> str:
        category = normalize_label(category)
        if category not in self.unknown_subtypes:
            raise CatalogValidationError(f"Bilinmeyen ana tür: {category}")
        return self.unknown_subtypes[category]


REQUIRED_RECORD_FIELDS = frozenset(
    {
        "asset_id",
        "audio_path",
        "sha256",
        "category",
        "subtype",
        "review_status",
        "taxonomy_version",
        "source_recording_id",
        "source_uri",
        "license",
    }
)


def normalize_label(value: str) -> str:
    normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise CatalogValidationError("Etiket boş olamaz")
    return normalized


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_catalog_record(
    record: Mapping[str, Any],
    taxonomy: Taxonomy | None = None,
    *,
    require_existing_audio: bool = False,
) -> dict[str, Any]:
    """Validate and normalize one manifest record.

    Validation does not mutate the supplied mapping. Unknown subtypes are valid
    catalog labels, but they are intentionally not eligible for fingerprinting.
    """

    taxonomy = taxonomy or Taxonomy.load()
    missing = sorted(REQUIRED_RECORD_FIELDS - record.keys())
    if missing:
        raise CatalogValidationError(f"Eksik manifest alanları: {', '.join(missing)}")

    normalized = dict(record)
    category, subtype = taxonomy.validate_label_pair(record["category"], record["subtype"])
    normalized["category"] = category
    normalized["subtype"] = subtype

    try:
        status = ReviewStatus(str(record["review_status"]).strip().upper())
    except ValueError as exc:
        allowed = ", ".join(status.value for status in ReviewStatus)
        raise CatalogValidationError(f"Geçersiz review_status; izin verilenler: {allowed}") from exc
    normalized["review_status"] = status.value

    if str(record["taxonomy_version"]) != taxonomy.version:
        raise CatalogValidationError(
            f"Taksonomi sürümü {taxonomy.version} olmalı: {record['taxonomy_version']}"
        )

    digest = str(record["sha256"]).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise CatalogValidationError("sha256, 64 karakterlik hexadecimal değer olmalıdır")
    normalized["sha256"] = digest

    for field in ("asset_id", "audio_path", "source_recording_id", "source_uri", "license"):
        if not str(record[field]).strip():
            raise CatalogValidationError(f"{field} boş olamaz")

    audio_path = Path(str(record["audio_path"]))
    if require_existing_audio and not audio_path.is_file():
        raise CatalogValidationError(f"Ses dosyası bulunamadı: {audio_path}")
    return normalized


def fingerprint_eligibility(
    record: Mapping[str, Any], taxonomy: Taxonomy | None = None
) -> tuple[bool, str]:
    """Return whether a validated record may enter the fingerprint index."""

    taxonomy = taxonomy or Taxonomy.load()
    try:
        normalized = validate_catalog_record(record, taxonomy)
    except CatalogValidationError as exc:
        return False, f"Geçersiz kayıt: {exc}"

    if normalized["review_status"] != ReviewStatus.APPROVED.value:
        return False, "İnsan onayı tamamlanmamış"
    if record.get("agent_action") == "QUARANTINE" or record.get("quality_issues"):
        return False, "Kalite karantinasındaki kayıt parmak izi kataloğuna eklenemez"
    if normalized["subtype"] == taxonomy.unknown_for(normalized["category"]):
        return False, "UNKNOWN alt tür parmak izi kataloğuna eklenemez"
    return True, "Onaylı ve indekslemeye uygun"
