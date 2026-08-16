"""Read-only lookup of known Shazam reference audio for human A/B review."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ReferenceAudio:
    category: str
    subtype: str
    name: str
    path: Path
    hash_count: int
    catalog: str


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReferenceAudioLibrary:
    def __init__(
        self,
        aircraft_db: Path | None = None,
        category_db: Path | None = None,
    ) -> None:
        self.aircraft_db = aircraft_db or ROOT / "models" / "aircraft_fingerprints.sqlite3"
        self.category_db = category_db or ROOT / "models" / "category_fingerprints_v1.sqlite3"

    def references_for(
        self, category: str, subtype: str, *, exclude_path: Path | None = None
    ) -> list[ReferenceAudio]:
        category, subtype = category.upper(), subtype.upper()
        if not subtype or subtype.startswith("UNKNOWN"):
            return []
        db = self.aircraft_db if category == "AIRCRAFT" else self.category_db
        key = subtype if category == "AIRCRAFT" else f"{category}::{subtype}"
        if not db.is_file():
            return []

        connection = sqlite3.connect(db)
        try:
            rows = connection.execute(
                "SELECT reference_name, source_path, hash_count FROM tracks "
                "WHERE aircraft_type = ? ORDER BY hash_count DESC, reference_name",
                (key,),
            ).fetchall()
        finally:
            connection.close()

        excluded = exclude_path.resolve() if exclude_path and exclude_path.exists() else None
        excluded_hash = _file_hash(excluded) if excluded else None
        references: list[ReferenceAudio] = []
        for name, raw_path, hash_count in rows:
            path = Path(raw_path)
            if not path.is_file():
                continue
            resolved = path.resolve()
            if excluded and resolved == excluded:
                continue
            if excluded_hash and _file_hash(resolved) == excluded_hash:
                continue
            references.append(
                ReferenceAudio(category, subtype, str(name), resolved, int(hash_count), db.name)
            )
        return references
