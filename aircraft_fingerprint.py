"""Shazam-style acoustic fingerprinting for aircraft reference recordings."""

from __future__ import annotations

import hashlib
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import stft


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


@dataclass(frozen=True)
class FingerprintConfig:
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    peak_neighborhood: int = 15
    peak_percentile: float = 75.0
    max_peaks_per_second: int = 12
    fan_value: int = 5
    min_time_delta: int = 1
    max_time_delta: int = 80
    frequency_bin_size: int = 2


@dataclass(frozen=True)
class AircraftMatch:
    aircraft_type: str
    reference_name: str
    matched_hashes: int
    aligned_hashes: int
    query_hashes: int
    confidence: float
    accepted: bool

    def as_dict(self) -> dict:
        return {
            "aircraft_type": self.aircraft_type,
            "reference_name": self.reference_name,
            "matched_hashes": self.matched_hashes,
            "aligned_hashes": self.aligned_hashes,
            "query_hashes": self.query_hashes,
            "confidence": self.confidence,
            "accepted": self.accepted,
        }


def load_audio(path: str | Path, sample_rate: int = 22050) -> np.ndarray:
    samples, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise ValueError(f"Ses kaydı boş: {path}")
    return samples


def _spectral_peaks(samples: np.ndarray, cfg: FingerprintConfig) -> list[tuple[int, int]]:
    if samples.ndim != 1:
        samples = np.mean(samples, axis=-1)
    if len(samples) < cfg.n_fft:
        samples = np.pad(samples, (0, cfg.n_fft - len(samples)))
    _, _, spectrum = stft(
        samples,
        fs=cfg.sample_rate,
        nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_length,
        nfft=cfg.n_fft,
        boundary=None,
        padded=False,
    )
    magnitude_db = 20.0 * np.log10(np.abs(spectrum) + 1e-10)
    local_max = maximum_filter(
        magnitude_db,
        size=(cfg.peak_neighborhood, cfg.peak_neighborhood),
        mode="nearest",
    )
    threshold = float(np.percentile(magnitude_db, cfg.peak_percentile))
    frequency_indices, time_indices = np.where(
        (magnitude_db == local_max) & (magnitude_db >= threshold)
    )

    # Dense broadband recordings can otherwise generate tens of thousands of
    # hashes per minute. Keep only the strongest landmarks in each one-second
    # region. This preserves robust peaks while keeping the SQLite index small.
    frames_per_second = max(1, round(cfg.sample_rate / cfg.hop_length))
    candidates: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    for freq, time in zip(frequency_indices, time_indices):
        second = int(time) // frames_per_second
        candidates[second].append(
            (float(magnitude_db[freq, time]), int(time), int(freq))
        )

    peaks: list[tuple[int, int]] = []
    for items in candidates.values():
        strongest = sorted(items, reverse=True)[:cfg.max_peaks_per_second]
        peaks.extend((time, freq) for _, time, freq in strongest)
    return sorted(peaks, key=lambda item: (item[0], item[1]))


def fingerprint_samples(
    samples: np.ndarray,
    cfg: FingerprintConfig | None = None,
) -> list[tuple[str, int]]:
    cfg = cfg or FingerprintConfig()
    peaks = _spectral_peaks(np.asarray(samples, dtype=np.float32), cfg)
    hashes: list[tuple[str, int]] = []
    for anchor_index, (anchor_time, anchor_freq) in enumerate(peaks):
        paired = 0
        for target_time, target_freq in peaks[anchor_index + 1:]:
            delta = target_time - anchor_time
            if delta < cfg.min_time_delta:
                continue
            if delta > cfg.max_time_delta:
                break
            f1 = anchor_freq // cfg.frequency_bin_size
            f2 = target_freq // cfg.frequency_bin_size
            digest = hashlib.sha1(f"{f1}|{f2}|{delta}".encode("ascii")).hexdigest()[:20]
            hashes.append((digest, anchor_time))
            paired += 1
            if paired >= cfg.fan_value:
                break
    return hashes


class AircraftFingerprintDatabase:
    """Persistent inverted index for time-aligned landmark hashes."""

    def __init__(
        self,
        db_path: str | Path,
        config: FingerprintConfig | None = None,
        min_aligned_hashes: int = 8,
        min_confidence: float = 0.05,
    ):
        self.db_path = Path(db_path)
        self.config = config or FingerprintConfig()
        self.min_aligned_hashes = min_aligned_hashes
        self.min_confidence = min_confidence

    @property
    def exists(self) -> bool:
        return self.db_path.is_file()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path))
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY,
                    aircraft_type TEXT NOT NULL,
                    reference_name TEXT NOT NULL,
                    source_path TEXT NOT NULL UNIQUE,
                    hash_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS fingerprints (
                    hash TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    anchor_time INTEGER NOT NULL,
                    FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_fingerprints_hash ON fingerprints(hash);
                """
            )

    def reset(self) -> None:
        """Clear generated index tables without touching any source audio."""
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS fingerprints;
                DROP TABLE IF EXISTS tracks;
                """
            )
        self.initialize()

    def compact(self) -> None:
        """Reclaim free SQLite pages left by a rebuild."""
        with closing(self._connect()) as connection:
            connection.execute("VACUUM")

    def add_reference(self, audio_path: str | Path, aircraft_type: str) -> int:
        path = Path(audio_path).resolve()
        hashes = fingerprint_samples(load_audio(path, self.config.sample_rate), self.config)
        if not hashes:
            raise ValueError(f"Parmak izi üretilemedi: {path}")
        self.initialize()
        with closing(self._connect()) as connection, connection:
            existing = connection.execute(
                "SELECT id FROM tracks WHERE source_path = ?", (str(path),)
            ).fetchone()
            if existing:
                connection.execute("DELETE FROM tracks WHERE id = ?", (existing[0],))
            cursor = connection.execute(
                """
                INSERT INTO tracks(aircraft_type, reference_name, source_path, hash_count)
                VALUES (?, ?, ?, ?)
                """,
                (aircraft_type, path.stem, str(path), len(hashes)),
            )
            track_id = int(cursor.lastrowid)
            connection.executemany(
                "INSERT INTO fingerprints(hash, track_id, anchor_time) VALUES (?, ?, ?)",
                ((digest, track_id, anchor_time) for digest, anchor_time in hashes),
            )
        return len(hashes)

    def index_reference_tree(self, reference_root: str | Path) -> dict[str, int]:
        root = Path(reference_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Referans klasörü bulunamadı: {root}")
        indexed: dict[str, int] = {}
        for type_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            aircraft_type = type_dir.name.upper().replace(" ", "_")
            for audio_path in sorted(type_dir.rglob("*")):
                if audio_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                    indexed[str(audio_path)] = self.add_reference(audio_path, aircraft_type)
        return indexed

    def match_samples(self, samples: np.ndarray) -> AircraftMatch | None:
        if not self.exists:
            return None
        query_hashes = fingerprint_samples(samples, self.config)
        if not query_hashes:
            return None
        query_by_hash: dict[str, list[int]] = defaultdict(list)
        for digest, query_time in query_hashes:
            query_by_hash[digest].append(query_time)
        offset_votes: Counter[tuple[int, int]] = Counter()
        total_matches: Counter[int] = Counter()
        hash_values = list(query_by_hash)
        with closing(self._connect()) as connection, connection:
            for start in range(0, len(hash_values), 800):
                batch = hash_values[start:start + 800]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"""
                    SELECT hash, track_id, anchor_time FROM fingerprints
                    WHERE hash IN ({placeholders})
                    """,
                    batch,
                )
                for digest, track_id, reference_time in rows:
                    for query_time in query_by_hash[digest]:
                        offset_votes[(track_id, reference_time - query_time)] += 1
                        total_matches[track_id] += 1
            if not offset_votes:
                return None
            (best_track_id, _), aligned = offset_votes.most_common(1)[0]
            row = connection.execute(
                "SELECT aircraft_type, reference_name FROM tracks WHERE id = ?",
                (best_track_id,),
            ).fetchone()
        if row is None:
            return None
        confidence = min(1.0, aligned / max(1, len(query_hashes)))
        accepted = aligned >= self.min_aligned_hashes and confidence >= self.min_confidence
        return AircraftMatch(
            aircraft_type=row[0] if accepted else "UNKNOWN_AIRCRAFT",
            reference_name=row[1],
            matched_hashes=int(total_matches[best_track_id]),
            aligned_hashes=int(aligned),
            query_hashes=len(query_hashes),
            confidence=float(confidence),
            accepted=accepted,
        )

    def match_file(self, audio_path: str | Path) -> AircraftMatch | None:
        return self.match_samples(load_audio(audio_path, self.config.sample_rate))

    def list_references(self) -> Iterable[tuple[str, str, int]]:
        if not self.exists:
            return []
        with closing(self._connect()) as connection, connection:
            return list(
                connection.execute(
                    """
                    SELECT aircraft_type, reference_name, hash_count
                    FROM tracks ORDER BY aircraft_type, reference_name
                    """
                )
            )
