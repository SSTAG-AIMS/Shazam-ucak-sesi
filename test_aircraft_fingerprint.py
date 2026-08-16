"""Small functional tests for the aircraft fingerprint layer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from aircraft_fingerprint import AircraftFingerprintDatabase, FingerprintConfig


class AircraftFingerprintTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sr = 22050
        rng = np.random.default_rng(42)
        seconds = 8
        t = np.arange(seconds * self.sr) / self.sr
        carrier = (
            0.40 * np.sin(2 * np.pi * (210 + 32 * t) * t)
            + 0.22 * np.sin(2 * np.pi * 690 * t)
            + 0.12 * np.sin(2 * np.pi * 1370 * t)
        )
        envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 1.7 * t) ** 2
        self.reference = (carrier * envelope + 0.01 * rng.normal(size=t.size)).astype(
            np.float32
        )

    def test_shifted_excerpt_matches_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference_path = root / "a320.wav"
            wavfile.write(reference_path, self.sr, self.reference)
            database = AircraftFingerprintDatabase(
                root / "fingerprints.sqlite3",
                config=FingerprintConfig(peak_percentile=65.0),
                min_aligned_hashes=4,
                min_confidence=0.01,
            )
            database.add_reference(reference_path, "AIRBUS_A320")
            excerpt = self.reference[2 * self.sr:7 * self.sr]
            match = database.match_samples(excerpt)
            self.assertIsNotNone(match)
            self.assertTrue(match.accepted)
            self.assertEqual(match.aircraft_type, "AIRBUS_A320")


if __name__ == "__main__":
    unittest.main()
