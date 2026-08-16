import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from aircraft_audio_comparison_v1 import compare_audio, load_audio, log_mel_spectrogram


class AircraftAudioComparisonTests(unittest.TestCase):
    def test_identical_audio_has_higher_similarity_than_different_tone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); sr = 22050; time = np.arange(sr * 2) / sr
            first = root / "first.wav"; same = root / "same.wav"; other = root / "other.wav"
            sf.write(first, np.sin(2 * np.pi * 220 * time) * .3, sr)
            sf.write(same, np.sin(2 * np.pi * 220 * time) * .3, sr)
            sf.write(other, np.sin(2 * np.pi * 4000 * time) * .3, sr)
            match = compare_audio(first, same); mismatch = compare_audio(first, other)
            self.assertGreater(match["combined_similarity"], mismatch["combined_similarity"])
            samples, loaded_sr = load_audio(first)
            self.assertEqual(log_mel_spectrogram(samples, loaded_sr).shape[0], 64)
            self.assertIn("doğruluk", match["disclaimer"])


if __name__ == "__main__":
    unittest.main()
