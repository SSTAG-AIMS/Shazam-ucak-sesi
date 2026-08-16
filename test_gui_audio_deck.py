import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication

from gui_aircraft_reference_intake_v1 import AudioDeck


class AudioDeckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_live_volume_changes_signal_amplitude_and_keeps_seek(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            sf.write(path, np.sin(np.linspace(0, 300, 44100)).astype(np.float32) * .05, 22050)
            deck = AudioDeck("TEST", "#00ffff", lambda: None); deck.set_audio(path)
            with patch("gui_aircraft_reference_intake_v1.sd.play") as play, patch("gui_aircraft_reference_intake_v1.sd.stop"):
                deck.volume.setValue(25); deck.play(); low = float(np.max(np.abs(play.call_args.args[0])))
                calls = play.call_count; deck.volume.setValue(75); high = float(np.max(np.abs(play.call_args.args[0])))
                self.assertGreater(play.call_count, calls)
                self.assertAlmostEqual(high / low, 3.0, places=1)
                deck.pause(); deck.seek.setValue(500)
                self.assertAlmostEqual(deck.position(), deck.duration() / 2, places=1)
            deck.close()


if __name__ == "__main__":
    unittest.main()
