import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from aircraft_reference_intake_v1 import ReferenceIntakeError, stage_reference


class AircraftReferenceIntakeTests(unittest.TestCase):
    def test_stages_provenance_and_never_auto_approves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            sf.write(audio, np.sin(np.linspace(0, 1000, 44100)).astype(np.float32) * 0.2, 22050)
            record = stage_reference(
                audio, aircraft_type="Airbus A321", icao_type="A21N",
                physical_airframe_id="ABC123", source_uri="https://example.test/audio",
                license_name="CC BY 4.0", inbox=root / "inbox", queue=root / "queue.jsonl",
            )
            self.assertEqual(record["proposed_subtype"], "AIRBUS_A321")
            self.assertFalse(record["fingerprint_indexed"])
            self.assertFalse(record["human_approved"])

    def test_rejects_unknown_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            sf.write(audio, np.ones(22050, dtype=np.float32) * 0.1, 22050)
            with self.assertRaises(ReferenceIntakeError):
                stage_reference(
                    audio, aircraft_type="A", icao_type="A21N", physical_airframe_id="X",
                    source_uri="https://example.test/a", license_name="ALL RIGHTS RESERVED",
                    inbox=Path(tmp) / "inbox", queue=Path(tmp) / "queue.jsonl",
                )

    def test_rejects_empty_type_and_invalid_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            sf.write(audio, np.ones(22050, dtype=np.float32) * 0.1, 22050)
            for aircraft_type, source in [("", "https://example.test/a"), ("A321", "https:///missing-host")]:
                with self.assertRaises(ReferenceIntakeError):
                    stage_reference(
                        audio, aircraft_type=aircraft_type, icao_type="A21N", physical_airframe_id="X",
                        source_uri=source, license_name="CC BY 4.0",
                        inbox=Path(tmp) / "inbox", queue=Path(tmp) / "queue.jsonl",
                    )


if __name__ == "__main__":
    unittest.main()
