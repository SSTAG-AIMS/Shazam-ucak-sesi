import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from prepare_aircraft_reference_lab_v1 import TYPES, prepare_lab


class PrepareAircraftReferenceLabTests(unittest.TestCase):
    def test_creates_two_isolated_samples_per_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source"; lab = root / "lab"
            for aircraft_type in TYPES:
                folder = source / aircraft_type; folder.mkdir(parents=True)
                for index in range(2):
                    sf.write(folder / f"HEX{index}_{index}.wav", np.ones(22050, dtype=np.float32) * .1, 22050)
            manifest = prepare_lab(lab, source)
            self.assertEqual(manifest["sample_count"], len(TYPES))
            self.assertEqual(manifest["gold_reference_count"], len(TYPES))
            self.assertTrue((lab / "test_manifest.json").is_file())
            self.assertTrue((lab / "workspace").is_dir())


if __name__ == "__main__":
    unittest.main()
