import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LAB = ROOT / "Test_Folder" / "AIRCRAFT_REFERENCE_LAB_V1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class AircraftLabManifestIntegrityTests(unittest.TestCase):
    def test_test_folder_contains_only_manifest_records(self):
        payload = json.loads((LAB / "test_manifest.json").read_text(encoding="utf-8"))
        declared = {Path(row["audio_path"]).resolve() for row in payload["records"]}
        actual = set((LAB / "TEST_SESLERI").rglob("*.wav"))
        self.assertEqual(actual, declared)

    def test_test_audio_is_not_an_exact_gold_reference_copy(self):
        test_hashes = {digest(path) for path in (LAB / "TEST_SESLERI").rglob("*.wav")}
        gold_hashes = {digest(path) for path in (LAB / "ALTIN_REFERANSLAR").rglob("*.wav")}
        self.assertFalse(test_hashes & gold_hashes)


if __name__ == "__main__":
    unittest.main()
