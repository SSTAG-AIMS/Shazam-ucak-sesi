import unittest
from pathlib import Path

from aircraft_reference_prediction_v1 import predict_aircraft
from prepare_aircraft_reference_lab_v1 import LAB, prepare_lab


class AircraftReferencePredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = prepare_lab()

    def test_known_lab_audio_is_predicted_by_agent_without_shazam(self):
        row = self.manifest["records"][0]
        path = Path(row["audio_path"])
        result = predict_aircraft(path)
        self.assertEqual(result["method"], "ADVANCED_AUDIO_REACT_COUNCIL_V3")
        evidence = "\n".join(result["evidence"])
        self.assertIn("GELİŞMİŞ SES MODELLERİ", evidence)
        self.assertIn("BEATs-SVM", evidence)
        self.assertIn("AST-SVM", evidence)
        self.assertIn("PANNs-CNN14-SVM", evidence)
        self.assertIn("CLAP-SVM", evidence)
        self.assertIn("Multi-Embedding Fusion", evidence)
        self.assertIn("foundation_result", result)


if __name__ == "__main__":
    unittest.main()
