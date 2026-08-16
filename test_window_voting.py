"""Tests for multi-window audio voting."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from noise_detector import (
    AirportNoiseSystem,
    _load_and_chunk_ml,
    has_dominant_aircraft_evidence,
)
from window_voting import (
    aggregate_window_probabilities,
    select_audio_windows,
)


class SelectAudioWindowsTest(unittest.TestCase):
    def test_short_clip_is_padded_to_one_window(self) -> None:
        windows, starts = select_audio_windows(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            5,
            hop_samples=2,
            max_windows=5,
        )

        np.testing.assert_array_equal(starts, [0])
        np.testing.assert_array_equal(windows, [[1.0, 2.0, 3.0, 0.0, 0.0]])

    def test_file_chunking_includes_the_final_tail_window(self) -> None:
        sample_rate = 22_050
        audio = np.arange(8 * sample_rate, dtype=np.float32)

        with patch(
            "noise_detector.librosa.load",
            return_value=(audio, sample_rate),
        ):
            windows, _, starts = _load_and_chunk_ml(
                "unused.wav", return_starts=True
            )

        np.testing.assert_array_equal(
            starts,
            [0, int(2.5 * sample_rate), int(3.0 * sample_rate)],
        )
        self.assertEqual(len(windows), 3)
        np.testing.assert_array_equal(windows[-1][0], audio[-5 * sample_rate])

    def test_file_chunking_caps_long_recording_at_five_windows(self) -> None:
        sample_rate = 22_050
        audio = np.arange(30 * sample_rate, dtype=np.float32)

        with patch(
            "noise_detector.librosa.load",
            return_value=(audio, sample_rate),
        ):
            windows, _, starts = _load_and_chunk_ml(
                "unused.wav", return_starts=True
            )

        self.assertEqual(len(windows), 5)
        self.assertEqual(len(starts), 5)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 25 * sample_rate)

    def test_long_clip_uses_five_windows_including_both_ends(self) -> None:
        audio = np.arange(30, dtype=np.float32)
        windows, starts = select_audio_windows(
            audio, 6, hop_samples=3, max_windows=5
        )

        self.assertEqual(windows.shape, (5, 6))
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 24)
        np.testing.assert_array_equal(windows[:, 0], starts)

    def test_exact_window_is_not_duplicated(self) -> None:
        audio = np.arange(6, dtype=np.float32)
        windows, starts = select_audio_windows(audio, 6, max_windows=5)

        self.assertEqual(windows.shape, (1, 6))
        np.testing.assert_array_equal(starts, [0])


class AggregateWindowProbabilitiesTest(unittest.TestCase):
    def test_majority_vote_wins(self) -> None:
        result = aggregate_window_probabilities(
            np.array(
                [
                    [0.51, 0.49],
                    [0.52, 0.48],
                    [0.51, 0.49],
                    [0.01, 0.99],
                    [0.01, 0.99],
                ]
            ),
            ["A", "B"],
        )

        self.assertEqual(result["winner"], "A")
        self.assertEqual(result["vote_counts"], {"A": 3, "B": 2})
        self.assertAlmostEqual(result["vote_share"], 0.6)
        self.assertEqual(len(result["window_predictions"]), 5)

    def test_probability_mean_breaks_vote_tie(self) -> None:
        result = aggregate_window_probabilities(
            np.array(
                [
                    [0.51, 0.49],
                    [0.52, 0.48],
                    [0.10, 0.90],
                    [0.20, 0.80],
                ]
            ),
            ["A", "B"],
        )

        self.assertEqual(result["winner"], "B")
        self.assertEqual(result["vote_counts"], {"A": 2, "B": 2})
        self.assertAlmostEqual(result["confidence"], 0.6675)

    def test_rows_are_normalized_before_aggregation(self) -> None:
        result = aggregate_window_probabilities(
            np.array([[2.0, 1.0], [4.0, 1.0]]),
            ["A", "B"],
        )

        self.assertEqual(result["winner"], "A")
        self.assertAlmostEqual(
            result["mean_probabilities"]["A"],
            ((2.0 / 3.0) + (4.0 / 5.0)) / 2.0,
        )

    def test_strict_vote_majority_is_not_overturned_by_confidence(self) -> None:
        result = aggregate_window_probabilities(
            np.array(
                [
                    [0.34, 0.33, 0.33],
                    [0.34, 0.33, 0.33],
                    [0.34, 0.33, 0.33],
                    [0.01, 0.98, 0.01],
                    [0.01, 0.98, 0.01],
                ]
            ),
            ["A", "B", "C"],
        )

        self.assertEqual(result["winner"], "A")
        self.assertEqual(result["vote_counts"], {"A": 3, "B": 2, "C": 0})


class MultiWindowInferenceIntegrationTest(unittest.TestCase):
    def test_beats_head_runs_each_selected_window_and_votes(self) -> None:
        class SequencedEncoder:
            def __init__(self) -> None:
                self.outputs = iter(
                    [
                        torch.tensor([[[4.0, 1.0]]]),
                        torch.tensor([[[1.0, 4.0]]]),
                        torch.tensor([[[3.0, 1.0]]]),
                    ]
                )

            def extract_features(self, waveform, padding_mask):
                return next(self.outputs).to(waveform.device), None

        system = AirportNoiseSystem.__new__(AirportNoiseSystem)
        system.beats_model = SimpleNamespace(encoder=SequencedEncoder())
        ten_seconds = np.zeros(10 * 22050, dtype=np.float32)

        result = system._infer_beats_head_multi_window(
            ten_seconds, torch.nn.Identity(), ["A", "B"]
        )

        self.assertEqual(result["winner"], "A")
        self.assertEqual(result["vote_counts"], {"A": 2, "B": 1})
        self.assertEqual(result["n_windows"], 3)
        self.assertEqual(result["window_starts_s"], [0.0, 2.5, 5.0])

    def test_main_category_path_exposes_the_same_vote_metadata(self) -> None:
        class StubSVM:
            def predict_proba(self, features):
                value = float(features[0, 0])
                if value < 0.5:
                    return np.array([[0.51, 0.49]])
                if value < 1.5:
                    return np.array([[0.52, 0.48]])
                return np.array([[0.01, 0.99]])

        system = AirportNoiseSystem.__new__(AirportNoiseSystem)
        system.ml_model = StubSVM()
        system.ml_le = SimpleNamespace(classes_=np.array(["A", "B"]))
        windows = [
            np.array([0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
        ]
        starts = np.array([0, 55_125, 110_250], dtype=np.int64)

        with patch(
            "noise_detector._load_and_chunk_ml",
            return_value=(windows, np.zeros(3, dtype=np.float32), starts),
        ), patch(
            "noise_detector.extract_features_ml",
            side_effect=lambda chunk: np.array([chunk[0]], dtype=np.float32),
        ):
            labels, times, summary = system._classify_ml("unused.wav")

        self.assertEqual(labels, ["A", "A", "B"])
        np.testing.assert_allclose(times, [0.0, 2.5, 5.0])
        self.assertEqual(summary, {"A": 66.7, "B": 33.3})
        self.assertEqual(system._last_window_voting["winner"], "A")
        self.assertEqual(system._last_window_voting["vote_counts"], {"A": 2, "B": 1})
        self.assertEqual(system._last_window_voting["n_windows"], 3)


class AircraftEvidenceGateTest(unittest.TestCase):
    def test_only_dominant_aircraft_opens_identification_gate(self) -> None:
        self.assertTrue(
            has_dominant_aircraft_evidence(
                {"AIRCRAFT": 42.1, "AMBIENT": 26.3, "TRAFFIC": 5.3}
            )
        )
        self.assertFalse(
            has_dominant_aircraft_evidence(
                {"AMBIENT": 65.2, "AIRCRAFT": 17.4, "TRAFFIC": 8.7}
            )
        )

    def test_empty_and_non_aircraft_summaries_are_rejected(self) -> None:
        self.assertFalse(has_dominant_aircraft_evidence(None))
        self.assertFalse(has_dominant_aircraft_evidence({}))
        self.assertFalse(
            has_dominant_aircraft_evidence({"TRAFFIC": 100.0})
        )
        self.assertFalse(
            has_dominant_aircraft_evidence({"OTHER": 100.0})
        )


if __name__ == "__main__":
    unittest.main()
