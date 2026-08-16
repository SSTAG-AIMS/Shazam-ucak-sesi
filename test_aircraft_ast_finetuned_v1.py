"""Unit tests for the experimental AST multi-window adapter."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aircraft_ast_finetuned_v1 import AircraftASTFineTunedV1


class _TensorResult:
    def __init__(self, values) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _TorchStub:
    class _InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    @staticmethod
    def inference_mode():
        return _TorchStub._InferenceMode()

    @staticmethod
    def softmax(logits, dim=-1):
        values = np.asarray(logits, dtype=np.float64)
        values = np.exp(values - values.max(axis=dim, keepdims=True))
        return _TensorResult(values / values.sum(axis=dim, keepdims=True))


class AircraftASTMultiWindowTest(unittest.TestCase):
    def test_windows_are_spread_across_long_recording(self) -> None:
        audio = np.arange(40, dtype=np.float32)

        windows = AircraftASTFineTunedV1._windows(audio, sample_rate=2)

        self.assertEqual(len(windows), 3)
        np.testing.assert_array_equal(
            [window[0] for window in windows], [0.0, 10.0, 20.0]
        )

    def test_predict_file_uses_majority_vote_not_probability_mean(self) -> None:
        adapter = AircraftASTFineTunedV1()
        adapter._torch = _TorchStub()
        adapter._extractor = lambda window, **kwargs: {"input_values": window}
        outputs = iter(
            [
                np.log([[0.51, 0.49]]),
                np.log([[0.51, 0.49]]),
                np.log([[0.01, 0.99]]),
            ]
        )
        adapter._model = SimpleNamespace(
            config=SimpleNamespace(id2label={0: "A", 1: "B"}),
            __call__=None,
        )

        class Model:
            config = adapter._model.config

            def __call__(self, **inputs):
                return SimpleNamespace(logits=next(outputs))

        adapter._model = Model()

        with patch(
            "aircraft_ast_finetuned_v1.librosa.load",
            return_value=(np.zeros(20, dtype=np.float32), 1),
        ):
            result = adapter.predict_file("unused.wav")

        self.assertEqual(result["predicted"], "A")
        self.assertEqual(result["vote_counts"], {"A": 2, "B": 1})
        self.assertAlmostEqual(result["vote_share"], 2 / 3)
        self.assertEqual(result["method"], "ast_multi_window_vote")


if __name__ == "__main__":
    unittest.main()
