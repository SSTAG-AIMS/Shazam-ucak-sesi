"""Utilities for stable multi-window audio classification."""

from __future__ import annotations

import numpy as np


def select_audio_windows(
    samples: np.ndarray,
    window_samples: int,
    *,
    hop_samples: int | None = None,
    max_windows: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return up to ``max_windows`` full windows spread across an audio clip."""
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    hop = window_samples // 2 if hop_samples is None else int(hop_samples)
    if hop <= 0:
        raise ValueError("hop_samples must be positive")

    if audio.size <= window_samples:
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[: audio.size] = audio
        return padded[None, :], np.array([0], dtype=np.int64)

    last_start = audio.size - window_samples
    starts = list(range(0, last_start + 1, hop))
    if starts[-1] != last_start:
        starts.append(last_start)

    if len(starts) > max_windows:
        selected = np.linspace(0, len(starts) - 1, max_windows)
        indices = np.rint(selected).astype(np.int64)
        starts = [starts[index] for index in indices]

    windows = np.stack(
        [audio[start : start + window_samples] for start in starts]
    ).astype(np.float32, copy=False)
    return windows, np.asarray(starts, dtype=np.int64)


def aggregate_window_probabilities(
    probabilities: np.ndarray,
    classes: list[str] | tuple[str, ...],
) -> dict:
    """Aggregate window predictions with majority vote and soft tie-breaking."""
    probs = np.asarray(probabilities, dtype=np.float64)
    # Label encoders often expose ``numpy.str_`` values.  Convert them to
    # plain strings so the result can be safely serialized by the GUI/export
    # layer as well as by JSON-based reports.
    class_names = [str(label) for label in classes]
    if probs.ndim != 2 or probs.shape[0] == 0:
        raise ValueError("probabilities must be a non-empty 2D array")
    if probs.shape[1] != len(class_names) or not class_names:
        raise ValueError("probability columns must match classes")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative")

    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("each probability row must have positive mass")
    probs = probs / row_sums

    predicted_indices = np.argmax(probs, axis=1)
    vote_counts_array = np.bincount(
        predicted_indices, minlength=len(class_names)
    )
    top_vote_count = int(vote_counts_array.max())
    tied_indices = np.flatnonzero(vote_counts_array == top_vote_count)
    mean_probs = probs.mean(axis=0)
    winner_index = int(tied_indices[np.argmax(mean_probs[tied_indices])])

    return {
        "winner": class_names[winner_index],
        "winner_index": winner_index,
        "confidence": float(mean_probs[winner_index]),
        "vote_share": float(top_vote_count / probs.shape[0]),
        "vote_counts": {
            label: int(count)
            for label, count in zip(class_names, vote_counts_array)
        },
        "window_predictions": [
            class_names[index] for index in predicted_indices
        ],
        "window_confidences": [
            float(probs[row, index])
            for row, index in enumerate(predicted_indices)
        ],
        "mean_probabilities": {
            label: float(value)
            for label, value in zip(class_names, mean_probs)
        },
    }
