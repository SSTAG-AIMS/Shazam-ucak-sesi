"""Evaluate project classifiers and weighted consensus on a sealed manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from catalog_review import read_jsonl
from multi_model_consensus import ModelVote, WeightedConsensus
from noise_detector import AirportNoiseSystem


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "cache" / "sealed_consensus_benchmark_pilot_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "multi_model_consensus_pilot_v1.json"
MODEL_PREFS = ("efficientnet", "cnn", "svm", "beats")


def probabilities_from_result(result: dict[str, Any]) -> dict[str, float]:
    frame_probs = result.get("frame_probs")
    if frame_probs is None:
        frame_probs = []
    raw_classes = result.get("class_names")
    classes = [str(value) for value in raw_classes] if raw_classes is not None else []
    if len(frame_probs) and classes:
        average = np.mean(np.asarray(frame_probs, dtype=np.float64), axis=0)
        return {label: float(value) for label, value in zip(classes, average)}
    summary = result.get("summary") or {}
    return {str(label): float(value) / 100.0 for label, value in summary.items()}


def evaluate(
    manifest_path: Path,
    output_path: Path,
    *,
    model_preferences: tuple[str, ...] = MODEL_PREFS,
    include_clap: bool = False,
) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError("Kapalı test manifesti boş")
    system = AirportNoiseSystem()
    available = {
        "efficientnet": system.eff_model is not None,
        "cnn": system.cnn_model is not None,
        "svm": system.ml_model is not None,
        "beats": system.beats_model is not None,
    }
    enabled = [name for name in model_preferences if available.get(name, False)]
    clap = None
    if include_clap:
        from clap_zero_shot import ClapZeroShotClassifier

        clap = ClapZeroShotClassifier()
        clap.load(local_files_only=True)
        enabled.append("clap")
    if len(enabled) < 3:
        raise RuntimeError(f"Uzlaşma için en az 3 model gerekli; bulunan: {enabled}")

    engine = WeightedConsensus(minimum_participating_models=3)
    predictions: dict[str, list[str]] = {name: [] for name in enabled}
    true_labels: list[str] = []
    consensus_labels: list[str] = []
    details = []

    for index, row in enumerate(rows, start=1):
        path = Path(row["path"])
        true_label = str(row["label"])
        true_labels.append(true_label)
        votes = []
        model_results = {}
        print(f"[ConsensusEval] {index}/{len(rows)} {path.name}")
        for model_name in enabled:
            if model_name == "clap":
                probabilities = clap.predict_file(path, sorted(engine.allowed_categories))
            else:
                result = system.analyze_for_gui(
                    str(path), model_pref=model_name, identify_subtype=False
                )
                probabilities = probabilities_from_result(result)
            vote = ModelVote(model_name, probabilities)
            votes.append(vote)
            normalized = vote.normalized_probabilities(engine.allowed_categories)
            predicted = max(normalized, key=normalized.get) if normalized else "UNKNOWN_CATEGORY"
            predictions[model_name].append(predicted)
            model_results[model_name] = {
                "predicted": predicted,
                "probabilities": normalized,
            }
        consensus = engine.decide(votes)
        consensus_labels.append(consensus.label)
        details.append(
            {
                "path": str(path),
                "true_label": true_label,
                "models": model_results,
                "consensus": consensus.as_dict(),
            }
        )

    labels = sorted(set(true_labels))
    metrics = {}
    for model_name, values in {**predictions, "consensus": consensus_labels}.items():
        metrics[model_name] = {
            "accuracy": float(accuracy_score(true_labels, values)),
            "macro_f1": float(
                f1_score(true_labels, values, labels=labels, average="macro", zero_division=0)
            ),
            "coverage": float(np.mean([value != "UNKNOWN_CATEGORY" for value in values])),
        }
    report = {
        "manifest": str(manifest_path),
        "sample_count": len(rows),
        "classes": labels,
        "enabled_models": enabled,
        "calibration_status": "PILOT_ONLY_NOT_FOR_PRODUCTION_WEIGHTS",
        "metrics": metrics,
        "details": details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Kapalı sette çoklu model uzlaşmasını ölç")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-clap",
        action="store_true",
        help="Yerel önbellekteki CLAP modelini beşinci kanıt olarak kullan",
    )
    args = parser.parse_args()
    report = evaluate(args.manifest, args.output, include_clap=args.include_clap)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
