"""Add CLAP evidence to an existing sealed four-model pilot report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from clap_zero_shot import ClapZeroShotClassifier
from multi_model_consensus import ModelVote, WeightedConsensus


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs" / "multi_model_consensus_pilot_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "multi_model_consensus_pilot_clap_v1.json"


def augment(input_path: Path, output_path: Path) -> dict[str, Any]:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    details = report.get("details") or []
    if not details:
        raise ValueError("Mevcut pilot raporunda ayrıntılı kayıt bulunamadı")

    engine = WeightedConsensus(minimum_participating_models=3)
    clap = ClapZeroShotClassifier()
    clap.load(local_files_only=True)
    true_labels: list[str] = []
    clap_labels: list[str] = []
    consensus_labels: list[str] = []

    for index, detail in enumerate(details, start=1):
        path = Path(detail["path"])
        true_label = str(detail["true_label"])
        print(f"[CLAPEval] {index}/{len(details)} {path.name}")
        probabilities = clap.predict_file(path, sorted(engine.allowed_categories))
        predicted = max(probabilities, key=probabilities.get)
        detail.setdefault("models", {})["clap"] = {
            "predicted": predicted,
            "probabilities": probabilities,
        }
        votes = [
            ModelVote(name, model_result.get("probabilities") or {})
            for name, model_result in detail["models"].items()
        ]
        consensus = engine.decide(votes)
        detail["consensus_with_clap"] = consensus.as_dict()
        true_labels.append(true_label)
        clap_labels.append(predicted)
        consensus_labels.append(consensus.label)

    labels = sorted(set(true_labels))
    report["enabled_models_with_clap"] = [
        *report.get("enabled_models", []), "clap"
    ]
    report["metrics"]["clap"] = {
        "accuracy": float(accuracy_score(true_labels, clap_labels)),
        "macro_f1": float(
            f1_score(true_labels, clap_labels, labels=labels, average="macro", zero_division=0)
        ),
        "coverage": 1.0,
    }
    report["metrics"]["consensus_with_clap"] = {
        "accuracy": float(accuracy_score(true_labels, consensus_labels)),
        "macro_f1": float(
            f1_score(
                true_labels,
                consensus_labels,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "coverage": float(
            np.mean([label != "UNKNOWN_CATEGORY" for label in consensus_labels])
        ),
    }
    report["calibration_status"] = "PILOT_ONLY_NOT_FOR_PRODUCTION_WEIGHTS"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Mevcut pilot raporuna CLAP kanıtı ekle")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = augment(args.input, args.output)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
