"""Weighted, abstaining consensus for independent audio classifiers.

Shazam fingerprints are intentionally excluded: an exact recording match is a
separate evidence channel, not another semantic-classification vote.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from dataset_catalog import Taxonomy, normalize_label


@dataclass(frozen=True)
class ModelVote:
    model_name: str
    probabilities: Mapping[str, float]
    weight: float = 1.0
    minimum_confidence: float = 0.45

    def normalized_probabilities(self, allowed: set[str]) -> dict[str, float]:
        values = {
            normalize_label(label): max(0.0, float(probability))
            for label, probability in self.probabilities.items()
            if normalize_label(label) in allowed
        }
        total = sum(values.values())
        if total <= 0.0:
            return {}
        return {label: value / total for label, value in values.items()}


@dataclass(frozen=True)
class ConsensusResult:
    label: str
    suggested_label: str
    accepted: bool
    strength: str
    weighted_confidence: float
    margin: float
    agreement_ratio: float
    vote_counts: Mapping[str, int]
    weighted_scores: Mapping[str, float]
    participating_models: int
    abstaining_models: tuple[str, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


class WeightedConsensus:
    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        *,
        minimum_participating_models: int = 3,
        strong_agreement: float = 0.80,
        normal_agreement: float = 0.60,
        strong_confidence: float = 0.65,
        normal_confidence: float = 0.55,
        minimum_margin: float = 0.15,
    ) -> None:
        self.taxonomy = taxonomy or Taxonomy.load()
        self.allowed_categories = set(self.taxonomy.categories)
        self.minimum_participating_models = minimum_participating_models
        self.strong_agreement = strong_agreement
        self.normal_agreement = normal_agreement
        self.strong_confidence = strong_confidence
        self.normal_confidence = normal_confidence
        self.minimum_margin = minimum_margin

    def decide(self, votes: Sequence[ModelVote]) -> ConsensusResult:
        weighted = defaultdict(float)
        counts: Counter[str] = Counter()
        abstaining: list[str] = []
        participating = 0
        total_weight = 0.0

        for vote in votes:
            if vote.weight <= 0:
                abstaining.append(vote.model_name)
                continue
            probabilities = vote.normalized_probabilities(self.allowed_categories)
            if not probabilities:
                abstaining.append(vote.model_name)
                continue
            label, confidence = max(probabilities.items(), key=lambda item: item[1])
            if confidence < vote.minimum_confidence:
                abstaining.append(vote.model_name)
                continue
            participating += 1
            total_weight += vote.weight
            counts[label] += 1
            for candidate, probability in probabilities.items():
                weighted[candidate] += vote.weight * probability

        if total_weight > 0:
            scores = {
                label: weighted.get(label, 0.0) / total_weight
                for label in sorted(self.allowed_categories)
            }
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            suggested, confidence = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = confidence - runner_up
            agreement = counts[suggested] / participating if participating else 0.0
        else:
            scores = {label: 0.0 for label in sorted(self.allowed_categories)}
            suggested, confidence, margin, agreement = "OTHER", 0.0, 0.0, 0.0

        reasons = []
        strength = "INSUFFICIENT"
        accepted = False
        if participating < self.minimum_participating_models:
            reasons.append("INSUFFICIENT_MODELS")
        elif margin < self.minimum_margin:
            reasons.append("LOW_MARGIN")
        elif agreement >= self.strong_agreement and confidence >= self.strong_confidence:
            accepted = True
            strength = "STRONG"
        elif agreement >= self.normal_agreement and confidence >= self.normal_confidence:
            accepted = True
            strength = "NORMAL"
        else:
            if agreement < self.normal_agreement:
                reasons.append("MODEL_DISAGREEMENT")
            if confidence < self.normal_confidence:
                reasons.append("LOW_WEIGHTED_CONFIDENCE")

        return ConsensusResult(
            label=suggested if accepted else "UNKNOWN_CATEGORY",
            suggested_label=suggested,
            accepted=accepted,
            strength=strength,
            weighted_confidence=float(confidence),
            margin=float(margin),
            agreement_ratio=float(agreement),
            vote_counts=dict(counts),
            weighted_scores=scores,
            participating_models=participating,
            abstaining_models=tuple(abstaining),
            reasons=tuple(reasons),
        )
