from __future__ import annotations

from dataclasses import dataclass
import math

from drug_interaction_detection.features.extractors import FeatureVector


@dataclass
class SoftmaxModel:
    labels: list[str]
    weights: dict[str, dict[str, float]]
    bias: dict[str, float]

    @classmethod
    def initialize(cls, labels: list[str]) -> "SoftmaxModel":
        return cls(
            labels=labels,
            weights={label: {} for label in labels},
            bias={label: 0.0 for label in labels},
        )

    def scores(self, features: FeatureVector) -> dict[str, float]:
        scores: dict[str, float] = {}
        for label in self.labels:
            score = self.bias[label]
            for feature_name, value in features.items():
                score += self.weights[label].get(feature_name, 0.0) * value
            scores[label] = score
        return scores

    def probabilities(self, features: FeatureVector) -> dict[str, float]:
        scores = self.scores(features)
        max_score = max(scores.values())
        exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
        denominator = sum(exp_scores.values()) or 1.0
        return {label: value / denominator for label, value in exp_scores.items()}

    def predict(self, features: FeatureVector) -> tuple[str, float]:
        probabilities = self.probabilities(features)
        label = max(probabilities, key=probabilities.get)
        return label, probabilities[label]

    def fit(
        self,
        examples: list[tuple[FeatureVector, str]],
        *,
        epochs: int,
        learning_rate: float,
        l2_penalty: float,
    ) -> None:
        for _ in range(epochs):
            for features, gold_label in examples:
                probabilities = self.probabilities(features)
                for label in self.labels:
                    target = 1.0 if label == gold_label else 0.0
                    error = probabilities[label] - target
                    self.bias[label] -= learning_rate * error
                    for feature_name, value in features.items():
                        current = self.weights[label].get(feature_name, 0.0)
                        gradient = error * value + l2_penalty * current
                        updated = current - learning_rate * gradient
                        if updated == 0.0:
                            self.weights[label].pop(feature_name, None)
                        else:
                            self.weights[label][feature_name] = updated

