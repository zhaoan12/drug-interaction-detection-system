from __future__ import annotations

from collections import Counter

from drug_interaction_detection.domain import PreparedExample


class MajorityClassifier:
    def __init__(self) -> None:
        self.majority_label = ""

    def fit(self, examples: list[PreparedExample], target_name: str) -> None:
        counts = Counter(getattr(example, target_name) for example in examples)
        self.majority_label = counts.most_common(1)[0][0]

    def predict(self) -> str:
        return self.majority_label

