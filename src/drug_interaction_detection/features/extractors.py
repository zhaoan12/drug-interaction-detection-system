from __future__ import annotations

from collections import Counter

from drug_interaction_detection.domain import PreparedExample


FeatureVector = dict[str, float]


def _add_token_features(features: FeatureVector, prefix: str, values: list[str]) -> None:
    for value in values:
        features[f"{prefix}:{value.lower()}"] = 1.0


def pair_features(example: PreparedExample, include_reference: bool = True) -> FeatureVector:
    features: FeatureVector = {
        f"drug_a:{example.drug_a.generic_name}": 1.0,
        f"drug_b:{example.drug_b.generic_name}": 1.0,
        f"class_pair:{example.drug_a.drug_class}|{example.drug_b.drug_class}": 1.0,
        f"class_pair:{example.drug_b.drug_class}|{example.drug_a.drug_class}": 1.0,
    }
    if include_reference:
        _add_token_features(features, "drug_a.mechanism", example.drug_a.mechanisms)
        _add_token_features(features, "drug_b.mechanism", example.drug_b.mechanisms)
        _add_token_features(features, "drug_a.warning", example.drug_a.warnings)
        _add_token_features(features, "drug_b.warning", example.drug_b.warnings)
        shared_enzymes = sorted(set(example.drug_a.enzymes).intersection(example.drug_b.enzymes))
        shared_transporters = sorted(set(example.drug_a.transporters).intersection(example.drug_b.transporters))
        for enzyme in shared_enzymes:
            features[f"shared_enzyme:{enzyme}"] = 1.0
        for transporter in shared_transporters:
            features[f"shared_transporter:{transporter}"] = 1.0
    evidence_tokens = Counter()
    for sentence in example.evidence:
        for token in sentence.lower().replace(",", "").split():
            evidence_tokens[token] += 1
    for token, count in evidence_tokens.items():
        features[f"evidence_token:{token}"] = float(count)
    return features


def top_feature_attribution(weights: dict[str, float], features: FeatureVector, limit: int = 5) -> dict[str, float]:
    contributions = {
        feature_name: round(value * weights.get(feature_name, 0.0), 5)
        for feature_name, value in features.items()
        if weights.get(feature_name, 0.0) != 0.0
    }
    ordered = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    return dict(ordered[:limit])

