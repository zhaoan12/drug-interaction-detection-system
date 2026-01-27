from __future__ import annotations

from collections import Counter

from drug_interaction_detection.domain import PreparedExample


FeatureVector = dict[str, float]


def _add_token_features(features: FeatureVector, prefix: str, values: list[str]) -> None:
    for value in values:
        features[f"{prefix}:{value.lower()}"] = 1.0


def _add_shared_token_features(features: FeatureVector, prefix: str, left: list[str], right: list[str]) -> None:
    for value in sorted(set(left).intersection(right)):
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
        _add_token_features(features, "drug_a.contraindication", example.drug_a.contraindications)
        _add_token_features(features, "drug_b.contraindication", example.drug_b.contraindications)
        _add_token_features(features, "drug_a.indication", example.drug_a.indications)
        _add_token_features(features, "drug_b.indication", example.drug_b.indications)
        _add_shared_token_features(features, "shared_enzyme", example.drug_a.enzymes, example.drug_b.enzymes)
        _add_shared_token_features(features, "shared_transporter", example.drug_a.transporters, example.drug_b.transporters)
        _add_shared_token_features(features, "shared_mechanism", example.drug_a.mechanisms, example.drug_b.mechanisms)
        _add_shared_token_features(features, "shared_warning", example.drug_a.warnings, example.drug_b.warnings)
        features["drug_a.warning_count"] = float(len(example.drug_a.warnings))
        features["drug_b.warning_count"] = float(len(example.drug_b.warnings))
        features["drug_a.enzyme_count"] = float(len(example.drug_a.enzymes))
        features["drug_b.enzyme_count"] = float(len(example.drug_b.enzymes))
        features["shared_enzyme_count"] = float(len(set(example.drug_a.enzymes).intersection(example.drug_b.enzymes)))
        features["shared_warning_count"] = float(len(set(example.drug_a.warnings).intersection(example.drug_b.warnings)))
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
