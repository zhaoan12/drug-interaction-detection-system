from __future__ import annotations

from drug_interaction_detection.config.settings import ResolvedSettings
from drug_interaction_detection.domain import PredictionResult, PreparedExample
from drug_interaction_detection.features.extractors import pair_features, top_feature_attribution
from drug_interaction_detection.modeling.pipeline import ArtifactBundle, load_artifacts
from drug_interaction_detection.retrieval.store import LocalDrugFactStore


class InferenceEngine:
    def __init__(self, settings: ResolvedSettings, artifacts: ArtifactBundle | None = None) -> None:
        self.settings = settings
        self.store = LocalDrugFactStore(settings.reference_dir)
        self.artifacts = artifacts or load_artifacts(None, settings)

    def predict(self, drug_a: str, drug_b: str, pair_id: str = "adhoc") -> PredictionResult:
        fact_a = self.store.get(drug_a)
        fact_b = self.store.get(drug_b)
        evidence = self._build_evidence(fact_a.warnings, fact_b.warnings)
        example = PreparedExample(
            pair_id=pair_id,
            drug_a=fact_a,
            drug_b=fact_b,
            interaction_label="no_known_interaction",
            severity="none",
            split="test",
            source="runtime_request",
            evidence=evidence,
        )
        features = pair_features(example, include_reference=self.artifacts.feature_mode == "reference")
        interaction_probabilities = self._apply_probability_floor(self.artifacts.interaction_model.probabilities(features))
        severity_probabilities = self._apply_probability_floor(self.artifacts.severity_model.probabilities(features))
        interaction_label = max(interaction_probabilities, key=interaction_probabilities.get)
        severity_label = max(severity_probabilities, key=severity_probabilities.get)
        interaction_confidence = interaction_probabilities[interaction_label]
        severity_confidence = severity_probabilities[severity_label]
        attributions = top_feature_attribution(self.artifacts.interaction_model.weights[interaction_label], features)
        return PredictionResult(
            pair_id=pair_id,
            drug_a=fact_a.generic_name,
            drug_b=fact_b.generic_name,
            interaction_label=interaction_label,
            interaction_confidence=round(interaction_confidence, 5),
            severity=severity_label,
            severity_confidence=round(severity_confidence, 5),
            evidence=evidence,
            feature_attribution=attributions,
            interaction_probabilities=self._rounded_distribution(interaction_probabilities),
            severity_probabilities=self._rounded_distribution(severity_probabilities),
            risk_summary=self._risk_summary(interaction_label, severity_label, evidence),
            low_confidence=interaction_confidence < 0.5,
        )

    def _build_evidence(self, warnings_a: list[str], warnings_b: list[str]) -> list[str]:
        merged = warnings_a[:2] + warnings_b[:2]
        return merged[: self.settings.settings.inference.max_evidence_items]

    def _rounded_distribution(self, probabilities: dict[str, float]) -> dict[str, float]:
        return {label: round(value, 5) for label, value in sorted(probabilities.items())}

    def _apply_probability_floor(self, probabilities: dict[str, float]) -> dict[str, float]:
        floor = self.settings.settings.inference.probability_floor
        adjusted = {label: max(value, floor) for label, value in probabilities.items()}
        total = sum(adjusted.values()) or 1.0
        return {label: value / total for label, value in adjusted.items()}

    def _risk_summary(self, interaction_label: str, severity: str, evidence: list[str]) -> str:
        evidence_clause = evidence[0] if evidence else "no warning evidence available"
        return f"{interaction_label} risk with {severity} severity; leading evidence: {evidence_clause}"
