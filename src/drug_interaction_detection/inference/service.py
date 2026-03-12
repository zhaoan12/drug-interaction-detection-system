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
        interaction_label, interaction_confidence = self.artifacts.interaction_model.predict(features)
        severity_label, severity_confidence = self.artifacts.severity_model.predict(features)
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
        )

    def _build_evidence(self, warnings_a: list[str], warnings_b: list[str]) -> list[str]:
        merged = warnings_a[:2] + warnings_b[:2]
        return merged[: self.settings.settings.inference.max_evidence_items]

