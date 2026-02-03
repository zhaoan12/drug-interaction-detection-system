from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


SeverityLabel = Literal["none", "low", "moderate", "high", "critical"]
InteractionLabel = Literal[
    "no_known_interaction",
    "bleeding_risk",
    "serotonin_syndrome",
    "qt_prolongation",
    "cyp_inhibition",
]


class DrugFact(BaseModel):
    drug_id: str
    generic_name: str
    drug_class: str
    normalized_names: list[str] = Field(default_factory=list)
    mechanisms: list[str] = Field(default_factory=list)
    indications: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    transporters: list[str] = Field(default_factory=list)
    enzymes: list[str] = Field(default_factory=list)


class DrugPairRecord(BaseModel):
    pair_id: str
    drug_a: str
    drug_b: str
    interaction_label: InteractionLabel
    severity: SeverityLabel
    evidence: list[str] = Field(default_factory=list)
    source: str

    @field_validator("drug_a", "drug_b")
    @classmethod
    def _normalize_drug_name(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")


class PreparedExample(BaseModel):
    pair_id: str
    drug_a: DrugFact
    drug_b: DrugFact
    interaction_label: InteractionLabel
    severity: SeverityLabel
    split: Literal["train", "validation", "test"]
    source: str
    evidence: list[str]


class PredictionResult(BaseModel):
    pair_id: str
    drug_a: str
    drug_b: str
    interaction_label: str
    interaction_confidence: float
    severity: str
    severity_confidence: float
    evidence: list[str] = Field(default_factory=list)
    feature_attribution: dict[str, float] = Field(default_factory=dict)
    interaction_probabilities: dict[str, float] = Field(default_factory=dict)
    severity_probabilities: dict[str, float] = Field(default_factory=dict)
    risk_summary: str = ""


def normalize_drug_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def canonical_pair_key(drug_a: str, drug_b: str) -> tuple[str, str]:
    return tuple(sorted((normalize_drug_name(drug_a), normalize_drug_name(drug_b))))
