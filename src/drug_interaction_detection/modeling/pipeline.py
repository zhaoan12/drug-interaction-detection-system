from __future__ import annotations

from dataclasses import dataclass

from drug_interaction_detection.config.settings import ResolvedSettings
from drug_interaction_detection.data.dataset import DatasetBundle
from drug_interaction_detection.features.extractors import FeatureVector, pair_features
from drug_interaction_detection.modeling.baselines import MajorityClassifier
from drug_interaction_detection.modeling.softmax import SoftmaxModel
from drug_interaction_detection.utils.io import write_json


@dataclass
class ArtifactBundle:
    interaction_model: SoftmaxModel
    severity_model: SoftmaxModel
    interaction_baseline: MajorityClassifier
    severity_baseline: MajorityClassifier
    feature_mode: str


def _vectorize(bundle: DatasetBundle, include_reference: bool) -> list[tuple[FeatureVector, str, str, str]]:
    rows: list[tuple[FeatureVector, str, str, str]] = []
    for example in bundle.examples:
        rows.append(
            (
                pair_features(example, include_reference=include_reference),
                example.interaction_label,
                example.severity,
                example.split,
            )
        )
    return rows


def train_models(settings: ResolvedSettings, bundle: DatasetBundle, include_reference: bool | None = None) -> ArtifactBundle:
    if include_reference is None:
        include_reference = settings.settings.training.full_feature_set
    rows = _vectorize(bundle, include_reference=include_reference)
    train_rows = [(features, interaction, severity) for features, interaction, severity, split in rows if split == "train"]
    interaction_labels = bundle.label_space()
    severity_labels = bundle.severity_space()
    interaction_model = SoftmaxModel.initialize(interaction_labels)
    severity_model = SoftmaxModel.initialize(severity_labels)
    interaction_model.fit(
        [(features, interaction) for features, interaction, _ in train_rows],
        epochs=settings.settings.training.epochs,
        learning_rate=settings.settings.training.learning_rate,
        l2_penalty=settings.settings.training.l2_penalty,
    )
    severity_model.fit(
        [(features, severity) for features, _, severity in train_rows],
        epochs=settings.settings.training.epochs,
        learning_rate=settings.settings.training.learning_rate,
        l2_penalty=settings.settings.training.l2_penalty,
    )
    interaction_baseline = MajorityClassifier()
    interaction_baseline.fit(bundle.by_split("train"), "interaction_label")
    severity_baseline = MajorityClassifier()
    severity_baseline.fit(bundle.by_split("train"), "severity")
    artifacts = ArtifactBundle(
        interaction_model=interaction_model,
        severity_model=severity_model,
        interaction_baseline=interaction_baseline,
        severity_baseline=severity_baseline,
        feature_mode="reference" if include_reference else "structure_only",
    )
    save_artifacts(settings, artifacts)
    return artifacts


def save_artifacts(settings: ResolvedSettings, bundle: ArtifactBundle) -> None:
    payload = {
        "feature_mode": bundle.feature_mode,
        "interaction_model": {
            "labels": bundle.interaction_model.labels,
            "weights": bundle.interaction_model.weights,
            "bias": bundle.interaction_model.bias,
        },
        "severity_model": {
            "labels": bundle.severity_model.labels,
            "weights": bundle.severity_model.weights,
            "bias": bundle.severity_model.bias,
        },
        "baselines": {
            "interaction_majority": bundle.interaction_baseline.majority_label,
            "severity_majority": bundle.severity_baseline.majority_label,
        },
    }
    write_json(settings.artifacts_dir / f"model_{bundle.feature_mode}.json", payload)


def load_artifacts(path: str | None, settings: ResolvedSettings) -> ArtifactBundle:
    artifact_name = path or f"model_{'reference' if settings.settings.training.full_feature_set else 'structure_only'}.json"
    from drug_interaction_detection.utils.io import read_json

    payload = read_json(settings.artifacts_dir / artifact_name)
    interaction_model = SoftmaxModel(
        labels=payload["interaction_model"]["labels"],
        weights=payload["interaction_model"]["weights"],
        bias=payload["interaction_model"]["bias"],
    )
    severity_model = SoftmaxModel(
        labels=payload["severity_model"]["labels"],
        weights=payload["severity_model"]["weights"],
        bias=payload["severity_model"]["bias"],
    )
    interaction_baseline = MajorityClassifier()
    interaction_baseline.majority_label = payload["baselines"]["interaction_majority"]
    severity_baseline = MajorityClassifier()
    severity_baseline.majority_label = payload["baselines"]["severity_majority"]
    return ArtifactBundle(
        interaction_model=interaction_model,
        severity_model=severity_model,
        interaction_baseline=interaction_baseline,
        severity_baseline=severity_baseline,
        feature_mode=payload["feature_mode"],
    )
