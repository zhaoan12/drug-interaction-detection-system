from __future__ import annotations

from dataclasses import asdict, dataclass

from drug_interaction_detection.config.settings import ResolvedSettings
from drug_interaction_detection.data.dataset import DatasetBundle
from drug_interaction_detection.evaluation.metrics import accuracy, confusion, macro_f1, per_label_report
from drug_interaction_detection.features.extractors import pair_features
from drug_interaction_detection.modeling.pipeline import ArtifactBundle
from drug_interaction_detection.utils.io import write_json


@dataclass
class EvaluationResult:
    split: str
    interaction_accuracy: float
    interaction_macro_f1: float
    severity_accuracy: float
    severity_macro_f1: float
    baseline_interaction_accuracy: float
    baseline_severity_accuracy: float
    feature_mode: str
    interaction_confusion: dict[str, dict[str, int]]
    severity_confusion: dict[str, dict[str, int]]
    interaction_per_label: dict[str, dict[str, float]]
    severity_per_label: dict[str, dict[str, float]]
    interaction_errors: list[dict[str, str]]
    severity_errors: list[dict[str, str]]


def evaluate_bundle(
    settings: ResolvedSettings,
    dataset: DatasetBundle,
    artifacts: ArtifactBundle,
    *,
    split: str = "test",
) -> EvaluationResult:
    examples = dataset.by_split(split)
    interaction_gold = [item.interaction_label for item in examples]
    severity_gold = [item.severity for item in examples]
    interaction_predictions: list[str] = []
    severity_predictions: list[str] = []
    interaction_baseline: list[str] = []
    severity_baseline: list[str] = []
    interaction_errors: list[dict[str, str]] = []
    severity_errors: list[dict[str, str]] = []
    for example in examples:
        features = pair_features(example, include_reference=artifacts.feature_mode == "reference")
        interaction_label, _ = artifacts.interaction_model.predict(features)
        severity_label, _ = artifacts.severity_model.predict(features)
        interaction_predictions.append(interaction_label)
        severity_predictions.append(severity_label)
        if interaction_label != example.interaction_label:
            interaction_errors.append(
                {
                    "pair_id": example.pair_id,
                    "drug_a": example.drug_a.generic_name,
                    "drug_b": example.drug_b.generic_name,
                    "gold": example.interaction_label,
                    "predicted": interaction_label,
                }
            )
        if severity_label != example.severity:
            severity_errors.append(
                {
                    "pair_id": example.pair_id,
                    "drug_a": example.drug_a.generic_name,
                    "drug_b": example.drug_b.generic_name,
                    "gold": example.severity,
                    "predicted": severity_label,
                }
            )
        interaction_baseline.append(artifacts.interaction_baseline.predict())
        severity_baseline.append(artifacts.severity_baseline.predict())
    result = EvaluationResult(
        split=split,
        interaction_accuracy=accuracy(interaction_gold, interaction_predictions),
        interaction_macro_f1=macro_f1(interaction_gold, interaction_predictions),
        severity_accuracy=accuracy(severity_gold, severity_predictions),
        severity_macro_f1=macro_f1(severity_gold, severity_predictions),
        baseline_interaction_accuracy=accuracy(interaction_gold, interaction_baseline),
        baseline_severity_accuracy=accuracy(severity_gold, severity_baseline),
        feature_mode=artifacts.feature_mode,
        interaction_confusion=confusion(interaction_gold, interaction_predictions),
        severity_confusion=confusion(severity_gold, severity_predictions),
        interaction_per_label=per_label_report(interaction_gold, interaction_predictions),
        severity_per_label=per_label_report(severity_gold, severity_predictions),
        interaction_errors=interaction_errors[:5],
        severity_errors=severity_errors[:5],
    )
    write_json(settings.reports_dir / f"evaluation_{artifacts.feature_mode}_{split}.json", asdict(result))
    return result


def render_markdown_report(result: EvaluationResult, ablation: EvaluationResult | None = None) -> str:
    lines = [
        f"# Evaluation Report ({result.feature_mode})",
        "",
        f"- split: `{result.split}`",
        f"- interaction accuracy: `{result.interaction_accuracy:.3f}`",
        f"- interaction macro F1: `{result.interaction_macro_f1:.3f}`",
        f"- severity accuracy: `{result.severity_accuracy:.3f}`",
        f"- severity macro F1: `{result.severity_macro_f1:.3f}`",
        f"- interaction baseline accuracy: `{result.baseline_interaction_accuracy:.3f}`",
        f"- severity baseline accuracy: `{result.baseline_severity_accuracy:.3f}`",
        "",
        "## Confusion Overview",
        "",
        f"- interaction: `{result.interaction_confusion}`",
        f"- severity: `{result.severity_confusion}`",
        "",
        "## Per-Label Quality",
        "",
        f"- interaction: `{result.interaction_per_label}`",
        f"- severity: `{result.severity_per_label}`",
        "",
        "## Example Errors",
        "",
        f"- interaction: `{result.interaction_errors}`",
        f"- severity: `{result.severity_errors}`",
    ]
    if ablation is not None:
        lines.extend(
            [
                "",
                "## Ablation Comparison",
                "",
                f"- full interaction accuracy: `{result.interaction_accuracy:.3f}` vs structure-only `{ablation.interaction_accuracy:.3f}`",
                f"- full severity accuracy: `{result.severity_accuracy:.3f}` vs structure-only `{ablation.severity_accuracy:.3f}`",
            ]
        )
    return "\n".join(lines)
