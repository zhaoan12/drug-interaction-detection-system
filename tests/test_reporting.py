from drug_interaction_detection.data.dataset import prepare_dataset
from drug_interaction_detection.evaluation.reporting import evaluate_bundle, render_markdown_report
from drug_interaction_detection.modeling.pipeline import train_models


def test_evaluation_generates_report(settings):
    bundle = prepare_dataset(settings)
    artifacts = train_models(settings, bundle)
    result = evaluate_bundle(settings, bundle, artifacts, split="test")
    report = render_markdown_report(result)
    assert "Evaluation Report" in report

