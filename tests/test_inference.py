from drug_interaction_detection.data.dataset import prepare_dataset
from drug_interaction_detection.inference.service import InferenceEngine
from drug_interaction_detection.modeling.pipeline import train_models


def test_inference_returns_prediction_payload(settings):
    bundle = prepare_dataset(settings)
    train_models(settings, bundle)
    engine = InferenceEngine(settings)
    result = engine.predict("warfarin", "ibuprofen")
    assert result.interaction_label
    assert result.severity
    assert result.evidence
    assert result.interaction_probabilities[result.interaction_label] == result.interaction_confidence
    assert result.severity_probabilities[result.severity] == result.severity_confidence
    assert "leading evidence" in result.risk_summary
    assert all(value >= settings.settings.inference.probability_floor for value in result.interaction_probabilities.values())
    assert isinstance(result.low_confidence, bool)
