from drug_interaction_detection.data.dataset import prepare_dataset
from drug_interaction_detection.features.extractors import pair_features
from drug_interaction_detection.modeling.pipeline import train_models


def test_train_models_produces_serialized_artifacts(settings):
    bundle = prepare_dataset(settings)
    artifacts = train_models(settings, bundle)
    assert artifacts.interaction_model.labels
    assert (settings.artifacts_dir / "model_reference.json").exists()
    assert artifacts.training_summary["train_examples"] >= 1
    assert artifacts.training_summary["feature_vocabulary_size"] >= 1


def test_pair_features_include_reference_overlap_signals(settings):
    bundle = prepare_dataset(settings)
    warfarin_case = next(example for example in bundle.examples if example.pair_id == "pair-001")
    features = pair_features(warfarin_case)
    assert features["shared_enzyme:cyp2c9"] == 1.0
    assert features["drug_a.warning_count"] >= 1.0
    assert "drug_b.contraindication:active gastrointestinal bleeding" in features
