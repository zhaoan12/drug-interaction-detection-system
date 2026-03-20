from drug_interaction_detection.data.dataset import prepare_dataset
from drug_interaction_detection.modeling.pipeline import train_models


def test_train_models_produces_serialized_artifacts(settings):
    bundle = prepare_dataset(settings)
    artifacts = train_models(settings, bundle)
    assert artifacts.interaction_model.labels
    assert (settings.artifacts_dir / "model_reference.json").exists()

