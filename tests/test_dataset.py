from drug_interaction_detection.data.dataset import prepare_dataset, summarize_dataset


def test_prepare_dataset_builds_examples(settings):
    bundle = prepare_dataset(settings)
    assert len(bundle.examples) >= 18
    summary = summarize_dataset(bundle)
    assert summary["num_examples"] == len(bundle.examples)
    assert summary["interaction_labels"]["bleeding_risk"] >= 1

