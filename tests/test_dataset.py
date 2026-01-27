from drug_interaction_detection.data.dataset import prepare_dataset, summarize_dataset, validate_raw_records
from drug_interaction_detection.domain import DrugPairRecord
from drug_interaction_detection.retrieval.store import LocalDrugFactStore
import pytest


def test_prepare_dataset_builds_examples(settings):
    bundle = prepare_dataset(settings)
    assert len(bundle.examples) >= 18
    summary = summarize_dataset(bundle)
    assert summary["num_examples"] == len(bundle.examples)
    assert summary["interaction_labels"]["bleeding_risk"] >= 1
    assert summary["sources"]["synthetic_literature"] == len(bundle.examples)


def test_validate_raw_records_rejects_duplicate_canonical_pairs():
    records = [
        DrugPairRecord(
            pair_id="pair-001",
            drug_a="warfarin",
            drug_b="ibuprofen",
            interaction_label="bleeding_risk",
            severity="high",
            source="test",
        ),
        DrugPairRecord(
            pair_id="pair-002",
            drug_a="ibuprofen",
            drug_b="warfarin",
            interaction_label="bleeding_risk",
            severity="high",
            source="test",
        ),
    ]
    with pytest.raises(ValueError, match="Duplicate drug pair"):
        validate_raw_records(records)


def test_local_drug_store_supports_alias_lookup(settings):
    store = LocalDrugFactStore(settings.reference_dir)
    fact = store.get("Warfarin")
    assert fact.generic_name == "warfarin"
