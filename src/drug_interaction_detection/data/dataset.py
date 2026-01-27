from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from collections import Counter

from drug_interaction_detection.config.settings import ResolvedSettings
from drug_interaction_detection.domain import DrugPairRecord, PreparedExample, canonical_pair_key
from drug_interaction_detection.retrieval.store import LocalDrugFactStore
from drug_interaction_detection.utils.io import read_jsonl, write_json
from drug_interaction_detection.utils.splitting import deterministic_split


@dataclass
class DatasetBundle:
    examples: list[PreparedExample]

    def by_split(self, split: str) -> list[PreparedExample]:
        return [example for example in self.examples if example.split == split]

    def label_space(self) -> list[str]:
        return sorted({example.interaction_label for example in self.examples})

    def severity_space(self) -> list[str]:
        return sorted({example.severity for example in self.examples})


def prepare_dataset(settings: ResolvedSettings) -> DatasetBundle:
    store = LocalDrugFactStore(settings.reference_dir)
    raw_records = [DrugPairRecord.model_validate(item) for item in read_jsonl(settings.raw_pairs)]
    validate_raw_records(raw_records)
    examples: list[PreparedExample] = []
    for record in raw_records:
        split = deterministic_split(
            record.pair_id,
            train_fraction=settings.settings.dataset.train_fraction,
            validation_fraction=settings.settings.dataset.validation_fraction,
        )
        example = PreparedExample(
            pair_id=record.pair_id,
            drug_a=store.get(record.drug_a),
            drug_b=store.get(record.drug_b),
            interaction_label=record.interaction_label,
            severity=record.severity,
            split=split,
            source=record.source,
            evidence=record.evidence,
        )
        examples.append(example)
    bundle = DatasetBundle(examples=_repair_training_coverage(examples, settings.settings.dataset.minimum_examples_per_label))
    serialized = [example.model_dump(mode="json") for example in bundle.examples]
    write_json(settings.artifacts_dir / "prepared_dataset.json", serialized)
    return bundle


def validate_raw_records(records: list[DrugPairRecord]) -> None:
    seen_pair_ids: set[str] = set()
    seen_pairs: dict[tuple[str, str], str] = {}
    for record in records:
        if record.pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate pair_id detected: {record.pair_id}")
        seen_pair_ids.add(record.pair_id)
        pair_key = canonical_pair_key(record.drug_a, record.drug_b)
        if pair_key[0] == pair_key[1]:
            raise ValueError(f"Self-interaction record is not allowed: {record.pair_id}")
        existing = seen_pairs.get(pair_key)
        if existing is not None:
            raise ValueError(
                "Duplicate drug pair detected across records: "
                f"{record.pair_id} conflicts with {existing}"
            )
        seen_pairs[pair_key] = record.pair_id


def _repair_training_coverage(examples: list[PreparedExample], minimum_examples_per_label: int) -> list[PreparedExample]:
    train_counts = Counter(item.interaction_label for item in examples if item.split == "train")
    by_label: dict[str, list[PreparedExample]] = {}
    for example in sorted(examples, key=lambda item: item.pair_id):
        by_label.setdefault(example.interaction_label, []).append(example)
    repaired: list[PreparedExample] = []
    reassigned_ids: set[str] = set()
    for label, items in by_label.items():
        needed = max(0, minimum_examples_per_label - train_counts.get(label, 0))
        if needed == 0:
            continue
        for candidate in items:
            if candidate.split != "train" and candidate.pair_id not in reassigned_ids:
                repaired.append(candidate.model_copy(update={"split": "train"}))
                reassigned_ids.add(candidate.pair_id)
                needed -= 1
                if needed == 0:
                    break
    for example in examples:
        if example.pair_id in reassigned_ids:
            continue
        repaired.append(example)
    repaired.sort(key=lambda item: item.pair_id)
    return repaired


def summarize_dataset(bundle: DatasetBundle) -> dict[str, object]:
    summary: dict[str, object] = {
        "num_examples": len(bundle.examples),
        "splits": {},
        "interaction_labels": {},
        "severity_labels": {},
        "sources": {},
    }
    for split in ("train", "validation", "test"):
        summary["splits"][split] = len(bundle.by_split(split))  # type: ignore[index]
    for label in bundle.label_space():
        summary["interaction_labels"][label] = sum(1 for item in bundle.examples if item.interaction_label == label)  # type: ignore[index]
    for label in bundle.severity_space():
        summary["severity_labels"][label] = sum(1 for item in bundle.examples if item.severity == label)  # type: ignore[index]
    source_counts = Counter(item.source for item in bundle.examples)
    summary["sources"] = dict(sorted(source_counts.items()))  # type: ignore[index]
    return summary


def iter_pairs(bundle: DatasetBundle, split: str | None = None) -> Iterable[PreparedExample]:
    if split is None:
        yield from bundle.examples
        return
    yield from bundle.by_split(split)
