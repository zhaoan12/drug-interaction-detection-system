from __future__ import annotations

from pathlib import Path

from drug_interaction_detection.domain import DrugFact, normalize_drug_name
from drug_interaction_detection.utils.io import read_json


class LocalDrugFactStore:
    """A filesystem-backed stand-in for S3-compatible object storage."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._index = self._build_index()

    def _build_index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for path in sorted(self.root.glob("*.json")):
            payload = read_json(path)
            fact = DrugFact.model_validate(payload)
            candidates = {path.stem, fact.generic_name, *fact.normalized_names}
            for candidate in candidates:
                index[normalize_drug_name(candidate)] = path
        return index

    def get(self, drug_name: str) -> DrugFact:
        normalized = normalize_drug_name(drug_name)
        path = self._index.get(normalized)
        if path is None:
            raise FileNotFoundError(f"Drug fact not found for {normalized}: {path}")
        return DrugFact.model_validate(read_json(path))
