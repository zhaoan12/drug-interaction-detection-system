from __future__ import annotations

from pathlib import Path

from drug_interaction_detection.domain import DrugFact
from drug_interaction_detection.utils.io import read_json


class LocalDrugFactStore:
    """A filesystem-backed stand-in for S3-compatible object storage."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, drug_name: str) -> DrugFact:
        normalized = drug_name.strip().lower().replace(" ", "_")
        path = self.root / f"{normalized}.json"
        if not path.exists():
            raise FileNotFoundError(f"Drug fact not found for {normalized}: {path}")
        return DrugFact.model_validate(read_json(path))

