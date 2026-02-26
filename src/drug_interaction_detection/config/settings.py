from __future__ import annotations

from pathlib import Path
from typing import Any
import tomllib

from pydantic import BaseModel


class ProjectConfig(BaseModel):
    name: str
    random_seed: int


class PathConfig(BaseModel):
    raw_pairs: str
    reference_dir: str
    artifacts_dir: str
    reports_dir: str
    tracking_dir: str


class DatasetConfig(BaseModel):
    train_fraction: float
    validation_fraction: float
    minimum_examples_per_label: int


class TrainingConfig(BaseModel):
    epochs: int
    learning_rate: float
    l2_penalty: float
    full_feature_set: bool = True


class InferenceConfig(BaseModel):
    max_evidence_items: int
    probability_floor: float


class ServiceConfig(BaseModel):
    host: str
    port: int


class Settings(BaseModel):
    project: ProjectConfig
    paths: PathConfig
    dataset: DatasetConfig
    training: TrainingConfig
    inference: InferenceConfig
    service: ServiceConfig

    def resolve(self, root: Path) -> "ResolvedSettings":
        return ResolvedSettings(
            root=root,
            settings=self,
            raw_pairs=root / self.paths.raw_pairs,
            reference_dir=root / self.paths.reference_dir,
            artifacts_dir=root / self.paths.artifacts_dir,
            reports_dir=root / self.paths.reports_dir,
            tracking_dir=root / self.paths.tracking_dir,
        )


class ResolvedSettings(BaseModel):
    root: Path
    settings: Settings
    raw_pairs: Path
    reference_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    tracking_dir: Path

    model_config = {"arbitrary_types_allowed": True}

    def ensure_directories(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)


def load_settings(path: str | Path) -> ResolvedSettings:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        payload: dict[str, Any] = tomllib.load(handle)
    settings = Settings.model_validate(payload)
    root = config_path.resolve().parent.parent
    resolved = settings.resolve(root)
    resolved.ensure_directories()
    return resolved

