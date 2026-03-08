from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
import json


@dataclass
class ExperimentRun:
    run_name: str
    started_at: str
    params: dict[str, object]
    metrics: dict[str, float]
    artifacts: dict[str, str]


def record_run(path: Path, run: ExperimentRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(run), sort_keys=True))
        handle.write("\n")


def make_run_name(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}"


def current_timestamp() -> str:
    return datetime.now(UTC).isoformat()
