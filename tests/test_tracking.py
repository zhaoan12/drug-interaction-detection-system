from __future__ import annotations

import json
from pathlib import Path

from drug_interaction_detection.tracking.experiment import ExperimentRun, record_run


def test_record_run_appends_jsonl():
    path = Path("artifacts/test_tracking_runs.jsonl")
    if path.exists():
        path.unlink()
    record_run(
        path,
        ExperimentRun(
            run_name="train-001",
            started_at="2026-02-28T15:00:00Z",
            params={"feature_mode": "reference"},
            metrics={"interaction_accuracy": 0.75},
            artifacts={"model": "artifacts/model_reference.json"},
        ),
    )
    record_run(
        path,
        ExperimentRun(
            run_name="evaluate-001",
            started_at="2026-02-28T16:00:00Z",
            params={"split": "test"},
            metrics={"interaction_accuracy": 0.8},
            artifacts={"report": "reports/evaluation_reference.md"},
        ),
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["run_name"] == "evaluate-001"
    path.unlink()
