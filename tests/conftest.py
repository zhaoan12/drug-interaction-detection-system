from __future__ import annotations

from pathlib import Path

import pytest

from drug_interaction_detection.config.settings import load_settings


@pytest.fixture()
def settings():
    return load_settings(Path("configs/default.toml"))

