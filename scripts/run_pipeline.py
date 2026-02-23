from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    config = root / "configs" / "default.toml"
    run([sys.executable, "-m", "drug_interaction_detection.cli.main", "--config", str(config), "prepare-dataset"])
    run([sys.executable, "-m", "drug_interaction_detection.cli.main", "--config", str(config), "train"])
    run([sys.executable, "-m", "drug_interaction_detection.cli.main", "--config", str(config), "evaluate"])


if __name__ == "__main__":
    main()

