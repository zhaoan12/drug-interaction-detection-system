# Drug Interaction Detection System

This repository implements a research-grade, production-minded drug-drug interaction
(DDI) detection system. It includes deterministic data preparation, feature
engineering from structured drug metadata, retrieval-backed inference, compact
local models, experiment tracking, evaluation reporting, a REST-style HTTP API,
AWS deployment scaffolding, and comprehensive tests.

## Highlights

- `src/` package layout with typed Python 3.11+ modules
- deterministic dataset building and split assignment
- object-storage style retrieval of drug facts during training and inference
- compact pure-Python softmax models for interaction class and severity
- baseline comparison, ablations, markdown report generation, and local run tracking
- CLI for dataset preparation, training, evaluation, batch prediction, and serving
- lightweight HTTP inference service with monitoring hooks
- reproducibility-focused documentation, architecture notes, and infrastructure scaffolding

## Repository Layout

- `configs/`: experiment and deployment configuration
- `data/`: sample raw pairs and reference drug facts
- `docs/`: architecture, reproducibility, and ADRs
- `examples/`: request and batch examples
- `infrastructure/`: Docker and Terraform scaffolding for AWS deployment
- `scripts/`: deterministic helper entry points
- `src/drug_interaction_detection/`: implementation modules
- `tests/`: unit and integration-style pytest coverage

## Quick Start

```bash
python -m pip install -e .[dev]
python scripts/run_pipeline.py
python -m drug_interaction_detection.cli.main predict --config configs/default.toml --drug-a warfarin --drug-b ibuprofen
python -m drug_interaction_detection.cli.main serve --config configs/default.toml --port 8080
```

## Default Workflow

1. Prepare the dataset from `data/raw/drug_pairs.jsonl`
2. Train the full model and baselines
3. Evaluate the trained artifacts and emit markdown reports
4. Serve the model through the built-in HTTP API or run batch prediction jobs

The sample data is intentionally small so the full pipeline remains fast and
deterministic in local environments. The code structure is designed to scale to a
real dataset and S3-compatible reference store.

## Documentation

- [`docs/architecture.md`](/C:/Users/ghost/Downloads/drug-interaction-detection-system/docs/architecture.md)
- [`docs/api.md`](/C:/Users/ghost/Downloads/drug-interaction-detection-system/docs/api.md)
- [`docs/reproducibility.md`](/C:/Users/ghost/Downloads/drug-interaction-detection-system/docs/reproducibility.md)
- [`docs/experiments.md`](/C:/Users/ghost/Downloads/drug-interaction-detection-system/docs/experiments.md)
- [`docs/model_card.md`](/C:/Users/ghost/Downloads/drug-interaction-detection-system/docs/model_card.md)
- [`docs/adr/0001-local-object-store-retrieval.md`](/C:/Users/ghost/Downloads/drug-interaction-detection-system/docs/adr/0001-local-object-store-retrieval.md)
