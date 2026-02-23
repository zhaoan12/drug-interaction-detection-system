# Architecture

## System Goals

The system is designed as a compact but production-minded DDI platform with
clear separation between dataset construction, retrieval, feature engineering,
modeling, evaluation, inference, and deployment scaffolding.

## Major Components

- `config.settings`: loads TOML configuration and resolves repository-relative paths
- `data.dataset`: validates raw pairs, hydrates drug facts from object storage, and produces deterministic splits
- `retrieval.store`: abstracts S3-compatible retrieval behind a local file-backed store for reproducibility
- `features.extractors`: converts drug facts and evidence into sparse interpretable features
- `modeling.softmax`: trains a compact pure-Python multiclass softmax model
- `evaluation.reporting`: compares learned models with majority baselines and produces machine-readable and markdown reports
- `inference.service`: performs retrieval-backed inference and exposes feature attributions
- `api.server`: provides `/health`, `/metrics`, and `/predict` endpoints
- `tracking.experiment`: appends structured run metadata for reproducible experiment history

## Data Flow

1. Raw drug pair records are loaded from JSONL.
2. Drug-level facts are retrieved from object-storage style JSON objects.
3. Deterministic split assignment is computed from pair IDs.
4. Sparse features are generated from drug classes, mechanisms, warnings, shared enzymes, and evidence text.
5. Separate compact models predict interaction label and severity.
6. Evaluation, tracking, and API layers consume the same serialized artifacts.

## Design Notes

- The retrieval layer deliberately resembles an S3 object key lookup. Replacing the
  local implementation with a boto3-backed client would not change the pipeline contract.
- The compact model avoids heavy scientific dependencies so the repository remains
  testable in restricted environments.
- The feature schema is transparent by construction, which makes error analysis and
  ablation reporting straightforward.

