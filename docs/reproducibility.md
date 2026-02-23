# Reproducibility

## Determinism

- dataset splits are deterministic and derived from `pair_id`
- the sample dataset is versioned directly in the repository
- model training uses fixed hyperparameters from `configs/default.toml`
- reports and artifacts are written to stable repository-relative locations

## Running the Full Pipeline

```bash
python -m pip install -e .[dev]
python scripts/run_pipeline.py
```

## Artifact Contract

- `artifacts/prepared_dataset.json`: hydrated and split-aware examples
- `artifacts/model_reference.json`: full model using reference-derived features
- `artifacts/model_structure_only.json`: ablation artifact without reference features
- `reports/evaluation_reference.json`: structured evaluation payload
- `reports/evaluation_reference.md`: markdown report for research review
- `artifacts/tracking/runs.jsonl`: append-only experiment log

## Limitations

- the repository ships with a compact illustrative dataset instead of a clinical corpus
- local retrieval stands in for external object storage
- the pure-Python learner is optimized for clarity and portability, not large-scale throughput

