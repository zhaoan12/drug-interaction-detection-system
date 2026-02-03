# API

## Endpoints

### `GET /health`

Returns a simple liveness payload:

```json
{"status": "ok", "service": "ddi-research-system"}
```

### `GET /metrics`

Returns aggregate in-process metrics:

- total requests
- total errors
- average latency in milliseconds
- prediction counts by interaction label

### `POST /predict`

Request body:

```json
{
  "pair_id": "example-001",
  "drug_a": "warfarin",
  "drug_b": "ibuprofen"
}
```

Response body includes:

- predicted interaction label
- confidence
- predicted severity
- retrieved evidence
- top feature attributions
- per-label probability distributions
- a compact natural-language risk summary

### `POST /batch-predict`

Request body:

```json
{
  "requests": [
    {"pair_id": "batch-001", "drug_a": "warfarin", "drug_b": "fluconazole"},
    {"pair_id": "batch-002", "drug_a": "metformin", "drug_b": "lisinopril"}
  ]
}
```

Response body includes a `results` array with the same payload shape returned by
`POST /predict`.

## Serving

```bash
python -m drug_interaction_detection.cli.main --config configs/default.toml serve --port 8080
```
