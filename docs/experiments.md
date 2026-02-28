# Experiments

## Run Tracking

Training, evaluation, and batch inference commands append JSONL records to
`artifacts/tracking/runs.jsonl`. Each record captures the run name, timestamp,
parameters, headline metrics, and emitted artifact paths so local experiments
remain auditable without requiring an external tracking service.

## Baselines

- interaction label majority classifier
- severity majority classifier

## Main Model

The primary model is a multiclass softmax classifier over sparse interpretable
features. Two independent heads are trained:

- interaction class head
- severity head

## Ablation

The default evaluation command trains:

- full model with reference-derived features
- structure-only model without retrieved warnings/mechanisms/shared biochemical features

This allows the report to quantify the value of retrieval-enriched metadata.
