# Drug Interaction Detection System

## Goal
Build a research-grade drug-drug interaction detection system that predicts interaction classes and severity between drug pairs.

## Expanded Scope
This project should go far beyond a simple classifier. It should include:
- reproducible data pipelines
- dataset versioning assumptions and clear data contracts
- feature engineering from drug metadata
- retrieval of supporting drug reference information from object storage
- a training pipeline for a compact model
- evaluation suite with baselines and ablations
- inference service with API
- experiment tracking
- monitoring hooks
- AWS-oriented deployment scaffolding
- extensive documentation in a research style
- diagrams, examples, and reproducibility instructions
- tests across data, training, inference, and API layers

## Desired Architecture
- monorepo or organized Python project
- src/ package layout
- configs/ for experiments
- docs/ for research and engineering documentation
- tests/
- scripts/
- infrastructure/ for AWS deployment scaffolding
- notebooks/ only if necessary, not as the main implementation

## Functional Requirements
- ingest structured drug pair data
- retrieve supporting drug facts from S3-compatible storage during inference
- train a compact local model
- provide prediction output with label, confidence, and retrieved evidence
- REST API for inference
- CLI for training and batch prediction
- evaluation report generation
- baseline models and comparison table

## Non-Functional Requirements
- strong README
- architecture doc
- experiment docs
- reproducibility
- clear assumptions and limitations
- clean commits
- production-style code quality
- comprehensive tests

## Deliverable Quality Bar
This should look like a serious ML systems/research engineering project rather than a class assignment.