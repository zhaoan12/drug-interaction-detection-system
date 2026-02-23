# ADR 0001: Local Object Store Retrieval Contract

## Status

Accepted

## Context

The project brief requires retrieval of supporting drug reference information from
S3-compatible object storage during inference. The execution environment for this
repository is intentionally local and deterministic.

## Decision

Implement a retrieval contract backed by local JSON objects stored under
`data/reference/`, with one object per normalized drug name.

## Consequences

- local runs remain deterministic and testable without cloud credentials
- the retrieval abstraction mirrors object-key lookup semantics
- production deployment can replace the implementation with a real S3 client while
  preserving the interface used by data preparation and inference

