# Model Card

## Intended Use

This system is intended for research workflows, engineering prototypes, and
pipeline demonstrations around drug-drug interaction detection.

## Model Type

- compact sparse softmax classifier for interaction label
- compact sparse softmax classifier for severity

## Inputs

- normalized drug names
- retrieved structured drug facts
- generated evidence snippets from reference warnings

## Outputs

- interaction class
- class confidence
- severity level
- severity confidence
- retrieved evidence and feature attribution

## Limitations

- not validated for clinical decision-making
- trained on a small repository-shipped dataset
- relies on structured metadata quality and coverage

