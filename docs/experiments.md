# Experiments

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

