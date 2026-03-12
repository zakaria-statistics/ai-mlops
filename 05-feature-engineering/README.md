# Phase 5: Feature Engineering

> Combined with Phase 4 (Data Preparation).

## What Was Done

Feature engineering is handled in `notebooks/02-data-preparation.py` (section 4):

| Feature | Source | Rationale |
|---------|--------|-----------|
| house_age | 2015 - yr_built | yr_built weak predictor (r=0.05), age is more intuitive |
| has_renovation | Binary from yr_renovated | Mostly zeros, binary flag is cleaner |

## Why Combined

The feature engineering for this dataset is straightforward — two derived features. Splitting it into a separate notebook would add overhead without value.

## Future Work

If more complex feature engineering is needed later (scaling, polynomial features, target encoding for zipcode, etc.), add a `notebooks/02b-feature-engineering.py` notebook.

## Next Step

-> Proceed to [06-modeling](../06-modeling/) for baseline model training.
