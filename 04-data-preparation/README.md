# Phase 4: Data Preparation

> Clean, transform, and split the dataset based on EDA findings from Phase 3.

## Table of Contents

1. [Goal](#goal)
2. [How to Run](#how-to-run)
3. [Steps Applied](#steps-applied)
4. [Output](#output)

---

## Goal

Transform raw data into model-ready train/test sets by applying the actions identified in [Phase 3 EDA](../03-data-exploration/README.md).

## How to Run

```bash
cd /root/claude/ai_mlops
source .venv/bin/activate
pip install scikit-learn pandas numpy
```

Open `notebooks/02-data-preparation.py` in VS Code and run cells with `# %%` markers.

Output saves to `data/processed/`.

## Steps Applied

| # | Step | Detail | EDA Finding |
|---|------|--------|-------------|
| 1 | Drop columns | id, date, sqft_above, zipcode | Not predictive or redundant; lat/long capture location |
| 2 | Cap outliers | Bedrooms capped at 10 | 33-bedroom data entry error |
| 3 | Engineer house_age | 2015 - yr_built | yr_built weak predictor (r=0.05) |
| 4 | Engineer has_renovation | Binary from yr_renovated | Mostly zeros, simplify |
| 5 | Log transform target | log1p(price) | Skew 4.02 -> near-normal |
| 6 | Train/test split | 80/20, random_state=42 | Reproducible split |

## Output

```
data/processed/
  X_train.csv    # 17,290 rows x 16 features
  X_test.csv     #  4,323 rows x 16 features
  y_train.csv    # log-transformed prices
  y_test.csv     # log-transformed prices
```

## Next Step

-> Feature engineering is included in this phase (see step 3-4). Proceed to [06-modeling](../06-modeling/) for baseline training.
