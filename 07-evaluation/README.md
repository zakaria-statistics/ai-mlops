# Phase 7: Evaluation

> Validate model performance, tune hyperparameters, select the best model.

## Table of Contents

1. [Goal](#goal)
2. [How to Run](#how-to-run)
3. [Steps](#steps)
4. [Output](#output)

---

## Goal

Answer three questions:
1. Are Phase 6 results stable or a lucky split?
2. Can hyperparameter tuning improve performance?
3. Where does the best model fail?

## How to Run

```bash
cd /root/claude/ai_mlops
source .venv/bin/activate
pip install scikit-learn matplotlib pandas numpy  # required
pip install xgboost                               # optional
```

Open `notebooks/04-evaluation.py` in VS Code and run cells with `# %%` markers.

### Libraries

| Library | Purpose | Required |
|---------|---------|----------|
| pandas | Data manipulation | Yes |
| numpy | Numerical operations | Yes |
| scikit-learn | Cross-validation, RandomizedSearchCV, metrics | Yes |
| matplotlib | Error analysis plots | Yes |
| xgboost | XGBoost tuning (falls back to RF if missing) | No |

## Steps

| # | Step | Detail |
|---|------|--------|
| 1 | Cross-validation | 5-fold CV on top 3 models from Phase 6 |
| 2 | Hyperparameter tuning | RandomizedSearchCV (30 combinations) on HGB + XGBoost |
| 3 | Final comparison | Tuned vs default models on held-out test set |
| 4 | Error analysis | By price range, worst predictions, residuals, feature importance |
| 5 | Final verdict | Select best model with interpretable metrics |

## Output

- CV scores with confidence intervals
- Best hyperparameters for HGB and XGBoost
- Tuned vs default comparison table
- Error analysis plots (by price range, residuals, feature importance)
- Final model selection with business-readable metrics (MAE in $, MAPE in %)

## Next Step

-> Proceed to [08-packaging](../08-packaging/) to serialize the selected model for deployment.
