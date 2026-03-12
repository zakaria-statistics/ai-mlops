# Phase 6: Modeling

> Train progressively complex models, understand WHY each level improves on the last.

## Table of Contents

1. [Goal](#goal)
2. [How to Run](#how-to-run)
3. [Model Progression](#model-progression)
4. [Output](#output)

---

## Goal

Train 5 model types on the prepared data, comparing performance at each step to build intuition for when and why to use each algorithm.

## How to Run

```bash
cd /root/claude/ai_mlops
source .venv/bin/activate
pip install scikit-learn matplotlib pandas numpy  # required
pip install xgboost                               # optional, for Level 5 XGBoost
```

### Libraries

| Library | Purpose | Required |
|---------|---------|----------|
| pandas | Data loading and manipulation | Yes |
| numpy | Numerical operations, log transforms | Yes |
| scikit-learn | All models (Linear, Ridge, Lasso, Tree, RF, HGB) | Yes |
| matplotlib | Comparison charts, residual plots | Yes |
| xgboost | Optional XGBoost model (Level 5) | No |

Open `notebooks/03-modeling.py` in VS Code and run cells with `# %%` markers.

## Model Progression

```
Level 1: Linear Regression
   ↓ "Why not enough?" → Assumes linear, additive effects
Level 2: Ridge / Lasso
   ↓ "Why not enough?" → Still linear, regularization helps little with 16 features
Level 3: Decision Tree
   ↓ "Why not enough?" → High variance, overfits or underfits
Level 4: Random Forest (bagging)
   ↓ "Why not enough?" → Good, but trees are independent
Level 5: Gradient Boosting (boosting)
   ↓ "Final comparison"
Level 6: Side-by-side comparison table + residual analysis
```

## Output

- Comparison table: RMSE, MAE ($), R², MAPE (%) for all models
- Feature importance plots (Random Forest)
- Residual analysis plots (best model)
- No model files saved yet — that's [Phase 8 (Packaging)](../08-packaging/)

## Next Step

-> Proceed to [07-evaluation](../07-evaluation/) for cross-validation, hyperparameter tuning, and final model selection.
