# 11 — Evaluation Deep Dive: Cross-Validation, Tuning, and Model Selection
> How to properly judge models — bias-variance diagnosis, K-fold, hyperparameter search, and when to stop

## Table of Contents
1. [Bias-Variance Revisited](#1-bias-variance)
2. [Learning Curves](#2-learning-curves)
3. [Cross-Validation](#3-cross-validation)
4. [Hyperparameter Tuning](#4-hyperparameter-tuning)
5. [Nested Cross-Validation](#5-nested-cv)
6. [Model Comparison](#6-model-comparison)
7. [When to Stop](#7-when-to-stop)

---

## 1. Bias-Variance

```
Total Error = Bias² + Variance + Irreducible Noise

Underfitting (high bias):    train error HIGH, test error HIGH
Overfitting (high variance): train error LOW,  test error HIGH
Good fit:                    train error LOW,  test error LOW (and close to each other)
```

---

## 2. Learning Curves

Plot performance vs training set size:

```
Score
  │
  │  ─────────────── Training score
  │  ·
  │    ·
  │      · · · · · · Test score
  │
  └──────────────────→ Training set size

High Bias (underfitting):     High Variance (overfitting):
Score                          Score
  │ ──────────                   │ ────────────── train
  │ · · · · · · ·               │
  │   (converge to LOW score)    │      · · · · · test
  │   (both bad)                 │   (big gap)
  └──────────→ size              └──────────→ size
  Fix: more complex model        Fix: more data or regularize
```

---

## 3. Cross-Validation

### K-Fold Algorithm

```
1. Shuffle data
2. Split into K equal folds
3. For i = 1 to K:
   a. Use fold i as validation set
   b. Use remaining K-1 folds as training set
   c. Train model, evaluate on fold i → scoreᵢ
4. Final score = mean(scores) ± std(scores)
```

```
K=5:
Fold 1: [VAL │train│train│train│train]  → score₁
Fold 2: [train│VAL │train│train│train]  → score₂
Fold 3: [train│train│VAL │train│train]  → score₃
Fold 4: [train│train│train│VAL │train]  → score₄
Fold 5: [train│train│train│train│VAL ]  → score₅

Result: 0.85 ± 0.03 (mean ± std)
```

### Stratified K-Fold

Same as K-Fold but preserves class distribution in each fold.

```
Data: 90% class A, 10% class B

Regular K-Fold might give:  Fold 1: 95%/5%  Fold 2: 85%/15%  (uneven)
Stratified K-Fold gives:   Fold 1: 90%/10% Fold 2: 90%/10%  (preserved)
```

Always use stratified for classification, especially imbalanced.

---

## 4. Hyperparameter Tuning

### GridSearchCV

```
Try ALL combinations:

param_grid = {
  'max_depth': [3, 5, 10],
  'n_estimators': [50, 100, 200]
}

Total: 3 × 3 = 9 combinations
Each evaluated with K-fold CV (K=5) → 9 × 5 = 45 model fits
```

### RandomizedSearchCV

```
Sample N random combinations from the search space:

param_distributions = {
  'max_depth': [3, 5, 7, 10, 15, 20],
  'n_estimators': [50, 100, 150, 200, 300],
  'min_samples_split': [2, 5, 10, 20]
}

Total possible: 6 × 5 × 4 = 120 combinations
RandomizedSearchCV with n_iter=20: tries 20 random combos instead of all 120
```

> **Key Intuition:** Random search often outperforms grid search because it explores more values per hyperparameter. If one parameter doesn't matter much, grid wastes time testing all its values while random search naturally allocates more unique values to important parameters.

```
Grid (3×3):                  Random (9 points):
  │ · · ·                      │    ·    ·
  │ · · ·   only 3 unique      │ ·     ·    9 unique
  │ · · ·   values per axis    │   ·  ·     values per axis
  └──────→                     │  ·    ·
                               └──────────→
```

---

## 5. Nested CV

Problem: if you use CV to both tune hyperparameters AND estimate performance, you're optimistically biased.

```
Nested CV:
  Outer loop (K=5): estimates TRUE performance
    Inner loop (K=5): tunes hyperparameters

For each outer fold:
  1. Hold out outer test fold
  2. On remaining data, run GridSearchCV (inner K-fold) to find best params
  3. Train with best params on outer training data
  4. Evaluate on outer test fold → unbiased score

Total: 5 outer × 5 inner × N param combos = many fits (but honest)
```

---

## 6. Model Comparison

Train multiple models, compare CV scores:

```
Model           │ CV Score (mean ± std)
────────────────┼──────────────────────
Logistic Reg    │ 0.79 ± 0.02
KNN             │ 0.75 ± 0.04
SVM (RBF)       │ 0.82 ± 0.03
Decision Tree   │ 0.72 ± 0.05
Random Forest   │ 0.84 ± 0.02
XGBoost         │ 0.85 ± 0.02
```

To check if the difference is statistically significant:
```
Paired t-test on CV fold scores:

XGBoost folds:  [0.84, 0.87, 0.83, 0.86, 0.85]
SVM folds:      [0.80, 0.84, 0.81, 0.83, 0.82]
Differences:    [0.04, 0.03, 0.02, 0.03, 0.03]

t-statistic and p-value tell you if the difference is real or noise
p < 0.05 → statistically significant difference
```

---

## 7. When to Stop

**Diminishing returns:**
```
Iteration 1:  baseline → 0.70
Iteration 2:  feature eng → 0.78  (+8%)
Iteration 3:  better model → 0.83 (+5%)
Iteration 4:  tuning → 0.84       (+1%)
Iteration 5:  ensemble → 0.845    (+0.5%)
Iteration 6:  more tuning → 0.847 (+0.2%)  ← stop here

The cost of each marginal improvement increases exponentially.
```

**Occam's Razor:** If two models perform similarly, pick the simpler one. It will generalize better and be easier to maintain.

---

## What to Look for in the Application Lab

In the application lab, you'll:
1. Plot learning curves to diagnose bias vs variance
2. Implement K-Fold CV from scratch, then use sklearn cross_val_score
3. Run GridSearchCV and RandomizedSearchCV, compare time and results
4. Compare 6+ models on Titanic with CV scores
5. Test statistical significance of differences
6. Select a final model and evaluate on the held-out test set ONCE
