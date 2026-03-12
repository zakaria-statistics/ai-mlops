# Techniques & Concepts — Mathematical Formulas

> Transformations, regularization, validation, and ensemble methods used in our pipeline.

## Table of Contents

1. [Log Transform](#1-log-transform)
2. [Regularization (L1 vs L2)](#2-regularization-l1-vs-l2)
3. [Bagging vs Boosting](#3-bagging-vs-boosting)
4. [Feature Importance](#4-feature-importance)
5. [Hyperparameter Tuning](#5-hyperparameter-tuning)
6. [Bias-Variance Tradeoff](#6-bias-variance-tradeoff)
7. [Overfitting Detection](#7-overfitting-detection)

---

## 1. Log Transform

**Why:** Price distribution is right-skewed (skew = 4.02). Most ML algorithms work
better with normally distributed targets.

### Forward Transform

```
log_price = log(1 + price)     ← log1p (handles price = 0 safely)
```

### Inverse Transform (to get predictions back in dollars)

```
price = exp(log_price) - 1     ← expm1
```

### Effect on Distribution

```
Before (price):
  ████████████████████▏                          skew = 4.02
  ↑ most homes        ↑ long tail (mansions)

After (log_price):
        ▕████████████████████▏                   skew ≈ 0
              roughly normal
```

### Why log1p / expm1 Instead of log / exp

```
log(0) = -∞       ← breaks if price = 0
log1p(0) = 0      ← safe

exp(big number) = overflow
expm1(x) = exp(x) - 1  ← numerically stable
```

---

## 2. Regularization (L1 vs L2)

**Why:** Prevent model from fitting noise by penalizing coefficient magnitude.

### General Regularized Loss

```
L(β) = Σᵢ (yᵢ - ŷᵢ)²  +  penalty(β)
        ─────────────      ──────────
        fit the data       keep it simple
```

### L2 (Ridge) — Penalty = α · Σⱼ βⱼ²

```
Geometry: constraint region is a CIRCLE (smooth)

         β₂
          │  ╭──╮
          │ │    │   ← all coefficients shrink
          │  ╰──╯      but stay non-zero
          └──────── β₁
```

### L1 (Lasso) — Penalty = α · Σⱼ |βⱼ|

```
Geometry: constraint region is a DIAMOND (corners)

         β₂
          │  ╱╲
          │╱    ╲    ← solution often hits a corner
          │╲    ╱      where some βⱼ = 0 (feature dropped)
          │  ╲╱
          └──────── β₁
```

### Effect of α (Regularization Strength)

```
α → 0:    no penalty, overfits like plain linear regression
α → ∞:    all coefficients → 0, underfits (predicts the mean)
sweet spot: found via cross-validation
```

---

## 3. Bagging vs Boosting

### Bagging (Random Forest)

```
Training data D
  ├── Bootstrap sample D₁* → Tree T₁ → prediction ŷ₁
  ├── Bootstrap sample D₂* → Tree T₂ → prediction ŷ₂
  ├── Bootstrap sample D₃* → Tree T₃ → prediction ŷ₃
  └── ... (B trees, trained INDEPENDENTLY)

Final: ŷ = (1/B) · (ŷ₁ + ŷ₂ + ... + ŷ_B)   ← average
```

**Bootstrap sample:** Draw n rows WITH replacement (some rows appear multiple times,
~37% of rows are left out per tree).

**Variance reduction:**

```
Var(ŷ) = ρ·σ² + (1-ρ)·σ²/B
          ─────   ───────────
          can't    shrinks with
          reduce   more trees (B)

ρ = correlation between trees
σ² = variance of single tree
```

Feature randomization (`max_features='sqrt'`) reduces ρ.

### Boosting (Gradient Boosting)

```
F₀ = ȳ (mean)
  │
  ├── residuals r₁ = y - F₀    → fit tree h₁ to r₁
  │   F₁ = F₀ + η·h₁
  │
  ├── residuals r₂ = y - F₁    → fit tree h₂ to r₂
  │   F₂ = F₁ + η·h₂
  │
  └── ... (M iterations, each tree corrects PREVIOUS errors)

Final: ŷ = F₀ + η·h₁ + η·h₂ + ... + η·hₘ
```

### Key Difference

```
Bagging:   PARALLEL trees, average → reduces VARIANCE
Boosting:  SEQUENTIAL trees, accumulate → reduces BIAS
```

---

## 4. Feature Importance

### Linear Coefficients

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ...

Importance(feature j) = |βⱼ|   (after feature scaling)
Sign: positive β → higher feature = higher price
      negative β → higher feature = lower price
```

**Caveat:** Only meaningful when features are on the same scale.

### Lasso Feature Selection

```
If βⱼ = 0  →  feature j is not useful (Lasso dropped it)
If βⱼ ≠ 0  →  feature j contributes to predictions
```

### Tree-Based Importance (Mean Decrease in Impurity)

```
Importance(feature j) = Σ over all nodes using feature j:
                         (fraction of samples) × (variance reduction)

Normalized: Σⱼ Importance(j) = 1
```

For a single split on feature j at node t:

```
ΔVariance = n_t/n · [ Var(t) - n_left/n_t · Var(t_left) - n_right/n_t · Var(t_right) ]
```

**For Random Forest:** averaged across all trees.

---

## 5. Hyperparameter Tuning

### RandomizedSearchCV

```
Given: parameter grid with C total combinations
Sample: n_iter random combinations (n_iter = 30 in our project)
For each combination:
  Run K-fold CV (K = 5)
  Record mean CV score

Select: combination with best mean CV score
```

### Why Randomized vs Grid Search

```
Grid:       tries ALL combinations → C = 3×3×3×3×3 = 243 combos (HGB grid)
Randomized: tries n random combos → 30 combos

Each combo runs 5-fold CV, so:
Grid:       243 × 5 = 1,215 model fits
Randomized: 30 × 5  = 150 model fits
```

Research shows: random search finds good solutions with far fewer evaluations,
because not all hyperparameters are equally important.

---

## 6. Bias-Variance Tradeoff

### Expected Prediction Error

```
E[(y - ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ²
               ─────────   ──────   ──
               systematic  model    irreducible
               error       instability  noise
```

### Bias

```
Bias(ŷ) = E[ŷ] - y_true
```

High bias = model is too simple, misses patterns (underfitting).

### Variance

```
Var(ŷ) = E[ (ŷ - E[ŷ])² ]
```

High variance = model changes a lot with different training data (overfitting).

### Our Models on This Spectrum

```
High Bias ←──────────────────────────────→ High Variance

Linear Reg    Ridge/Lasso    Random Forest    Decision Tree
(underfits)                  (balanced)       (unlimited, overfits)
                         Gradient Boosting
                           (best balance)
```

---

## 7. Overfitting Detection

### Train vs Test Error

```
Healthy:      Train RMSE ≈ Test RMSE     (good generalization)
Overfitting:  Train RMSE << Test RMSE    (memorized training data)
Underfitting: Train RMSE ≈ Test RMSE     (both high)
```

### Our Results

```
Decision Tree (unlimited):  train 0.029  test 0.249  → extreme overfitting
Decision Tree (pruned):     train 0.191  test 0.218  → mild overfitting
Random Forest:              train 0.138  test 0.185  → moderate
Gradient Boosting:          train 0.135  test 0.164  → good
XGBoost:                    train 0.114  test 0.162  → good
Linear Regression:          train 0.253  test 0.257  → underfit (both high)
```

### Detection Rule (Used in Our Code)

```
if train_rmse < test_rmse × 0.8:
    flag "⚠ Possible overfitting"
```

### Residual Analysis

Healthy residuals should be:
- **Random** (no patterns in residual-vs-predicted plot)
- **Centered at 0** (mean residual ≈ 0)
- **Normally distributed** (bell curve in histogram)
- **Constant variance** (no fan shape — homoscedasticity)
