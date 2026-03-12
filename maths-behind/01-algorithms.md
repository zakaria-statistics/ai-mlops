# Algorithms — Mathematical Formulas

> Every model used in our training pipeline, from simple to complex.

## Table of Contents

1. [Linear Regression](#1-linear-regression)
2. [Ridge Regression (L2)](#2-ridge-regression-l2)
3. [Lasso Regression (L1)](#3-lasso-regression-l1)
4. [Decision Tree](#4-decision-tree)
5. [Random Forest](#5-random-forest)
6. [Gradient Boosting](#6-gradient-boosting)
7. [XGBoost](#7-xgboost)

---

## 1. Linear Regression

**Goal:** Find the hyperplane that minimizes squared errors.

### Prediction

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

In matrix form:

```
ŷ = Xβ
```

Where:
- `X` = feature matrix (n × p)
- `β` = coefficient vector (p × 1)
- `ŷ` = predicted values (n × 1)

### Loss Function (Ordinary Least Squares)

```
L(β) = Σᵢ (yᵢ - ŷᵢ)²  =  ‖y - Xβ‖²
```

### Closed-Form Solution (Normal Equation)

```
β = (XᵀX)⁻¹ Xᵀy
```

### Assumptions
- Linear relationship between features and target
- Each feature has constant, additive effect
- Errors are normally distributed with constant variance

### In Our Project
- 16 features → 16 coefficients + 1 intercept
- Result: R² = 0.77, MAPE = 20% — misses non-linear patterns

---

## 2. Ridge Regression (L2)

**Goal:** Linear regression + penalty on large coefficients to prevent overfitting.

### Loss Function

```
L(β) = Σᵢ (yᵢ - ŷᵢ)²  +  α · Σⱼ βⱼ²
        ─────────────      ──────────
         OLS term          L2 penalty
```

### Closed-Form Solution

```
β = (XᵀX + αI)⁻¹ Xᵀy
```

### How α Works

```
α = 0     → pure linear regression (no penalty)
α → ∞    → all coefficients shrink toward 0
```

- **Shrinks** all coefficients but never zeros them out
- Handles **multicollinearity** (correlated features)

### In Our Project
- Tested α ∈ {0.01, 0.1, 1.0, 10.0, 100.0}
- Minimal improvement over plain linear regression (features aren't highly collinear after prep)

---

## 3. Lasso Regression (L1)

**Goal:** Linear regression + penalty that can zero out coefficients (automatic feature selection).

### Loss Function

```
L(β) = Σᵢ (yᵢ - ŷᵢ)²  +  α · Σⱼ |βⱼ|
        ─────────────      ────────────
         OLS term          L1 penalty
```

### Key Difference from Ridge

```
Ridge (L2):  penalty = Σ βⱼ²    → shrinks toward 0, never reaches 0
Lasso (L1):  penalty = Σ |βⱼ|   → can shrink TO exactly 0
```

### Geometric Intuition

```
Ridge:  constraint region is a circle   → coefficients slide along curve
Lasso:  constraint region is a diamond  → coefficients hit corners (= 0)
```

### In Our Project
- α = 0.001
- Some features zeroed out → tells us which features Lasso considers useless

---

## 4. Decision Tree

**Goal:** Partition feature space into rectangular regions using binary splits.

### Split Criterion (Regression — Variance Reduction)

For a node with data S, splitting into left (Sₗ) and right (Sᵣ):

```
Gain = Var(S) - [ |Sₗ|/|S| · Var(Sₗ) + |Sᵣ|/|S| · Var(Sᵣ) ]
```

Where variance:

```
Var(S) = (1/|S|) · Σᵢ∈S (yᵢ - ȳₛ)²
```

### Prediction

Each leaf predicts the **mean** of training samples in that region:

```
ŷ = ȳ_leaf  =  (1/|S_leaf|) · Σᵢ∈S_leaf yᵢ
```

### Pruning (Regularization)

```
max_depth = 10     → tree can't grow deeper than 10 levels
min_samples_leaf = 20  → each leaf must have ≥ 20 samples
```

### In Our Project
- **Unlimited tree:** train RMSE 0.029, test RMSE 0.249 → extreme overfitting
- **Pruned tree:** train RMSE 0.191, test RMSE 0.218 → better but still high variance

---

## 5. Random Forest

**Goal:** Average many decision trees trained on random subsets to reduce variance.

### Algorithm (Bagging + Feature Randomization)

```
For b = 1 to B (number of trees):
  1. Draw bootstrap sample S*ᵦ (sample n rows WITH replacement)
  2. Grow tree Tᵦ on S*ᵦ, at each split:
     - Select m random features (m = √p for regression)
     - Find best split among those m features
  3. Grow until stopping criteria (max_depth, min_samples_leaf)
```

### Prediction (Average of Trees)

```
ŷ = (1/B) · Σᵦ Tᵦ(x)
```

### Why It Works — Variance Reduction

For B independent trees each with variance σ²:

```
Var(average) = σ² / B
```

Trees aren't fully independent (correlated), so actual reduction is less,
but feature randomization (`max_features='sqrt'`) reduces correlation between trees.

### Feature Importance (Mean Decrease in Impurity)

```
Importance(feature j) = Σ over all trees, all nodes using feature j:
                         (weighted variance reduction at that node)
```

Normalized so all importances sum to 1.

### In Our Project
- B = 200 trees, max_depth = 20, max_features = √16 ≈ 4
- R² = 0.88, MAE = $79K

---

## 6. Gradient Boosting

**Goal:** Train trees sequentially — each new tree fits the residual errors of all previous trees.

### Algorithm

```
Initialize: F₀(x) = ȳ  (predict the mean)

For m = 1 to M (number of iterations):
  1. Compute residuals:  rᵢₘ = yᵢ - Fₘ₋₁(xᵢ)
  2. Fit a tree hₘ(x) to residuals rₘ
  3. Update model:  Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)
```

Where:
- `Fₘ(x)` = model after m trees
- `hₘ(x)` = new tree fitted to residuals
- `η` = learning rate (step size, e.g., 0.05)

### Final Prediction

```
ŷ = F₀(x) + η · h₁(x) + η · h₂(x) + ... + η · hₘ(x)
  = ȳ + η · Σₘ hₘ(x)
```

### Learning Rate Tradeoff

```
Small η (0.01):  needs more trees (M), slower, often better generalization
Large η (0.1):   needs fewer trees, faster, risk overfitting
```

### HistGradientBoosting (Our Implementation)

Uses histogram-based splits instead of exact splits:
- Bins continuous features into ~256 bins
- Much faster for large datasets
- Inspired by LightGBM

### In Our Project
- M = 500, η = 0.05, max_depth = 8
- R² = 0.91, MAE = $66K

---

## 7. XGBoost

**Goal:** Gradient boosting with built-in regularization and optimized implementation.

### Objective Function

```
Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₘ Ω(hₘ)
       ──────────     ─────────
       loss term      regularization
```

Where regularization for each tree:

```
Ω(h) = γ · T + (1/2) · λ · Σⱼ wⱼ²
        ─────   ──────────────────
        penalty   L2 penalty on
        on #leaves  leaf weights
```

- `T` = number of leaves
- `wⱼ` = weight (prediction value) of leaf j
- `γ` = min loss reduction to make a split (pruning)
- `λ` = L2 regularization on leaf weights

### Key Hyperparameters

```
subsample = 0.8        → use 80% of rows per tree (stochastic)
colsample_bytree = 0.8 → use 80% of features per tree
min_child_weight = 1   → minimum sum of instance weight in a leaf
```

### Difference from Standard Gradient Boosting

| Aspect | Standard GB | XGBoost |
|--------|-------------|---------|
| Regularization | None built-in | L1 + L2 on leaf weights |
| Split finding | Exact greedy | Approximate + histogram |
| Missing values | Must impute | Learns best direction |
| Parallelism | Sequential trees | Parallel feature splits |

### In Our Project
- n_estimators = 500, η = 0.05, max_depth = 6
- R² = 0.91, MAE = $64K — best model
