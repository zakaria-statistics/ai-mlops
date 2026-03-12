# Metrics & KPIs — Mathematical Formulas

> Every evaluation metric used to measure and compare our models.

## Table of Contents

1. [RMSE — Root Mean Squared Error](#1-rmse--root-mean-squared-error)
2. [MAE — Mean Absolute Error](#2-mae--mean-absolute-error)
3. [R² — Coefficient of Determination](#3-r--coefficient-of-determination)
4. [MAPE — Mean Absolute Percentage Error](#4-mape--mean-absolute-percentage-error)
5. [Cross-Validation Score](#5-cross-validation-score)
6. [Metric Comparison](#6-metric-comparison)

---

## 1. RMSE — Root Mean Squared Error

**Purpose:** Primary optimization metric. Penalizes large errors more than small ones.

### Formula

```
RMSE = √[ (1/n) · Σᵢ (yᵢ - ŷᵢ)² ]
```

Expanded:

```
RMSE = √[ (1/n) · ( (y₁-ŷ₁)² + (y₂-ŷ₂)² + ... + (yₙ-ŷₙ)² ) ]
```

### Properties
- Same units as target (log-price in our case)
- Sensitive to outliers (squaring amplifies large errors)
- Lower = better, 0 = perfect
- Always ≥ MAE

### In Our Project
- Computed on **log scale** (since target is log-transformed)
- Used for both train and test to detect overfitting:
  ```
  train RMSE << test RMSE  →  overfitting
  train RMSE ≈ test RMSE   →  good generalization
  ```

---

## 2. MAE — Mean Absolute Error

**Purpose:** Interpretable error in real dollars. Treats all errors equally.

### Formula

```
MAE = (1/n) · Σᵢ |yᵢ - ŷᵢ|
```

### In Our Project
- Computed in **dollar scale** after inverse transform:
  ```
  y_dollars = expm1(y_log)       # inverse of log1p
  MAE = (1/n) · Σ |actual$ - predicted$|
  ```
- Business meaning: "On average, the prediction is off by $X"

### MAE vs RMSE

```
Error set: [1, 1, 1, 10]

MAE  = (1+1+1+10) / 4 = 3.25
RMSE = √((1+1+1+100)/4) = √25.75 = 5.07
```

RMSE punishes the outlier (10) more heavily. If you care about big misses, use RMSE.
If you want a "typical" error, use MAE.

---

## 3. R² — Coefficient of Determination

**Purpose:** How much variance the model explains. Scale-free (0 to 1).

### Formula

```
R² = 1 - (SS_res / SS_tot)
```

Where:

```
SS_res = Σᵢ (yᵢ - ŷᵢ)²       ← residual sum of squares (model errors)
SS_tot = Σᵢ (yᵢ - ȳ)²        ← total sum of squares (data variance)
```

### Interpretation

```
R² = 1.0   → model explains 100% of variance (perfect)
R² = 0.91  → model explains 91% of variance (our best)
R² = 0.0   → model is no better than predicting the mean
R² < 0     → model is WORSE than predicting the mean
```

### Geometric Meaning

```
Total variance:    |←————————————————————→|  100%
                   |←——— explained ———→|←?→|
                         R² = 0.91       9% unexplained
```

---

## 4. MAPE — Mean Absolute Percentage Error

**Purpose:** Error as a percentage of actual value. Business-friendly.

### Formula

```
MAPE = (100/n) · Σᵢ | (yᵢ - ŷᵢ) / yᵢ |
```

### Interpretation

```
MAPE = 12%  →  "predictions are off by about 12% on average"
```

A $500K home with 12% MAPE → error of ~$60K
A $1M home with 12% MAPE → error of ~$120K

### Caveats
- Asymmetric: overprediction and underprediction have different % impact
- Undefined when actual = 0
- Biased toward low predictions (underpredicting a $100K home by $50K = 50%, overpredicting = 33%)

---

## 5. Cross-Validation Score

**Purpose:** Estimate model performance on unseen data without touching the test set.

### K-Fold CV Formula

```
CV_score = (1/K) · Σₖ Score(fold_k)
```

### Standard Deviation (Confidence)

```
CV_std = √[ (1/K) · Σₖ (Scoreₖ - CV_score)² ]
```

### Reporting Format

```
RMSE: 0.164 ± 0.003
       ─────   ─────
       mean     std across 5 folds
```

Small std → stable performance across different data splits.
Large std → model is sensitive to which data it sees.

### In Our Project (5-Fold)

```
Data split into 5 parts, each ~3,458 rows:

Fold 1: [VAL  ][TRAIN][TRAIN][TRAIN][TRAIN]  → RMSE₁
Fold 2: [TRAIN][VAL  ][TRAIN][TRAIN][TRAIN]  → RMSE₂
Fold 3: [TRAIN][TRAIN][VAL  ][TRAIN][TRAIN]  → RMSE₃
Fold 4: [TRAIN][TRAIN][TRAIN][VAL  ][TRAIN]  → RMSE₄
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][VAL  ]  → RMSE₅

Final = mean(RMSE₁..₅) ± std(RMSE₁..₅)
```

---

## 6. Metric Comparison

### When to Use Each

| Metric | Best For | Units | Sensitive to Outliers? |
|--------|----------|-------|----------------------|
| RMSE | Optimization, model selection | Same as target | Yes (squaring) |
| MAE | Interpretable average error | Dollars | No |
| R² | Explaining model quality | Dimensionless (0-1) | Moderate |
| MAPE | Business communication | Percentage | No (but asymmetric) |
| CV Score | Reliable generalization estimate | Same as chosen metric | Depends on metric |

### Relationships

```
RMSE ≥ MAE           (always, equality only when all errors are equal)
RMSE = MAE           (when all errors are the same magnitude)
RMSE >> MAE          (when there are large outlier errors)

R² = 1 - (RMSE² · n / SS_tot)    (direct relationship)
```
