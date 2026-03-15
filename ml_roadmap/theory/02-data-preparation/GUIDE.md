# Data Preparation — Splits, Scaling, Encoding, and Leakage
> The math and logic behind transforming raw data into model-ready inputs.

## Table of Contents
1. [Train / Test / Validation Splits](#1-train--test--validation-splits) — Why and how to partition data
2. [Stratification](#2-stratification) — Preserving class proportions in splits
3. [Feature Scaling](#3-feature-scaling) — StandardScaler and MinMaxScaler formulas
4. [Encoding Categorical Variables](#4-encoding-categorical-variables) — One-hot and ordinal math
5. [Handling Missing Values](#5-handling-missing-values) — Strategies and tradeoffs
6. [Handling Imbalanced Data](#6-handling-imbalanced-data) — SMOTE intuition and class weights
7. [Data Leakage](#7-data-leakage) — What it is and how to avoid it
8. [By-Hand Preparation Example](#8-by-hand-preparation-example) — Full pipeline on a tiny dataset
9. [What to Look for in the Application Lab](#9-what-to-look-for-in-the-application-lab)

---

## 1. Train / Test / Validation Splits

### Why Split?

You need to answer: "How well will this model perform on **data it has never seen**?"

If you train and evaluate on the same data, you are measuring **memorization**, not
**generalization**.

```
  ┌────────────────────────────────────────────────────────┐
  │                    Full Dataset                        │
  ├────────────────────────────┬───────────┬───────────────┤
  │      Training Set         │ Val Set   │   Test Set    │
  │       (60-80%)            │ (10-20%)  │   (10-20%)    │
  │                           │           │               │
  │  Model LEARNS from this   │ Tune      │  Final score  │
  │                           │ hyperparams│ (touch ONCE) │
  └────────────────────────────┴───────────┴───────────────┘
```

### The Three Roles

| Set | Purpose | When used | Typical size |
|-----|---------|-----------|-------------|
| Training | Model learns parameters (weights) | During training | 60-80% |
| Validation | Tune hyperparameters, detect overfitting | During development | 10-20% |
| Test | Final unbiased performance estimate | ONCE, at the very end | 10-20% |

> **Key Intuition:** The test set is like a sealed exam. If you peek at it during
> development (even indirectly, by tuning based on test results), your final score
> is no longer trustworthy. The validation set exists so you have something to
> peek at safely.

### Simple vs Cross-Validation

When data is limited, a single split wastes too much data. **K-Fold Cross-Validation**
uses all data for both training and validation:

```
  5-Fold Cross-Validation:

  Fold 1: [VAL][Train][Train][Train][Train]  → score₁
  Fold 2: [Train][VAL][Train][Train][Train]  → score₂
  Fold 3: [Train][Train][VAL][Train][Train]  → score₃
  Fold 4: [Train][Train][Train][VAL][Train]  → score₄
  Fold 5: [Train][Train][Train][Train][VAL]  → score₅

  Final score = mean(score₁, ..., score₅)   ± std for confidence
```

Each data point serves as validation exactly once. More reliable estimate than a
single split, especially with small datasets.

---

## 2. Stratification

### The Problem

Random splitting can create **unrepresentative** subsets, especially with imbalanced classes.

```
  Full dataset: 90% class A, 10% class B  (n=100)

  Random split (bad luck):
    Train (80): 75 A, 5 B   → 93.75% A, 6.25% B    ← distorted
    Test  (20): 15 A, 5 B   → 75% A, 25% B          ← distorted

  Stratified split (preserves proportions):
    Train (80): 72 A, 8 B   → 90% A, 10% B          ← matches original
    Test  (20): 18 A, 2 B   → 90% A, 10% B          ← matches original
```

### When to Stratify

- Classification problems (always recommended)
- When any class has < 15-20% frequency
- When dataset is small (< 1000 samples)
- Regression: can stratify on binned target values

---

## 3. Feature Scaling

### Why Scale?

Many algorithms are sensitive to the **magnitude** of features.

```
  Feature 1: age     = [25, 30, 35, 40, 45]     range: 20
  Feature 2: income  = [30000, 50000, 70000]     range: 40000

  Without scaling, income DOMINATES the distance calculation:

  Distance = √[(age₁-age₂)² + (income₁-income₂)²]

  Example:  √[(25-45)² + (30000-70000)²]
          = √[400 + 1,600,000,000]
          ≈ √1,600,000,000
          ≈ 40,000

  The age difference (20 years) is completely invisible!
```

### StandardScaler (Z-score normalization)

Transforms each feature to have **mean = 0** and **standard deviation = 1**.

```
  z = (x - μ) / σ

  where:
    μ = mean of the feature (computed on TRAINING data only)
    σ = std of the feature (computed on TRAINING data only)

  After transformation:
    mean(z) = 0
    std(z) = 1
```

**Properties:**
- Centers data around zero
- Preserves outliers (no bounded range)
- Best when data is approximately normal

### MinMaxScaler

Transforms each feature to a fixed range, typically [0, 1].

```
  x_scaled = (x - x_min) / (x_max - x_min)

  where:
    x_min = minimum value (computed on TRAINING data only)
    x_max = maximum value (computed on TRAINING data only)

  After transformation:
    min(x_scaled) = 0
    max(x_scaled) = 1
```

To scale to a different range [a, b]:

```
  x_scaled = a + (x - x_min) · (b - a) / (x_max - x_min)
```

**Properties:**
- Bounded output [0, 1]
- Sensitive to outliers (one extreme value compresses everything else)
- Good for algorithms that expect bounded inputs (neural networks, KNN)

### When to Use Which

| Scaler | Use when | Avoid when |
|--------|----------|------------|
| StandardScaler | Data ~normal, features have different units | — |
| MinMaxScaler | Need bounded [0,1] range, no extreme outliers | Outliers present |
| RobustScaler* | Outliers present | — |
| None | Tree-based models (they split on rank, not magnitude) | — |

*RobustScaler uses median and IQR instead of mean and std:
```
  x_scaled = (x - median) / IQR
```

### By-Hand Scaling Example

```
  Feature "age": [20, 30, 40, 50, 60]

  ── StandardScaler ──
  μ = 40,  σ = √[(400+100+0+100+400)/5] = √200 = 14.14

  z(20) = (20-40)/14.14 = -1.41
  z(30) = (30-40)/14.14 = -0.71
  z(40) = (40-40)/14.14 =  0.00
  z(50) = (50-40)/14.14 = +0.71
  z(60) = (60-40)/14.14 = +1.41

  Result: [-1.41, -0.71, 0.00, +0.71, +1.41]

  ── MinMaxScaler ──
  x_min = 20,  x_max = 60,  range = 40

  scaled(20) = (20-20)/40 = 0.00
  scaled(30) = (30-20)/40 = 0.25
  scaled(40) = (40-20)/40 = 0.50
  scaled(50) = (50-20)/40 = 0.75
  scaled(60) = (60-20)/40 = 1.00

  Result: [0.00, 0.25, 0.50, 0.75, 1.00]
```

---

## 4. Encoding Categorical Variables

ML models need numbers. Categorical features must be converted.

### One-Hot Encoding (for nominal categories)

Each category becomes its own binary column.

```
  Original:              One-Hot Encoded:

  color                  color_red  color_blue  color_green
  ─────                  ─────────  ──────────  ───────────
  red                       1          0           0
  blue                      0          1           0
  green                     0          0           1
  red                       1          0           0
  blue                      0          1           0
```

**Math representation:** For a feature with k categories, create k binary vectors.
Each vector eᵢ has a 1 in position i and 0 elsewhere.

```
  red   → [1, 0, 0]    = e₁
  blue  → [0, 1, 0]    = e₂
  green → [0, 0, 1]    = e₃
```

**Drop-one variant:** Use k-1 columns to avoid the "dummy variable trap"
(perfect multicollinearity, since column_k = 1 - sum of others).

```
  color_blue  color_green     (red is the "reference" category)
  ──────────  ───────────
      0           0           ← this IS red (both 0)
      1           0
      0           1
      0           0
      1           0
```

### Ordinal Encoding (for ordinal categories)

Assign integers that preserve the natural order.

```
  education       encoded
  ─────────       ───────
  high_school  →    0
  bachelors    →    1
  masters      →    2
  phd          →    3
```

> **Key Intuition:** Never use ordinal encoding for nominal data. If you encode
> {red=0, blue=1, green=2}, the model thinks green > blue > red, and that
> green - blue = blue - red. These relationships are meaningless for colors.
> One-hot avoids this by treating each category as independent.

---

## 5. Handling Missing Values

### Strategies

```
  Strategy               Formula / Logic                    When to use
  ────────────────────────────────────────────────────────────────────────
  Drop rows              Remove rows with NaN               <5% missing, MCAR
  Drop columns           Remove entire feature              >50% missing
  Mean imputation        x_missing = x̄                     Numerical, normal dist
  Median imputation      x_missing = median(x)             Numerical, skewed
  Mode imputation        x_missing = mode(x)               Categorical
  Constant               x_missing = c (e.g., 0, "Unknown") Domain knowledge
  Forward/backward fill  x_missing = previous/next value   Time series
  Model-based (KNN)      Predict missing from k neighbors  Complex patterns
```

### Missing Data Types

```
  MCAR (Missing Completely At Random)
    Missingness has NO relationship to any data
    Example: sensor randomly fails

  MAR (Missing At Random)
    Missingness depends on OTHER observed variables
    Example: younger people skip "income" question

  MNAR (Missing Not At Random)
    Missingness depends on the MISSING VALUE ITSELF
    Example: high-income people refuse to report income
    Hardest case — simple imputation introduces bias
```

### By-Hand Example

```
  Data:  [10, 12, NaN, 14, 11, NaN, 13]

  Non-missing values: [10, 12, 14, 11, 13]

  Mean imputation:    NaN → (10+12+14+11+13)/5 = 60/5 = 12.0
  Result: [10, 12, 12, 14, 11, 12, 13]

  Median imputation:  sorted non-missing = [10, 11, 12, 13, 14]
                      median = 12
  Result: [10, 12, 12, 14, 11, 12, 13]    (same here by coincidence)
```

> **Key Intuition:** Mean/median imputation reduces variance because you are
> replacing unknown values with the "average." This can weaken correlations.
> For critical features with lots of missing data, consider adding a binary
> indicator column: `feature_was_missing = [0, 0, 1, 0, 0, 1, 0]`.

---

## 6. Handling Imbalanced Data

### The Problem

When one class dominates, a model can achieve high accuracy by **always predicting
the majority class**.

```
  Dataset: 950 "not fraud" + 50 "fraud" = 1000 total

  Dumb model: always predict "not fraud"
  Accuracy = 950/1000 = 95%  ← looks great, catches zero fraud!
```

### Strategy 1: Class Weights

Penalize misclassification of the minority class more heavily in the loss function.

```
  Standard loss:    L = -(1/n) Σ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]

  Weighted loss:    L = -(1/n) Σ [wᵢ · (yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ))]

  where wᵢ = n / (2 · n_class_k)  for "balanced" weights

  Example:
    w_majority = 1000 / (2 · 950) = 0.526
    w_minority = 1000 / (2 · 50)  = 10.0     ← 19x heavier penalty!
```

The model pays a much higher cost for missing a fraud case, so it learns to
be more sensitive to them.

### Strategy 2: SMOTE (Synthetic Minority Oversampling Technique)

Instead of duplicating minority examples, SMOTE creates **new synthetic examples**
by interpolating between existing minority neighbors.

```
  Algorithm:
  1. Pick a minority sample xᵢ
  2. Find its k nearest minority neighbors
  3. Randomly pick one neighbor xₙ
  4. Create synthetic point:

     x_new = xᵢ + λ · (xₙ - xᵢ)

     where λ ~ Uniform(0, 1)

  Visually:

     xᵢ ●─────────────────● xₙ
              ↑
         x_new = somewhere on this line
              (random λ picks the spot)
```

**Example with 2D data:**

```
  xᵢ = [2, 4]     (existing minority point)
  xₙ = [6, 8]     (nearest minority neighbor)
  λ = 0.3          (random)

  x_new = [2, 4] + 0.3 · ([6, 8] - [2, 4])
        = [2, 4] + 0.3 · [4, 4]
        = [2, 4] + [1.2, 1.2]
        = [3.2, 5.2]    ← new synthetic minority point
```

```
  Before SMOTE:                After SMOTE:

  y│ ○ ○ ○ ○ ○                y│ ○ ○ ○ ○ ○
   │ ○ ○ ○ ○                   │ ○ ○ ○ ○
   │ ○ ○ ○                     │ ○ ○ ○
   │   ● ●                     │   ● ● ◆ ◆
   │                            │ ◆ ● ◆ ●
   └──────── x                  └──────── x

  ○ = majority     ● = minority     ◆ = synthetic minority
```

> **Key Intuition:** SMOTE only on the TRAINING set, never on test/validation.
> Creating synthetic test examples would inflate your metrics. Also, SMOTE
> can create noisy samples if minority points are surrounded by majority —
> that is why variants like SMOTE-ENN exist (clean up borderline synthetics).

### Strategy 3: Undersample the Majority

Remove random majority samples to balance. Simple but loses information.

```
  Before: 950 majority + 50 minority
  After:  50 majority + 50 minority  ← threw away 900 samples!
```

Only viable when you have abundant data.

---

## 7. Data Leakage

### What Is It?

Data leakage occurs when information from **outside the training set** bleeds into
the training process, making your model look better than it actually is.

```
  ┌─────────── The Wall ───────────┐
  │                                │
  │  TRAINING SIDE    │  TEST SIDE │
  │                   │            │
  │  Everything the   │ Must be    │
  │  model sees       │ invisible  │
  │  during fitting   │ to model   │
  │                   │            │
  └───────────────────┴────────────┘

  Leakage = any information that crosses this wall
```

### Common Sources of Leakage

**1. Scaling before splitting (the #1 beginner mistake)**

```
  WRONG:
    scaler.fit(ALL_DATA)          ← test statistics leak into training
    scaler.transform(ALL_DATA)
    X_train, X_test = split(ALL_DATA)

  RIGHT:
    X_train, X_test = split(ALL_DATA)
    scaler.fit(X_train)           ← only learn from training
    scaler.transform(X_train)
    scaler.transform(X_test)      ← apply same transformation
```

When you fit the scaler on all data, the training process "knows" the test set's
mean and standard deviation. This is information from the future.

**2. Feature contains future information**

```
  Predicting: "Will this customer buy?"
  Feature:    "total_purchases_this_month"  ← includes the purchase you're predicting!
```

**3. Target leakage through proxy features**

```
  Predicting: "Does patient have disease X?"
  Feature:    "prescribed_drug_for_X"       ← only prescribed IF diagnosed!
```

### The Pipeline Solution

```
  ┌──────────────────────────────────────────┐
  │           sklearn Pipeline               │
  │                                          │
  │  Step 1: Imputer    ──fit on train only──│──► transform train & test
  │  Step 2: Scaler     ──fit on train only──│──► transform train & test
  │  Step 3: Encoder    ──fit on train only──│──► transform train & test
  │  Step 4: Model      ──fit on train only──│──► predict on test
  │                                          │
  └──────────────────────────────────────────┘

  pipeline.fit(X_train, y_train)     ← fits ALL steps on training only
  pipeline.predict(X_test)           ← transforms and predicts test safely
```

> **Key Intuition:** Ask yourself for every feature and every transformation:
> "Would I have access to this information at prediction time in the real world?"
> If the answer is no, it is leakage.

---

## 8. By-Hand Preparation Example

### Starting Dataset (5 rows)

```
  id │ age  │ income  │ city    │ rating   │ purchased
  ───┼──────┼─────────┼─────────┼──────────┼──────────
   1 │  25  │ 40000   │ Austin  │ low      │    0
   2 │  35  │  NaN    │ Dallas  │ medium   │    1
   3 │  45  │ 80000   │ Austin  │ high     │    1
   4 │  30  │ 55000   │ Dallas  │ low      │    0
   5 │  50  │ 90000   │ Austin  │ medium   │    1
```

**Step 1: Split first (before any transformation)**

```
  Train (rows 1,3,4):               Test (rows 2,5):
  age  income  city    rating  y    age  income  city    rating  y
  25   40000   Austin  low     0    35   NaN     Dallas  medium  1
  45   80000   Austin  high    1    50   90000   Austin  medium  1
  30   55000   Dallas  low     0
```

**Step 2: Handle missing values (fit on train, apply to test)**

```
  income in train: [40000, 80000, 55000]
  median_income = 55000  (computed from TRAIN only)

  Test row 2: income NaN → 55000
```

**Step 3: Encode categoricals**

```
  Ordinal encode "rating": low=0, medium=1, high=2

  One-hot encode "city":
    Austin → city_Dallas=0
    Dallas → city_Dallas=1
    (drop "city_Austin" to avoid dummy trap)

  Train:
  age  income  rating  city_Dallas  y
  25   40000     0         0        0
  45   80000     2         0        1
  30   55000     0         1        0

  Test:
  age  income  rating  city_Dallas  y
  35   55000     1         1        1
  50   90000     1         0        1
```

**Step 4: Scale numerical features (fit on train, transform both)**

```
  StandardScaler on "age" (train only):
    μ_age = (25+45+30)/3 = 33.33
    σ_age = √[((25-33.33)² + (45-33.33)² + (30-33.33)²)/3]
          = √[(69.39 + 136.09 + 11.09)/3] = √72.19 = 8.50

  StandardScaler on "income" (train only):
    μ_inc = (40000+80000+55000)/3 = 58333
    σ_inc = √[((40000-58333)²+(80000-58333)²+(55000-58333)²)/3]
          = √[(336108889+469708889+11108889)/3] = √272308889 = 16502

  Train (scaled):
  age     income   rating  city_Dallas  y
  -0.98   -1.11      0         0        0
   1.37    1.31      2         0        1
  -0.39   -0.20      0         1        0

  Test (scaled using TRAIN's μ and σ):
  age     income   rating  city_Dallas  y
   0.20   -0.20      1         1        1
   1.96    1.92      1         0        1
```

The test set is transformed using parameters learned **only from the training set**.
No leakage.

---

## 9. What to Look for in the Application Lab

| Theory concept | What you will see in code |
|---|---|
| Train/test split | `train_test_split(X, y, test_size=0.2, random_state=42)` |
| Stratification | `train_test_split(..., stratify=y)` |
| StandardScaler | `scaler.fit(X_train)` then `scaler.transform(X_test)` |
| MinMaxScaler | Same fit/transform pattern |
| One-hot encoding | `pd.get_dummies()` or `OneHotEncoder()` |
| Ordinal encoding | `OrdinalEncoder(categories=[ordered_list])` |
| Missing values | `SimpleImputer(strategy='median')` |
| Class weights | `LogisticRegression(class_weight='balanced')` |
| SMOTE | `from imblearn.over_sampling import SMOTE` |
| Pipeline (no leakage) | `Pipeline([('scaler', ...), ('model', ...)])` |

**Questions to ask yourself during the lab:**
1. Did I split BEFORE any transformation? (leakage check)
2. Did I fit scalers/encoders on training data only?
3. Are my class proportions preserved in the split? (stratification check)
4. If classes are imbalanced, what strategy am I using?
5. How many missing values per feature? Which imputation strategy makes sense?
