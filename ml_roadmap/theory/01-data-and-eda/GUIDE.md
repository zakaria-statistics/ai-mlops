# Data Types, Distributions, and Exploratory Data Analysis
> Understanding your data before modeling — the math behind summary statistics, correlation, and outlier detection.

## Table of Contents
1. [Types of Data](#1-types-of-data) — Numerical vs categorical and their sub-types
2. [Distributions](#2-distributions) — Normal, skewed, uniform and what they mean
3. [Central Tendency](#3-central-tendency) — Mean, median, mode with formulas
4. [Spread and Dispersion](#4-spread-and-dispersion) — Variance, standard deviation, IQR
5. [Correlation](#5-correlation) — Pearson and Spearman with derivations
6. [Outlier Detection](#6-outlier-detection) — IQR rule and Z-score methods
7. [Visualization Types](#7-visualization-types) — Which plot for which question
8. [By-Hand EDA Example](#8-by-hand-eda-example) — Full walkthrough on a tiny dataset
9. [What to Look for in the Application Lab](#9-what-to-look-for-in-the-application-lab)

---

## 1. Types of Data

```
Data Types
├── Numerical (quantitative — can do math on it)
│   ├── Continuous    values on a spectrum, can be any decimal
│   │   Examples: temperature (36.7°C), price ($142.50), height (1.82m)
│   │
│   └── Discrete      countable integers, no in-between values
│       Examples: number of rooms (3), children (2), orders (17)
│
├── Categorical (qualitative — labels or groups)
│   ├── Nominal       no natural order, just names
│   │   Examples: color {red, blue}, city {NYC, LA}, blood type {A, B, O, AB}
│   │
│   └── Ordinal       has a natural order, but gaps are not equal
│       Examples: rating {low, medium, high}, education {HS, BS, MS, PhD}
│
└── Special Types
    ├── Binary        two values only: {0, 1} or {yes, no}
    ├── DateTime      timestamps, dates
    └── Text          free-form strings (needs NLP preprocessing)
```

> **Key Intuition:** The data type determines everything downstream —
> which statistics you can compute, which plots make sense, and which
> ML algorithms work. You cannot take the "mean" of a nominal variable,
> and you should not treat ordinal as continuous without thinking about it.

### Why It Matters for ML

| Data Type | Can compute mean? | Can compute median? | Needs encoding? | Common encoding |
|-----------|:-:|:-:|:-:|---|
| Continuous | Yes | Yes | No | — |
| Discrete | Yes | Yes | Sometimes | — |
| Nominal | No | No | Yes | One-hot |
| Ordinal | No | Yes | Yes | Ordinal (integer) |
| Binary | Yes (as 0/1) | Yes | No | Already numeric |

---

## 2. Distributions

A distribution describes **how values are spread** across possible outcomes.

### Normal (Gaussian) Distribution

The most important distribution in statistics. Bell-shaped, symmetric.

```
  Probability
      ▲
      │          ·  ·
      │        ·      ·
      │      ·    μ     ·
      │     ·     │      ·
      │    ·      │       ·
      │   ·       │        ·
      │  ·        │         ·
      │ ·         │          ·
      │·          │           ·
      └───────────┼────────────── Value
            μ-2σ  μ  μ+2σ
```

**Formula (probability density function):**

```
  f(x) = (1 / (σ√(2π))) · e^(-(x-μ)²/(2σ²))

  where:
    μ = mean (center)
    σ = standard deviation (width)
    e ≈ 2.718 (Euler's number)
```

**The 68-95-99.7 rule:**
```
  68.3% of data falls within μ ± 1σ
  95.4% of data falls within μ ± 2σ
  99.7% of data falls within μ ± 3σ
```

### Skewed Distributions

```
  Right-skewed (positive)        Left-skewed (negative)
  (long tail on the right)       (long tail on the left)

      ▲                              ▲
      │ ·                            │           ·
      │ · ·                          │         · ·
      │ ·   ·                        │       ·   ·
      │ ·     · ·                    │   · ·     ·
      │ ·        · · · ·            │ · · ·        ·
      └──────────────────►          └──────────────────►

  mode < median < mean            mean < median < mode
  Example: income, home prices    Example: age at retirement
```

### Uniform Distribution

Every value equally likely. Flat.

```
      ▲
      │ ┌──────────────┐
      │ │              │     f(x) = 1 / (b - a)    for a ≤ x ≤ b
      │ │              │           = 0              otherwise
      │ │              │
      └─┴──────────────┴──►
        a              b
```

> **Key Intuition:** Many ML algorithms assume or work best with normally distributed
> features. When data is heavily skewed, transformations like log(x) or sqrt(x) can
> help pull it closer to normal.

---

## 3. Central Tendency

Three ways to answer "what is a typical value?"

### Mean (arithmetic average)

```
  μ = (1/n) Σᵢ xᵢ  =  (x₁ + x₂ + ... + xₙ) / n
```

- Uses every data point
- Sensitive to outliers (one extreme value pulls it)

### Median (middle value)

```
  Sort the data: x₍₁₎ ≤ x₍₂₎ ≤ ... ≤ x₍ₙ₎

  If n is odd:   median = x₍₍ₙ₊₁₎/₂₎
  If n is even:  median = (x₍ₙ/₂₎ + x₍ₙ/₂₊₁₎) / 2
```

- Only looks at position, not magnitude
- Robust to outliers

### Mode (most frequent value)

```
  mode = value with the highest frequency
  (a dataset can have 0, 1, or multiple modes)
```

### By-Hand Example

```
  Data: [2, 3, 3, 5, 100]    n = 5

  Mean   = (2 + 3 + 3 + 5 + 100) / 5 = 113 / 5 = 22.6   ← pulled by 100
  Median = 3rd value (sorted) = 3                          ← robust
  Mode   = 3 (appears twice)                               ← most common
```

> **Key Intuition:** When mean and median diverge significantly, you have skew
> or outliers. The median is almost always the better "typical value" for skewed data.
> This is why median income is more informative than mean income.

---

## 4. Spread and Dispersion

How spread out is the data? Central tendency alone is not enough.

### Variance

```
  Population variance:   σ² = (1/N) Σᵢ (xᵢ - μ)²

  Sample variance:       s² = (1/(n-1)) Σᵢ (xᵢ - x̄)²
                                  ▲
                                  │
                         Bessel's correction: dividing by (n-1) instead of n
                         gives an unbiased estimate when working with a sample
```

**Why squared?** Squaring does two things:
1. Makes all deviations positive (no cancellation)
2. Penalizes large deviations more heavily

### Standard Deviation

```
  σ = √(σ²)      or      s = √(s²)
```

Same units as the original data (variance is in squared units).

### Interquartile Range (IQR)

```
  Sort the data, then find:

  Q1 = 25th percentile (median of lower half)
  Q2 = 50th percentile (median = middle)
  Q3 = 75th percentile (median of upper half)

  IQR = Q3 - Q1

  ┌────────┬────────┬────────┬────────┐
  │  25%   │  25%   │  25%   │  25%   │
  min     Q1       Q2       Q3       max
           │◄──────IQR──────►│
```

IQR captures the spread of the **middle 50%** of the data — immune to outliers.

### By-Hand Example

```
  Data: [1, 3, 5, 7, 9, 11, 13]    n = 7

  Mean x̄ = (1+3+5+7+9+11+13)/7 = 49/7 = 7.0

  Deviations from mean:
    (1-7)² = 36
    (3-7)² = 16
    (5-7)² =  4
    (7-7)² =  0
    (9-7)² =  4
    (11-7)²= 16
    (13-7)²= 36
              ───
    Sum    = 112

  Sample variance  s² = 112 / (7-1) = 112/6 = 18.67
  Std deviation    s  = √18.67 ≈ 4.32

  Q1 = median of [1, 3, 5] = 3
  Q3 = median of [9, 11, 13] = 11
  IQR = 11 - 3 = 8
```

---

## 5. Correlation

Measures **linear relationship** between two variables.

### Pearson Correlation Coefficient (r)

```
  r = Σᵢ (xᵢ - x̄)(yᵢ - ȳ)
      ─────────────────────────────
      √[Σᵢ (xᵢ - x̄)²] · √[Σᵢ (yᵢ - ȳ)²]

  Equivalently:

  r = Cov(X, Y) / (σ_X · σ_Y)

  where Cov(X,Y) = (1/(n-1)) Σᵢ (xᵢ - x̄)(yᵢ - ȳ)
```

**Properties:**
- -1 ≤ r ≤ +1
- r = +1: perfect positive linear relationship
- r = -1: perfect negative linear relationship
- r = 0: no linear relationship (could still be non-linear!)

```
  r ≈ +1           r ≈ 0             r ≈ -1          r = 0 (but related!)

   y│    ··          y│  · ·  ·        y│··               y│    ·
    │   ··            │ ·  · ·          │ ··               │  ·   ·
    │  ··             │·  ·  ·          │  ··              │ ·     ·
    │ ··              │ · ·  ·          │   ··             │·       ·
    │··               │  ·· ·           │    ··            │ ·     ·
    └──────x          └──────x          └──────x           └──────x
                                                          (parabola)
```

### Spearman Rank Correlation (ρ)

Measures **monotonic** relationships (not just linear). Works on ranks instead of raw values.

```
  Step 1: Replace each value with its rank (1, 2, 3, ...)
  Step 2: Compute Pearson r on the ranks

  Shortcut formula (when no tied ranks):

  ρ = 1 - (6 · Σᵢ dᵢ²) / (n(n² - 1))

  where dᵢ = rank(xᵢ) - rank(yᵢ)
```

**When to use which:**
| | Pearson r | Spearman ρ |
|---|---|---|
| Measures | Linear relationship | Monotonic relationship |
| Assumes | Both variables ~normal, continuous | Any ordinal or continuous |
| Sensitive to outliers | Yes | No (uses ranks) |

### By-Hand Example: Pearson r

```
  Data (n=5):
  x: [1, 2, 3, 4, 5]    x̄ = 3
  y: [2, 4, 5, 4, 8]    ȳ = 4.6

  i │  xᵢ  │  yᵢ  │ xᵢ-x̄ │ yᵢ-ȳ  │ (xᵢ-x̄)(yᵢ-ȳ) │ (xᵢ-x̄)² │ (yᵢ-ȳ)²
  ──┼──────┼──────┼──────┼───────┼───────────────┼─────────┼────────
  1 │  1   │  2   │ -2   │ -2.6  │    5.2        │   4     │  6.76
  2 │  2   │  4   │ -1   │ -0.6  │    0.6        │   1     │  0.36
  3 │  3   │  5   │  0   │  0.4  │    0.0        │   0     │  0.16
  4 │  4   │  4   │  1   │ -0.6  │   -0.6        │   1     │  0.36
  5 │  5   │  8   │  2   │  3.4  │    6.8        │   4     │ 11.56
  ──┼──────┼──────┼──────┼───────┼───────────────┼─────────┼────────
                            Sums:│   12.0        │  10     │ 19.20

  r = 12.0 / (√10 · √19.20)
    = 12.0 / (3.162 · 4.382)
    = 12.0 / 13.856
    = 0.866

  Strong positive linear correlation.
```

---

## 6. Outlier Detection

### Method 1: IQR Rule (Tukey's Fences)

```
  Lower fence = Q1 - 1.5 · IQR
  Upper fence = Q3 + 1.5 · IQR

  Any point outside [lower fence, upper fence] is an outlier.
```

**Why 1.5?** For normally distributed data, this captures ~99.3% of values.
The 1.5 multiplier is a convention by John Tukey — not derived from first principles
but works well empirically.

```
  ◄──outlier──┤ Q1 ├─────IQR─────┤ Q3 ├──outlier──►
              │    │              │    │
  ◄──1.5·IQR──┤    ├──────────────┤    ├──1.5·IQR──►
```

### Method 2: Z-Score

```
  z = (x - μ) / σ

  Convention: |z| > 3  →  outlier  (more than 3 standard deviations from mean)
  Sometimes:  |z| > 2  is used for stricter filtering
```

**Limitation:** Z-score uses mean and std, which are themselves distorted by outliers.
This is a catch-22. For heavily contaminated data, use IQR or robust methods instead.

### By-Hand Example: Both Methods

```
  Data: [10, 12, 11, 13, 12, 11, 50]    n = 7

  ── IQR Method ──
  Sorted: [10, 11, 11, 12, 12, 13, 50]

  Q1 = 11 (median of [10, 11, 11])
  Q3 = 13 (median of [12, 13, 50])
  IQR = 13 - 11 = 2

  Lower fence = 11 - 1.5(2) = 8
  Upper fence = 13 + 1.5(2) = 16

  50 > 16  →  OUTLIER

  ── Z-Score Method ──
  μ = (10+12+11+13+12+11+50)/7 = 119/7 = 17.0
  σ = √[(1/7)·Σ(xᵢ-17)²]

  Deviations²: 49 + 25 + 36 + 16 + 25 + 36 + 1089 = 1276
  σ = √(1276/7) = √182.3 = 13.5

  z(50) = (50 - 17) / 13.5 = 2.44

  |z| = 2.44 < 3  →  NOT an outlier by the |z|>3 rule!

  Notice: The outlier inflated μ and σ so much that the z-score
  method failed to catch it. The IQR method is more robust here.
```

> **Key Intuition:** The IQR method is generally more robust than Z-scores
> because quartiles are not affected by extreme values. Use Z-scores when
> your data is approximately normal and not heavily contaminated.

---

## 7. Visualization Types

### Which Plot for Which Question

```
┌──────────────────────┬────────────────────┬──────────────────────────┐
│ Question             │ Plot Type          │ Data Types               │
├──────────────────────┼────────────────────┼──────────────────────────┤
│ Distribution shape?  │ Histogram          │ 1 numerical              │
│                      │ KDE plot           │ 1 numerical (smooth)     │
│                      │ Box plot           │ 1 numerical              │
├──────────────────────┼────────────────────┼──────────────────────────┤
│ Relationship?        │ Scatter plot       │ 2 numerical              │
│                      │ Line plot          │ numerical over time      │
│                      │ Heatmap            │ correlation matrix       │
├──────────────────────┼────────────────────┼──────────────────────────┤
│ Comparison?          │ Bar chart          │ categorical vs numerical │
│                      │ Grouped box plot   │ categorical vs numerical │
├──────────────────────┼────────────────────┼──────────────────────────┤
│ Composition?         │ Stacked bar        │ parts of a whole         │
│                      │ Pie chart          │ few categories only      │
├──────────────────────┼────────────────────┼──────────────────────────┤
│ Many variables?      │ Pair plot          │ all numerical combos     │
│                      │ Parallel coords    │ high-dimensional         │
└──────────────────────┴────────────────────┴──────────────────────────┘
```

### Anatomy of a Box Plot

```
                          ┌─── maximum (or upper fence)
                          │
                     ─────┤
                    │     │
                    │     │ ─── Q3 (75th percentile)
                    │     │
                    ├─────┤ ─── Q2 / median (50th)
                    │     │
                    │     │ ─── Q1 (25th percentile)
                    │     │
                     ─────┤
                          │
                          └─── minimum (or lower fence)

              ○               ─── outlier (beyond 1.5·IQR)
```

### Anatomy of a Histogram

```
  Count
   ▲
  8│     ██
  7│     ██
  6│  ██ ██
  5│  ██ ██ ██
  4│  ██ ██ ██
  3│  ██ ██ ██ ██
  2│  ██ ██ ██ ██ ██
  1│  ██ ██ ██ ██ ██ ██
   └──────────────────────► Value
     10  20  30  40  50  60
         ◄──bin width──►
```

Bin width matters:
- Too few bins: lose detail (undersmoothing)
- Too many bins: noisy (oversmoothing)
- Rule of thumb: √n bins, or Sturges' rule k = 1 + 3.322·log₁₀(n)

---

## 8. By-Hand EDA Example

### Dataset: 5 Houses

```
  id │ sqft │ bedrooms │ city    │ price($k)
  ───┼──────┼──────────┼─────────┼──────────
   1 │ 1200 │    2     │ Austin  │   250
   2 │ 1500 │    3     │ Austin  │   310
   3 │ 1800 │    3     │ Dallas  │   280
   4 │ 2200 │    4     │ Dallas  │   420
   5 │ 3500 │    5     │ Austin  │   650
```

**Step 1: Identify data types**
```
  sqft      → numerical, continuous
  bedrooms  → numerical, discrete
  city      → categorical, nominal
  price     → numerical, continuous (this is our target)
```

**Step 2: Central tendency and spread for `price`**
```
  Mean  = (250 + 310 + 280 + 420 + 650) / 5 = 1910 / 5 = 382.0
  Sorted prices: [250, 280, 310, 420, 650]
  Median = 310  (3rd of 5)
  Mode   = none (all unique)

  Mean > Median → right-skewed (pulled by the 650 house)

  Variance s² = [(250-382)² + (310-382)² + (280-382)² + (420-382)² + (650-382)²] / 4
              = [17424 + 5184 + 10404 + 1444 + 71824] / 4
              = 106280 / 4
              = 26570

  Std dev s = √26570 = 162.97
```

**Step 3: Correlation between sqft and price**
```
  sqft: [1200, 1500, 1800, 2200, 3500]   x̄ = 2040
  price: [250, 310, 280, 420, 650]       ȳ = 382

  Σ(xᵢ-x̄)(yᵢ-ȳ) = (-840)(-132) + (-540)(-72) + (-240)(-102)
                   + (160)(38) + (1460)(268)
                 = 110880 + 38880 + 24480 + 6080 + 391280
                 = 571600

  Σ(xᵢ-x̄)² = 705600 + 291600 + 57600 + 25600 + 2131600 = 3212000
  Σ(yᵢ-ȳ)² = 17424 + 5184 + 10404 + 1444 + 71824 = 106280

  r = 571600 / √(3212000 · 106280)
    = 571600 / √(341388960000)
    = 571600 / 584457
    = 0.978

  Very strong positive correlation: bigger houses cost more.
```

**Step 4: Outlier check on price**
```
  Sorted: [250, 280, 310, 420, 650]
  Q1 = 280,  Q3 = 420,  IQR = 140

  Lower fence = 280 - 1.5(140) = 280 - 210 = 70
  Upper fence = 420 + 1.5(140) = 420 + 210 = 630

  650 > 630  →  marginally an outlier (worth investigating)
```

---

## 9. What to Look for in the Application Lab

| Theory concept | What you will see in code |
|---|---|
| Data types | `df.dtypes`, `df.info()` |
| Central tendency | `df.describe()` gives mean, median (50%), quartiles |
| Distribution shape | `df['col'].hist()`, `sns.histplot()` |
| Correlation | `df.corr()`, `sns.heatmap(df.corr())` |
| Outliers | `sns.boxplot()`, manual IQR filtering |
| Scatter relationships | `sns.scatterplot(x='sqft', y='price')` |
| Pair-wise overview | `sns.pairplot(df)` |

**Questions to ask yourself during the lab:**
1. What types are my features? Do I need to encode any?
2. Are any features heavily skewed? Should I transform them?
3. Which features correlate most strongly with my target?
4. Are there outliers that might distort my model?
5. Are any features highly correlated with each other? (multicollinearity risk)
