# Logistic Regression — Full Mathematical Derivation
> From the sigmoid function to cross-entropy loss to multi-class extension, every formula derived from scratch.

## Table of Contents
1. [Why Linear Regression Fails for Classification](#1-why-linear-regression-fails-for-classification) — The fundamental problem
2. [The Sigmoid Function](#2-the-sigmoid-function) — Derivation and properties
3. [The Logistic Regression Model](#3-the-logistic-regression-model) — Probability interpretation
4. [The Decision Boundary](#4-the-decision-boundary) — Where the model draws the line
5. [Log-Loss / Binary Cross-Entropy](#5-log-loss--binary-cross-entropy) — Cost function derivation
6. [Gradient Descent for Logistic Regression](#6-gradient-descent-for-logistic-regression) — Weight update rule
7. [Multi-Class Extension](#7-multi-class-extension) — Softmax and one-vs-rest
8. [Classification Metrics](#8-classification-metrics) — Accuracy, precision, recall, F1, ROC/AUC
9. [By-Hand Example](#9-by-hand-example) — Full logistic regression on 5 data points
10. [What to Look for in the Application Lab](#10-what-to-look-for-in-the-application-lab)

---

## 1. Why Linear Regression Fails for Classification

### The Problem

For classification, we want to predict a **probability** P(y=1|X) between 0 and 1.
Linear regression outputs any real number from -inf to +inf.

```
  Linear regression for classification:

  ŷ = w₀ + w₁x

   ŷ│
  2 │                          ·····
    │                     ····
  1 │─ ─ ─ ─ ─ ─ ─ ─····─ ─ ─ ─ ─ ─    ← should max out at 1
    │              ····
  0 │─ ─ ─ ─ ····─ ─ ─ ─ ─ ─ ─ ─ ─ ─   ← should min out at 0
    │      ····
 -1 │ ····
    └────────────────────────────────► x

  Problems:
  1. Predictions > 1 and < 0 (not valid probabilities)
  2. Sensitive to outliers (one extreme point shifts the line)
  3. Assumes linear relationship between x and probability
```

### The Solution: Wrap It in a Sigmoid

Instead of predicting y directly, predict the **log-odds** linearly,
then transform to a probability:

```
  Linear:     z = w₀ + w₁x₁ + w₂x₂ + ... + wₘxₘ     (any real number)
  Sigmoid:    ŷ = σ(z) = 1 / (1 + e⁻ᶻ)                (bounded 0 to 1)
```

---

## 2. The Sigmoid Function

### Definition

```
  ┌───────────────────────────────────┐
  │                                   │
  │   σ(z) = 1 / (1 + e⁻ᶻ)          │
  │                                   │
  └───────────────────────────────────┘
```

### The Curve

```
  σ(z)
  1.0 │                          ─────────
      │                      ···
      │                   ··
      │                 ·
  0.5 │─ ─ ─ ─ ─ ─ ─ · ─ ─ ─ ─ ─ ─ ─ ─    ← σ(0) = 0.5
      │              ·
      │            ··
      │         ···
  0.0 │─────────
      └───────────────┼───────────────── z
                      0

  z → -∞:  σ(z) → 0    (e⁻ᶻ → ∞, so 1/(1+∞) → 0)
  z = 0:   σ(z) = 0.5  (e⁰ = 1, so 1/(1+1) = 0.5)
  z → +∞:  σ(z) → 1    (e⁻ᶻ → 0, so 1/(1+0) → 1)
```

### Key Properties

**1. Output is always between 0 and 1** — perfect for probability.

**2. Symmetric around 0.5:**
```
  σ(-z) = 1 - σ(z)

  Proof:
    σ(-z) = 1 / (1 + eᶻ)
           = eᶻ / (eᶻ + eᶻ·eᶻ)  ... actually, simpler:
           = 1 / (1 + eᶻ)
           = (1 + e⁻ᶻ - 1) / ... let's do it directly:

    1 - σ(z) = 1 - 1/(1+e⁻ᶻ)
             = (1+e⁻ᶻ-1) / (1+e⁻ᶻ)
             = e⁻ᶻ / (1+e⁻ᶻ)
             = 1 / (eᶻ+1)
             = 1 / (1+eᶻ)
             = σ(-z)  ✓
```

**3. Beautiful derivative (this is why sigmoid is special):**

```
  σ'(z) = σ(z) · (1 - σ(z))

  Derivation using quotient rule:

    σ(z) = (1 + e⁻ᶻ)⁻¹

    σ'(z) = -1 · (1 + e⁻ᶻ)⁻² · (-e⁻ᶻ)        [chain rule]
          = e⁻ᶻ / (1 + e⁻ᶻ)²

    Now factor:
          = [1 / (1 + e⁻ᶻ)] · [e⁻ᶻ / (1 + e⁻ᶻ)]
          = σ(z) · [e⁻ᶻ / (1 + e⁻ᶻ)]

    Note: e⁻ᶻ / (1 + e⁻ᶻ) = (1 + e⁻ᶻ - 1) / (1 + e⁻ᶻ) = 1 - 1/(1+e⁻ᶻ) = 1 - σ(z)

    Therefore:  σ'(z) = σ(z) · (1 - σ(z))  ✓
```

The derivative is maximized at z=0 where σ(0)=0.5: σ'(0) = 0.5 · 0.5 = 0.25

```
  σ'(z)
  0.25│          ·
      │        ·   ·
      │      ·       ·
      │    ·           ·
      │  ·               ·
      │·                   ·
  0.0 │─────────────────────── z
                 0
  Maximum gradient at z=0, where the model is most "uncertain"
```

> **Key Intuition:** The derivative σ'(z) = σ(1-σ) means the gradient is largest
> when the model is uncertain (σ ≈ 0.5) and smallest when it is confident
> (σ ≈ 0 or σ ≈ 1). This makes learning self-regulating: the model adjusts
> weights more aggressively when it is unsure, and barely changes when confident.

---

## 3. The Logistic Regression Model

### Probability Interpretation

```
  z = w₀ + w₁x₁ + w₂x₂ + ... + wₘxₘ    (linear combination)

  P(y = 1 | X) = σ(z) = 1 / (1 + e⁻ᶻ)

  P(y = 0 | X) = 1 - σ(z) = 1 / (1 + eᶻ)
```

The model outputs a **probability**, and we convert to a class label with a threshold:

```
  ŷ_class = 1    if P(y=1|X) ≥ 0.5    (equivalently, if z ≥ 0)
          = 0    if P(y=1|X) < 0.5    (equivalently, if z < 0)
```

### Log-Odds (Logit) Interpretation

What is z actually measuring? Rearrange the sigmoid:

```
  σ(z) = p    where p = P(y=1|X)

  p = 1 / (1 + e⁻ᶻ)
  p(1 + e⁻ᶻ) = 1
  1 + e⁻ᶻ = 1/p
  e⁻ᶻ = 1/p - 1 = (1-p)/p
  eᶻ = p/(1-p)
  z = ln(p/(1-p))

  ┌─────────────────────────────────────────────┐
  │                                             │
  │  z = ln(p/(1-p)) = ln(odds)  = "log-odds"  │
  │                                             │
  │  The linear part models the LOG-ODDS,       │
  │  not the probability directly.              │
  └─────────────────────────────────────────────┘
```

**Odds and log-odds examples:**

```
  Probability    Odds (p/(1-p))    Log-odds (z)
  ───────────    ──────────────    ────────────
     0.1            0.111            -2.20
     0.2            0.25             -1.39
     0.5            1.0               0.00
     0.8            4.0               1.39
     0.9            9.0               2.20
     0.99          99.0               4.60
```

> **Key Intuition:** A weight w₁ = 0.5 means: for each unit increase in x₁,
> the **log-odds** of y=1 increase by 0.5. In odds terms: the odds are
> multiplied by e^0.5 ≈ 1.65 (65% increase in odds). This is NOT the same
> as a 0.5 increase in probability.

---

## 4. The Decision Boundary

The decision boundary is where P(y=1|X) = 0.5, which means z = 0.

### Single Feature

```
  z = w₀ + w₁x = 0
  x = -w₀/w₁              ← a single threshold point

  Example: w₀ = -3, w₁ = 1
  Boundary at x = 3

  P(y=1)
  1.0 │                      ─────
      │                  ···
      │               ··
  0.5 │─ ─ ─ ─ ─ ─ · ─ ─ ─ ─ ─ ─
      │           ·
      │        ··
  0.0 │─────···
      └──────────┼──────────────► x
                 3
              boundary
```

### Two Features

```
  z = w₀ + w₁x₁ + w₂x₂ = 0

  Solving for x₂:
    x₂ = -(w₀ + w₁x₁) / w₂    ← a line in 2D space

  Example: w₀ = -3, w₁ = 1, w₂ = 1
  Boundary: x₁ + x₂ = 3

   x₂│
     │╲  P(y=1) region
   3 │ ╲
     │  ╲
     │   ╲  ← decision boundary (line)
     │    ╲
     │     ╲
   0 │──────╲────────► x₁
     0       3
        P(y=0) region
```

> **Key Intuition:** Logistic regression always creates a **linear** decision
> boundary (a line in 2D, a plane in 3D, a hyperplane in higher dimensions).
> If the true boundary between classes is curved, logistic regression will
> underfit. This is its main limitation.

---

## 5. Log-Loss / Binary Cross-Entropy

### Why Not Use MSE?

If we use MSE with the sigmoid, the cost function becomes **non-convex**
(multiple local minima). Gradient descent could get stuck.

```
  MSE with sigmoid:                    Log-loss with sigmoid:

  J│ ·                                 J│ \
   │  · ·     ·                         │  \
   │     · · ·                          │   \
   │                                    │    \
   │        · ·                         │     \
   │           · ·                      │      \_____
   └──────────────── w                  └──────────────── w
    Non-convex! Local minima           Convex! One global minimum
```

### Derivation from Maximum Likelihood

We want to find weights that make the **observed labels most probable**.

For a single sample:
```
  P(yᵢ | Xᵢ; w) = ŷᵢʸⁱ · (1 - ŷᵢ)¹⁻ʸⁱ

  where ŷᵢ = σ(Xᵢw)

  When yᵢ = 1:  P = ŷᵢ¹ · (1-ŷᵢ)⁰ = ŷᵢ         (want ŷᵢ high)
  When yᵢ = 0:  P = ŷᵢ⁰ · (1-ŷᵢ)¹ = 1-ŷᵢ        (want ŷᵢ low)
```

For all n samples (assuming independence):
```
  Likelihood:      L(w) = Πᵢ₌₁ⁿ ŷᵢʸⁱ · (1-ŷᵢ)¹⁻ʸⁱ

  Log-likelihood:  ℓ(w) = Σᵢ₌₁ⁿ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
```

We maximize log-likelihood, or equivalently **minimize the negative** log-likelihood:

```
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  J(w) = -(1/n) Σᵢ₌₁ⁿ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]   │
  │                                                              │
  │  Binary Cross-Entropy / Log-Loss                             │
  └──────────────────────────────────────────────────────────────┘

  where ŷᵢ = σ(Xᵢw)
```

### Understanding the Loss Per Sample

```
  When y = 1:  loss = -log(ŷ)
  When y = 0:  loss = -log(1-ŷ)

  Loss
   ▲
   │ \
   │  \    -log(ŷ) when y=1         -log(1-ŷ) when y=0
   │   \                                     /
   │    \                                  /
   │     \                               /
   │      \                            /
   │       \___                    ___/
   │           ────────    ────────
  0│
   └──────────────────────────────── ŷ
   0           0.5              1

  When y=1: model pays INFINITE cost for predicting ŷ→0 (confident and WRONG)
            model pays ZERO cost for predicting ŷ→1 (confident and RIGHT)
  When y=0: mirror image
```

> **Key Intuition:** Log-loss punishes confident wrong predictions **severely**
> (approaching infinity), while rewarding confident correct predictions.
> This is much harsher than MSE, which treats a wrong prediction of 0.9
> the same whether the model was trying to predict 0 or 1.

---

## 6. Gradient Descent for Logistic Regression

### Computing the Gradient

We need ∂J/∂wⱼ. This is where the beautiful sigmoid derivative pays off.

```
  J(w) = -(1/n) Σᵢ [yᵢ·log(σ(zᵢ)) + (1-yᵢ)·log(1-σ(zᵢ))]

  where zᵢ = Xᵢw

  Step 1: derivative of J with respect to σ:
    ∂J/∂σ = -(1/n) Σᵢ [yᵢ/σ(zᵢ) - (1-yᵢ)/(1-σ(zᵢ))]

  Step 2: derivative of σ with respect to z (we derived this!):
    ∂σ/∂z = σ(z)(1-σ(z))

  Step 3: derivative of z with respect to wⱼ:
    ∂z/∂wⱼ = xᵢⱼ

  Chain rule: ∂J/∂wⱼ = ∂J/∂σ · ∂σ/∂z · ∂z/∂wⱼ

  = -(1/n) Σᵢ [yᵢ/σᵢ - (1-yᵢ)/(1-σᵢ)] · σᵢ(1-σᵢ) · xᵢⱼ

  = -(1/n) Σᵢ [yᵢ(1-σᵢ) - (1-yᵢ)σᵢ] · xᵢⱼ

  = -(1/n) Σᵢ [yᵢ - yᵢσᵢ - σᵢ + yᵢσᵢ] · xᵢⱼ

  = -(1/n) Σᵢ [yᵢ - σᵢ] · xᵢⱼ

  = (1/n) Σᵢ [σ(zᵢ) - yᵢ] · xᵢⱼ

  = (1/n) Σᵢ (ŷᵢ - yᵢ) · xᵢⱼ
```

The result is remarkably simple:

```
  ┌─────────────────────────────────────────────┐
  │                                             │
  │  ∂J/∂wⱼ = (1/n) Σᵢ (ŷᵢ - yᵢ) · xᵢⱼ       │
  │                                             │
  │  This is IDENTICAL in form to the           │
  │  linear regression gradient!                │
  │                                             │
  │  The only difference: ŷᵢ = σ(Xᵢw)          │
  │  instead of ŷᵢ = Xᵢw                       │
  └─────────────────────────────────────────────┘
```

### The Weight Update Rule

```
  wⱼ := wⱼ - η · (1/n) Σᵢ (ŷᵢ - yᵢ) · xᵢⱼ

  In vector form:
  w := w - η · (1/n) Xᵀ(σ(Xw) - y)
```

### The Algorithm

```
  LOGISTIC REGRESSION GRADIENT DESCENT:
  ──────────────────────────────────────
  1. Initialize: w = [0, 0, ..., 0]
  2. Repeat until convergence:
     a. Compute linear output:   z = Xw
     b. Apply sigmoid:           ŷ = σ(z) = 1/(1+e⁻ᶻ)
     c. Compute gradient:        ∇J = (1/n) Xᵀ(ŷ - y)
     d. Update weights:          w = w - η · ∇J
  3. Return w
```

> **Key Intuition:** The gradient (ŷᵢ - yᵢ)·xᵢⱼ has a natural interpretation.
> If yᵢ=1 but ŷᵢ=0.2 (underpredicting), then (ŷᵢ-yᵢ) = -0.8 is negative,
> so wⱼ gets INCREASED (making the prediction higher). If xᵢⱼ is large,
> the correction is larger — the model "blames" features with big values more.

---

## 7. Multi-Class Extension

### One-vs-Rest (OvR)

Train K separate binary classifiers, one per class.

```
  3-class problem: {cat, dog, bird}

  Classifier 1: cat vs {dog, bird}     → P(cat|X)
  Classifier 2: dog vs {cat, bird}     → P(dog|X)
  Classifier 3: bird vs {cat, dog}     → P(bird|X)

  Final prediction: argmax of the three probabilities

  Example:
    P(cat|X) = 0.7,  P(dog|X) = 0.2,  P(bird|X) = 0.4
    Predict: cat

  Note: probabilities don't sum to 1 (they come from separate models)
```

### Softmax Regression (Multinomial Logistic)

Generalize sigmoid to K classes, with probabilities that sum to 1.

```
  For each class k, compute a linear score:
    zₖ = wₖ₀ + wₖ₁x₁ + ... + wₖₘxₘ

  Softmax function:

  ┌──────────────────────────────────────┐
  │                                      │
  │  P(y=k|X) = eᶻᵏ / Σⱼ₌₁ᴷ eᶻʲ       │
  │                                      │
  │  Softmax: normalizes K scores        │
  │  into a probability distribution     │
  └──────────────────────────────────────┘
```

**Example with 3 classes:**

```
  Scores: z = [2.0, 1.0, 0.5]

  Exponentials: e² = 7.389,  e¹ = 2.718,  e⁰·⁵ = 1.649
  Sum = 7.389 + 2.718 + 1.649 = 11.756

  P(class 1) = 7.389 / 11.756 = 0.628
  P(class 2) = 2.718 / 11.756 = 0.231
  P(class 3) = 1.649 / 11.756 = 0.140
                                 ─────
                          Sum =  0.999 ≈ 1.0  ✓
```

**Softmax loss (categorical cross-entropy):**

```
  J(W) = -(1/n) Σᵢ Σₖ yᵢₖ · log(P(y=k|Xᵢ))

  where yᵢₖ = 1 if sample i belongs to class k, 0 otherwise (one-hot)
```

> **Key Intuition:** When K=2, softmax reduces to sigmoid. Softmax is
> the natural generalization. With sigmoid you need one set of weights;
> with softmax for K classes you need K sets of weights (one per class).

---

## 8. Classification Metrics

### The Confusion Matrix

```
                        Predicted
                    Positive  Negative
               ┌──────────┬──────────┐
  Actual  Pos  │    TP     │    FN    │
               ├──────────┼──────────┤
  Actual  Neg  │    FP     │    TN    │
               └──────────┴──────────┘

  TP = True Positive   (predicted 1, actually 1) ← correct
  TN = True Negative   (predicted 0, actually 0) ← correct
  FP = False Positive  (predicted 1, actually 0) ← Type I error
  FN = False Negative  (predicted 0, actually 1) ← Type II error
```

### Accuracy

```
  Accuracy = (TP + TN) / (TP + TN + FP + FN)

  = correct predictions / all predictions
```

**Problem:** Misleading with imbalanced data (95% accuracy by always predicting majority).

### Precision

```
  Precision = TP / (TP + FP)

  "Of everything I PREDICTED positive, how many were actually positive?"

  High precision = few false alarms
  Important when: cost of false positive is high
  Example: spam filter (don't want to delete real email)
```

### Recall (Sensitivity, True Positive Rate)

```
  Recall = TP / (TP + FN)

  "Of everything that IS positive, how many did I catch?"

  High recall = few missed positives
  Important when: cost of false negative is high
  Example: cancer detection (don't want to miss a case)
```

### F1 Score

```
  F1 = 2 · (Precision · Recall) / (Precision + Recall)

  = harmonic mean of precision and recall
```

The harmonic mean penalizes imbalance: if either precision or recall is low,
F1 will be low.

```
  Precision = 0.9, Recall = 0.1:
    Arithmetic mean = 0.50  ← misleadingly high
    F1 (harmonic)   = 0.18  ← captures the imbalance
```

### By-Hand Metrics Example

```
  Confusion matrix:
                 Pred +    Pred -
  Actual +  │     40    │    10    │   (50 actual positives)
  Actual -  │      5    │    45    │   (50 actual negatives)

  Accuracy  = (40 + 45) / 100 = 0.85
  Precision = 40 / (40 + 5) = 40/45 = 0.889
  Recall    = 40 / (40 + 10) = 40/50 = 0.800
  F1        = 2 · (0.889 · 0.800) / (0.889 + 0.800)
            = 2 · 0.711 / 1.689
            = 0.842
```

### ROC Curve and AUC

The **ROC curve** (Receiver Operating Characteristic) plots True Positive Rate
vs False Positive Rate at **every possible threshold**.

```
  TPR = Recall = TP / (TP + FN)
  FPR = FP / (FP + TN)

  TPR (Recall)
  1.0 │·····················─────
      │                   ··
      │                 ··
      │               ··        ← ROC curve (good model)
      │             ··
      │           ··
      │        ···
      │     ···
      │  ···              ╱ ← diagonal (random guess, AUC=0.5)
      │···              ╱
  0.0 │───────────────────── FPR
      0                   1.0
```

**AUC** (Area Under the ROC Curve):
```
  AUC = 1.0  →  perfect classifier
  AUC = 0.5  →  random guess (diagonal line)
  AUC < 0.5  →  worse than random (predictions are inverted)
```

> **Key Intuition:** AUC measures the probability that the model ranks a
> random positive sample higher than a random negative sample. It is
> threshold-independent, making it ideal for comparing models without
> committing to a specific threshold.

### Choosing a Threshold

The default threshold is 0.5, but you can adjust it:

```
  Lower threshold (e.g., 0.3):
    More things predicted positive → Higher recall, lower precision
    Use when: missing a positive is costly (cancer screening)

  Higher threshold (e.g., 0.7):
    Fewer things predicted positive → Lower recall, higher precision
    Use when: false positives are costly (criminal conviction)

  Precision
  1.0 │·
      │ ·
      │  ··
      │    ··
      │      ···
      │         ····
      │             ·····
  0.0 │                  ·····
      └───────────────────────── Recall
      0                       1.0
      Precision-Recall tradeoff curve
```

---

## 9. By-Hand Example

### Dataset (5 samples, 1 feature)

```
  i │  x  │  y (class)
  ──┼─────┼───────────
  1 │  1  │    0
  2 │  2  │    0
  3 │  3  │    1
  4 │  4  │    1
  5 │  5  │    1
```

### Step 1: Initialize weights

```
  w₀ = 0,  w₁ = 0,  η = 0.1
```

### Step 2: Forward pass (iteration 1)

```
  z = w₀ + w₁·x = 0 + 0·x = 0  for all samples

  ŷ = σ(0) = 0.5  for all samples

  i │  x │  y  │  z   │  ŷ = σ(z)  │  ŷ - y
  ──┼────┼─────┼──────┼────────────┼───────
  1 │  1 │  0  │  0   │   0.5      │  +0.5
  2 │  2 │  0  │  0   │   0.5      │  +0.5
  3 │  3 │  1  │  0   │   0.5      │  -0.5
  4 │  4 │  1  │  0   │   0.5      │  -0.5
  5 │  5 │  1  │  0   │   0.5      │  -0.5
```

### Step 3: Compute gradients

```
  ∂J/∂w₀ = (1/5) Σ (ŷᵢ - yᵢ) · 1
          = (1/5)(0.5 + 0.5 - 0.5 - 0.5 - 0.5)
          = (1/5)(-0.5)
          = -0.1

  ∂J/∂w₁ = (1/5) Σ (ŷᵢ - yᵢ) · xᵢ
          = (1/5)(0.5·1 + 0.5·2 + (-0.5)·3 + (-0.5)·4 + (-0.5)·5)
          = (1/5)(0.5 + 1.0 - 1.5 - 2.0 - 2.5)
          = (1/5)(-4.5)
          = -0.9
```

### Step 4: Update weights

```
  w₀ = 0 - 0.1·(-0.1) = 0.01
  w₁ = 0 - 0.1·(-0.9) = 0.09
```

### Step 5: Forward pass (iteration 2)

```
  i │  x │  z = 0.01 + 0.09x │  ŷ = σ(z)  │  y  │  ŷ-y
  ──┼────┼────────────────────┼────────────┼─────┼──────
  1 │  1 │  0.10              │  0.525     │  0  │ +0.525
  2 │  2 │  0.19              │  0.547     │  0  │ +0.547
  3 │  3 │  0.28              │  0.570     │  1  │ -0.430
  4 │  4 │  0.37              │  0.591     │  1  │ -0.409
  5 │  5 │  0.46              │  0.613     │  1  │ -0.387

  The model is starting to predict higher probabilities for larger x.
  Not there yet, but moving in the right direction.
```

### Step 6: Compute log-loss (iteration 2)

```
  J = -(1/5)[0·log(0.525) + 1·log(1-0.525)
           + 0·log(0.547) + 1·log(1-0.547)
           + 1·log(0.570) + 0·log(1-0.570)
           + 1·log(0.591) + 0·log(1-0.591)
           + 1·log(0.613) + 0·log(1-0.613)]

    = -(1/5)[log(0.475) + log(0.453) + log(0.570) + log(0.591) + log(0.613)]

    = -(1/5)[(-0.744) + (-0.792) + (-0.562) + (-0.527) + (-0.489)]

    = -(1/5)(-3.114) = 0.623

  (After many more iterations, loss will decrease and weights will converge
   to approximately w₀ ≈ -3.5, w₁ ≈ 1.3, giving a decision boundary at x ≈ 2.7)
```

### Final Converged Model (after many iterations)

```
  Approximate solution: w₀ ≈ -3.5,  w₁ ≈ 1.3

  Decision boundary: z = 0  →  -3.5 + 1.3x = 0  →  x = 2.69

  i │  x │  z = -3.5+1.3x │  ŷ = σ(z) │  y │ Correct?
  ──┼────┼─────────────────┼───────────┼────┼─────────
  1 │  1 │  -2.2           │  0.10     │  0 │   Yes
  2 │  2 │  -0.9           │  0.29     │  0 │   Yes
  3 │  3 │   0.4           │  0.60     │  1 │   Yes
  4 │  4 │   1.7           │  0.85     │  1 │   Yes
  5 │  5 │   3.0           │  0.95     │  1 │   Yes

  All 5 samples classified correctly!

  Confusion matrix:
               Pred +  Pred -
  Actual +  │    3   │   0    │
  Actual -  │    0   │   2    │

  Accuracy  = 5/5 = 1.0
  Precision = 3/3 = 1.0
  Recall    = 3/3 = 1.0
  F1        = 1.0
```

---

## 10. What to Look for in the Application Lab

| Theory concept | What you will see in code |
|---|---|
| Sigmoid | `from scipy.special import expit` or built into `LogisticRegression` |
| Logistic model | `LogisticRegression()` |
| Probabilities | `model.predict_proba(X)` |
| Decision boundary | `model.predict(X)` uses threshold=0.5 internally |
| Log-loss | `log_loss(y_true, y_pred_proba)` |
| Weights | `model.coef_`, `model.intercept_` |
| Confusion matrix | `confusion_matrix(y_true, y_pred)` |
| Precision/Recall/F1 | `classification_report(y_true, y_pred)` |
| ROC/AUC | `roc_auc_score()`, `roc_curve()` |
| Multi-class | `LogisticRegression(multi_class='multinomial')` |
| Regularization | `LogisticRegression(C=1.0)` — note: C = 1/α (inverse!) |

**Questions to ask yourself during the lab:**
1. What are the learned weights? Which features push toward class 1 vs class 0?
2. Is accuracy misleading because of class imbalance? Check precision/recall.
3. What does the ROC curve look like? Is AUC close to 1?
4. If I change the threshold from 0.5 to 0.3, how does precision/recall shift?
5. For multi-class: are some classes easier to classify than others?
6. What does `model.predict_proba()` show for borderline cases near the decision boundary?
