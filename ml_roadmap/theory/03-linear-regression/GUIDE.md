# Linear Regression — Full Mathematical Derivation
> From the cost function to gradient descent to regularization, every formula derived from scratch.

## Table of Contents
1. [The Model](#1-the-model) — What linear regression computes
2. [The Cost Function (OLS)](#2-the-cost-function-ols) — MSE derivation and intuition
3. [The Normal Equation](#3-the-normal-equation) — Closed-form solution
4. [Gradient Descent](#4-gradient-descent) — Iterative optimization from scratch
5. [Learning Rate](#5-learning-rate) — Why it matters and how to choose
6. [R-Squared (R²)](#6-r-squared-r) — Measuring goodness of fit
7. [Ridge Regression (L2)](#7-ridge-regression-l2) — Adding the squared penalty
8. [Lasso Regression (L1)](#8-lasso-regression-l1) — Adding the absolute penalty
9. [ElasticNet](#9-elasticnet) — Combining L1 and L2
10. [Geometric Intuition of Regularization](#10-geometric-intuition-of-regularization) — Diamond vs circle
11. [By-Hand Example](#11-by-hand-example) — Full regression on 4 data points
12. [What to Look for in the Application Lab](#12-what-to-look-for-in-the-application-lab)

---

## 1. The Model

Linear regression predicts a continuous output as a weighted sum of inputs.

**Single feature:**
```
  ŷ = w₁x + w₀

  where:
    x  = input feature
    w₁ = weight (slope) — how much ŷ changes per unit change in x
    w₀ = bias (intercept) — the prediction when x = 0
    ŷ  = predicted value
```

**Multiple features:**
```
  ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₘxₘ

  In vector form:
    ŷ = Xw

  where:
    X = [n × (m+1)] matrix  (n samples, m features + 1 bias column of 1s)
    w = [(m+1) × 1] vector  (m weights + 1 bias)
```

**Expanded matrix form:**
```
  ┌ ŷ₁ ┐   ┌ 1  x₁₁  x₁₂ ··· x₁ₘ ┐   ┌ w₀ ┐
  │ ŷ₂ │   │ 1  x₂₁  x₂₂ ··· x₂ₘ │   │ w₁ │
  │  ⋮  │ = │ ⋮   ⋮    ⋮       ⋮   │ · │ w₂ │
  │ ŷₙ │   │ 1  xₙ₁  xₙ₂ ··· xₙₘ │   │  ⋮ │
  └     ┘   └                       ┘   │ wₘ │
                                        └    ┘
```

The goal: find the weight vector **w** that makes the predictions **ŷ** as close
as possible to the true values **y**.

---

## 2. The Cost Function (OLS)

**Ordinary Least Squares (OLS)** defines "close" using squared errors.

### Mean Squared Error

```
  J(w) = (1/2n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²

       = (1/2n) Σᵢ₌₁ⁿ (yᵢ - Xᵢw)²

  In matrix form:
    J(w) = (1/2n) (y - Xw)ᵀ(y - Xw)
```

**Why squared?**
1. Errors can be positive or negative — squaring makes all positive
2. Large errors are penalized more than small ones (quadratic growth)
3. The function is **differentiable everywhere** (needed for gradient descent)
4. Produces a **convex** function — one global minimum, no local traps

**Why the 1/2?** Pure convenience: it cancels with the 2 from the power rule
when we take the derivative. Has no effect on where the minimum is.

```
  J(w)
   ▲
   │\                    /
   │ \                  /
   │  \               /        ← convex: any local min IS the global min
   │   \            /
   │    \         /
   │     \      /
   │      \   /
   │       \_/  ← minimum at w*
   └────────────────────► w
```

---

## 3. The Normal Equation

The closed-form solution: set the gradient to zero and solve algebraically.

### Derivation

Start with the cost function in matrix form:

```
  J(w) = (1/2n) (y - Xw)ᵀ(y - Xw)
```

Expand:

```
  J(w) = (1/2n) (yᵀy - yᵀXw - wᵀXᵀy + wᵀXᵀXw)

  Since yᵀXw is a scalar, yᵀXw = (yᵀXw)ᵀ = wᵀXᵀy

  J(w) = (1/2n) (yᵀy - 2wᵀXᵀy + wᵀXᵀXw)
```

Take the gradient with respect to w and set to zero:

```
  ∂J/∂w = (1/2n) (-2Xᵀy + 2XᵀXw) = 0

  Drop the constants:
    -Xᵀy + XᵀXw = 0
    XᵀXw = Xᵀy
```

Solve for w:

```
  ┌─────────────────────────────────┐
  │                                 │
  │   w* = (XᵀX)⁻¹ Xᵀy            │
  │                                 │
  │   The Normal Equation           │
  └─────────────────────────────────┘
```

**When to use / not use:**
| | Normal Equation | Gradient Descent |
|---|---|---|
| Time complexity | O(m³) for matrix inverse | O(m·n·iterations) |
| Works when m > 10,000 features | Slow | Fine |
| Works when n > 1,000,000 samples | Fine | Fine |
| Requires feature scaling | No | Yes |
| Always finds exact solution | Yes (if XᵀX invertible) | Approximate |

---

## 4. Gradient Descent

When the normal equation is too expensive, we optimize **iteratively**.

### The Gradient

The gradient tells us the **direction of steepest increase** of J(w).
We move **opposite** to the gradient (downhill).

For a single weight wⱼ:

```
  ∂J/∂wⱼ = (1/n) Σᵢ₌₁ⁿ (ŷᵢ - yᵢ) · xᵢⱼ

  In words:
    For each sample, compute (prediction error) × (feature value)
    Average over all samples
```

**Derivation for single-feature case:**

```
  J(w₀, w₁) = (1/2n) Σ (yᵢ - w₀ - w₁xᵢ)²

  ∂J/∂w₀ = (1/n) Σ (ŷᵢ - yᵢ) · 1        [derivative of -w₀ with respect to w₀ = -1]
          = (1/n) Σ (ŷᵢ - yᵢ)              → average error

  ∂J/∂w₁ = (1/n) Σ (ŷᵢ - yᵢ) · xᵢ        [derivative of -w₁xᵢ with respect to w₁ = -xᵢ]
          = (1/n) Σ (error · feature)        → weighted average error
```

In vector form for all weights simultaneously:

```
  ∇J(w) = (1/n) Xᵀ(Xw - y)
```

### The Update Rule

```
  ┌──────────────────────────────────┐
  │                                  │
  │   w := w - η · ∇J(w)            │
  │                                  │
  │   Component-wise:                │
  │   wⱼ := wⱼ - η · ∂J/∂wⱼ        │
  │                                  │
  │   where η = learning rate        │
  └──────────────────────────────────┘
```

### The Algorithm

```
  GRADIENT DESCENT:
  ─────────────────
  1. Initialize weights: w = [0, 0, ..., 0]  (or small random)
  2. Repeat until convergence:
     a. Compute predictions:  ŷ = Xw
     b. Compute errors:       e = ŷ - y
     c. Compute gradient:     ∇J = (1/n) Xᵀe
     d. Update weights:       w = w - η · ∇J
     e. Check convergence:    if |J_new - J_old| < ε, stop
  3. Return w
```

### Visualizing the Descent

```
  J(w₀, w₁) contour plot         Path of gradient descent
  (bird's eye view of the bowl)

       w₁                              w₁
        ▲                               ▲
        │  ╭─────────╮                  │  ╭─────────╮
        │ ╭┤         ├╮                 │ ╭┤    ·    ├╮
        │╭┤│  ╭───╮  │├╮               │╭┤│  · ───╮ │├╮
        ││││ ╭┤ · ├╮ ││││              ││││·╭┤   ├╮ ││││
        │╰┤│ ╰┤   ├╯ │├╯               │╰┤│·╰┤   ├╯ │├╯
        │ ╰┤  ╰───╯  ├╯               │ ╰┤ ·╰───╯  ├╯
        │  ╰─────────╯                │  ╰──·──────╯
        └──────────────► w₀           └──────·──────► w₀
                                           start
     · = minimum (target)
                                     Each · is one iteration
```

### Variants

```
  Batch Gradient Descent:     Use ALL n samples per update
                              Slow but stable path

  Stochastic GD (SGD):       Use 1 random sample per update
                              Fast but noisy path

  Mini-batch GD:              Use b samples (e.g., 32) per update
                              Best of both worlds — standard in practice
```

---

## 5. Learning Rate

The learning rate η controls **step size** along the gradient.

```
  η too small:                η just right:              η too large:

  J│                          J│                          J│
   │\                          │\                          │
   │ ·                         │ ·                         │·       ·
   │  ·                        │  ·                        │ ·     ·
   │   ·                       │   ·                       │  ·   ·
   │    ·                      │    ·                      │   · ·
   │     ·                     │     ·_                    │    ·
   │      · · · ·              │                           │     ·
   │           (still going)   │  (converged)              │  (diverging!)
   └──────────── iter          └──────────── iter          └──────────── iter
```

**Typical starting values:** η = 0.01, 0.001, 0.1

**Learning rate schedules:** Start large, decay over time.
```
  η(t) = η₀ / (1 + decay_rate · t)
```

---

## 6. R-Squared (R²)

### Derivation

R² measures what **fraction of the variance** in y is explained by the model.

```
  Total sum of squares (how much y varies):
    SS_tot = Σᵢ (yᵢ - ȳ)²

  Residual sum of squares (what the model could NOT explain):
    SS_res = Σᵢ (yᵢ - ŷᵢ)²

  R² = 1 - SS_res / SS_tot
```

**Interpretation:**

```
  R² = 1.0    →  SS_res = 0       →  perfect predictions
  R² = 0.0    →  SS_res = SS_tot  →  model is no better than predicting ȳ
  R² < 0.0    →  SS_res > SS_tot  →  model is WORSE than predicting ȳ (possible!)
```

> **Key Intuition:** R² compares your model to the dumbest possible baseline:
> always predicting the mean. R² = 0.85 means your model explains 85% of the
> variance that the mean-baseline cannot. It does NOT mean 85% of predictions
> are correct.

### Adjusted R² (penalizes adding useless features)

```
  R²_adj = 1 - [(1-R²)(n-1)] / [n-m-1]

  where n = samples, m = features

  Adding a useless feature: R² can only increase (or stay same)
  but R²_adj can DECREASE, exposing the useless feature.
```

---

## 7. Ridge Regression (L2)

### The Problem Ridge Solves

When features are correlated or there are many features relative to samples,
OLS weights can become very large and unstable. The model overfits.

### The Ridge Cost Function

Add a **penalty** proportional to the **squared magnitude** of weights:

```
  J_ridge(w) = (1/2n) Σ (yᵢ - ŷᵢ)²  +  α · Σⱼ wⱼ²
                    ╰──── OLS loss ────╯    ╰── L2 penalty ──╯

  In matrix form:
    J_ridge(w) = (1/2n)(y-Xw)ᵀ(y-Xw) + α · wᵀw
```

**α** (alpha) is the regularization strength:
- α = 0: pure OLS (no regularization)
- α → ∞: all weights → 0 (extreme regularization, underfitting)

### Why It Shrinks Weights

The gradient of the Ridge cost:

```
  ∂J_ridge/∂wⱼ = (1/n) Σᵢ (ŷᵢ - yᵢ)·xᵢⱼ  +  2α·wⱼ
                    ╰──── OLS gradient ────╯    ╰─ pull toward 0 ─╯
```

The 2α·wⱼ term **always pushes weights toward zero**. The larger a weight gets,
the stronger the push back. This prevents any single weight from dominating.

### Ridge Normal Equation

```
  w_ridge = (XᵀX + 2nα·I)⁻¹ Xᵀy

  The αI term adds a positive value to the diagonal of XᵀX,
  which guarantees invertibility (fixes multicollinearity).
```

### Update Rule

```
  wⱼ := wⱼ - η · [(1/n) Σᵢ (ŷᵢ - yᵢ)·xᵢⱼ + 2α·wⱼ]

  Equivalently:
  wⱼ := wⱼ(1 - 2ηα) - η·(1/n) Σᵢ (ŷᵢ - yᵢ)·xᵢⱼ
              ▲
              │
      "weight decay" — each step shrinks wⱼ by factor (1 - 2ηα)
```

> **Key Intuition:** Ridge never makes weights exactly zero. It shrinks them
> all proportionally. Every feature stays in the model, just with reduced
> influence. Use Ridge when you believe all features are somewhat relevant.

---

## 8. Lasso Regression (L1)

### The Lasso Cost Function

Replace the squared penalty with an **absolute value** penalty:

```
  J_lasso(w) = (1/2n) Σ (yᵢ - ŷᵢ)²  +  α · Σⱼ |wⱼ|
                    ╰──── OLS loss ────╯    ╰── L1 penalty ──╯
```

### Why Lasso Zeros Out Weights

The subgradient of |wⱼ|:

```
  ∂|wⱼ|/∂wⱼ = +1   if wⱼ > 0
             = -1   if wⱼ < 0
             ∈ [-1,+1] if wⱼ = 0   (subdifferential)
```

This creates a **constant force** pulling toward zero, regardless of how small
the weight already is:

```
  Ridge pull:  2α·wⱼ   → gets weaker as wⱼ → 0 (never reaches zero)
  Lasso pull:  α·sign(wⱼ) → constant force (can push all the way to zero)

  Ridge:                     Lasso:
  Force                      Force
   ▲                          ▲
   │     /                    │   ┌─── +α
   │   /                      │   │
   │ /                        │   │
   ├──────── wⱼ               ├───┴──── wⱼ
   │/                         │   │
   │                          │   │
   │                          │   └─── -α
```

**Result:** Lasso performs **automatic feature selection**. Unimportant features
get their weights driven to exactly zero and are effectively removed.

> **Key Intuition:** Lasso is like a budget. The α penalty is a fixed "tax"
> per feature used. If a feature does not pay for itself in reduced error,
> its weight gets zeroed out. Ridge is like a proportional tax — it reduces
> everything but never eliminates anything.

---

## 9. ElasticNet

Combines both L1 and L2 penalties:

```
  J_elastic(w) = (1/2n) Σ (yᵢ - ŷᵢ)²  +  α · [ρ·Σ|wⱼ| + (1-ρ)·Σwⱼ²]
                                                  ╰─ L1 ─╯   ╰── L2 ──╯

  where:
    α = overall regularization strength
    ρ = L1 ratio (0 = pure Ridge, 1 = pure Lasso, between = mix)
```

**When to use ElasticNet:**
- Many correlated features: Lasso arbitrarily picks one, Ridge keeps all.
  ElasticNet tends to keep groups of correlated features together.
- When you want some feature selection (L1) but also stability (L2).

---

## 10. Geometric Intuition of Regularization

The regularization penalty defines a **constraint region** in weight space.
The solution is where the OLS contours first touch this constraint.

```
  Lasso (L1): Diamond                 Ridge (L2): Circle

       w₂                                  w₂
        ▲                                   ▲
        │    /╲                             │   ╭──╮
        │   / ╱╲                            │  ╱╭──╮╲
        │  / ╱  ╲                           │ ╱╱    ╲╲
        │ / ╱    ╲                          │╱╱      ╲╲
   ─────╳──╱──────╲──────► w₁         ─────╳╱────────╲╳──► w₁
        │╲ ╲     ╱                          │╲        ╱│
        │ ╲ ╲  ╱                            │ ╲╲    ╱╱
        │  ╲  ╲╱                            │  ╲╰──╯╱
        │   ╲╱                              │   ╰──╯
        │                                   │

   ● = OLS solution (unconstrained)
   ╳ = regularized solution (where contour hits constraint)

   Diamond has CORNERS on axes    Circle is smooth — no axis preference
   → solution likely lands on     → solution lands anywhere on the edge
     a corner (some wⱼ = 0)        (all wⱼ small but non-zero)
```

The OLS loss contours are ellipses centered at the unconstrained optimum.
As you expand the ellipses outward from the constrained solution, the first
contact point is the regularized answer:

- **Lasso diamond:** Corners sit on the axes. The ellipse is very likely to
  first touch a corner, setting one or more weights to exactly zero.
- **Ridge circle:** Smooth surface. The ellipse touches smoothly, giving
  non-zero values to all weights.

---

## 11. By-Hand Example

### Dataset (4 points, 1 feature)

```
  x: [1, 2, 3, 4]
  y: [2, 4, 5, 4]

  Goal: find ŷ = w₀ + w₁·x
```

**Step 1: Set up the design matrix**

```
  X = ┌ 1  1 ┐    y = ┌ 2 ┐
      │ 1  2 │        │ 4 │
      │ 1  3 │        │ 5 │
      └ 1  4 ┘        └ 4 ┘
```

**Step 2: Normal equation  w = (XᵀX)⁻¹Xᵀy**

```
  XᵀX = ┌ 1 1 1 1 ┐ ┌ 1  1 ┐   ┌ 4   10 ┐
        │ 1 2 3 4 │·│ 1  2 │ = │ 10  30 │
        └         ┘ │ 1  3 │   └        ┘
                     └ 1  4 ┘

  Xᵀy = ┌ 1 1 1 1 ┐ ┌ 2 ┐   ┌ 15 ┐
        │ 1 2 3 4 │·│ 4 │ = │ 41 │
        └         ┘ │ 5 │   └    ┘
                     └ 4 ┘

  (XᵀX)⁻¹ = (1/det) · ┌  30  -10 ┐
                        │ -10    4 │
                        └          ┘

  det = 4·30 - 10·10 = 120 - 100 = 20

  (XᵀX)⁻¹ = ┌  1.5  -0.5 ┐
             │ -0.5   0.2 │
             └             ┘

  w = (XᵀX)⁻¹ · Xᵀy = ┌  1.5  -0.5 ┐ ┌ 15 ┐   ┌ 1.5·15 + (-0.5)·41 ┐   ┌ 2.0 ┐
                        │ -0.5   0.2 │·│ 41 │ = │ -0.5·15 + 0.2·41   │ = │ 0.7 │
                        └             ┘ └    ┘   └                     ┘   └     ┘

  Result: ŷ = 2.0 + 0.7x
```

**Step 3: Predictions and R²**

```
  x=1: ŷ = 2.0 + 0.7(1) = 2.7    error = 2 - 2.7 = -0.7
  x=2: ŷ = 2.0 + 0.7(2) = 3.4    error = 4 - 3.4 =  0.6
  x=3: ŷ = 2.0 + 0.7(3) = 4.1    error = 5 - 4.1 =  0.9
  x=4: ŷ = 2.0 + 0.7(4) = 4.8    error = 4 - 4.8 = -0.8

  SS_res = 0.49 + 0.36 + 0.81 + 0.64 = 2.30

  ȳ = (2+4+5+4)/4 = 3.75
  SS_tot = (2-3.75)² + (4-3.75)² + (5-3.75)² + (4-3.75)²
         = 3.0625 + 0.0625 + 1.5625 + 0.0625
         = 4.75

  R² = 1 - 2.30/4.75 = 1 - 0.484 = 0.516

  The model explains about 52% of the variance.
```

**Step 4: One iteration of gradient descent (to verify)**

```
  Initialize: w₀ = 0, w₁ = 0,  η = 0.01

  Predictions: ŷ = [0, 0, 0, 0]
  Errors (ŷ-y): [-2, -4, -5, -4]

  ∂J/∂w₀ = (1/4)·(-2-4-5-4) = -3.75
  ∂J/∂w₁ = (1/4)·[(-2)(1)+(-4)(2)+(-5)(3)+(-4)(4)] = (1/4)·(-41) = -10.25

  Update:
  w₀ = 0 - 0.01·(-3.75) = 0.0375
  w₁ = 0 - 0.01·(-10.25) = 0.1025

  After one step: ŷ = 0.0375 + 0.1025x
  (Still far from the normal equation answer — needs more iterations)
```

---

## 12. What to Look for in the Application Lab

| Theory concept | What you will see in code |
|---|---|
| Linear model | `LinearRegression()`, `model.coef_`, `model.intercept_` |
| Cost function | `mean_squared_error(y_true, y_pred)` |
| Normal equation | Happens inside `model.fit()` for LinearRegression |
| Gradient descent | `SGDRegressor(loss='squared_error')` |
| R² score | `model.score(X, y)` or `r2_score(y_true, y_pred)` |
| Ridge | `Ridge(alpha=1.0)` |
| Lasso | `Lasso(alpha=1.0)` |
| ElasticNet | `ElasticNet(alpha=1.0, l1_ratio=0.5)` |
| Learning rate | `SGDRegressor(eta0=0.01)` |
| Regularization path | Plot R² or MSE vs different alpha values |

**Questions to ask yourself during the lab:**
1. What are the learned weights? Which features have the largest coefficients?
2. Is training R² much higher than test R²? (overfitting signal)
3. Does Ridge/Lasso improve test performance over plain OLS?
4. For Lasso: which features got zeroed out? Does that make domain sense?
5. How sensitive is performance to the choice of α?
