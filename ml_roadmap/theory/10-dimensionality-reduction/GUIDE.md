# 10 — Dimensionality Reduction: PCA and t-SNE
> Compress high-dimensional data while preserving information — eigenvalues, eigenvectors, and projection

## Table of Contents
1. [The Curse of Dimensionality](#1-the-curse)
2. [PCA — Full Derivation](#2-pca)
3. [Choosing the Number of Components](#3-choosing-components)
4. [t-SNE](#4-t-sne)
5. [Feature Selection vs Extraction](#5-selection-vs-extraction)
6. [By-Hand Example: PCA on 2D Data](#6-by-hand-example)

---

## 1. The Curse

As dimensions increase, data becomes sparse and distances lose meaning.

```
1D: 10 points fill a line nicely      ••••••••••
2D: 10 points in a square are sparse   •   •
                                          •  •
                                        •    •
                                          •
                                        •   •
3D: 10 points in a cube are very sparse
100D: 10 points are basically alone — every point is far from every other
```

**Why it matters:**
- KNN breaks down (all distances become similar)
- Models overfit (more features than samples)
- Computation slows down
- Visualization is impossible above 3D

---

## 2. PCA

**Principal Component Analysis** finds the directions of maximum variance and projects data onto them.

### Step-by-Step Algorithm

```
1. Center the data:        X_centered = X - mean(X)
2. Compute covariance:     C = (1/n) Xᵀ X
3. Eigendecomposition:     C = V Λ Vᵀ
4. Sort by eigenvalue:     λ₁ ≥ λ₂ ≥ ... ≥ λₘ
5. Pick top K eigenvectors: V_k = [v₁, v₂, ..., vₖ]
6. Project:                Z = X · V_k
```

### The Math

**Covariance matrix** (m×m, where m = number of features):
```
C = (1/n) Xᵀ X

Cᵢⱼ = covariance between feature i and feature j
Cᵢᵢ = variance of feature i

C is symmetric and positive semi-definite
```

**Eigendecomposition:**
```
Cv = λv

v = eigenvector (direction of a principal component)
λ = eigenvalue (variance explained in that direction)

Key property: eigenvectors are orthogonal (perpendicular to each other)
```

**Explained variance ratio:**
```
Variance explained by component k = λₖ / Σᵢ λᵢ

Cumulative variance = Σᵢ₌₁ᵏ λᵢ / Σᵢ₌₁ᵐ λᵢ
```

> **Key Intuition:** PCA rotates your coordinate system so that the first axis (PC1) points in the direction of maximum variance, the second (PC2) is perpendicular and captures the next most variance, etc. By keeping only the top K components, you compress the data while losing minimal information.

```
Original axes:          PCA axes:
  y│    ·  · ·           PC2│
   │  ·  ··              ╱  │   · ··
   │ ·· ·                   │ · ··  ·
   │· ·                     │··  ·
   └──────── x              └──────── PC1
                           (direction of max spread)
```

---

## 3. Choosing Components

**Rule of thumb:** Keep enough components to explain 95% of variance.

```
Cumulative
Variance
  │                    ___________
  │                ___╱
  │            ___╱
  │         __╱
  │       _╱
  │     _╱
  │   _╱
  │  ╱
  │_╱
  └────────────────────→ # Components
  1  2  3  4  5 ... 50

If 95% is reached at K=5, use 5 components instead of 50.
```

---

## 4. t-SNE

**t-distributed Stochastic Neighbor Embedding** — for visualization only.

### Idea
1. In high-D: compute pairwise similarities using Gaussian distribution
2. In low-D (2D): compute similarities using t-distribution
3. Minimize KL divergence between the two distributions using gradient descent

```
High-D similarity: pᵢⱼ = exp(-||xᵢ-xⱼ||² / 2σ²) / Σ...
Low-D similarity:  qᵢⱼ = (1 + ||yᵢ-yⱼ||²)⁻¹ / Σ...

Minimize: KL(P||Q) = Σᵢⱼ pᵢⱼ · log(pᵢⱼ/qᵢⱼ)
```

### Why t-distribution?
The heavier tails of the t-distribution allow moderate distances in high-D to become larger distances in low-D, making clusters more visible.

### PCA vs t-SNE

| | PCA | t-SNE |
|---|---|---|
| Purpose | Dimensionality reduction | Visualization only |
| Deterministic | Yes | No (different runs → different results) |
| Preserves | Global structure (variance) | Local structure (neighborhoods) |
| Can project new data | Yes (Z = X · V_k) | No (non-parametric) |
| Speed | Fast | Slow (O(n²)) |
| Use for preprocessing | Yes | No |

---

## 5. Selection vs Extraction

```
Feature Selection: pick a SUBSET of original features
  [f1, f2, f3, f4, f5] → [f1, f3, f5]
  (interpretable — you keep real features)

Feature Extraction: create NEW features as combinations
  [f1, f2, f3, f4, f5] → [PC1, PC2]
  where PC1 = 0.3·f1 + 0.5·f2 + 0.1·f3 + ...
  (lower dimension but less interpretable)
```

---

## 6. By-Hand Example

### PCA on 5 Points in 2D → 1D

```
Points: (2,3), (3,5), (5,4), (6,7), (4,6)
```

**Step 1 — Center:**
```
mean_x = (2+3+5+6+4)/5 = 4.0
mean_y = (3+5+4+7+6)/5 = 5.0

Centered:
(-2,-2), (-1,0), (1,-1), (2,2), (0,1)
```

**Step 2 — Covariance matrix:**
```
C = (1/n) Xᵀ X = (1/5) × [sum of outer products]

Var(x) = (4+1+1+4+0)/5 = 2.0
Var(y) = (4+0+1+4+1)/5 = 2.0
Cov(x,y) = (4+0-1+4+0)/5 = 1.4

C = [2.0  1.4]
    [1.4  2.0]
```

**Step 3 — Eigenvalues:**
```
det(C - λI) = 0
(2-λ)² - 1.96 = 0
λ² - 4λ + 2.04 = 0
λ = (4 ± √(16-8.16)) / 2 = (4 ± 2.8) / 2

λ₁ = 3.4    (first principal component — captures most variance)
λ₂ = 0.6    (second — captures the rest)
```

**Step 4 — Eigenvectors:**
```
For λ₁=3.4: (C - 3.4I)v = 0
[-1.4  1.4] v = 0  →  v₁ = [1/√2, 1/√2] = [0.707, 0.707]
[ 1.4 -1.4]

For λ₂=0.6:
v₂ = [-1/√2, 1/√2] = [-0.707, 0.707]
```

**Step 5 — Project onto PC1:**
```
z = X_centered · v₁

(-2,-2) · (0.707, 0.707) = -2.83
(-1, 0) · (0.707, 0.707) = -0.71
( 1,-1) · (0.707, 0.707) =  0.00
( 2, 2) · (0.707, 0.707) =  2.83
( 0, 1) · (0.707, 0.707) =  0.71

Projected 1D: [-2.83, -0.71, 0.00, 2.83, 0.71]
```

**Variance explained:**
```
PC1: λ₁/(λ₁+λ₂) = 3.4/4.0 = 85%
PC2: λ₂/(λ₁+λ₂) = 0.6/4.0 = 15%

With just PC1, we retain 85% of the information.
```

---

## What to Look for in the Application Lab

In the application lab, you'll:
1. Implement PCA from scratch (center, covariance, eigen, project) on MNIST digits
2. Compare with sklearn PCA
3. Plot cumulative explained variance — find the 95% threshold
4. Visualize 64-dimensional digits in 2D using both PCA and t-SNE
5. Use PCA as preprocessing before KNN — speed vs accuracy tradeoff
