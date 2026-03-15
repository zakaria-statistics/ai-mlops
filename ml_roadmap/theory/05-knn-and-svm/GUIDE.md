# Lab 05 Theory — K-Nearest Neighbors & Support Vector Machines
> Distance-based classifiers: find neighbors or find the widest street between classes.

## Table of Contents
1. [KNN: Core Idea](#1-knn-core-idea) — Classify by proximity
2. [Distance Metrics](#2-distance-metrics) — Euclidean, Manhattan, Minkowski
3. [Choosing K](#3-choosing-k) — The bias-variance knob
4. [Weighted KNN](#4-weighted-knn) — Closer neighbors matter more
5. [Curse of Dimensionality](#5-curse-of-dimensionality) — Why KNN breaks in high dimensions
6. [KNN By-Hand Example](#6-knn-by-hand-example) — 2D classification step by step
7. [SVM: Core Idea](#7-svm-core-idea) — Maximum margin classification
8. [The Hyperplane Equation](#8-the-hyperplane-equation) — w-dot-x + b = 0
9. [The Margin](#9-the-margin) — Why 2/||w|| and how to maximize it
10. [Hard Margin Optimization](#10-hard-margin-optimization) — The constrained problem
11. [Soft Margin and the C Parameter](#11-soft-margin-and-the-c-parameter) — Allowing mistakes
12. [The Kernel Trick](#12-the-kernel-trick) — Non-linear boundaries without explicit mapping
13. [Why Scaling Is Critical](#13-why-scaling-is-critical) — For both KNN and SVM
14. [SVM By-Hand Example](#14-svm-by-hand-example) — Linear SVM margin in 2D
15. [What to Look for in the Application Lab](#15-what-to-look-for-in-the-application-lab)

---

## 1. KNN: Core Idea

KNN is the simplest possible classifier: store all training data, and when a
new point arrives, find the K closest training points and let them vote.

```
   Training data          New point ?

   o o          x x        ?
   o   o      x   x
     o o        x x

   "Find K nearest neighbors to ?, majority class wins"
```

There is NO training phase. All computation happens at prediction time.
This makes KNN a **lazy learner** — it memorizes rather than generalizes.

---

## 2. Distance Metrics

### Euclidean Distance (L2)

The straight-line distance between two points in n-dimensional space:

```
d(x, y) = sqrt( sum_i (xi - yi)^2 )

In 2D:  d = sqrt( (x1-y1)^2 + (x2-y2)^2 )
```

Example: points A=(1,2) and B=(4,6)

```
d = sqrt( (4-1)^2 + (6-2)^2 )
  = sqrt( 9 + 16 )
  = sqrt(25)
  = 5
```

### Manhattan Distance (L1)

Sum of absolute differences — distance along axes only (like city blocks):

```
d(x, y) = sum_i |xi - yi|

In 2D:  d = |x1-y1| + |x2-y2|
```

Example: same points A=(1,2) and B=(4,6)

```
d = |4-1| + |6-2| = 3 + 4 = 7
```

### Minkowski Distance (General Form)

Both Euclidean and Manhattan are special cases of Minkowski:

```
d(x, y) = ( sum_i |xi - yi|^p )^(1/p)

p = 1  -->  Manhattan
p = 2  -->  Euclidean
p -> inf -> Chebyshev (max of |xi - yi|)
```

### When to Use Which

```
Euclidean (p=2):  Default, works well when features are similar scale
Manhattan (p=1):  Better for high-dimensional data, more robust to outliers
Chebyshev (p=inf): When worst-case dimension difference matters
```

> **Key Intuition:** Manhattan distance treats all dimensions equally in
> terms of contribution. Euclidean amplifies large differences in any
> single dimension (because of squaring). This matters when features
> have different scales.

---

## 3. Choosing K

K is the number of neighbors that vote. It controls the bias-variance tradeoff:

```
K=1:  Decision boundary follows every point (high variance, low bias)
      Memorizes noise. Overfits.

K=N:  Every point votes. Always predicts majority class.
      Ignores all structure. Underfits.

K=sqrt(N): Common starting heuristic.
```

```
  K=1 boundary          K=15 boundary         K=N boundary
  (very jagged)         (smoother)            (straight line)

  +---------+           +---------+           +---------+
  |o/x x    |           |ooo|x x  |           |         |
  |o o/x x  |           |o o|x x  |           | ALL = x |
  |o/ x x   |           |ooo|x x  |           |         |
  +---------+           +---------+           +---------+
  Overfitting           Good balance          Underfitting
```

Rules:
- Use ODD K for binary classification (avoids ties)
- Use cross-validation to find optimal K
- Typical range: K = 3 to 20

---

## 4. Weighted KNN

Standard KNN: each of K neighbors gets 1 vote regardless of distance.
Weighted KNN: closer neighbors get more influence.

```
Weight function:   w_i = 1 / d(x_query, x_i)

Prediction:  class = argmax_c  sum_{i in neighbors}  w_i * I(y_i = c)
```

Example with K=3 neighbors:

```
Neighbor 1:  class=A, distance=1.0  -->  weight = 1/1.0 = 1.00
Neighbor 2:  class=B, distance=2.0  -->  weight = 1/2.0 = 0.50
Neighbor 3:  class=B, distance=5.0  -->  weight = 1/5.0 = 0.20

Vote for A: 1.00
Vote for B: 0.50 + 0.20 = 0.70

Standard KNN: B wins (2 vs 1)
Weighted KNN: A wins (1.00 vs 0.70)
```

> **Key Intuition:** Weighted KNN is more robust to large K values. Even
> with K=100, distant points contribute almost nothing. This makes the
> choice of K less critical.

---

## 5. Curse of Dimensionality

As dimensions increase, distances become meaningless. Here is why:

### The Volume Explosion

Consider a unit hypercube [0,1]^d. What fraction of the volume is within
0.1 of the boundary?

```
Interior volume = (1 - 2*0.1)^d = 0.8^d

d=1:   0.8^1  = 0.80   -->  20% near boundary
d=10:  0.8^10 = 0.107  -->  89% near boundary
d=50:  0.8^50 = 0.000014 --> 99.999% near boundary
d=100: effectively 0    --> ALL points are near boundary
```

### Distance Concentration

In high dimensions, the ratio of farthest to nearest neighbor approaches 1:

```
For random points in d dimensions:

    max_distance / min_distance  -->  1  as d --> infinity

This means: ALL points are approximately equidistant.
KNN cannot distinguish "near" from "far".
```

### Concrete Example

With 100 uniformly distributed points:

```
d=2:   Need points in a circle of radius ~0.1 to cover 1% of space
d=10:  Need radius ~0.8 to cover 1% of space
d=100: Need radius ~0.955 to cover 1% of space
       (neighbors are basically the ENTIRE dataset)
```

> **Key Intuition:** In high dimensions, your "nearest" neighbor isn't
> meaningfully closer than your farthest point. KNN needs dimensionality
> reduction (PCA, feature selection) to work beyond ~20 features.

---

## 6. KNN By-Hand Example

**Dataset:** 6 points in 2D, classify the query point Q=(3,3).

```
Point  x1  x2  Class
A      1   1   0
B      2   1   0
C      1   3   0
D      4   4   1
E      5   3   1
F      5   5   1
```

```
  x2
  5 |          . F(1)
  4 |       . D(1)
  3 | . C(0)  ? Q    . E(1)
  2 |
  1 | . A(0)  . B(0)
    +--+--+--+--+--+-- x1
       1  2  3  4  5
```

**Step 1: Compute Euclidean distances to Q=(3,3)**

```
d(Q,A) = sqrt((3-1)^2 + (3-1)^2) = sqrt(4+4) = sqrt(8) = 2.83
d(Q,B) = sqrt((3-2)^2 + (3-1)^2) = sqrt(1+4) = sqrt(5) = 2.24
d(Q,C) = sqrt((3-1)^2 + (3-3)^2) = sqrt(4+0) = sqrt(4) = 2.00
d(Q,D) = sqrt((3-4)^2 + (3-4)^2) = sqrt(1+1) = sqrt(2) = 1.41
d(Q,E) = sqrt((3-5)^2 + (3-3)^2) = sqrt(4+0) = sqrt(4) = 2.00
d(Q,F) = sqrt((3-5)^2 + (3-5)^2) = sqrt(4+4) = sqrt(8) = 2.83
```

**Step 2: Sort by distance**

```
Rank  Point  Distance  Class
1     D      1.41      1
2     C      2.00      0     (tie)
3     E      2.00      1     (tie)
4     B      2.24      0
5     A      2.83      0     (tie)
6     F      2.83      1     (tie)
```

**Step 3: Classify with different K**

```
K=1:  {D}         --> Class 1 (1 vote for 1)
K=3:  {D, C, E}   --> Class 1 (2 votes for 1, 1 for 0)
K=5:  {D,C,E,B,A} --> Class 0 (3 votes for 0, 2 for 1)
```

K=1 and K=3 predict class 1. K=5 predicts class 0.
The answer depends on K — this is why cross-validation matters.

---

## 7. SVM: Core Idea

SVM finds the **maximum margin** hyperplane that separates two classes.

```
  Margin = the "street" between classes
  Support vectors = points sitting on the edge of the street

    |<--- margin --->|
    |                |
  o | o              | x   x
  o |   o    street  |   x
  o | o              | x   x
    |                |
    decision boundary
    (middle of street)
```

Why maximum margin? It gives the best generalization. A wider street means
the classifier is more confident and more robust to new data.

---

## 8. The Hyperplane Equation

A hyperplane in n dimensions is defined by:

```
w . x + b = 0

where:
  w = weight vector (normal to the hyperplane), determines orientation
  x = input point
  b = bias term, determines offset from origin
  w . x = dot product = w1*x1 + w2*x2 + ... + wn*xn
```

```
  In 2D, the hyperplane is a LINE:
  w1*x1 + w2*x2 + b = 0

  The vector w is PERPENDICULAR to this line.

       w
       ^
       |
  -----+------  <-- the line w.x + b = 0
       |
```

For a point x:
```
w . x + b > 0   -->  one side (class +1)
w . x + b < 0   -->  other side (class -1)
w . x + b = 0   -->  ON the hyperplane
```

---

## 9. The Margin

The distance from a point x to the hyperplane w.x + b = 0 is:

```
distance = |w . x + b| / ||w||

where ||w|| = sqrt(w1^2 + w2^2 + ... + wn^2)  (norm of w)
```

We define the margin boundaries as:

```
w . x + b = +1   (positive class boundary)
w . x + b = -1   (negative class boundary)
w . x + b =  0   (decision boundary, in the middle)
```

The distance between the two margin boundaries:

```
margin = 2 / ||w||
```

**Derivation:**

```
The +1 boundary is at distance 1/||w|| from the decision boundary.
The -1 boundary is at distance 1/||w|| from the decision boundary.
Total margin = 1/||w|| + 1/||w|| = 2/||w||.
```

> **Key Intuition:** To MAXIMIZE the margin (2/||w||), we need to
> MINIMIZE ||w||. This is why the SVM optimization minimizes ||w||^2.
> We use ||w||^2 instead of ||w|| because it's easier to differentiate
> (no square root) and has the same optimum.

---

## 10. Hard Margin Optimization

The SVM optimization problem (hard margin — no misclassifications allowed):

```
MINIMIZE:    (1/2) * ||w||^2

SUBJECT TO:  yi * (w . xi + b) >= 1    for all i = 1, ..., n

where:
  yi = +1 or -1  (class label)
  xi = training point
```

What does the constraint mean?

```
If yi = +1:   w . xi + b >= +1   (positive points on or beyond +1 boundary)
If yi = -1:   w . xi + b <= -1   (negative points on or beyond -1 boundary)

Combined:  yi * (w . xi + b) >= 1
```

```
            yi*(w.xi+b) >= 1
                |
  NEGATIVE      |  margin  |      POSITIVE
  class -1      |          |      class +1
                |          |
  o  o  o  [sv] |          | [sv]  x  x  x
  o  o          |          |          x  x
                |          |
            w.x+b=-1   w.x+b=0   w.x+b=+1

  [sv] = support vector (constraint is ACTIVE: equality holds)
```

Points where yi*(w.xi+b) = 1 exactly are the **support vectors**.
They are the only points that matter — moving any other point doesn't
change the solution.

---

## 11. Soft Margin and the C Parameter

Real data is rarely linearly separable. Soft margin SVM allows some
points to violate the margin using **slack variables** xi_i:

```
MINIMIZE:    (1/2) * ||w||^2  +  C * sum_i(xi_i)

SUBJECT TO:  yi * (w . xi + b) >= 1 - xi_i
             xi_i >= 0    for all i
```

What the slack variable means:

```
xi_i = 0:       Point is correctly classified, on or beyond margin
0 < xi_i < 1:   Point is inside the margin but correctly classified
xi_i = 1:       Point is exactly on the decision boundary
xi_i > 1:       Point is MISCLASSIFIED
```

```
  Correctly      Inside        On boundary    Misclassified
  classified     margin        xi_i = 1       xi_i > 1
  xi_i = 0       0<xi_i<1

  o              o       |     o    |         |    o
  (beyond        (in the |          |         |  (wrong
   margin)        street)|          |         |   side)
```

### The C Parameter

```
C = infinity:  No violations allowed (hard margin)
C = large:     Few violations, narrow margin (risk of overfitting)
C = small:     Many violations OK, wide margin (risk of underfitting)
```

```
  Large C (tight fit)          Small C (relaxed fit)

  o o|   |x x x               o  o  |         |  x  x
  o  |   |  x x               o  o  |         |  x  x
  o o|   |x   x                  o  |   o     |  x
  narrow margin                wide margin (some o's inside)
```

> **Key Intuition:** C controls the tradeoff between a wide margin
> (good generalization) and few classification errors (good training
> accuracy). It's the regularization knob for SVM.

---

## 12. The Kernel Trick

When data is not linearly separable, we can map it to a higher dimension
where it IS separable. The kernel trick does this without explicitly
computing the high-dimensional coordinates.

### The Problem

```
  1D data: not linearly separable

  ---x--x--o--o--o--x--x---

  No single point can split x from o.
```

### The Mapping Idea

```
  Map to 2D: phi(x) = (x, x^2)

  x^2 |     x           x
      |      x         x
      |        o  o  o
      |
      +--x--x--o--o--o--x--x-- x

  NOW a line can separate them in 2D!
```

### Why "Trick"?

SVM only needs DOT PRODUCTS between points: phi(xi) . phi(xj).
A kernel function K(xi, xj) computes this dot product directly
without ever computing phi(x):

```
K(xi, xj) = phi(xi) . phi(xj)

We never compute phi(x) explicitly!
This saves massive computation in high/infinite dimensions.
```

### Common Kernels

```
Linear:       K(x, y) = x . y
              (no mapping, standard dot product)

Polynomial:   K(x, y) = (x . y + c)^d
              Maps to polynomial feature space of degree d

RBF (Gaussian): K(x, y) = exp(-gamma * ||x - y||^2)
              Maps to INFINITE dimensional space
              gamma = 1/(2*sigma^2)

              gamma large: tight influence, complex boundary
              gamma small: wide influence, smooth boundary
```

### RBF Kernel Intuition

```
K(x, y) = exp(-gamma * ||x - y||^2)

When ||x - y|| = 0:  K = exp(0) = 1     (identical points)
When ||x - y|| is large: K -> 0          (distant points)

gamma controls how fast K decays with distance:

  K
  1 |\.
    | \  .
    |  \    .     gamma = 10 (sharp drop)
    |   \      .
    |    '--------  gamma = 0.1 (slow drop)
  0 +------------- distance
```

> **Key Intuition:** RBF kernel measures SIMILARITY between points.
> Each support vector creates a "bump" of influence. The decision
> boundary is the contour where these bumps balance between classes.

---

## 13. Why Scaling Is Critical

Both KNN and SVM use distances. Unscaled features break them.

### Example: Predicting house quality

```
Feature 1: Square footage   (range: 500 - 5000)
Feature 2: Number of rooms  (range: 1 - 10)

House A: (1000 sqft, 5 rooms)
House B: (1010 sqft, 1 room)
House C: (1000 sqft, 9 rooms)

Euclidean distance A->B: sqrt((1010-1000)^2 + (1-5)^2)
                       = sqrt(100 + 16) = sqrt(116) = 10.8

Euclidean distance A->C: sqrt((1000-1000)^2 + (9-5)^2)
                       = sqrt(0 + 16) = sqrt(16) = 4.0

B is "farther" from A than C because sqft dominates.
But the sqft difference (10) is tiny, while room difference (4) is huge!
```

### After StandardScaler (z-score normalization):

```
z = (x - mean) / std

Now both features have mean=0, std=1.
Distances reflect ACTUAL differences proportionally.
```

### For SVM specifically:

```
Without scaling: The margin is dominated by large-scale features.
The SVM essentially ignores small-scale features.

With scaling: All features contribute equally to the margin.
```

> **Key Intuition:** ALWAYS scale your data before KNN or SVM.
> StandardScaler or MinMaxScaler. No exceptions.

---

## 14. SVM By-Hand Example

**Dataset:** 4 points in 2D, find the maximum margin hyperplane.

```
Point  x1  x2  Class (y)
A      1   1   -1
B      2   1   -1
C      3   3   +1
D      4   3   +1
```

```
  x2
  4 |
  3 |          . C(+1)  . D(+1)
  2 |
  1 | . A(-1)  . B(-1)
    +--+--+--+--+-- x1
       1  2  3  4
```

**By inspection:** The separating line passes between (2,1)/(3,3) region.

**Step 1: Identify candidate support vectors**

The closest points between classes are B=(2,1) and C=(3,3).
These are likely support vectors.

**Step 2: The midpoint and direction**

```
Midpoint of B and C: M = ((2+3)/2, (1+3)/2) = (2.5, 2.0)
The line must pass through or near M.

Direction B->C: (3-2, 3-1) = (1, 2)
This is the direction of w (normal to the hyperplane).

So w = (1, 2) (proportional — we'll normalize).
```

**Step 3: Find the hyperplane equation**

```
Hyperplane: w1*x1 + w2*x2 + b = 0
Using w = (1, 2) and point M = (2.5, 2):

1*2.5 + 2*2.0 + b = 0
2.5 + 4.0 + b = 0
b = -6.5

Hyperplane: x1 + 2*x2 - 6.5 = 0
```

**Step 4: Verify constraints**

```
For B=(2,1), y=-1:  y*(w.x+b) = -1*(2+2-6.5) = -1*(-2.5) = 2.5 >= 1  OK
For C=(3,3), y=+1:  y*(w.x+b) = +1*(3+6-6.5) = +1*(2.5)  = 2.5 >= 1  OK
For A=(1,1), y=-1:  y*(w.x+b) = -1*(1+2-6.5) = -1*(-3.5) = 3.5 >= 1  OK
For D=(4,3), y=+1:  y*(w.x+b) = +1*(4+6-6.5) = +1*(3.5)  = 3.5 >= 1  OK
```

**Step 5: Compute margin**

```
||w|| = sqrt(1^2 + 2^2) = sqrt(5) = 2.236

Margin = 2 / ||w|| = 2 / 2.236 = 0.894

But we should normalize so constraints are tight (= 1) for SVs.
Scale: w' = w/2.5, b' = b/2.5

w' = (0.4, 0.8), b' = -2.6

Check B: -1*(0.4*2 + 0.8*1 - 2.6) = -1*(0.8+0.8-2.6) = -1*(-1) = 1  exact
Check C: +1*(0.4*3 + 0.8*3 - 2.6) = +1*(1.2+2.4-2.6) = +1*(1)  = 1  exact

||w'|| = sqrt(0.16 + 0.64) = sqrt(0.80) = 0.894
Margin = 2 / 0.894 = 2.236

Support vectors: B and C (constraints active at = 1)
```

---

## 15. What to Look for in the Application Lab

When you move to the coding lab:

1. **KNN:** Watch how K affects the decision boundary — plot it for K=1,5,15
2. **KNN:** Compare StandardScaler vs no scaling — accuracy will differ dramatically
3. **SVM:** Try `kernel='linear'` first, then `kernel='rbf'`
4. **SVM:** Tune C with cross-validation — see how it changes the margin
5. **SVM:** Tune gamma for RBF — too large overfits, too small underfits
6. **Both:** Always scale first with `StandardScaler` or `MinMaxScaler`
7. **Both:** Compare which does better on your specific dataset — there's no universal winner
8. **SVM:** Look at `model.support_vectors_` to see which points define the boundary
