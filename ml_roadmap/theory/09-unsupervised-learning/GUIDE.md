# 09 — Unsupervised Learning: Clustering and Anomaly Detection
> Finding structure in data without labels — K-Means, DBSCAN, Hierarchical clustering, and Isolation Forest

## Table of Contents
1. [No Labels — What Can We Do?](#1-no-labels)
2. [K-Means Clustering](#2-k-means-clustering)
3. [DBSCAN](#3-dbscan)
4. [Hierarchical Clustering](#4-hierarchical-clustering)
5. [Choosing K and Evaluating Clusters](#5-choosing-k)
6. [Anomaly Detection](#6-anomaly-detection)
7. [By-Hand Example: K-Means](#7-by-hand-example)

---

## 1. No Labels

In unsupervised learning, we only have X (features) — no y (target).

```
Supervised:    X → y (learn the mapping)
Unsupervised:  X → ??? (find structure)
```

What kind of structure?
- **Clustering:** points that are "similar" belong to the same group
- **Anomaly detection:** points that are "different" from everything else
- **Dimensionality reduction:** find the most informative directions (see Lab 10)

---

## 2. K-Means Clustering

### The Idea

Partition n data points into K clusters, where each point belongs to the cluster with the nearest centroid (center).

### Objective Function

```
Minimize J = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²

where:
  K = number of clusters
  Cₖ = set of points in cluster k
  μₖ = centroid (mean) of cluster k
  ||·||² = squared Euclidean distance
```

> **Key Intuition:** Minimize the total within-cluster distance. Points should be close to their cluster's center.

### Algorithm

```
1. Choose K (number of clusters)
2. Initialize K centroids randomly (or use K-Means++)
3. Repeat until convergence:
   a. ASSIGN: each point to the nearest centroid
      cluster(xᵢ) = argminₖ ||xᵢ - μₖ||²
   b. UPDATE: recalculate each centroid as the mean of its assigned points
      μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ
4. Stop when assignments don't change (or max iterations reached)
```

```
Step 0 (init):     Step 1 (assign):   Step 2 (update):   Step 3 (assign):
  · ·               · ·[A]             · ·                 · ·[A]
 · ·  ★A           · ·  ★A            · · ★A              · · ★A
  ·                  ·[A]               ·                    ·[A]
        · ·               · ·[B]            · ·                  · ·[B]
       ·  ★B             ·  ★B             · ★B                · ★B
         ·                 ·[B]              ·                    ·[B]

★ = centroid       [A/B] = assignment   ★ moved to mean    Converged!
(random init)                           of assigned points
```

### K-Means++ Initialization

Random initialization can lead to bad clusters. K-Means++ spreads initial centroids apart:

```
1. Pick first centroid randomly from data points
2. For each remaining centroid:
   - Compute distance D(x) from each point to nearest existing centroid
   - Pick next centroid with probability proportional to D(x)²
   (far points are more likely to be picked)
3. Repeat until K centroids chosen
```

### Limitations

- Must specify K in advance
- Assumes spherical clusters of similar size
- Sensitive to initialization (run multiple times)
- Can't handle non-convex shapes
- Sensitive to outliers (they pull centroids)

---

## 3. DBSCAN

**Density-Based Spatial Clustering of Applications with Noise**

### Parameters
```
ε (epsilon): radius of neighborhood
minPts: minimum points in neighborhood to be a "core point"
```

### Point Classification
```
Core point:    has ≥ minPts points within distance ε (including itself)
Border point:  has < minPts neighbors but is within ε of a core point
Noise point:   neither core nor border → OUTLIER

  ε=1.0, minPts=3

    ·₁  ·₂  ·₃    ← all within ε of each other → core points (cluster A)
              ·₄   ← within ε of ·₃ only, < minPts neighbors → border of A

                        ·₅   ← far from everything → NOISE
```

### Algorithm
```
1. For each point, count neighbors within ε
2. Mark core points (≥ minPts neighbors)
3. Connect core points that are within ε of each other → same cluster
4. Assign border points to nearest core point's cluster
5. Mark remaining points as noise
```

### Advantages over K-Means
- No need to specify K (finds it automatically)
- Handles arbitrary cluster shapes
- Identifies outliers/noise naturally
- Robust to outliers

### Disadvantages
- Struggles with varying density clusters
- Sensitive to ε and minPts choices
- Doesn't work well in very high dimensions

---

## 4. Hierarchical Clustering

### Agglomerative (Bottom-up)

```
1. Start: each point is its own cluster
2. Find the two closest clusters
3. Merge them into one cluster
4. Repeat until one cluster remains (or desired K reached)
```

### Linkage Methods (how to measure cluster distance)

```
Single linkage:    min distance between any two points in different clusters
                   → tends to create long, chain-like clusters

Complete linkage:  max distance between any two points
                   → tends to create compact, spherical clusters

Average linkage:   average of all pairwise distances
                   → compromise

Ward's method:     minimize increase in total within-cluster variance
                   → tends to create equal-sized clusters (most common)
```

### Dendrogram

```
Height
(distance)
  │
  ├──────────────┐
  │         ┌────┤
  │    ┌────┤    │
  │    │    │    │
  ──┬──┤    │    │
    │  │    │    │
    A  B    C    D  E

Cut at any height to get clusters:
  Cut high → 2 clusters: {A,B,C} and {D,E}
  Cut low  → 4 clusters: {A}, {B,C}, {D}, {E}
```

---

## 5. Choosing K

### Elbow Method (for K-Means)
```
Plot inertia (J = total within-cluster distance) vs K:

Inertia
  │╲
  │  ╲
  │    ╲___
  │        ╲______
  │               ────────
  └────────────────────────→ K
  1   2   3   4   5   6   7

The "elbow" (where improvement slows) suggests optimal K.
Here: K ≈ 3
```

### Silhouette Score

For each point i:
```
a(i) = mean distance to other points in same cluster (cohesion)
b(i) = mean distance to points in nearest other cluster (separation)

s(i) = (b(i) - a(i)) / max(a(i), b(i))

Range: [-1, +1]
  +1: point is well-assigned (far from other clusters)
   0: point is on the boundary
  -1: point is misassigned (closer to another cluster)

Overall score: mean of s(i) across all points
```

---

## 6. Anomaly Detection

### Isolation Forest

> **Key Intuition:** Anomalies are few and different. If you randomly split data with hyperplanes, anomalies get isolated quickly (short path in the tree), while normal points take many splits.

```
Normal point:           Anomaly:
Takes many splits       Isolated quickly
to isolate              (few splits)

  ┌─────────┐            ┌──────────┐
  │·· ·· · ·│            │·· ·  · · │
  │ ·· · · ·│            │ ·· ·· · ·│
  │· ·· · ··│──split──   │· ·       │──split──→ isolated!
  │ · ·· ·· │            │  ★       │
  └─────────┘            └──────────┘
  needs 5+ splits        needs 1-2 splits
  → normal               → anomaly
```

**Anomaly score:** average path length across many random trees. Short path = anomaly.

### Local Outlier Factor (LOF)

Compares local density of a point to local density of its neighbors:
```
LOF(x) ≈ density of neighbors / density of x

LOF ≈ 1: similar density to neighbors (normal)
LOF >> 1: much less dense than neighbors (anomaly)
```

---

## 7. By-Hand Example

### K-Means on 6 Points, K=2

```
Points: A(1,1), B(1.5,2), C(3,4), D(5,7), E(3.5,5), F(4.5,5)
```

**Iteration 0 — Initialize centroids randomly:**
```
μ₁ = A = (1, 1)
μ₂ = D = (5, 7)
```

**Iteration 1 — Assign:**
```
Distance to μ₁(1,1) and μ₂(5,7):

A(1,1):    d₁ = 0,     d₂ = √(16+36) = 7.21  → Cluster 1
B(1.5,2):  d₁ = √(0.25+1) = 1.12,  d₂ = √(12.25+25) = 6.10  → Cluster 1
C(3,4):    d₁ = √(4+9) = 3.61,     d₂ = √(4+9) = 3.61  → Tie! assign to 1
D(5,7):    d₁ = √(16+36) = 7.21,   d₂ = 0  → Cluster 2
E(3.5,5):  d₁ = √(6.25+16) = 4.72, d₂ = √(2.25+4) = 2.50  → Cluster 2
F(4.5,5):  d₁ = √(12.25+16) = 5.31, d₂ = √(0.25+4) = 2.06  → Cluster 2

Cluster 1: {A, B, C}
Cluster 2: {D, E, F}
```

**Iteration 1 — Update centroids:**
```
μ₁ = mean of {A, B, C} = ((1+1.5+3)/3, (1+2+4)/3) = (1.83, 2.33)
μ₂ = mean of {D, E, F} = ((5+3.5+4.5)/3, (7+5+5)/3) = (4.33, 5.67)
```

**Iteration 2 — Reassign:**
```
A(1,1):    d₁ = 1.56, d₂ = 5.62 → Cluster 1 ✓
B(1.5,2):  d₁ = 0.47, d₂ = 4.59 → Cluster 1 ✓
C(3,4):    d₁ = 2.01, d₂ = 2.10 → Cluster 1 ✓ (barely!)
D(5,7):    d₁ = 5.50, d₂ = 1.49 → Cluster 2 ✓
E(3.5,5):  d₁ = 3.12, d₂ = 1.08 → Cluster 2 ✓
F(4.5,5):  d₁ = 3.68, d₂ = 0.69 → Cluster 2 ✓

Same assignments → CONVERGED!
```

**Final clusters:**
```
Cluster 1: {A(1,1), B(1.5,2), C(3,4)} — lower-left group
Cluster 2: {D(5,7), E(3.5,5), F(4.5,5)} — upper-right group

Inertia = Σ||xᵢ-μₖ||² for all points
        = (1.56² + 0.47² + 2.01²) + (1.49² + 1.08² + 0.69²)
        = (2.43 + 0.22 + 4.04) + (2.22 + 1.17 + 0.48)
        = 6.69 + 3.87 = 10.56
```

---

## What to Look for in the Application Lab

In the application lab, you'll:
1. Implement K-Means from scratch in numpy — watch centroids move each iteration
2. Compare with sklearn KMeans (should get same result)
3. Use the elbow method and silhouette score to pick K
4. Apply DBSCAN and see it handle noise without specifying K
5. Build a dendrogram with hierarchical clustering
6. Use Isolation Forest to find anomalous customers
