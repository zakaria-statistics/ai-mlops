# Lab 07 Theory — Ensemble Methods
> Combine weak models into strong ones: the power of collective decision-making.

## Table of Contents
1. [Why Ensembles Work](#1-why-ensembles-work) — Wisdom of crowds
2. [Bagging](#2-bagging) — Bootstrap aggregating
3. [Random Forest](#3-random-forest) — Bagging + random feature subsets
4. [Boosting: Core Idea](#4-boosting-core-idea) — Sequential error correction
5. [AdaBoost](#5-adaboost) — Sample reweighting
6. [Gradient Boosting](#6-gradient-boosting) — Fit on residuals
7. [XGBoost](#7-xgboost) — Regularized gradient boosting
8. [Stacking](#8-stacking) — Meta-learner on top of base models
9. [Voting](#9-voting) — Hard vs soft voting
10. [Comparison Table](#10-comparison-table) — Bagging vs boosting tradeoffs
11. [What to Look for in the Application Lab](#11-what-to-look-for-in-the-application-lab)

---

## 1. Why Ensembles Work

### The Wisdom of Crowds Argument

One decision tree: ~70% accuracy (high variance, unstable).
100 decision trees combined: ~90%+ accuracy. Why?

**Mathematical argument:** If you have N independent classifiers, each
with accuracy p > 0.5, the majority vote accuracy approaches 1 as N grows.

```
Single model accuracy: p = 0.6 (barely better than random)

Majority vote of 3 independent models:
  P(majority correct) = P(all 3 right) + P(exactly 2 right)
  = p^3 + 3*p^2*(1-p)
  = 0.6^3 + 3*0.6^2*0.4
  = 0.216 + 0.432 = 0.648

Majority vote of 11 models:
  P(>=6 correct) = sum_{k=6}^{11} C(11,k) * p^k * (1-p)^(11-k)
  = 0.753

Majority vote of 101 models:
  P(>=51 correct) = 0.869

Majority vote of 1001 models:
  P(>=501 correct) = 0.999+
```

> **Key Intuition:** Each model makes different errors. When you combine
> them, errors cancel out and correct predictions reinforce each other.
> BUT this only works if the models are DIFFERENT (diverse). If all
> models make the same errors, combining them doesn't help.

### Bias-Variance Decomposition

```
Total Error = Bias^2 + Variance + Irreducible Noise

Bagging:  Reduces VARIANCE (averaging removes noise)
Boosting: Reduces BIAS (sequential correction fixes systematic errors)

  Error
    |
    |  Variance                    Variance
    |  ========                    ==
    |  Bias                        Bias
    |  ====                        ========
    |  Single Tree                 Boosted Trees
    |
    |  Variance
    |  ==
    |  Bias
    |  ====
    |  Bagged Trees (RF)
```

---

## 2. Bagging

**B**ootstrap **Agg**regating. Train multiple models on different random
subsets of the data, then combine their predictions.

### Bootstrap Sampling

Draw N samples WITH REPLACEMENT from a dataset of N points:

```
Original data: [A, B, C, D, E, F, G, H]  (8 points)

Bootstrap sample 1: [A, A, C, D, D, F, G, H]  (B,E missing; A,D repeated)
Bootstrap sample 2: [B, B, C, C, E, F, G, H]
Bootstrap sample 3: [A, C, D, E, F, F, G, G]
...

Each sample:
  - Same size as original (N)
  - ~63.2% of original points appear (some repeated)
  - ~36.8% are left out (out-of-bag samples)
```

Why 63.2%? Probability a point is NOT picked in N draws:

```
P(not picked) = (1 - 1/N)^N  -->  1/e = 0.368  as N -> infinity

So P(picked at least once) = 1 - 1/e = 0.632
```

### Bagging Algorithm

```
INPUT: Training data D, number of models T

FOR t = 1 to T:
    D_t = bootstrap sample from D
    Train model M_t on D_t

PREDICTION (classification):
    y_hat = majority vote of {M_1(x), M_2(x), ..., M_T(x)}

PREDICTION (regression):
    y_hat = (1/T) * sum_t M_t(x)
```

```
  Original Data
  +----------+
  | x x o o  |
  | o x o x  |
  | x o x o  |
  +----------+
       |
       | bootstrap sampling
       v
  +------+  +------+  +------+  +------+
  |Sample1|  |Sample2|  |Sample3|  |Sample4|
  +------+  +------+  +------+  +------+
       |         |         |         |
       v         v         v         v
  [Tree 1]  [Tree 2]  [Tree 3]  [Tree 4]   <-- parallel, independent
       |         |         |         |
       v         v         v         v
   pred: x   pred: o   pred: x   pred: x
                    \       |       /
                     \      |      /
                      v     v     v
                    VOTE: x wins (3 vs 1)
```

---

## 3. Random Forest

Random Forest = Bagging with decision trees + **random feature subsets**.

### The Extra Randomness

At each split in each tree, only consider a random subset of features:

```
Standard tree:     Consider ALL d features at each split
Random Forest:     Consider only m features at each split

Typical m values:
  Classification: m = sqrt(d)
  Regression:     m = d/3

If d = 100 features:
  Classification: try only 10 random features per split
  Regression:     try only 33 random features per split
```

### Why Random Features Help

```
Without random features:
  If feature X is very strong, ALL trees split on X first.
  Trees are highly correlated -> averaging barely helps.

With random features:
  Sometimes X isn't in the random subset.
  Trees are forced to find DIFFERENT good splits.
  Trees become LESS correlated -> averaging helps much more.
```

```
  Variance of average = (1/N) * avg_variance + ((N-1)/N) * avg_covariance

  If trees are correlated (high covariance):
    Averaging barely reduces total variance.

  If trees are uncorrelated (low covariance):
    Averaging divides variance by N.

  Random features REDUCE correlation between trees.
```

### Out-of-Bag (OOB) Score

Each tree doesn't see ~36.8% of the data. Use those unseen points to
estimate test accuracy — for FREE, no separate validation set needed.

```
For each training point x_i:
    Find all trees where x_i was NOT in the bootstrap sample
    Let those trees vote on x_i
    Compare vote result to true label

OOB_accuracy = fraction of correctly classified OOB predictions
```

> **Key Intuition:** OOB score is essentially cross-validation for free.
> Each point is evaluated by the ~1/3 of trees that never saw it.

### Feature Importance

Two common methods:

```
1. Impurity-based importance:
   For each feature, sum the total information gain (or Gini decrease)
   across all splits in all trees that use that feature.

   importance(feature_j) = sum over all nodes that split on j:
                           (weighted impurity decrease at that node)

2. Permutation importance:
   Randomly shuffle feature j's values.
   Measure how much accuracy drops.
   Large drop = important feature.
```

---

## 4. Boosting: Core Idea

Boosting builds models **sequentially**. Each new model focuses on the
mistakes of the previous models.

```
  Round 1:           Round 2:           Round 3:
  Basic model        Focus on errors    Focus on remaining
                     from Round 1       errors

  o o | x x          o o | x x          o o | x x
  o o | x x          o O | x x          o o | x x
  o o | X x          o o | x x          o o | x x
      |                  |   |              /|
  "X is wrong"       "O is wrong"       "fix remaining"

  weight X more      weight O more       ...
  next round         next round
```

```
  Bagging:   Build all models INDEPENDENTLY (parallel)
  Boosting:  Build models SEQUENTIALLY (each depends on previous)

  Bagging:   Reduces VARIANCE
  Boosting:  Reduces BIAS (and can reduce variance too)
```

---

## 5. AdaBoost

**Ada**ptive **Boost**ing. Reweight training samples so misclassified
points get more attention.

### Algorithm

```
INITIALIZE: Sample weights w_i = 1/N for all i

FOR t = 1 to T:
    1. Train weak learner h_t on weighted data

    2. Compute weighted error:
       epsilon_t = sum_{i: h_t(x_i) != y_i} w_i / sum_i w_i

    3. Compute model weight:
       alpha_t = (1/2) * ln((1 - epsilon_t) / epsilon_t)

    4. Update sample weights:
       w_i <- w_i * exp(-alpha_t * y_i * h_t(x_i))

       If y_i * h_t(x_i) = +1 (correct):  w_i <- w_i * exp(-alpha_t)  (decrease)
       If y_i * h_t(x_i) = -1 (wrong):    w_i <- w_i * exp(+alpha_t)  (increase)

    5. Normalize weights so they sum to 1

FINAL PREDICTION:
    H(x) = sign( sum_t alpha_t * h_t(x) )
```

### Understanding alpha_t

```
alpha_t = (1/2) * ln((1 - epsilon) / epsilon)

epsilon = 0.5 (random):  alpha = 0       (model is useless, weight = 0)
epsilon = 0.3 (decent):  alpha = 0.42    (moderate weight)
epsilon = 0.1 (good):    alpha = 1.10    (high weight)
epsilon = 0.01 (great):  alpha = 2.30    (very high weight)
epsilon = 0   (perfect): alpha = infinity

Better models get HIGHER weight in the final vote.
```

```
  alpha
    |
  3 |                                      .
    |                                   .
  2 |                                .
    |                            .
  1 |                        .
    |                   .
  0 +-----------.-------------------- epsilon
    0    0.1   0.2   0.3   0.4   0.5

  alpha is positive when epsilon < 0.5 (better than random)
  alpha is zero when epsilon = 0.5 (random guessing)
```

### Weight Update Intuition

```
After a round with alpha = 0.5:

  Correctly classified:  w_new = w_old * exp(-0.5) = w_old * 0.607
  Misclassified:         w_new = w_old * exp(+0.5) = w_old * 1.649

  Misclassified points get ~2.7x more weight than correct ones.
  Next model is "forced" to focus on these hard cases.
```

> **Key Intuition:** AdaBoost builds a weighted committee of experts.
> Good experts (low error) get more voting power. Each new expert
> specializes in what previous experts got wrong.

---

## 6. Gradient Boosting

Instead of reweighting samples, fit each new model to the **residuals**
(errors) of the current ensemble.

### Core Idea

```
Step 0:  F_0(x) = mean(y)                          (start with simple prediction)
Step 1:  r_1 = y - F_0(x)                          (compute residuals)
         Fit tree h_1 to r_1                        (learn to predict errors)
         F_1(x) = F_0(x) + eta * h_1(x)            (add correction)

Step 2:  r_2 = y - F_1(x)                          (new residuals)
         Fit tree h_2 to r_2                        (learn remaining errors)
         F_2(x) = F_1(x) + eta * h_2(x)            (add correction)

...

Step T:  F_T(x) = F_0(x) + eta * sum_{t=1}^{T} h_t(x)
```

Where eta = learning rate (shrinkage), typically 0.01 to 0.3.

### Why "Gradient"?

For squared error loss L = (1/2)(y - F(x))^2:

```
Negative gradient of loss w.r.t. F(x):

  -dL/dF = -(F(x) - y) = y - F(x) = residual!

So fitting to residuals IS gradient descent in function space.
```

For other loss functions, the "residual" is replaced by the negative
gradient of whatever loss you're using (log loss for classification, etc.).

### Learning Rate (Shrinkage)

```
eta = 1.0:   Full correction each step (aggressive, risk of overfitting)
eta = 0.1:   Only 10% correction each step (conservative, needs more trees)
eta = 0.01:  1% correction (very conservative, needs many trees)

Lower eta + more trees = better generalization (but slower training)
```

```
  Residuals over boosting rounds:

  |residual|
      |
  2.0 | \
      |  \
  1.0 |   \                  eta=1.0 (fast but overfit)
      |    \
  0.5 |     '---------
      |       \               eta=0.1 (slower but generalizes)
  0.1 |        '----------
      +--+--+--+--+--+--+--
         10 20 30 40 50 60  rounds
```

### Concrete Example (Regression)

```
Data: x=[1,2,3,4,5], y=[2.1, 3.9, 6.2, 7.8, 10.1]

Step 0: F_0(x) = mean(y) = 6.02 for all x

Residuals r_1:
  x=1: 2.1 - 6.02 = -3.92
  x=2: 3.9 - 6.02 = -2.12
  x=3: 6.2 - 6.02 = +0.18
  x=4: 7.8 - 6.02 = +1.78
  x=5: 10.1 - 6.02 = +4.08

Step 1: Fit a tree h_1 to predict these residuals from x.
  Say the tree learns: h_1(x) = 2*x - 6  (roughly)

  With eta=0.1:
  F_1(x) = 6.02 + 0.1 * (2x - 6)
  F_1(1) = 6.02 + 0.1*(-4) = 5.62    (closer to 2.1 than 6.02)
  F_1(3) = 6.02 + 0.1*(0)  = 6.02    (about right)
  F_1(5) = 6.02 + 0.1*(4)  = 6.42    (closer to 10.1 than 6.02)

Step 2: Compute new residuals r_2 = y - F_1(x), fit h_2 to those...
```

---

## 7. XGBoost

E**X**treme **G**radient **Boost**ing. Gradient boosting with regularization,
second-order approximation, and engineering optimizations.

### Objective Function

```
Obj = sum_i L(y_i, y_hat_i) + sum_t Omega(h_t)
      |___________________|   |_______________|
       loss on training data   regularization on trees

Omega(h) = gamma * T + (1/2) * lambda * sum_j w_j^2

where:
  T      = number of leaves in tree h
  w_j    = prediction value at leaf j
  gamma  = penalty per leaf (encourages fewer leaves)
  lambda = L2 penalty on leaf values (encourages smaller predictions)
```

### Taylor Expansion Trick

Instead of computing gradients numerically, XGBoost uses a second-order
Taylor expansion of the loss function:

```
L(y, y_hat + delta) ~= L(y, y_hat) + g * delta + (1/2) * h * delta^2

where:
  g = dL/d(y_hat)     (first derivative = gradient)
  h = d^2L/d(y_hat)^2  (second derivative = Hessian)
```

For squared error: g = y_hat - y, h = 1
For log loss: g = sigmoid(y_hat) - y, h = sigmoid(y_hat) * (1 - sigmoid(y_hat))

This gives a closed-form optimal leaf value:

```
w_j* = -G_j / (H_j + lambda)

where:
  G_j = sum of gradients g_i for all points in leaf j
  H_j = sum of hessians h_i for all points in leaf j
```

And the optimal split gain:

```
Gain = (1/2) * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma

       |__________________________|   |__________________________|   |________________________________|
        left child contribution        right child contribution       parent (no split) contribution

  minus gamma = penalty for adding a leaf

  If Gain < 0: don't split (the regularization says it's not worth it)
```

> **Key Intuition:** XGBoost adds a cost for model complexity directly
> into the objective. Regular gradient boosting only minimizes loss.
> XGBoost minimizes loss PLUS complexity. This is why it generalizes better.

### Histogram Binning

```
Standard gradient boosting:
  Sort feature values at each split: O(N * log(N))
  Try every possible threshold

XGBoost with histogram binning:
  Bin continuous values into ~256 buckets: O(N)
  Try only bucket boundaries as thresholds

  Speed improvement: ~10x faster for large datasets
```

---

## 8. Stacking

Train multiple diverse models, then train a **meta-learner** on their
outputs.

```
  Training data
  +----------+
  |          |
  +----------+
       |
       |--> [Model 1: Random Forest]  --> predictions_1
       |--> [Model 2: SVM]           --> predictions_2
       |--> [Model 3: KNN]           --> predictions_3
       |
       v
  +----------------------------------+
  | Meta-features:                   |
  | [pred_1, pred_2, pred_3, ...]    |
  +----------------------------------+
       |
       v
  [Meta-learner: Logistic Regression]
       |
       v
  Final prediction
```

### How to Train Without Overfitting

Use cross-validation to generate meta-features:

```
1. Split training data into K folds
2. For each base model:
   a. For each fold k:
      - Train on all folds EXCEPT k
      - Predict on fold k (these are the meta-features for fold k)
   b. Combine all fold predictions = full set of meta-features
3. Train meta-learner on these meta-features
```

```
  Fold 1: Train on [2,3,4,5] -> Predict fold 1 -> meta_1
  Fold 2: Train on [1,3,4,5] -> Predict fold 2 -> meta_2
  Fold 3: Train on [1,2,4,5] -> Predict fold 3 -> meta_3
  Fold 4: Train on [1,2,3,5] -> Predict fold 4 -> meta_4
  Fold 5: Train on [1,2,3,4] -> Predict fold 5 -> meta_5

  Stack: [meta_1, meta_2, meta_3, meta_4, meta_5] -> train meta-learner
```

> **Key Intuition:** Stacking learns which models to trust on which
> types of inputs. If Model 1 is good for some regions of the feature
> space and Model 2 for others, the meta-learner can learn this.

---

## 9. Voting

### Hard Voting

Each model casts one vote for a class. Majority wins.

```
Model 1 predicts: Cat
Model 2 predicts: Dog
Model 3 predicts: Cat
Model 4 predicts: Cat
Model 5 predicts: Dog

Hard vote: Cat (3 vs 2)
```

### Soft Voting

Average the predicted PROBABILITIES, pick the class with highest average.

```
                 P(Cat)  P(Dog)
Model 1:         0.90    0.10
Model 2:         0.30    0.70
Model 3:         0.60    0.40
Model 4:         0.55    0.45
Model 5:         0.40    0.60

Average:         0.55    0.45

Soft vote: Cat (0.55 > 0.45)
```

> **Key Intuition:** Soft voting is usually better because it accounts
> for CONFIDENCE. A model that predicts Cat with 90% confidence should
> count more than one that predicts Cat with 51% confidence.

---

## 10. Comparison Table

```
+------------------+-------------------+-------------------+
| Aspect           | Bagging (RF)      | Boosting (GB/XGB) |
+------------------+-------------------+-------------------+
| Training         | Parallel          | Sequential        |
| Independence     | Models independent| Each depends on   |
|                  |                   | previous          |
+------------------+-------------------+-------------------+
| Reduces          | VARIANCE          | BIAS              |
| (primarily)      | (averaging out    | (correcting       |
|                  |  noise)           |  errors)          |
+------------------+-------------------+-------------------+
| Base learners    | Full trees        | Shallow trees     |
|                  | (high variance)   | (stumps/depth 3-6)|
+------------------+-------------------+-------------------+
| Overfitting      | Hard to overfit   | CAN overfit if    |
|                  | with more trees   | too many rounds   |
+------------------+-------------------+-------------------+
| Key parameters   | n_estimators      | n_estimators      |
|                  | max_features      | learning_rate     |
|                  |                   | max_depth         |
+------------------+-------------------+-------------------+
| Typical accuracy | Good              | Often better      |
|                  |                   | (but tuning needed)|
+------------------+-------------------+-------------------+
| Robustness       | Very robust       | Sensitive to noise|
| to noise         | (noise averaged)  | (may boost noise) |
+------------------+-------------------+-------------------+
| Parallelizable   | Yes               | No (sequential)   |
+------------------+-------------------+-------------------+
| When to use      | Default first try | When you need     |
|                  | Good enough,robust| max accuracy and  |
|                  |                   | can tune carefully|
+------------------+-------------------+-------------------+
```

---

## 11. What to Look for in the Application Lab

When you move to the coding lab:

1. **Random Forest:** Start with `n_estimators=100`, compare to a single `DecisionTreeClassifier`
2. **Feature importance:** Plot `rf.feature_importances_` — which features drive decisions?
3. **OOB score:** Set `oob_score=True`, compare to cross-validation score
4. **Gradient Boosting:** Try `GradientBoostingClassifier` with `learning_rate=0.1, n_estimators=200`
5. **XGBoost:** Install `xgboost`, compare `XGBClassifier` to sklearn's `GradientBoostingClassifier`
6. **Learning curves:** Plot train vs test accuracy as `n_estimators` increases
   - RF: test accuracy plateaus, train stays high
   - GB: test accuracy peaks then may decrease (overfitting)
7. **Tuning interaction:** `learning_rate` and `n_estimators` are inversely related — lower rate needs more trees
8. **VotingClassifier:** Combine your best RF, SVM, and LogReg with soft voting
9. **Compare wall-clock time:** RF (parallel) vs GradientBoosting (sequential)
