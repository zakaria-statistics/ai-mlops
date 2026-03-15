# Lab 06 Theory — Decision Trees
> Recursive partitioning: split the data at the question that gives the most information.

## Table of Contents
1. [Core Idea](#1-core-idea) — Trees as a sequence of yes/no questions
2. [Entropy](#2-entropy) — Measuring disorder
3. [Gini Impurity](#3-gini-impurity) — Faster alternative to entropy
4. [Information Gain](#4-information-gain) — Picking the best split
5. [How a Tree Picks the Best Split](#5-how-a-tree-picks-the-best-split) — The algorithm
6. [Regression Trees](#6-regression-trees) — Variance reduction instead of entropy
7. [Tree Pruning](#7-tree-pruning) — Controlling complexity
8. [Why Unlimited Trees Memorize Data](#8-why-unlimited-trees-memorize-data) — The overfitting problem
9. [By-Hand Example](#9-by-hand-example) — Build a 3-level tree from 8 data points
10. [What to Look for in the Application Lab](#10-what-to-look-for-in-the-application-lab)

---

## 1. Core Idea

A decision tree is a flowchart of binary questions that partition the
feature space into rectangular regions, each assigned a class.

```
                   [Age > 30?]
                   /          \
                YES            NO
               /                \
        [Income > 50K?]      [Student?]
        /           \         /       \
      YES           NO      YES       NO
       |             |        |         |
   Buy: YES      Buy: NO  Buy: YES  Buy: NO
```

Each internal node = a test on one feature.
Each leaf node = a class prediction.
Each path from root to leaf = a classification rule.

```
Feature space partitioning:

  Income                    The tree above creates these
    |                       rectangular regions:
 50K+--+--------+
    |  | YES    |           Age<=30 AND Student=Yes  -> YES
    |  |        | YES       Age<=30 AND Student=No   -> NO
    |  +--------+           Age>30  AND Income>50K   -> YES
    |  | NO     |           Age>30  AND Income<=50K  -> NO
    +--+--------+-----
       30              Age
```

---

## 2. Entropy

Entropy measures the **impurity** or **disorder** of a set. It comes from
information theory (Claude Shannon, 1948).

```
H(S) = -sum_{i=1}^{c} pi * log2(pi)

where:
  S = a set of examples
  c = number of classes
  pi = proportion of class i in S
```

### Binary case (2 classes):

```
H(S) = -p * log2(p) - (1-p) * log2(1-p)

where p = proportion of positive class
```

```
  H(S)
  1.0 |        .....
      |      .       .
      |    .           .
  0.5 |  .               .
      | .                   .
  0.0 +.---+---+---+---+----.
      0   0.2  0.4  0.6  0.8  1.0
                   p

  Maximum entropy = 1.0 at p = 0.5 (maximum uncertainty)
  Minimum entropy = 0.0 at p = 0 or p = 1 (pure set)
```

### Key values to memorize:

```
All one class (pure):     H = -1*log2(1) = 0
50/50 split:              H = -0.5*log2(0.5) - 0.5*log2(0.5)
                            = -0.5*(-1) - 0.5*(-1) = 1.0
75/25 split:              H = -0.75*log2(0.75) - 0.25*log2(0.25)
                            = -0.75*(-0.415) - 0.25*(-2)
                            = 0.311 + 0.500 = 0.811
```

> **Key Intuition:** Entropy answers "how surprised would I be if I
> randomly picked an element?" Pure sets = no surprise = low entropy.
> Mixed sets = high surprise = high entropy. The tree wants to REDUCE
> entropy at every split.

---

## 3. Gini Impurity

A faster-to-compute alternative to entropy:

```
G(S) = 1 - sum_{i=1}^{c} pi^2
```

### Binary case:

```
G(S) = 1 - p^2 - (1-p)^2 = 2p(1-p)
```

### Key values:

```
All one class (pure):     G = 1 - 1^2 = 0
50/50 split:              G = 1 - 0.5^2 - 0.5^2 = 0.5
75/25 split:              G = 1 - 0.75^2 - 0.25^2 = 1 - 0.5625 - 0.0625 = 0.375
```

### Entropy vs Gini Comparison:

```
  Value
  1.0 |        ...E...
      |      .   .G.   .
      |    .   .     .   .
  0.5 |  .  .           .  .
      | . .                . .
  0.0 +..---+---+---+---+--..
      0   0.2  0.4  0.6  0.8  1.0
                   p

  E = Entropy (peaks at 1.0)
  G = Gini    (peaks at 0.5)

  They are very similar in shape. In practice, they produce
  nearly identical trees. Gini is slightly faster (no log).
```

> **Key Intuition:** Gini answers "if I randomly label a point from
> this set using the class distribution, what's the probability I
> label it WRONG?" Pure sets = 0 chance of error. 50/50 = maximum
> chance of error.

---

## 4. Information Gain

Information gain = how much entropy DECREASES after a split:

```
IG(S, feature) = H(S_parent) - sum_{k} (n_k / n) * H(S_child_k)

where:
  H(S_parent) = entropy of the set before splitting
  S_child_k   = the k-th child set after splitting
  n_k          = number of elements in child k
  n            = total elements in parent
  n_k / n      = weight (proportion going to child k)
```

The tree picks the feature/threshold with the HIGHEST information gain.

### Example:

```
Parent set: 5 Yes, 5 No  -->  H = 1.0

Split option A:
  Left child:  4 Yes, 1 No   H = -0.8*log2(0.8) - 0.2*log2(0.2) = 0.722
  Right child: 1 Yes, 4 No   H = 0.722

  IG = 1.0 - (5/10)*0.722 - (5/10)*0.722 = 1.0 - 0.722 = 0.278

Split option B:
  Left child:  5 Yes, 0 No   H = 0.0  (pure!)
  Right child: 0 Yes, 5 No   H = 0.0  (pure!)

  IG = 1.0 - (5/10)*0.0 - (5/10)*0.0 = 1.0 - 0 = 1.0

  Split B is PERFECT. Maximum possible gain.
```

---

## 5. How a Tree Picks the Best Split

The algorithm is exhaustive search over all possible splits:

```
ALGORITHM: Find best split for a node

FOR each feature f in {f1, f2, ..., fd}:
    Sort values of feature f
    FOR each threshold t between consecutive distinct values:
        Split data into LEFT (f <= t) and RIGHT (f > t)
        Compute information gain (or Gini reduction)
        IF this gain > best_gain_so_far:
            best_gain = this gain
            best_feature = f
            best_threshold = t

Split on best_feature at best_threshold
Recurse on left child and right child
```

### For continuous features:

```
Values: [1, 3, 5, 7, 9]
Candidate thresholds: [2, 4, 6, 8]  (midpoints between consecutive values)

Try each: split at 2, split at 4, split at 6, split at 8
Pick the one with highest information gain.
```

### For categorical features:

```
Values: {Red, Blue, Green}
Candidate splits:
  {Red} vs {Blue, Green}
  {Blue} vs {Red, Green}
  {Green} vs {Red, Blue}

For m categories: 2^(m-1) - 1 possible splits
```

### Stopping conditions (when to stop recursing):

```
- Node is pure (all one class): entropy = 0
- Reached max_depth
- Node has fewer than min_samples_split examples
- Information gain < threshold
- Leaf would have fewer than min_samples_leaf examples
```

---

## 6. Regression Trees

For regression (continuous target), replace entropy with **variance reduction**:

```
Impurity measure = Variance = (1/n) * sum_i (yi - y_mean)^2

Variance reduction = Var(parent) - sum_k (n_k/n) * Var(child_k)
```

Leaf prediction = mean of target values in that leaf.

```
Example: Predict house price

                [sqft > 1500?]
                /             \
              YES               NO
              /                   \
    [rooms > 3?]              [age > 20?]
    /          \              /          \
  YES          NO           YES          NO
   |            |             |            |
 $350K        $250K         $150K        $200K
  (mean)      (mean)        (mean)       (mean)
```

> **Key Intuition:** Classification trees maximize purity (reduce entropy).
> Regression trees minimize variance (reduce spread of target values).
> Same algorithm, different impurity measure.

---

## 7. Tree Pruning

### Pre-pruning (stop growing early)

Set constraints BEFORE building:

```
max_depth = 5          Stop at depth 5
min_samples_split = 20 Don't split nodes with < 20 samples
min_samples_leaf = 5   Don't create leaves with < 5 samples
max_features = 'sqrt'  Only consider sqrt(d) features per split
min_impurity_decrease  Don't split if gain < threshold
```

### Post-pruning (grow full tree, then cut back)

```
1. Build a full tree (possibly overfitting)
2. Evaluate each internal node:
   - Compare accuracy WITH subtree vs REPLACING subtree with a leaf
   - If leaf is better (or nearly as good), prune the subtree
3. Use validation set or cross-validation to decide

Cost-complexity pruning (used by sklearn):
  R_alpha(T) = R(T) + alpha * |T|

  R(T) = misclassification rate of tree T
  |T|  = number of leaves
  alpha = complexity penalty (like regularization)

  Larger alpha --> simpler tree (more pruning)
```

```
  Full tree (overfit)            Pruned tree (generalized)

        [A]                            [A]
       /   \                          /   \
     [B]   [C]                      [B]  LEAF
    / \    / \                      / \
  [D] [E][F][G]                   [D] [E]
  / \
 [H][I]   <-- these captured noise
```

---

## 8. Why Unlimited Trees Memorize Data

An unpruned tree can perfectly classify ANY training set by creating
one leaf per training example:

```
8 training points -> tree can have up to 8 leaves
Each leaf contains exactly 1 point -> 100% training accuracy

But the decision boundary looks like this:

  Without limit              With max_depth=2
  (memorization)             (generalization)

  +--+--+--+--+             +--------+--------+
  |x |o |x |  |             |  x x   |  o o   |
  +--+--+--+--+             |  x     |  o     |
  |o |x |  |o |             +--------+--------+
  +--+--+--+--+             |  o     |  x     |
  |  |o |x |  |             |  o o   |  x x   |
  +--+--+--+--+             +--------+--------+

  Tiny regions =             Large regions =
  captures noise             captures pattern
```

### Why this happens mathematically:

```
With enough depth, the tree can ALWAYS reduce entropy to 0:

Depth 0: H = 1.0 (mixed)
Depth 1: H ~ 0.8 (slightly less mixed)
Depth 2: H ~ 0.5
...
Depth k: If each leaf has 1 point, H = 0 (pure by definition)

Training accuracy = 100%. Test accuracy = terrible.
The tree learned the noise, not the signal.
```

> **Key Intuition:** A decision tree with no constraints is a lookup
> table. It can represent any function of the training data. This is
> maximum variance, zero bias — the extreme overfitting end.
> Pruning adds bias to reduce variance.

---

## 9. By-Hand Example

**Build a 3-level tree from 8 data points.**

**Dataset:** Predict "Play Tennis?" from Outlook and Humidity.

```
#  Outlook    Humidity   Play?
1  Sunny      High       No
2  Sunny      High       No
3  Overcast   High       Yes
4  Rain       Normal     Yes
5  Rain       High       Yes
6  Rain       Normal     No
7  Overcast   Normal     Yes
8  Sunny      Normal     Yes
```

```
Total: 5 Yes, 3 No
```

### Level 0 — Root: Which feature to split on?

**Parent entropy:**

```
p(Yes) = 5/8,  p(No) = 3/8

H(parent) = -(5/8)*log2(5/8) - (3/8)*log2(3/8)
           = -(5/8)*(-0.678) - (3/8)*(-1.415)
           = 0.424 + 0.530
           = 0.954
```

**Try splitting on Outlook {Sunny, Overcast, Rain}:**

```
Sunny:    {#1:No, #2:No, #8:Yes}     -> 1 Yes, 2 No
Overcast: {#3:Yes, #7:Yes}           -> 2 Yes, 0 No
Rain:     {#4:Yes, #5:Yes, #6:No}    -> 2 Yes, 1 No

H(Sunny)    = -(1/3)*log2(1/3) - (2/3)*log2(2/3)
            = -(0.333)*(-1.585) - (0.667)*(-0.585)
            = 0.528 + 0.390 = 0.918

H(Overcast) = -(2/2)*log2(2/2) = -1*log2(1) = 0.0  (PURE!)

H(Rain)     = -(2/3)*log2(2/3) - (1/3)*log2(1/3)
            = 0.390 + 0.528 = 0.918

IG(Outlook) = 0.954 - (3/8)*0.918 - (2/8)*0.0 - (3/8)*0.918
            = 0.954 - 0.344 - 0.0 - 0.344
            = 0.266
```

**Try splitting on Humidity {High, Normal}:**

```
High:   {#1:No, #2:No, #3:Yes, #5:Yes}  -> 2 Yes, 2 No
Normal: {#4:Yes, #6:No, #7:Yes, #8:Yes} -> 3 Yes, 1 No

H(High)   = -(2/4)*log2(2/4) - (2/4)*log2(2/4)
          = -0.5*(-1) - 0.5*(-1) = 1.0

H(Normal) = -(3/4)*log2(3/4) - (1/4)*log2(1/4)
          = -(0.75)*(-0.415) - (0.25)*(-2.0)
          = 0.311 + 0.500 = 0.811

IG(Humidity) = 0.954 - (4/8)*1.0 - (4/8)*0.811
             = 0.954 - 0.500 - 0.406
             = 0.048
```

**Decision: Split on Outlook (IG=0.266 > 0.048)**

```
                     [Outlook?]
                    /     |     \
              Sunny    Overcast   Rain
              1Y,2N    2Y,0N     2Y,1N
```

### Level 1 — Overcast branch: DONE (pure)

```
Overcast: 2 Yes, 0 No  -->  H = 0  -->  Leaf: YES
```

### Level 1 — Sunny branch: Split on Humidity

```
Sunny subset: {#1:No(High), #2:No(High), #8:Yes(Normal)}

Only feature left to be useful: Humidity

High:   {#1:No, #2:No}    -> 0 Yes, 2 No  -> H = 0 (pure)
Normal: {#8:Yes}           -> 1 Yes, 0 No  -> H = 0 (pure)

IG = 0.918 - (2/3)*0.0 - (1/3)*0.0 = 0.918  (perfect split!)
```

### Level 1 — Rain branch: Split on Humidity

```
Rain subset: {#4:Yes(Normal), #5:Yes(High), #6:No(Normal)}

Humidity split:
High:   {#5:Yes}          -> 1 Yes, 0 No  -> H = 0 (pure)
Normal: {#4:Yes, #6:No}   -> 1 Yes, 1 No  -> H = 1.0

H(Rain) = 0.918
IG = 0.918 - (1/3)*0.0 - (2/3)*1.0 = 0.918 - 0.667 = 0.251
```

### Final Tree (3 levels):

```
                        [Outlook?]
                       /     |     \
                 Sunny    Overcast   Rain
                  /          |         \
          [Humidity?]      YES      [Humidity?]
           /      \                   /      \
        High    Normal             High    Normal
         |        |                  |        |
        NO       YES               YES    [1Y,1N]
                                            |
                                         (tie/leaf)
```

The Rain-Normal leaf has 1 Yes, 1 No — this would need more features
or data to resolve. With majority vote: Yes (or random).

### Verification: Classify each training point

```
#1 Sunny,High     -> Outlook=Sunny -> Humidity=High   -> NO   correct
#2 Sunny,High     -> Outlook=Sunny -> Humidity=High   -> NO   correct
#3 Overcast,High  -> Outlook=Overcast                 -> YES  correct
#4 Rain,Normal    -> Outlook=Rain  -> Humidity=Normal  -> YES  correct
#5 Rain,High      -> Outlook=Rain  -> Humidity=High   -> YES  correct
#6 Rain,Normal    -> Outlook=Rain  -> Humidity=Normal  -> YES  WRONG (actual: No)
#7 Overcast,Norm  -> Outlook=Overcast                 -> YES  correct
#8 Sunny,Normal   -> Outlook=Sunny -> Humidity=Normal -> YES  correct

Training accuracy: 7/8 = 87.5%
(#6 is misclassified due to the ambiguous Rain+Normal node)
```

---

## 10. What to Look for in the Application Lab

When you move to the coding lab:

1. **Visualize the tree:** Use `plot_tree()` — see which features it chose first (most important)
2. **Try different depths:** `max_depth=2` vs `max_depth=None` — see overfitting in action
3. **Compare Gini vs Entropy:** `criterion='gini'` vs `criterion='entropy'` — usually similar results
4. **Check feature importances:** `model.feature_importances_` — based on total information gain per feature
5. **Watch for overfitting:** Compare train vs test accuracy at different max_depth values
6. **No scaling needed:** Trees split on thresholds, so feature scale doesn't matter (unlike KNN/SVM)
7. **Categorical encoding matters:** Trees handle ordinal naturally but need encoding for nominal features
8. **Look at the leaves:** Each leaf should have a reasonable number of samples — tiny leaves = overfitting
