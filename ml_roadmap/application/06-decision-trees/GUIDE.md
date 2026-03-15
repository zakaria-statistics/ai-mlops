# Lab 06 — Decision Trees
> Implement Gini, entropy, and information gain from scratch — visualize tree structure

**Prerequisites:** Read `theory/06-decision-trees/GUIDE.md`
**Dataset:** Iris (`from sklearn.datasets import load_iris`)
**Libraries:** numpy, matplotlib, sklearn

---

## Steps

### Step 1: FROM SCRATCH — Gini Impurity
```python
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)
```

### Step 2: FROM SCRATCH — Entropy
```python
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))
```

### Step 3: FROM SCRATCH — Information Gain
```python
def information_gain(parent, left_child, right_child):
    weight_l = len(left_child) / len(parent)
    weight_r = len(right_child) / len(parent)
    return entropy(parent) - (weight_l * entropy(left_child) + weight_r * entropy(right_child))
```

### Step 4: FROM SCRATCH — Best Split Finder
```
For each feature:
    For each unique threshold:
        Split data into left (≤ threshold) and right (> threshold)
        Compute information gain
Return the feature and threshold with maximum gain
```

### Step 5: sklearn DecisionTreeClassifier
```
- Fit on Iris dataset (3 classes, 4 features)
- Print accuracy on train and test
```

### Step 6: Visualize the Tree
```
- sklearn.tree.plot_tree() — see the actual splits
- Read it: each node shows feature, threshold, Gini, samples, class
```

### Checkpoint: Can you read and explain the tree structure?

### Step 7: Overfitting Demo
```
- Unlimited depth: max_depth=None → train accuracy ~100%, test lower
- Pruned: max_depth=3 → train accuracy lower, test may be better
- Plot: train vs test accuracy for max_depth = 1 to 20
```
**Expected:** Gap between train and test grows with depth

### Step 8: Decision Tree for Regression
```
- DecisionTreeRegressor on a simple 1D problem (e.g., sin(x) + noise)
- Plot the step function prediction vs smooth actual function
- Show how max_depth affects the smoothness
```

### Step 9: Feature Importance
```
- model.feature_importances_ → bar chart of top features
- Compare with correlation-based feature ranking
- Trees can capture non-linear importance that correlation misses
```

---

## Checkpoint

- [ ] Gini and entropy computed by hand match sklearn's node values
- [ ] Can read and explain a tree visualization
- [ ] Understand overfitting in trees and how pruning helps
- [ ] Feature importance extracted and interpreted
