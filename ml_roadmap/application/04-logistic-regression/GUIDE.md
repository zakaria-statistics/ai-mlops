# Lab 04 — Logistic Regression from Scratch
> Implement sigmoid, log-loss, and gradient descent for classification — ROC, precision, recall

**Prerequisites:** Read `theory/04-logistic-regression/GUIDE.md`, complete Lab 02
**Dataset:** Titanic (prepared from Lab 02)
**Libraries:** numpy, matplotlib, sklearn

---

## Steps

### Step 1: FROM SCRATCH — Sigmoid Function
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
- Plot sigmoid for z in [-10, 10]
- Verify: sigmoid(0) = 0.5, sigmoid(large) ≈ 1, sigmoid(very negative) ≈ 0

### Step 2: FROM SCRATCH — Log-Loss (Binary Cross-Entropy)
```python
def log_loss(y, y_hat):
    epsilon = 1e-15  # avoid log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
```

### Step 3: FROM SCRATCH — Gradient Descent for Logistic Regression
```
Initialize w = zeros
Loop for 1000 iterations:
    z = X @ w
    ŷ = sigmoid(z)
    gradient = (1/n) * X.T @ (ŷ - y)    ← same form as linear reg!
    w = w - learning_rate * gradient
    loss = log_loss(y, ŷ)
    track loss
```

### Step 4: Plot Loss Convergence
```
- Loss should decrease smoothly
- Compare convergence with different learning rates
```

### Checkpoint: Training loss is decreasing

### Step 5: FROM SCRATCH — Predict
```python
y_pred = (sigmoid(X_test @ w) >= 0.5).astype(int)
accuracy = np.mean(y_pred == y_test)
```

### Step 6: sklearn Comparison
```
- LogisticRegression().fit(X_train, y_train)
- Compare weights and accuracy with scratch version
```
**Expected:** Very similar results

### Step 7: Confusion Matrix from Scratch
```
Compute: TP, TN, FP, FN by comparing y_pred vs y_test
Build the 2x2 matrix manually
Then verify with sklearn confusion_matrix
```

### Step 8: Precision, Recall, F1 by Hand
```
Precision = TP / (TP + FP)    "of predicted positive, how many correct?"
Recall    = TP / (TP + FN)    "of actual positive, how many found?"
F1        = 2 * P * R / (P + R)

Compute from your confusion matrix values
Verify with sklearn classification_report
```

### Step 9: ROC Curve
```
- Vary threshold from 0 to 1 (100 steps)
- At each threshold: compute TPR (recall) and FPR (FP/(FP+TN))
- Plot TPR vs FPR → this is the ROC curve
- Compute AUC (area under the curve) → sklearn roc_auc_score
```

### Step 10: Decision Boundary Visualization
```
- Pick 2 features (PCA if needed to reduce to 2D)
- Create meshgrid, predict on every point
- Contourf to show the boundary
- Scatter actual points on top
```

---

## Checkpoint

- [ ] Sigmoid implemented and plotted
- [ ] Gradient descent converges, weights match sklearn
- [ ] Confusion matrix computed by hand and verified
- [ ] Understand precision/recall tradeoff
- [ ] ROC curve plotted, AUC computed
