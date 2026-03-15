# Lab 03 — Linear Regression from Scratch
> Implement OLS, gradient descent, and regularization — then compare with sklearn

**Prerequisites:** Read `theory/03-linear-regression/GUIDE.md`
**Dataset:** California Housing (`from sklearn.datasets import fetch_california_housing`)
**Libraries:** numpy, matplotlib, sklearn

---

## Steps

### Step 1: FROM SCRATCH — Normal Equation
```
- Select 2-3 features for simplicity
- Add bias column of ones to X
- Compute: w = (XᵀX)⁻¹ Xᵀy (use np.linalg.inv)
- Print weights
```
**Expected:** Weight vector matching sklearn's coef_ and intercept_

### Step 2: FROM SCRATCH — Gradient Descent
```
- Initialize w = zeros
- Loop for 1000 iterations:
    ŷ = X @ w
    gradient = (1/n) * X.T @ (ŷ - y)
    w = w - learning_rate * gradient
    loss = (1/(2*n)) * np.sum((y - ŷ)**2)
    track loss
- Try learning_rate = 0.01
```

### Step 3: Plot Loss Curve
```
- Plot loss vs iteration number
- Should see smooth decrease and convergence
- Try different learning rates: 0.001, 0.01, 0.1 — plot all three
```
**Expected:** 0.001 converges slowly, 0.01 good, 0.1 may oscillate

### Checkpoint: Scratch weights ≈ Normal equation weights ≈ sklearn weights

### Step 4: sklearn LinearRegression
```
- from sklearn.linear_model import LinearRegression
- model.fit(X_train, y_train)
- Compare model.coef_ and model.intercept_ with your scratch result
```

### Step 5: Ridge from Scratch
```
- Modify gradient: gradient = (1/n) * X.T @ (ŷ - y) + 2*alpha*w
- (don't regularize the bias term)
- Try alpha = 0.1, 1.0, 10.0
- Compare weights: larger alpha → smaller weights
```

### Step 6: Lasso with sklearn
```
- from sklearn.linear_model import Lasso
- Fit with alpha=1.0
- Print weights — some should be exactly 0 (feature selection!)
- Compare: which features did Lasso eliminate?
```

### Step 7: Evaluate All Models
```
Comparison table:
| Model          | MAE   | RMSE  | R²    |
|----------------|-------|-------|-------|
| OLS (scratch)  |       |       |       |
| GD (scratch)   |       |       |       |
| sklearn Linear |       |       |       |
| Ridge α=1.0   |       |       |       |
| Lasso α=1.0   |       |       |       |
```

### Step 8: Residual Analysis
```
- Plot: residuals (y - ŷ) vs predicted values
- Check: should be randomly scattered around 0
- Plot: histogram of residuals (should be roughly normal)
- If pattern exists → model is missing something (non-linearity?)
```

---

## Checkpoint

- [ ] Normal equation and gradient descent give same weights
- [ ] sklearn matches your scratch implementation
- [ ] Ridge shrinks weights, Lasso zeros some out
- [ ] Understand residual analysis
