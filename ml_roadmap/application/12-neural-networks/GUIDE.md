# Lab 12: Neural Networks
> Build neurons and networks from scratch — then scale with PyTorch on Fashion-MNIST

## Table of Contents
1. [Setup](#setup) - Dataset and libraries
2. [Activation Functions](#step-1-activation-functions-from-scratch) - Sigmoid, ReLU, derivatives
3. [Single Neuron](#step-2-single-neuron-from-scratch) - Forward pass
4. [Loss Function](#step-3-binary-cross-entropy-from-scratch) - Measure error
5. [Weight Update](#step-4-weight-update-from-scratch) - Gradient descent
6. [Train Single Neuron](#step-5-train-single-neuron) - Binary classification
7. [2-Layer MLP](#step-6-2-layer-mlp-from-scratch) - Forward + backprop
8. [Train MLP](#step-7-train-scratch-mlp) - Multi-neuron training
9. [PyTorch MLP](#step-8-pytorch-mlp) - Same architecture, less code
10. [Full Fashion-MNIST](#step-9-full-fashion-mnist) - 10-class classification
11. [Experiments](#step-10-experiments) - Vary architecture and hyperparameters
12. [Comparison](#step-11-accuracy-comparison) - Scratch vs PyTorch progression

## Prerequisites
- Read `theory/12-neural-networks/GUIDE.md` first
- Strong grasp of: matrix multiplication, chain rule, partial derivatives
- Completed Labs 04 (logistic regression — single neuron is similar)

## Dataset
**Fashion-MNIST** — 10 classes of clothing items
- Via torchvision: `torchvision.datasets.FashionMNIST`
- 60,000 train / 10,000 test images, 28x28 grayscale
- Classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- For scratch implementation, use a 2-class subset first (e.g., T-shirt vs Trouser)

## Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

---

## Step 1: Activation Functions from Scratch

Implement with numpy:

1. **Sigmoid** and its derivative:
   ```python
   def sigmoid(z):
       return 1 / (1 + np.exp(-z))

   def sigmoid_derivative(z):
       s = sigmoid(z)
       return s * (1 - s)
   ```

2. **ReLU** and its derivative:
   ```python
   def relu(z):
       return np.maximum(0, z)

   def relu_derivative(z):
       return (z > 0).astype(float)
   ```

3. Plot both functions and their derivatives over range [-5, 5]
4. Note: sigmoid squashes to (0,1), saturates at extremes. ReLU is zero for negative, linear for positive.

**Expected output:** Sigmoid is S-shaped, derivative peaks at 0.25 at z=0. ReLU is a hockey stick, derivative is a step function.

---

## Step 2: Single Neuron from Scratch

1. Create a fake 2D dataset or extract 2-class subset from Fashion-MNIST (flatten to 784 features)
2. For simplicity, start with 2D fake data:
   ```python
   np.random.seed(42)
   X = np.random.randn(200, 2)
   y = (X[:, 0] + X[:, 1] > 0).astype(float)
   ```
3. Initialize weights and bias:
   ```python
   w = np.random.randn(2) * 0.01
   b = 0.0
   ```
4. Forward pass:
   ```python
   z = X @ w + b        # linear combination
   a = sigmoid(z)       # activation (prediction)
   ```

**Expected output:** `a` contains values between 0 and 1 — these are predicted probabilities.

---

## Step 3: Binary Cross-Entropy from Scratch

1. Implement the loss:
   ```python
   def binary_cross_entropy(y_true, y_pred):
       epsilon = 1e-15  # prevent log(0)
       y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
       return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
   ```
2. Compute loss with random weights — should be high (~0.7 = random guessing)
3. Compute with perfect predictions — should be ~0

**Expected output:** Initial loss around 0.6-0.7 (close to -log(0.5) = 0.693 for random guessing).

> **Checkpoint 1:** You have activation functions, a forward pass, and a loss function. All from scratch.

---

## Step 4: Weight Update from Scratch

Derive and implement the gradients:

1. The gradient of BCE loss with respect to weights:
   ```
   dL/dw = (1/n) * X.T @ (a - y)
   dL/db = (1/n) * sum(a - y)
   ```
   (This is the same as logistic regression — a single neuron with sigmoid IS logistic regression)

2. Update rule:
   ```python
   learning_rate = 0.1
   dw = (1/len(y)) * X.T @ (a - y)
   db = (1/len(y)) * np.sum(a - y)
   w = w - learning_rate * dw
   b = b - learning_rate * db
   ```

3. Run one update and check: did the loss decrease?

**Expected output:** Loss should drop after one weight update. If it increases, check gradient signs.

---

## Step 5: Train Single Neuron

1. Put it in a training loop:
   ```python
   losses = []
   for epoch in range(1000):
       # Forward
       z = X @ w + b
       a = sigmoid(z)
       loss = binary_cross_entropy(y, a)
       losses.append(loss)
       # Backward
       dw = (1/len(y)) * X.T @ (a - y)
       db = (1/len(y)) * np.sum(a - y)
       # Update
       w -= learning_rate * dw
       b -= learning_rate * db
   ```
2. Plot loss curve — should decrease and plateau
3. Compute accuracy: `(predictions == y).mean()`
4. Plot decision boundary on the 2D data

**Expected output:** Loss converges around epoch 200-500. Accuracy ~85-95% on the linearly separable fake data. Decision boundary is a straight line (single neuron = linear classifier).

---

## Step 6: 2-Layer MLP from Scratch

Now add a hidden layer. Architecture: Input(2) -> Hidden(8, ReLU) -> Output(1, Sigmoid)

1. Initialize weights:
   ```python
   W1 = np.random.randn(2, 8) * 0.01   # input to hidden
   b1 = np.zeros(8)
   W2 = np.random.randn(8, 1) * 0.01   # hidden to output
   b2 = np.zeros(1)
   ```

2. **Forward pass:**
   ```python
   z1 = X @ W1 + b1        # (n, 8)
   a1 = relu(z1)            # (n, 8)
   z2 = a1 @ W2 + b2       # (n, 1)
   a2 = sigmoid(z2)         # (n, 1)
   ```

3. **Backpropagation** — compute gradients layer by layer using chain rule:
   ```python
   # Output layer gradients
   dz2 = a2 - y.reshape(-1, 1)          # (n, 1)
   dW2 = (1/n) * a1.T @ dz2             # (8, 1)
   db2 = (1/n) * np.sum(dz2, axis=0)    # (1,)

   # Hidden layer gradients (chain rule through W2 and ReLU)
   da1 = dz2 @ W2.T                     # (n, 8)
   dz1 = da1 * relu_derivative(z1)      # (n, 8)
   dW1 = (1/n) * X.T @ dz1             # (2, 8)
   db1 = (1/n) * np.sum(dz1, axis=0)   # (8,)
   ```

4. **Update all weights:**
   ```python
   W2 -= lr * dW2
   b2 -= lr * db2
   W1 -= lr * dW1
   b1 -= lr * db1
   ```

**Expected output:** The shapes should all check out. This is the hardest part of the lab — if backprop shapes don't match, review the chain rule derivation in the theory guide.

> **Checkpoint 2:** You've implemented forward pass and backpropagation for a 2-layer network from scratch.

---

## Step 7: Train Scratch MLP

1. Create a non-linear dataset to show MLP advantage:
   ```python
   # XOR-like pattern or circles
   from sklearn.datasets import make_moons
   X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
   ```
2. Train the scratch MLP for 2000 epochs
3. Plot loss curve
4. Plot decision boundary — it should be curved (non-linear), unlike the single neuron
5. Accuracy should be higher than single neuron on this non-linear data

**Expected output:** MLP achieves ~95%+ accuracy on moons dataset. Single neuron gets ~50% (can't separate non-linear data). Decision boundary is curved.

---

## Step 8: PyTorch MLP

Implement the same architecture in PyTorch:

1. Define the model:
   ```python
   class MLP(nn.Module):
       def __init__(self, input_dim, hidden_dim, output_dim):
           super().__init__()
           self.fc1 = nn.Linear(input_dim, hidden_dim)
           self.relu = nn.ReLU()
           self.fc2 = nn.Linear(hidden_dim, output_dim)
           self.sigmoid = nn.Sigmoid()

       def forward(self, x):
           x = self.relu(self.fc1(x))
           x = self.sigmoid(self.fc2(x))
           return x
   ```

2. Train with Adam optimizer:
   ```python
   model = MLP(2, 8, 1)
   optimizer = optim.Adam(model.parameters(), lr=0.01)
   criterion = nn.BCELoss()
   ```

3. Training loop:
   ```python
   X_tensor = torch.FloatTensor(X)
   y_tensor = torch.FloatTensor(y).unsqueeze(1)

   for epoch in range(500):
       pred = model(X_tensor)
       loss = criterion(pred, y_tensor)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
   ```

4. Compare with scratch: same accuracy? Faster convergence with Adam?

**Expected output:** PyTorch version trains faster (Adam vs basic SGD) and reaches similar accuracy. Much less code.

> **Checkpoint 3:** You've seen the same network in numpy and PyTorch. PyTorch handles backprop automatically.

---

## Step 9: Full Fashion-MNIST

Scale to the real dataset with 10 classes:

1. Load and prepare data:
   ```python
   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
   train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
   test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
   train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
   test_loader = DataLoader(test_data, batch_size=64)
   ```

2. Build a larger MLP:
   ```python
   class FashionMLP(nn.Module):
       def __init__(self):
           super().__init__()
           self.flatten = nn.Flatten()
           self.fc1 = nn.Linear(784, 128)
           self.fc2 = nn.Linear(128, 64)
           self.fc3 = nn.Linear(64, 10)
           self.relu = nn.ReLU()

       def forward(self, x):
           x = self.flatten(x)
           x = self.relu(self.fc1(x))
           x = self.relu(self.fc2(x))
           x = self.fc3(x)  # raw logits, no softmax (CrossEntropyLoss handles it)
           return x
   ```

3. Use `nn.CrossEntropyLoss()` (includes softmax internally)
4. Train for 10-20 epochs, track train/val accuracy each epoch
5. Plot training and validation accuracy curves

**Expected output:** ~87-89% test accuracy after 10-20 epochs. Training accuracy should be slightly higher.

---

## Step 10: Experiments

Create a results table by varying:

1. **Hidden size:** 32, 64, 128, 256
2. **Learning rate:** 0.001, 0.01, 0.1
3. **Epochs:** 5, 10, 20

For each combination, record test accuracy and training time. Use a subset of combinations (not all 36):
- Fix lr=0.001, vary hidden size
- Fix hidden=128, vary lr
- Fix hidden=128, lr=0.001, vary epochs

```
| Hidden | LR    | Epochs | Test Acc | Train Time |
|--------|-------|--------|----------|------------|
| 32     | 0.001 | 10     |          |            |
| 64     | 0.001 | 10     |          |            |
| 128    | 0.001 | 10     |          |            |
| 256    | 0.001 | 10     |          |            |
| 128    | 0.01  | 10     |          |            |
| 128    | 0.1   | 10     |          |            |
```

**Expected output:** Larger hidden layers help up to a point (diminishing returns after 128). LR=0.1 may diverge or oscillate. LR=0.001 is stable. More epochs help but eventually overfit.

---

## Step 11: Accuracy Comparison

Create a summary bar chart:

```
| Model                         | Accuracy | Notes              |
|-------------------------------|----------|--------------------|
| Scratch single neuron (2-class) | ~85%   | Linear only        |
| Scratch MLP (2-class)          | ~95%   | Non-linear capable |
| PyTorch MLP (2-class)          | ~96%   | Adam optimizer     |
| PyTorch MLP (10-class)         | ~88%   | Full Fashion-MNIST |
| Larger PyTorch MLP (10-class)  | ~89%   | More params        |
```

Key takeaway: adding layers and neurons helps, but fully connected networks plateau on image data. CNNs (Lab 13) will do significantly better.

> **Checkpoint 4:** You've built neural networks from single neurons to multi-layer PyTorch models. You understand forward pass, backpropagation, and the effect of hyperparameters.

---

## Summary Deliverables
- [ ] Sigmoid and ReLU implemented with derivatives
- [ ] Single neuron trained from scratch (binary classification)
- [ ] 2-layer MLP with manual backpropagation
- [ ] PyTorch MLP matching scratch results
- [ ] Full Fashion-MNIST 10-class classifier
- [ ] Hyperparameter experiment table
- [ ] Accuracy progression chart (scratch -> PyTorch -> larger)
