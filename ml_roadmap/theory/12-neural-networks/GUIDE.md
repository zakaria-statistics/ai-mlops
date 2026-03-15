# 12 — Neural Networks: From a Single Neuron to Backpropagation
> The complete math of how neural networks learn — sigmoid, weight updates, forward pass, and backpropagation derived step by step

## Table of Contents
1. [The Single Neuron (Perceptron)](#1-the-single-neuron)
2. [Activation Functions](#2-activation-functions)
3. [Forward Pass](#3-forward-pass)
4. [Loss Functions](#4-loss-functions)
5. [Backpropagation — The Full Derivation](#5-backpropagation)
6. [Weight Update Rule](#6-weight-update-rule)
7. [Learning Rate](#7-learning-rate)
8. [Multi-Layer Perceptron (MLP)](#8-multi-layer-perceptron)
9. [Optimizers](#9-optimizers)
10. [Training Mechanics](#10-training-mechanics)
11. [By-Hand Example: Full Forward + Backward Pass](#11-by-hand-example)

---

## 1. The Single Neuron

A single neuron is the building block. It does three things:

```
Step 1: Weighted sum       z = w₁x₁ + w₂x₂ + ... + wₘxₘ + b = wᵀx + b
Step 2: Activation         a = σ(z)
Step 3: Output             ŷ = a
```

```
  x₁ ──w₁──╲
              ╲
  x₂ ──w₂───── Σ ──→ z = wᵀx + b ──→ σ(z) ──→ a (output)
              ╱                         ▲
  x₃ ──w₃──╱                    activation function
              │
         b (bias)
```

**What are weights?**
- w₁, w₂, ... are "importance multipliers" for each input
- Large |wᵢ| means feature xᵢ is important
- Positive wᵢ means xᵢ increases the output
- Negative wᵢ means xᵢ decreases the output

**What is bias?**
- b shifts the decision boundary
- Like the y-intercept of a line
- Allows the neuron to activate even when all inputs are zero

> **Key Intuition:** A single neuron is just a logistic regression! The "learning" is finding the right w and b so the neuron's output matches the desired target.

---

## 2. Activation Functions

Without activation functions, stacking layers of neurons is pointless — you'd just get another linear function. Activations introduce **non-linearity**.

### Sigmoid

```
σ(z) = 1 / (1 + e⁻ᶻ)

Output range: (0, 1)

              1 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                              ·····
                          ···
                        ·
                      ·
                    ·
              ···
         ·····
0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    -6  -4  -2   0   2   4   6
```

**Derivative (THIS IS CRUCIAL for backprop):**
```
σ'(z) = σ(z) · (1 - σ(z))

Derivation:
σ(z) = (1 + e⁻ᶻ)⁻¹

dσ/dz = -1 · (1 + e⁻ᶻ)⁻² · (-e⁻ᶻ)    [chain rule]

      = e⁻ᶻ / (1 + e⁻ᶻ)²

      = [1/(1+e⁻ᶻ)] · [e⁻ᶻ/(1+e⁻ᶻ)]

      = σ(z) · [(1+e⁻ᶻ-1)/(1+e⁻ᶻ)]

      = σ(z) · [1 - 1/(1+e⁻ᶻ)]

      = σ(z) · (1 - σ(z))
```

**Properties:**
- Max derivative = 0.25 (at z=0, where σ=0.5)
- Squashes everything to (0, 1)
- Problem: for large |z|, derivative ≈ 0 → **vanishing gradient!**

### Tanh

```
tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)

Output range: (-1, 1)

Derivative: tanh'(z) = 1 - tanh²(z)
```

- Zero-centered (unlike sigmoid)
- Still has vanishing gradient for large |z|

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)

              ╱
             ╱
            ╱
           ╱
──────────╱
    -4  -2  0   2   4

Derivative:
ReLU'(z) = { 0  if z < 0
           { 1  if z > 0
           { undefined at z = 0 (use 0 or 1 by convention)
```

**Why ReLU solved deep learning:**
- No vanishing gradient for positive z (derivative = 1)
- Computationally cheap (just max(0, z))
- Sparse activation (some neurons output 0 → efficiency)
- Problem: "dying ReLU" — neurons stuck at 0 if they go negative

### Comparison

| Function | Range | Derivative | Problem | Use case |
|----------|-------|-----------|---------|----------|
| Sigmoid | (0,1) | σ(1-σ), max 0.25 | Vanishing gradient | Output layer (binary) |
| Tanh | (-1,1) | 1-tanh², max 1.0 | Vanishing gradient | Hidden layers (old) |
| ReLU | [0,∞) | 0 or 1 | Dying neurons | Hidden layers (default) |
| Softmax | (0,1), sums to 1 | complex | — | Output layer (multi-class) |

---

## 3. Forward Pass

For a network with L layers, the forward pass computes layer by layer:

```
Layer 0 (input):   a⁰ = X

Layer 1:           z¹ = W¹a⁰ + b¹       (weighted sum)
                   a¹ = σ(z¹)            (activation)

Layer 2:           z² = W²a¹ + b²
                   a² = σ(z²)

...

Layer L (output):  zᴸ = Wᴸaᴸ⁻¹ + bᴸ
                   aᴸ = σ(zᴸ) = ŷ
```

**Matrix dimensions (for a network with layers of size n₀, n₁, n₂, ...):**
```
W¹: [n₁ × n₀]    (n₁ neurons, each connects to n₀ inputs)
b¹: [n₁ × 1]
a¹: [n₁ × 1]

General: Wˡ is [nₗ × nₗ₋₁]
```

```
Example: 3 inputs → 4 hidden → 2 output

Input (3)    Hidden (4)     Output (2)
  x₁ ──────── h₁ ──────── o₁
  x₂ ──╳───── h₂ ──╳───── o₂
  x₃ ──────── h₃ ──────
               h₄ ──────
       W¹[4×3]       W²[2×4]
       b¹[4×1]       b²[2×1]
```

---

## 4. Loss Functions

The loss measures how wrong our predictions are.

### Binary Cross-Entropy (for binary classification, sigmoid output)

```
L = -(1/n) Σᵢ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
```

Why this works:
```
When y=1: L = -log(ŷ)     → want ŷ close to 1, else loss is huge
When y=0: L = -log(1-ŷ)   → want ŷ close to 0, else loss is huge

  Loss
   ▲
   │ ╲
   │  ╲
   │   ╲  -log(ŷ)  (when y=1)
   │    ╲
   │     ╲____
   └──────────────→ ŷ
   0           1
```

### Mean Squared Error (for regression)

```
L = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

### Categorical Cross-Entropy (for multi-class, softmax output)

```
L = -(1/n) Σᵢ Σⱼ yᵢⱼ · log(ŷᵢⱼ)

where yᵢⱼ is one-hot encoded (1 for correct class, 0 elsewhere)
```

---

## 5. Backpropagation

> **This is the core algorithm that makes neural networks learn.**

Backpropagation = computing ∂L/∂w for EVERY weight using the **chain rule**, working backwards from the output.

### The Chain Rule

If L depends on a, which depends on z, which depends on w:

```
∂L/∂w = (∂L/∂a) · (∂a/∂z) · (∂z/∂w)
```

```
  w ──→ z = wx+b ──→ a = σ(z) ──→ L(y, a)

  To find how L changes with w:
  ∂L    ∂L   ∂a   ∂z
  ── = ── · ── · ──
  ∂w    ∂a   ∂z   ∂w
```

### Step-by-Step for a 2-Layer Network

Network: Input → Hidden (sigmoid) → Output (sigmoid) → Loss

```
Forward:
  z¹ = W¹x + b¹
  a¹ = σ(z¹)
  z² = W²a¹ + b²
  a² = σ(z²) = ŷ
  L = -(y·log(ŷ) + (1-y)·log(1-ŷ))
```

**Backward (output layer):**

```
Step 1: ∂L/∂a² = -(y/a² - (1-y)/(1-a²))
                = (a² - y) / (a²(1-a²))

Step 2: ∂a²/∂z² = σ'(z²) = a²(1-a²)     [sigmoid derivative!]

Step 3: ∂L/∂z² = (∂L/∂a²) · (∂a²/∂z²)
               = (a² - y) / (a²(1-a²)) · a²(1-a²)
               = (a² - y)                             ← simplifies beautifully!

This is called δ² (delta for output layer):
  δ² = a² - y = ŷ - y
```

**Gradients for output layer weights:**
```
∂L/∂W² = δ² · (a¹)ᵀ     (outer product of error and previous layer's activation)
∂L/∂b² = δ²
```

**Backward (hidden layer) — propagate the error back:**

```
Step 4: ∂L/∂a¹ = (W²)ᵀ · δ²     (error flows back through weights)

Step 5: ∂a¹/∂z¹ = σ'(z¹) = a¹(1-a¹)

Step 6: δ¹ = (W²)ᵀ · δ² ⊙ σ'(z¹)    (⊙ = element-wise multiply)
```

**Gradients for hidden layer weights:**
```
∂L/∂W¹ = δ¹ · xᵀ
∂L/∂b¹ = δ¹
```

### General Pattern

For any layer l:
```
δᴸ = (aᴸ - y)                               [output layer]
δˡ = (Wˡ⁺¹)ᵀ · δˡ⁺¹ ⊙ σ'(zˡ)              [hidden layers]

∂L/∂Wˡ = δˡ · (aˡ⁻¹)ᵀ
∂L/∂bˡ = δˡ
```

> **Key Intuition:** Backprop is just the chain rule applied systematically. The error at the output "flows backwards" through the network, weighted by the connection weights. Each layer computes its share of the blame.

---

## 6. Weight Update Rule

Once we have gradients, update every weight:

```
W = W - η · (∂L/∂W)
b = b - η · (∂L/∂b)

where η = learning rate
```

This is **gradient descent** — move each weight in the direction that reduces the loss.

```
Before update:        After update:
W ────────────→       W ────────→
         ∂L/∂W                      (moved opposite to gradient)
         ←──────
```

**Why subtract the gradient?**
- Gradient points in the direction of steepest INCREASE
- We want to DECREASE the loss
- So we go the OPPOSITE direction

---

## 7. Learning Rate

```
η too large:           η too small:           η just right:
  L                      L                      L
  │╲  ╱╲  ╱             │╲                     │╲
  │  ╲╱  ╲╱             │  ╲                   │  ╲
  │    diverges!         │    ╲.........        │    ╲__
  └──────→ epochs        └──────→ epochs        └──────→ epochs
                         takes forever          converges nicely
```

Typical values: 0.001, 0.01, 0.1 (try powers of 10)

---

## 8. Multi-Layer Perceptron

Stack multiple layers:

```
Input ──→ Hidden₁ ──→ Hidden₂ ──→ ... ──→ Output
     W¹,b¹       W²,b²                Wᴸ,bᴸ
```

**Why multiple layers?**

Single neuron: can only learn a line/hyperplane (linear boundary)
One hidden layer: can approximate any continuous function (Universal Approximation Theorem)
More layers: learn hierarchical features (simple → complex)

```
Layer 1: learns edges
Layer 2: learns shapes (from edges)
Layer 3: learns objects (from shapes)
```

**Parameter count:**
```
Layer with nᵢₙ inputs and nₒᵤₜ outputs:
  Weights: nᵢₙ × nₒᵤₜ
  Biases: nₒᵤₜ
  Total: nᵢₙ × nₒᵤₜ + nₒᵤₜ

Example: 784 → 128 → 64 → 10
  Layer 1: 784×128 + 128 = 100,480
  Layer 2: 128×64 + 64 = 8,256
  Layer 3: 64×10 + 10 = 650
  Total: 109,386 parameters
```

---

## 9. Optimizers

### SGD (Stochastic Gradient Descent)
```
w = w - η · ∇L
```
Simple but can be slow and get stuck.

### SGD with Momentum
```
v = β·v + (1-β)·∇L        (velocity = moving average of gradients)
w = w - η·v
```
Like a ball rolling downhill — builds up speed in consistent directions.

### Adam (Adaptive Moment Estimation)
```
m = β₁·m + (1-β₁)·∇L            (1st moment: mean of gradients)
v = β₂·v + (1-β₂)·(∇L)²         (2nd moment: mean of squared gradients)
m̂ = m / (1-β₁ᵗ)                  (bias correction)
v̂ = v / (1-β₂ᵗ)
w = w - η · m̂ / (√v̂ + ε)
```
- Adapts learning rate per parameter
- Default choice for most problems
- Typical: η=0.001, β₁=0.9, β₂=0.999, ε=1e-8

---

## 10. Training Mechanics

### Epochs, Batches, Mini-batches

```
1 Epoch = one complete pass through the entire training dataset

Dataset: 10,000 samples
Batch size: 32

Per epoch: 10000/32 = 313 mini-batches (iterations)
Each iteration: forward pass + backward pass + weight update on 32 samples
```

| Method | Batch size | Speed | Stability |
|--------|-----------|-------|-----------|
| Batch GD | All data | Slow per step | Smooth |
| SGD | 1 sample | Fast per step | Very noisy |
| Mini-batch | 32-256 | Good balance | Some noise (helps escape local minima) |

---

## 11. By-Hand Example

### Network: 2 inputs → 2 hidden (sigmoid) → 1 output (sigmoid)

```
  x₁ ──w₁──→ h₁ ──w₅──→ o₁
     ╲ w₃ ╱     ╲ w₆ ╱
      ╳         ╳
     ╱ w₂ ╲     ╱    ╲
  x₂ ──w₄──→ h₂ ──────→
```

**Initial values:**
```
Inputs:  x₁ = 0.5,  x₂ = 0.8
Target:  y = 1

Weights (randomly initialized):
  W¹ = [w₁ w₂] = [0.4  0.5]    (h₁ connections)
       [w₃ w₄]   [0.3  0.6]    (h₂ connections)
  b¹ = [0.1, 0.2]

  W² = [w₅ w₆] = [0.7  0.8]    (output connections)
  b² = [0.3]

Learning rate: η = 0.5
```

### FORWARD PASS

**Hidden layer:**
```
z₁ = w₁·x₁ + w₂·x₂ + b₁ = 0.4(0.5) + 0.5(0.8) + 0.1 = 0.2 + 0.4 + 0.1 = 0.7
a₁ = σ(0.7) = 1/(1+e⁻⁰·⁷) = 1/1.4966 = 0.6682

z₂ = w₃·x₁ + w₄·x₂ + b₂ = 0.3(0.5) + 0.6(0.8) + 0.2 = 0.15 + 0.48 + 0.2 = 0.83
a₂ = σ(0.83) = 1/(1+e⁻⁰·⁸³) = 1/1.4360 = 0.6963
```

**Output layer:**
```
z₃ = w₅·a₁ + w₆·a₂ + b₃ = 0.7(0.6682) + 0.8(0.6963) + 0.3
   = 0.4677 + 0.5570 + 0.3 = 1.3247

ŷ = σ(1.3247) = 1/(1+e⁻¹·³²⁴⁷) = 1/1.2660 = 0.7898
```

**Loss:**
```
L = -(y·log(ŷ) + (1-y)·log(1-ŷ))
  = -(1·log(0.7898) + 0·log(0.2102))
  = -log(0.7898)
  = 0.2360
```

### BACKWARD PASS

**Output layer delta:**
```
δ₃ = ŷ - y = 0.7898 - 1 = -0.2102
```

**Output layer gradients:**
```
∂L/∂w₅ = δ₃ · a₁ = -0.2102 × 0.6682 = -0.1404
∂L/∂w₆ = δ₃ · a₂ = -0.2102 × 0.6963 = -0.1463
∂L/∂b₃ = δ₃ = -0.2102
```

**Hidden layer deltas (propagate error back):**
```
δ₁ = (w₅ · δ₃) · σ'(z₁)
   = (0.7 × -0.2102) · a₁(1-a₁)
   = -0.1471 × 0.6682 × (1-0.6682)
   = -0.1471 × 0.2217
   = -0.0326

δ₂ = (w₆ · δ₃) · σ'(z₂)
   = (0.8 × -0.2102) · a₂(1-a₂)
   = -0.1682 × 0.6963 × (1-0.6963)
   = -0.1682 × 0.2116
   = -0.0356
```

**Hidden layer gradients:**
```
∂L/∂w₁ = δ₁ · x₁ = -0.0326 × 0.5 = -0.0163
∂L/∂w₂ = δ₁ · x₂ = -0.0326 × 0.8 = -0.0261
∂L/∂w₃ = δ₂ · x₁ = -0.0356 × 0.5 = -0.0178
∂L/∂w₄ = δ₂ · x₂ = -0.0356 × 0.8 = -0.0285
∂L/∂b₁ = δ₁ = -0.0326
∂L/∂b₂ = δ₂ = -0.0356
```

### WEIGHT UPDATE

```
w₁_new = 0.4 - 0.5×(-0.0163) = 0.4 + 0.0082 = 0.4082
w₂_new = 0.5 - 0.5×(-0.0261) = 0.5 + 0.0130 = 0.5130
w₃_new = 0.3 - 0.5×(-0.0178) = 0.3 + 0.0089 = 0.3089
w₄_new = 0.6 - 0.5×(-0.0285) = 0.6 + 0.0142 = 0.6142

w₅_new = 0.7 - 0.5×(-0.1404) = 0.7 + 0.0702 = 0.7702
w₆_new = 0.8 - 0.5×(-0.1463) = 0.8 + 0.0732 = 0.8732

b₁_new = 0.1 + 0.0163 = 0.1163
b₂_new = 0.2 + 0.0178 = 0.2178
b₃_new = 0.3 + 0.1051 = 0.4051
```

**All weights moved in the direction that would make ŷ closer to y=1.**

The output was 0.7898 (too low for target 1). After one update, if we do another forward pass with the new weights, the output would be higher (closer to 1). Repeat for thousands of iterations → the network learns.

### What Happened

```
Before: ŷ = 0.7898, Loss = 0.2360
  ↓ compute gradients via chain rule (backprop)
  ↓ update all 8 weights + 3 biases
After one step: ŷ will be closer to 1, Loss will decrease

After 1000 steps: ŷ ≈ 0.999, Loss ≈ 0.001
```

> **Key Intuition:** Each weight asks "how much did I contribute to the error?" via the chain rule, then adjusts proportionally. Weights that contributed more to the error change more. This is backpropagation — distributing blame backwards through the network.

---

## What to Look for in the Application Lab

In the application lab, you'll:
1. Implement sigmoid and ReLU from scratch — plot them and their derivatives
2. Build a single neuron — forward pass, loss, weight update on a toy problem
3. Build a 2-layer MLP from scratch in numpy — full forward + backward pass
4. Train it on Fashion-MNIST (2-class subset) and watch the loss decrease
5. Build the same network in PyTorch — compare results
6. Scale to 10 classes with softmax
7. Experiment with learning rates, hidden sizes, and number of epochs
