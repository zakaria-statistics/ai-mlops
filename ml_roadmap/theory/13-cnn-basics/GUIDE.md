# 13 — CNNs: Convolutional Neural Networks
> How neural networks see images — convolution, pooling, feature maps, and transfer learning

## Table of Contents
1. [Why Not Fully-Connected for Images?](#1-why-not-fc)
2. [The Convolution Operation](#2-convolution)
3. [Pooling](#3-pooling)
4. [CNN Architecture](#4-cnn-architecture)
5. [Parameter Counting](#5-parameter-counting)
6. [Transfer Learning](#6-transfer-learning)
7. [By-Hand Example: Convolution on a 5x5 Image](#7-by-hand-example)

---

## 1. Why Not FC?

A 224×224 color image = 224×224×3 = 150,528 input values.
First hidden layer of 1000 neurons: 150,528 × 1000 = **150 million weights!**

Problems:
- Way too many parameters → overfits and slow
- No spatial awareness (pixel at (0,0) treated same as pixel at (100,100))
- Doesn't recognize shifted/translated objects

CNNs solve this with: **weight sharing** (same filter everywhere) and **local connectivity** (each neuron sees only a small patch).

---

## 2. Convolution

A small **kernel/filter** (e.g., 3×3) slides over the image, computing element-wise multiply + sum at each position.

```
(I * K)(i,j) = Σₘ Σₙ I(i+m, j+n) · K(m,n)
```

```
Image (5×5):              Kernel (3×3):
┌─────────────┐           ┌──────┐
│ 1  0  1  0  1│           │1  0  1│
│ 0  1  0  1  0│           │0  1  0│
│ 1  0  1  0  1│           │1  0  1│
│ 0  1  0  1  0│           └──────┘
│ 1  0  1  0  1│
└─────────────┘

Position (0,0):
1×1 + 0×0 + 1×1 + 0×0 + 1×1 + 0×0 + 1×1 + 0×0 + 1×1 = 5

The kernel slides → produces output feature map (3×3 with valid padding)
```

**Stride:** How many pixels the kernel moves per step (default=1)
**Padding:**
- Valid (no padding): output shrinks → (n-k+1) × (n-k+1)
- Same (zero padding): output same size as input

**Output size:** (W - K + 2P) / S + 1, where W=input, K=kernel, P=padding, S=stride

---

## 3. Pooling

Reduces spatial dimensions, keeps important information.

**Max Pooling (2×2, stride 2):**
```
Input:           Output:
┌────┬────┐
│ 1 3│ 2 4│      ┌───┐
│ 5 2│ 6 1│  →   │5  6│
├────┼────┤      │8  7│
│ 3 8│ 7 2│      └───┘
│ 4 1│ 3 5│
└────┴────┘
Take max of each 2×2 block → halves each dimension
```

Why: translation invariance (small shifts don't change the max), reduces computation.

---

## 4. CNN Architecture

```
Input → [Conv → ReLU → Pool] × N → Flatten → Dense → Output

Typical:
  Image (32×32×3)
    → Conv2d(32 filters, 3×3) → ReLU → MaxPool(2×2)   → 16×16×32
    → Conv2d(64 filters, 3×3) → ReLU → MaxPool(2×2)   → 8×8×64
    → Flatten                                           → 4096
    → Dense(128) → ReLU                                 → 128
    → Dense(10) → Softmax                               → 10 classes
```

Multiple filters = multiple feature maps (each learns a different pattern: edges, corners, textures).

---

## 5. Parameter Counting

```
Conv layer:  (kernel_h × kernel_w × in_channels + 1) × out_channels
             (+1 for bias per filter)

Example: Conv2d(in=3, out=32, kernel=3×3)
  Params = (3×3×3 + 1) × 32 = 28 × 32 = 896

Dense layer: (in_features + 1) × out_features
  Flatten(4096) → Dense(128): (4096+1)×128 = 524,416

Note: Conv layers have FAR fewer params than dense layers!
```

---

## 6. Transfer Learning

Use a network pretrained on millions of images (e.g., ImageNet), adapt to your task.

```
Pretrained ResNet:
  [Conv layers — learned general features] → [Dense → 1000 ImageNet classes]
                    ↓                                      ↓
                 FREEZE these                         REPLACE with:
                 (don't retrain)                      [Dense → YOUR classes]

Training: only update the new dense layer (fast, works with small data)
Fine-tuning: optionally unfreeze last few conv layers too
```

> **Key Intuition:** Early conv layers learn universal features (edges, textures). Later layers learn task-specific features. Reusing early layers saves enormous training time.

---

## 7. By-Hand Example

### 3×3 Edge Detection Kernel on a 5×5 Image

```
Image:                    Kernel (horizontal edge):
┌─────────────┐           ┌────────┐
│ 0  0  0  0  0│           │-1 -1 -1│
│ 0  0  0  0  0│           │ 0  0  0│
│ 1  1  1  1  1│           │ 1  1  1│
│ 1  1  1  1  1│           └────────┘
│ 1  1  1  1  1│
└─────────────┘

Position (0,0): 0(-1)+0(-1)+0(-1) + 0(0)+0(0)+0(0) + 1(1)+1(1)+1(1) = 3
Position (1,0): 0(-1)+0(-1)+0(-1) + 1(0)+1(0)+1(0) + 1(1)+1(1)+1(1) = 3
Position (2,0): 1(-1)+1(-1)+1(-1) + 1(0)+1(0)+1(0) + 1(1)+1(1)+1(1) = 0

Output (3×3):
┌─────┐
│3 3 3│    ← edge detected (row where 0→1 transition happens)
│3 3 3│
│0 0 0│    ← no edge (uniform area)
└─────┘
```

The kernel detected the horizontal edge between the dark (0) and light (1) regions.

---

## What to Look for in the Application Lab

1. Implement 2D convolution from scratch with numpy
2. Apply edge/blur/sharpen kernels and see the output
3. Build a CNN in PyTorch for CIFAR-10
4. Compare MLP vs CNN accuracy (CNN should win significantly)
5. Use pretrained ResNet18 with transfer learning — see how fast it trains
