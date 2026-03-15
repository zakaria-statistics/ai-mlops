# 15 — RNNs and Sequence Models
> Neural networks with memory — RNN, LSTM gates, vanishing gradients, and sequence tasks

## Table of Contents
1. [Why Feedforward Fails for Sequences](#1-why-rnn)
2. [Simple RNN](#2-simple-rnn)
3. [Vanishing Gradient Problem](#3-vanishing-gradient)
4. [LSTM — Long Short-Term Memory](#4-lstm)
5. [GRU — Gated Recurrent Unit](#5-gru)
6. [Sequence Task Types](#6-task-types)
7. [By-Hand Example: RNN Forward Pass](#7-by-hand-example)

---

## 1. Why RNN?

Feedforward networks: fixed input size, no notion of order.
Sequences need: variable length, order matters, memory of past.

```
"The cat sat on the ___"  → need to remember "cat" to predict "mat"
Stock prices: today depends on yesterday and last week
```

---

## 2. Simple RNN

Each time step: combine current input with previous hidden state.

```
hₜ = tanh(Wₕ · hₜ₋₁ + Wₓ · xₜ + b)

hₜ = hidden state at time t (the "memory")
xₜ = input at time t
Wₕ = weight matrix for hidden-to-hidden (recurrent weights)
Wₓ = weight matrix for input-to-hidden
b  = bias
```

**Unrolled:**
```
x₁ ──→ [RNN] ──→ h₁ ──→ [RNN] ──→ h₂ ──→ [RNN] ──→ h₃ ──→ output
         ↑                  ↑                  ↑
        h₀                 h₁                 h₂
     (initial)          (memory)           (memory)

Same weights Wₕ, Wₓ, b shared across ALL time steps
```

---

## 3. Vanishing Gradient

During backpropagation through time (BPTT), gradients multiply through many time steps:

```
∂L/∂W = ∂L/∂hₜ · ∂hₜ/∂hₜ₋₁ · ∂hₜ₋₁/∂hₜ₋₂ · ... · ∂h₁/∂W
         └────────────────────────────────────────┘
                    T multiplications

Each ∂hₜ/∂hₜ₋₁ involves tanh derivative (max 1.0) and Wₕ

If |Wₕ| < 1: gradient → 0 exponentially (VANISHING) → forgets long-term
If |Wₕ| > 1: gradient → ∞ exponentially (EXPLODING) → training unstable
```

> **Key Intuition:** Simple RNNs can't learn long-range dependencies. After ~10-20 time steps, the gradient is essentially zero. This is why LSTM was invented.

---

## 4. LSTM

LSTM adds **gates** that control information flow, allowing long-term memory.

### The Four Components

```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)           ← Forget gate: what to ERASE from memory
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)           ← Input gate: what to WRITE to memory
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)        ← Candidate: new information to consider
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ              ← Cell state: updated memory
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)           ← Output gate: what to READ from memory
hₜ = oₜ ⊙ tanh(Cₜ)                      ← Hidden state: filtered output
```

```
                    ┌───────────────────────────────────────┐
                    │           Cell State (Cₜ)              │
         Cₜ₋₁  ───►│──[×forget]──[+input]──────────────────►│── Cₜ
                    │      ↑          ↑                      │
                    │      fₜ     iₜ ⊙ C̃ₜ                  │
                    │      │       │   │                     │
                    │      σ       σ  tanh                   │
                    │      │       │   │                     │
         hₜ₋₁ ────►├──────┴───────┴───┘                     │
           xₜ ────►│                        ┌──oₜ──tanh(Cₜ)─►── hₜ
                    │                        │   │               │
                    │                        σ   │               │
                    │                        │   │               │
                    └────────────────────────┴───┘               │
                                                                 └──► output
```

**Why it works:**
- Forget gate (fₜ): sigmoid → values between 0 and 1. Multiply with cell state. 0 = forget completely, 1 = keep completely.
- The cell state Cₜ has an **additive** path (not multiplicative like RNN), so gradients flow without vanishing.
- Gates learn WHEN to remember and WHEN to forget.

---

## 5. GRU

Simplified LSTM with 2 gates instead of 3 (faster, often similar performance):

```
zₜ = σ(Wz · [hₜ₋₁, xₜ])         ← Update gate (combines forget + input)
rₜ = σ(Wr · [hₜ₋₁, xₜ])         ← Reset gate
h̃ₜ = tanh(W · [rₜ ⊙ hₜ₋₁, xₜ])  ← Candidate
hₜ = (1-zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ   ← Output (interpolation)
```

No separate cell state — hidden state IS the memory.

---

## 6. Task Types

```
Many-to-One:     x₁,x₂,...,xₜ → y      (sentiment: text → positive/negative)
Many-to-Many:    x₁,x₂,...,xₜ → y₁,...,yₜ  (translation, POS tagging)
One-to-Many:     x → y₁,y₂,...,yₜ      (image captioning)
```

---

## 7. By-Hand Example

### Simple RNN, 3 Time Steps, Hidden Size = 2

```
Inputs: x₁=[1], x₂=[2], x₃=[3]  (scalar inputs)
h₀ = [0, 0]  (initial hidden state)

Wₓ = [0.5]   (1×1 → extended to hidden size by broadcasting)
     [0.3]
Wₕ = [0.1  0.2]
     [0.3  0.1]
b = [0, 0]
```

**Step 1 (t=1):**
```
z₁ = Wₕ·h₀ + Wₓ·x₁ + b
   = [0.1(0)+0.2(0)] + [0.5(1)] + [0] = [0.5]
     [0.3(0)+0.1(0)]   [0.3(1)]   [0]   [0.3]

h₁ = tanh([0.5, 0.3]) = [0.462, 0.291]
```

**Step 2 (t=2):**
```
z₂ = Wₕ·h₁ + Wₓ·x₂ + b
   = [0.1(0.462)+0.2(0.291)] + [0.5(2)] = [0.046+0.058] + [1.0] = [1.104]
     [0.3(0.462)+0.1(0.291)]   [0.3(2)]   [0.139+0.029]   [0.6]   [0.768]

h₂ = tanh([1.104, 0.768]) = [0.801, 0.646]
```

**Step 3 (t=3):**
```
z₃ = Wₕ·h₂ + Wₓ·x₃ + b
   = [0.1(0.801)+0.2(0.646)] + [0.5(3)] = [0.209] + [1.5] = [1.709]
     [0.3(0.801)+0.1(0.646)]   [0.3(3)]   [0.305]   [0.9]   [1.205]

h₃ = tanh([1.709, 1.205]) = [0.937, 0.835]
```

h₃ = [0.937, 0.835] encodes the entire sequence [1, 2, 3]. Use this for prediction.

---

## What to Look for in the Application Lab

1. Implement simple RNN cell from scratch — forward pass through time steps
2. Demonstrate vanishing gradient (multiply weight matrices many times)
3. Use PyTorch nn.RNN and nn.LSTM for time series prediction
4. Compare RNN vs LSTM vs ARIMA on the same data
5. Build LSTM for text sentiment classification
