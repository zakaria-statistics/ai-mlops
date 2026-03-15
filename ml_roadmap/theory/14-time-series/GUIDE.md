# 14 — Time Series
> Data with order — trend, seasonality, stationarity, and ARIMA

## Table of Contents
1. [Components of Time Series](#1-components)
2. [Stationarity](#2-stationarity)
3. [ACF and PACF](#3-acf-pacf)
4. [AR, MA, and ARIMA Models](#4-arima)
5. [Train/Test for Time Series](#5-train-test)
6. [By-Hand Example: AR(1)](#6-by-hand-example)

---

## 1. Components

```
y(t) = Trend + Seasonality + Residual

Trend:       long-term direction (up, down, flat)
Seasonality: repeating patterns at fixed intervals (weekly, yearly)
Residual:    random noise left after removing trend and seasonality
```

**Additive:** y = T + S + R (components are independent of level)
**Multiplicative:** y = T × S × R (seasonal swing grows with level)

---

## 2. Stationarity

A time series is **stationary** if its statistical properties (mean, variance) don't change over time.

Most time series models (ARIMA, AR, MA) require stationarity.

**ADF Test (Augmented Dickey-Fuller):**
- H₀: series has a unit root (non-stationary)
- p < 0.05 → reject H₀ → series IS stationary
- p > 0.05 → series is NOT stationary → needs differencing

**Differencing** to achieve stationarity:
```
First difference: y'(t) = y(t) - y(t-1)    removes linear trend
Second difference: y''(t) = y'(t) - y'(t-1)  removes quadratic trend
```

---

## 3. ACF and PACF

**ACF (Autocorrelation Function):** correlation of series with its own lagged values
```
ACF(k) = Corr(yₜ, yₜ₋ₖ)
```

**PACF (Partial ACF):** correlation at lag k after removing effects of intermediate lags

These help determine ARIMA order:
```
ACF cuts off after q lags     → MA(q)
PACF cuts off after p lags    → AR(p)
Both decay gradually          → ARMA(p,q)
```

---

## 4. ARIMA

### AR(p) — Autoregressive
```
yₜ = c + φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ + εₜ

"Today's value depends on the last p values"
```

### MA(q) — Moving Average
```
yₜ = c + εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θqεₜ₋q

"Today's value depends on the last q forecast errors"
```

### ARIMA(p, d, q)
```
p = AR order (how many past values)
d = differencing order (how many times to difference for stationarity)
q = MA order (how many past errors)
```

### Seasonal ARIMA: SARIMA(p,d,q)(P,D,Q,s)
Adds seasonal terms with period s (e.g., s=12 for monthly data with yearly seasonality).

---

## 5. Train/Test for Time Series

**NEVER random split.** Time series must be split chronologically.

```
WRONG: random shuffle
RIGHT: [──── Train ────│── Test ──]
        past            future
```

**Walk-forward validation:**
```
Step 1: Train on [1..50],  predict 51-55,  evaluate
Step 2: Train on [1..55],  predict 56-60,  evaluate
Step 3: Train on [1..60],  predict 61-65,  evaluate
...
Final: average all evaluation scores
```

---

## 6. By-Hand Example

### Fit AR(1) on 5 Data Points

```
Data: y₁=10, y₂=12, y₃=11, y₄=13, y₅=14
```

AR(1): yₜ = c + φ·yₜ₋₁ + εₜ

**Estimate φ (autocorrelation at lag 1):**
```
ȳ = (10+12+11+13+14)/5 = 12

Pairs (yₜ₋₁, yₜ): (10,12), (12,11), (11,13), (13,14)

φ ≈ Σ(yₜ₋₁ - ȳ)(yₜ - ȳ) / Σ(yₜ₋₁ - ȳ)²

Numerator: (-2)(0) + (0)(-1) + (-1)(1) + (1)(2) = 0+0-1+2 = 1
Denominator: 4+0+1+1 = 6

φ ≈ 1/6 = 0.167
c ≈ ȳ(1-φ) = 12(1-0.167) = 10.0

Model: yₜ = 10.0 + 0.167·yₜ₋₁
Prediction: y₆ = 10.0 + 0.167(14) = 12.33
```

---

## What to Look for in the Application Lab

1. Decompose Airline Passengers into trend + seasonality + residual
2. Test stationarity with ADF, apply differencing
3. Read ACF/PACF plots to choose ARIMA orders
4. Implement AR(1) from scratch
5. Fit ARIMA with statsmodels, forecast, and evaluate
