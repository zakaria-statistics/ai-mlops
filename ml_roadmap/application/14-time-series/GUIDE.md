# Lab 14: Time Series
> Decompose, test stationarity, and forecast — AR from scratch, ARIMA, walk-forward validation

## Table of Contents
1. [Setup](#setup) - Dataset and libraries
2. [Plot the Series](#step-1-plot-the-series) - Visual inspection
3. [Decompose](#step-2-decompose) - Trend, seasonal, residual
4. [Stationarity](#step-3-stationarity-test) - ADF test and differencing
5. [ACF and PACF](#step-4-acf-and-pacf) - Identify AR/MA orders
6. [AR from Scratch](#step-5-ar-from-scratch) - Simple autoregressive model
7. [ARIMA](#step-6-arima-with-statsmodels) - Full model fitting
8. [Train/Test Split](#step-7-train-test-split) - Chronological split
9. [Forecast and Plot](#step-8-forecast-and-plot) - Predictions vs actual
10. [Evaluate](#step-9-evaluate) - MAE, RMSE, MAPE
11. [Alternative Models](#step-10-alternative-models) - Exponential Smoothing or Prophet
12. [Walk-Forward Validation](#step-11-walk-forward-validation) - Realistic evaluation

## Prerequisites
- Read `theory/14-time-series/GUIDE.md` first
- Completed Labs 01-03 (data handling, regression concepts)
- Understanding of: autocorrelation, stationarity, differencing (from theory)

## Dataset
**Airline Passengers** — monthly international airline passengers, 1949-1960
- 144 observations, monthly frequency
- Clear trend (upward) + seasonality (annual cycle)
- Classic time series dataset, available everywhere:
  ```python
  # Option 1: seaborn
  import seaborn as sns
  df = sns.load_dataset('flights')

  # Option 2: direct CSV
  # https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
  ```

## Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

---

## Step 1: Plot the Series

1. Load and prepare:
   ```python
   df = sns.load_dataset('flights')
   df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str))
   ts = df.set_index('date')['passengers']
   ```
2. Plot the full time series
3. Identify visually:
   - **Trend:** is it going up, down, flat?
   - **Seasonality:** do you see repeating patterns?
   - **Variance:** is the spread constant or growing?

**Expected output:** Clear upward trend. Annual seasonality (peaks every summer). Variance increases over time (multiplicative seasonality).

---

## Step 2: Decompose

1. Decompose into components:
   ```python
   decomposition = seasonal_decompose(ts, model='multiplicative', period=12)
   decomposition.plot()
   ```
2. Try both `model='additive'` and `model='multiplicative'` — which fits better?
3. Examine each component:
   - **Trend:** smooth upward curve
   - **Seasonal:** repeating annual pattern
   - **Residual:** what's left — should look random

**Expected output:** Multiplicative model fits better (because variance grows with level). Seasonal component shows consistent annual cycle. Residuals should be roughly centered around 1.0.

> **Checkpoint 1:** You can visually decompose a time series and identify its components.

---

## Step 3: Stationarity Test

A stationary series has constant mean and variance over time. ARIMA requires stationarity.

1. Plot the original series — clearly non-stationary (trend + growing variance)
2. Run the Augmented Dickey-Fuller (ADF) test:
   ```python
   result = adfuller(ts)
   print(f'ADF Statistic: {result[0]:.4f}')
   print(f'p-value: {result[1]:.4f}')
   ```
   If p > 0.05, the series is non-stationary.

3. **First differencing:** `ts_diff = ts.diff().dropna()` — removes trend
4. Re-run ADF on differenced series — p-value should drop
5. If still non-stationary, try **second differencing** or **log transform + differencing**:
   ```python
   ts_log_diff = np.log(ts).diff().dropna()
   ```
6. Plot each transformation and its ADF result

**Expected output:** Original series: p >> 0.05 (non-stationary). After log + differencing: p < 0.05 (stationary). This tells us d=1 for ARIMA.

---

## Step 4: ACF and PACF

Use the stationary (differenced) series:

1. Plot ACF:
   ```python
   plot_acf(ts_log_diff, lags=36)
   ```
2. Plot PACF:
   ```python
   plot_pacf(ts_log_diff, lags=36)
   ```
3. Interpret:
   - **PACF cuts off at lag p** -> AR(p) component
   - **ACF cuts off at lag q** -> MA(q) component
   - Significant spikes at lag 12 indicate seasonal component
4. Use these to pick ARIMA(p, d, q) orders

**Expected output:** PACF suggests p=1 or p=2. ACF suggests q=1 or q=2. Lag 12 spikes in both indicate seasonal ARIMA may be needed.

> **Checkpoint 2:** You can determine ARIMA orders from ACF/PACF plots.

---

## Step 5: AR from Scratch

Implement a simple AR(1) model: y_t = phi * y_{t-1} + c + epsilon

1. Use the differenced (stationary) series
2. Fit the AR coefficient:
   ```python
   # y_t = phi * y_{t-1}
   y = ts_diff.values[1:]       # current values
   y_lag = ts_diff.values[:-1]  # lagged values

   # Least squares fit
   phi = np.sum(y * y_lag) / np.sum(y_lag ** 2)
   print(f"AR(1) coefficient: {phi:.4f}")
   ```
3. Predict: for each t, predict y_t = phi * y_{t-1}
4. Plot: actual vs predicted (on the differenced series)
5. Convert back to original scale by cumulative sum + first value

**Expected output:** AR(1) captures some of the pattern but misses seasonality. Predictions lag behind the actual values (a common AR(1) behavior).

---

## Step 6: ARIMA with statsmodels

1. Fit ARIMA with orders from ACF/PACF analysis:
   ```python
   model = ARIMA(ts, order=(2, 1, 2))
   fitted = model.fit()
   print(fitted.summary())
   ```
2. Check the summary: AIC, BIC, coefficient significance (p-values)
3. Try a few order combinations and compare AIC:
   ```
   | Order     | AIC    |
   |-----------|--------|
   | (1,1,1)   |        |
   | (2,1,1)   |        |
   | (2,1,2)   |        |
   | (1,1,2)   |        |
   ```
4. Pick the model with lowest AIC
5. Check residuals: `fitted.plot_diagnostics()` — should look like white noise

**Expected output:** ARIMA(2,1,2) or similar fits well. Residuals should be approximately normally distributed with no significant autocorrelation.

---

## Step 7: Train/Test Split

Time series splits must be chronological — no shuffling.

1. Use last 12 months as test:
   ```python
   train = ts[:-12]
   test = ts[-12:]
   print(f"Train: {train.index[0]} to {train.index[-1]}")
   print(f"Test: {test.index[0]} to {test.index[-1]}")
   ```
2. Fit ARIMA on train only
3. Never use future data during training

**Expected output:** Train: 1949-01 to 1959-12. Test: 1960-01 to 1960-12.

---

## Step 8: Forecast and Plot

1. Generate forecast:
   ```python
   model = ARIMA(train, order=(2, 1, 2))
   fitted = model.fit()
   forecast = fitted.forecast(steps=12)
   ```
2. Plot: full historical series + forecast + actual test values
3. Add confidence intervals if available:
   ```python
   pred = fitted.get_forecast(steps=12)
   ci = pred.conf_int()
   ```

**Expected output:** Forecast should follow the general upward trend. It may miss exact seasonality peaks. Confidence intervals widen as you forecast further ahead.

> **Checkpoint 3:** You've built a forecasting model and visualized predictions vs actuals.

---

## Step 9: Evaluate

Compute forecast metrics:

1. **MAE** — Mean Absolute Error:
   ```python
   mae = mean_absolute_error(test, forecast)
   ```
2. **RMSE** — Root Mean Squared Error:
   ```python
   rmse = np.sqrt(mean_squared_error(test, forecast))
   ```
3. **MAPE** — Mean Absolute Percentage Error:
   ```python
   mape = np.mean(np.abs((test - forecast) / test)) * 100
   ```
4. Print all three. MAPE gives the most intuitive interpretation (e.g., "off by 5% on average")

**Expected output:** MAPE around 5-15% depending on ARIMA order. Lower is better.

---

## Step 10: Alternative Models

Try Exponential Smoothing (Holt-Winters):

1. Fit with trend and seasonality:
   ```python
   hw_model = ExponentialSmoothing(
       train, trend='mul', seasonal='mul', seasonal_periods=12
   ).fit()
   hw_forecast = hw_model.forecast(12)
   ```
2. Plot: ARIMA forecast vs Holt-Winters forecast vs actual
3. Compute MAE/RMSE/MAPE for Holt-Winters
4. Compare with ARIMA — which is more accurate?

**Optional:** If Prophet is installed:
```python
from prophet import Prophet
df_prophet = pd.DataFrame({'ds': train.index, 'y': train.values})
m = Prophet(yearly_seasonality=True)
m.fit(df_prophet)
future = m.make_future_dataframe(periods=12, freq='MS')
pred = m.predict(future)
```

**Expected output:** Holt-Winters often handles multiplicative seasonality better than basic ARIMA on this dataset. Results table:

```
| Model          | MAE  | RMSE | MAPE  |
|----------------|------|------|-------|
| AR(1) scratch  |      |      |       |
| ARIMA(2,1,2)   |      |      |       |
| Holt-Winters   |      |      |       |
```

---

## Step 11: Walk-Forward Validation

More realistic than a single train/test split — retrain at each step:

1. Start with first 120 months as training
2. Forecast 1 month ahead
3. Add that month to training, retrain, forecast next month
4. Repeat for last 24 months:
   ```python
   history = list(train.values)
   predictions = []
   for t in range(len(test)):
       model = ARIMA(history, order=(2, 1, 2))
       fitted = model.fit()
       yhat = fitted.forecast(steps=1)[0]
       predictions.append(yhat)
       history.append(test.values[t])  # add actual observation
   ```
5. Plot walk-forward predictions vs actual
6. Compute metrics — should be better than static forecast (model adapts)

**Expected output:** Walk-forward predictions follow the actual series more closely than static 12-step forecast. MAPE should be lower.

> **Checkpoint 4:** You've implemented proper time series evaluation with walk-forward validation, compared multiple models, and understand the full forecasting pipeline.

---

## Summary Deliverables
- [ ] Time series plot with trend and seasonality identified
- [ ] Decomposition (multiplicative)
- [ ] ADF stationarity test — before and after differencing
- [ ] ACF/PACF plots with order interpretation
- [ ] AR(1) from scratch
- [ ] ARIMA fitted and evaluated
- [ ] Holt-Winters comparison
- [ ] Walk-forward validation results
- [ ] Model comparison table (MAE, RMSE, MAPE)
