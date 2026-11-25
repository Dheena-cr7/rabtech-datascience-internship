# Task 5: Time-Series Forecasting with SARIMA & Prophet

**RabTech Academy · Data Science with Python & R Internship**

**Notebook:** [`notebooks/demand_forecasting.ipynb`](notebooks/demand_forecasting.ipynb) (executed, all outputs included)
**Forecast table:** [`reports/forecast_july_2026.csv`](reports/forecast_july_2026.csv)

**Data:** RabTech Academy `daily-demand-series.csv`, daily demand for 1–30 June 2026 with marketing-event and holiday flags.
**Split:** train on 1–23 June, **test on the last 7 days (24–30 June)**.

## What's covered

| Requirement | Notebook section |
|---|---|
| Decomposition into Trend, Seasonality and Residual (classical and STL) | §2 |
| Augmented Dickey-Fuller test (plus KPSS), differencing, ACF/PACF | §3 |
| SARIMA with AIC grid-search hyperparameter selection (192 models) | §5 |
| SARIMAX with exogenous event and holiday flags | §6 |
| Prophet with hyperparameter tuning on an inner validation week | §7 |
| MAPE and RMSE on the test split vs baselines and Holt-Winters | §4, §8 |
| Residual diagnostics (Ljung-Box, Q-Q) | §9 |
| 30-day forecast projection with 80% and 95% prediction intervals | §10 |

## Key results

| Model | Test MAPE | Test RMSE |
|---|---|---|
| **SARIMA(0,1,0)(0,1,0)₇** | **0.19%** | **0.53** |
| Seasonal naive + drift | 0.45% | 1.13 |
| Prophet (tuned) | 0.74% | 1.39 |
| Holt-Winters | 0.78% | 1.59 |
| SARIMAX (event + holiday) | 2.36% | 3.84 |
| Naive | 8.25% | 17.03 |

- Trend: **+0.97 units/day**. Weekly seasonality peaks on Saturday (+19 units).
- The original series is non-stationary (ADF p = 0.65) and becomes stationary after first differencing (ADF p < 0.001).
- **July 2026 forecast: ≈ 5,063 units (+18.8% vs June)**. SARIMA and Prophet agree within about 1% per week.

## Reproduce

```bash
uv sync --locked
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/demand_forecasting.ipynb
```
