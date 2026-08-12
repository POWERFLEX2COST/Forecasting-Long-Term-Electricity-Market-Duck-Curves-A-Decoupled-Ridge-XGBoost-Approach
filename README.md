# Forecasting Long-Term Electricity Market Duck Curves: A Decoupled Ridge–XGBoost Approach

Code and data for the paper by Ángel Arcos Vargas and Luis Galiana Fajardo, proposing a hybrid Ridge Regression + XGBoost model to forecast the Spanish day-ahead electricity price profile (2026–2035) as a percentage of the monthly mean, without recursive feedback.

## Contents

- **`hybrid_model_code/`** — Python implementation of the hybrid model, five comparison baselines (Holt-Winters, SARIMAX, Prophet, LSTM, N-BEATS), the ablation study, and the backtesting/comparison harness (paper Sections 3–4).
- **`spain_electricity_price_pct_monthly_average_2019_2035.xlsx`** — Hourly dataset (2019–2035) with date/time, day type (working day/holiday), historical/projected flag, and price as % of monthly average.
