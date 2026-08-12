# =============================================================================
# BASELINE 1: SARIMAX with Fourier-term exogenous regressors
# (dynamic harmonic regression, Hyndman & Athanasopoulos 2021, ch. 12.1)
#
# Purpose in the comparison: SARIMA/ARIMA-family models are the classical
# statistical benchmark for time series forecasting and are exactly the
# family the paper (Sec. 3.2, main.tex) argues is structurally unsuited to
# decade-long hourly projection: (i) forecasts are produced by the model's
# internal recursive state-space propagation, so uncertainty/error compounds
# over the horizon; (ii) without an explicit structural (non-stationary)
# trend component, the model cannot represent the year-on-year deepening of
# the solar "duck curve" and will revert to a fixed seasonal pattern learned
# from 2019-2024.
#
# Implementation notes:
#   - A literal seasonal_order at hourly resolution (daily=24, weekly=168,
#     annual=8766) is computationally intractable in statsmodels' state-space
#     MLE. Instead we follow standard practice for multi-seasonal series:
#     non-seasonal ARMA errors + exogenous Fourier terms encoding the daily,
#     weekly and annual cycles ("dynamic harmonic regression").
#   - order=(2,0,2), no differencing: the % of monthly-mean target is
#     already a bounded, roughly stationary series; d=1 over an 87,648-hour
#     projection horizon would risk numerically runaway extrapolation, which
#     is a less interesting (and less fair) failure mode than the one the
#     paper actually predicts (flat/outdated seasonal pattern).
#   - Forecasting is done with a single .get_forecast(steps=n) call: this is
#     the standard, non-"manual-lag-feedback" way of using SARIMA, but it
#     still relies on the model's internal recursive propagation, which is
#     precisely what is being tested here.
# =============================================================================

import time
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import data_utils as du

MODEL_NAME = "SARIMAX"
ORDER = (2, 0, 2)

DAILY_PERIOD = 24.0
WEEKLY_PERIOD = 168.0
ANNUAL_PERIOD = 365.25 * 24.0
K_DAILY, K_WEEKLY, K_ANNUAL = 4, 2, 3


def fourier_terms(hours_since_origin: np.ndarray) -> pd.DataFrame:
    cols = {}
    for period, K, tag in [
        (DAILY_PERIOD, K_DAILY, "d"),
        (WEEKLY_PERIOD, K_WEEKLY, "w"),
        (ANNUAL_PERIOD, K_ANNUAL, "y"),
    ]:
        for k in range(1, K + 1):
            angle = 2 * np.pi * k * hours_since_origin / period
            cols[f"fourier_{tag}_sin{k}"] = np.sin(angle)
            cols[f"fourier_{tag}_cos{k}"] = np.cos(angle)
    return pd.DataFrame(cols)


def build_exog(timestamps: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
    hours = (timestamps - origin) / pd.Timedelta(hours=1)
    return fourier_terms(hours.values)


def main():
    t0 = time.time()
    df = du.load_data()
    df_train, df_test = du.train_test_split(df)
    origin = df_train["timestamp"].iloc[0]

    print(f"[SARIMAX] Training observations: {len(df_train)}")
    y_train = df_train["target"].values
    exog_train = build_exog(df_train["timestamp"], origin)

    print(f"[SARIMAX] Fitting SARIMAX{ORDER} with "
          f"{exog_train.shape[1]} Fourier exogenous regressors ...")
    model = SARIMAX(
        y_train, exog=exog_train.values, order=ORDER,
        trend="c", enforce_stationarity=False, enforce_invertibility=False,
    )
    res = model.fit(disp=False, method="lbfgs", maxiter=200)
    print(f"[SARIMAX] Fit done in {time.time() - t0:.1f}s. AIC={res.aic:.1f}")

    # ---- Backtest 2025 (blind, single multi-step forecast call) -----------
    print("[SARIMAX] Forecasting blind 2025 backtest ...")
    exog_test = build_exog(df_test["timestamp"], origin)
    fc_test = res.get_forecast(steps=len(df_test), exog=exog_test.values)
    df_test = df_test.copy()
    df_test["target_pred"] = fc_test.predicted_mean
    ci = fc_test.conf_int(alpha=0.05)
    df_test["ci_lower"] = ci[:, 0]
    df_test["ci_upper"] = ci[:, 1]

    metrics = du.compute_metrics(
        df_test["target"].values, df_test["target_pred"].values,
        df_test["target_naive"].values,
    )
    print(f"[SARIMAX] Backtest 2025 metrics: {metrics}")
    du.save_predictions(
        df_test[["timestamp", "target", "target_pred", "ci_lower", "ci_upper"]],
        "backtest_2025_sarimax.csv",
    )
    pd.DataFrame([metrics]).assign(model=MODEL_NAME).to_csv(
        f"{du.PRED_DIR}/metrics_backtest_sarimax.csv", index=False
    )

    # ---- Long-horizon projection 2026-2035 (stability check) --------------
    print("[SARIMAX] Forecasting 2026-2035 projection (stability check) ...")
    df_fut = du.build_future_calendar()
    exog_fut = build_exog(df_fut["timestamp"], origin)
    fc_fut = res.get_forecast(steps=len(df_fut), exog=exog_fut.values)
    df_fut["target_pred"] = fc_fut.predicted_mean
    ci_fut = fc_fut.conf_int(alpha=0.05)
    df_fut["ci_lower"] = ci_fut[:, 0]
    df_fut["ci_upper"] = ci_fut[:, 1]

    du.save_predictions(
        df_fut[["timestamp", "Año", "mes", "hora", "target_pred", "ci_lower", "ci_upper"]],
        "projection_2026_2035_sarimax.csv",
    )
    print(f"[SARIMAX] Done. Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
