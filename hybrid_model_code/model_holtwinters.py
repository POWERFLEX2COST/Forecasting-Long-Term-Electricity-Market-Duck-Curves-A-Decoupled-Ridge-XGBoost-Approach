# =============================================================================
# BASELINE 2: Holt-Winters / ETS exponential smoothing (damped additive trend
# + additive seasonal component, statsmodels ExponentialSmoothing)
#
# Purpose in the comparison: classical decomposition-based competitor,
# conceptually the closest classical analogue to the paper's own Ridge trend
# component (both decompose the series into trend + seasonal/periodic parts),
# but without an explicit, separately-regularised structural trend per
# hour-month-daytype cell.
#
# Implementation notes:
#   - statsmodels' ExponentialSmoothing supports a SINGLE seasonal period.
#     seasonal_periods=168 (one week) is used, the standard choice for
#     hourly electricity-price series, since it jointly captures the
#     intraday and weekday/weekend cycle; annual seasonality is left to the
#     (damped) trend component, exactly as classical HW would be applied in
#     practice by a analyst without further customisation.
#   - A DAMPED trend is used: with an 87,648-hour (10-year) projection
#     horizon, an undamped linear trend would risk diverging to physically
#     meaningless values; damping is the standard, best-practice safeguard
#     recommended for long-horizon HW forecasts (Hyndman & Athanasopoulos).
# =============================================================================

import time
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import data_utils as du

MODEL_NAME = "Holt-Winters (ETS)"
SEASONAL_PERIODS = 168  # one week, hourly resolution


def main():
    t0 = time.time()
    df = du.load_data()
    df_train, df_test = du.train_test_split(df)

    y_train = df_train["target"].values
    print(f"[HoltWinters] Training observations: {len(y_train)}")

    print("[HoltWinters] Fitting damped-trend additive Holt-Winters "
          f"(seasonal_periods={SEASONAL_PERIODS}) ...")
    model = ExponentialSmoothing(
        y_train, trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=SEASONAL_PERIODS,
        initialization_method="estimated",
    )
    res = model.fit(optimized=True)
    print(f"[HoltWinters] Fit done in {time.time() - t0:.1f}s. "
          f"AIC={res.aic:.1f}  phi(damping)={res.params.get('damping_trend'):.4f}")

    # ---- Backtest 2025 -----------------------------------------------------
    print("[HoltWinters] Forecasting blind 2025 backtest ...")
    df_test = df_test.copy()
    df_test["target_pred"] = np.asarray(res.forecast(steps=len(df_test)))

    metrics = du.compute_metrics(
        df_test["target"].values, df_test["target_pred"].values,
        df_test["target_naive"].values,
    )
    print(f"[HoltWinters] Backtest 2025 metrics: {metrics}")
    du.save_predictions(
        df_test[["timestamp", "target", "target_pred"]],
        "backtest_2025_holtwinters.csv",
    )
    pd.DataFrame([metrics]).assign(model=MODEL_NAME).to_csv(
        f"{du.PRED_DIR}/metrics_backtest_holtwinters.csv", index=False
    )

    # ---- Long-horizon projection 2026-2035 (stability check) ---------------
    print("[HoltWinters] Forecasting 2026-2035 projection (stability check) ...")
    df_fut = du.build_future_calendar()
    # Need one continuous forecast from end of training through end of 2035
    # so the internal level/trend/season states evolve consistently; 2025 is
    # re-forecast here too and discarded, to keep a single coherent call.
    total_steps = len(df_test) + len(df_fut)
    full_fc = np.asarray(res.forecast(steps=total_steps))
    df_fut = df_fut.copy()
    df_fut["target_pred"] = full_fc[len(df_test):]

    du.save_predictions(
        df_fut[["timestamp", "Año", "mes", "hora", "target_pred"]],
        "projection_2026_2035_holtwinters.csv",
    )
    print(f"[HoltWinters] Done. Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
