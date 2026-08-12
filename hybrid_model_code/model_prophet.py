# =============================================================================
# BASELINE 3: Prophet (Meta/Facebook) — additive trend + multiple seasonalities
#
# Purpose in the comparison: Prophet is, philosophically, the closest
# published competitor to the paper's own hybrid design: it decomposes the
# series into an explicit trend component (piecewise-linear, with automatic
# changepoint detection) plus additive seasonal (Fourier) terms, and — like
# the Ridge+XGBoost hybrid — is NON-recursive: once fitted, it predicts any
# future timestamp directly from the model equation, with no lag feedback.
# This makes it the fairest "apples-to-apples" benchmark for the paper's
# central claim that a structural-trend + residual-correction architecture
# is preferable to a purely statistical one.
#
# Implementation notes:
#   - Default hyperparameters (linear growth, automatic changepoints,
#     additive daily/weekly/yearly seasonality) are used throughout: this is
#     the standard, out-of-the-box way any Elsevier Energy reviewer would
#     expect Prophet to be applied, not a deliberately weakened setup.
# =============================================================================

import time
import numpy as np
import pandas as pd
from prophet import Prophet

import data_utils as du

MODEL_NAME = "Prophet"


def main():
    t0 = time.time()
    df = du.load_data()
    df_train, df_test = du.train_test_split(df)

    train_prophet = df_train[["timestamp", "target"]].rename(
        columns={"timestamp": "ds", "target": "y"}
    )
    print(f"[Prophet] Training observations: {len(train_prophet)}")

    model = Prophet(
        growth="linear",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
    )
    print("[Prophet] Fitting model (MAP estimation) ...")
    model.fit(train_prophet)
    print(f"[Prophet] Fit done in {time.time() - t0:.1f}s")

    # ---- Backtest 2025 -----------------------------------------------------
    print("[Prophet] Forecasting blind 2025 backtest ...")
    future_test = df_test[["timestamp"]].rename(columns={"timestamp": "ds"})
    fc_test = model.predict(future_test)

    df_test = df_test.copy()
    df_test["target_pred"] = fc_test["yhat"].values
    df_test["ci_lower"] = fc_test["yhat_lower"].values
    df_test["ci_upper"] = fc_test["yhat_upper"].values

    metrics = du.compute_metrics(
        df_test["target"].values, df_test["target_pred"].values,
        df_test["target_naive"].values,
    )
    print(f"[Prophet] Backtest 2025 metrics: {metrics}")
    du.save_predictions(
        df_test[["timestamp", "target", "target_pred", "ci_lower", "ci_upper"]],
        "backtest_2025_prophet.csv",
    )
    pd.DataFrame([metrics]).assign(model=MODEL_NAME).to_csv(
        f"{du.PRED_DIR}/metrics_backtest_prophet.csv", index=False
    )

    # ---- Long-horizon projection 2026-2035 (stability check) ---------------
    print("[Prophet] Forecasting 2026-2035 projection (stability check) ...")
    df_fut = du.build_future_calendar()
    future_proj = df_fut[["timestamp"]].rename(columns={"timestamp": "ds"})
    fc_fut = model.predict(future_proj)

    df_fut["target_pred"] = fc_fut["yhat"].values
    df_fut["ci_lower"] = fc_fut["yhat_lower"].values
    df_fut["ci_upper"] = fc_fut["yhat_upper"].values

    du.save_predictions(
        df_fut[["timestamp", "Año", "mes", "hora", "target_pred", "ci_lower", "ci_upper"]],
        "projection_2026_2035_prophet.csv",
    )
    print(f"[Prophet] Done. Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
