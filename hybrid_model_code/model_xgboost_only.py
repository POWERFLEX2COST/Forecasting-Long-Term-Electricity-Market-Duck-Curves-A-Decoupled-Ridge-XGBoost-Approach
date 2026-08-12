# =============================================================================
# ABLATION 2: XGBoost ALONE (no Ridge structural trend), predicting the raw
# target directly from calendar features PLUS the year, so it gets a fair
# chance to learn the multi-year trend itself via tree splits.
#
# Purpose: isolate the contribution of the explicit Ridge trend component.
# Same XGBoost hyperparameters/features as the residual corrector in the
# full hybrid (model_ridge_xgboost.py), plus `anio_norm` (year, centred at
# 2019) as an extra feature — the residual model in the hybrid never sees
# year at all, since that job belongs entirely to Ridge.
#
# Expected (and informative) failure mode: gradient-boosted trees CANNOT
# extrapolate beyond the numeric range seen in training for a split feature.
# Every 2026-2035 row has anio_norm > 5 (max seen in training, 2024-2019),
# so all of them fall into the same terminal leaves as the last training
# year and the projection should come out essentially FLAT/frozen at the
# boundary-year pattern — unlike Ridge's polynomial, which extrapolates
# smoothly (for better or worse) beyond the fitted range. This is a concrete,
# textbook justification for why the paper assigns the trend job to a
# parametric model (Ridge) and keeps XGBoost strictly to the residual
# (always-in-range, cyclic) job.
# =============================================================================

import time
import numpy as np
import pandas as pd
import xgboost as xgb

import data_utils as du

MODEL_NAME = "XGBoost only (no Ridge trend)"
SEED = 42
np.random.seed(SEED)

FEATURE_COLS = [
    "hora_sin", "hora_cos", "mes_sin", "mes_cos", "semana_sin", "semana_cos",
    "es_fin_semana", "hora_raw", "mes_raw", "hora_x_mes", "hora_x_diasem",
    "anio_norm",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = du.add_cyclic_features(df).copy()
    df["hora_x_mes"] = df["hora"] * df["mes"]
    df["hora_x_diasem"] = df["hora"] * df["dia_semana"]
    return df


def main():
    t0 = time.time()
    df = du.load_data()
    df = add_features(df)
    df_train_full, df_test = du.train_test_split(df)

    df_train = df_train_full[df_train_full["Año"] <= 2023]
    df_val = df_train_full[df_train_full["Año"] == 2024]

    print(f"[XGBoost-only] Train: {len(df_train)}, Val: {len(df_val)}")
    model = xgb.XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
        tree_method="hist", random_state=SEED, early_stopping_rounds=50,
    )
    model.fit(
        df_train[FEATURE_COLS].values, df_train["target"].values,
        eval_set=[(df_val[FEATURE_COLS].values, df_val["target"].values)],
        verbose=False,
    )
    print(f"[XGBoost-only] Fit done in {time.time() - t0:.1f}s "
          f"(best_iteration={model.best_iteration})")

    # ---- Backtest 2025 -----------------------------------------------------
    print("[XGBoost-only] Forecasting blind 2025 backtest ...")
    df_test = df_test.copy()
    df_test["target_pred"] = model.predict(df_test[FEATURE_COLS].values)

    metrics = du.compute_metrics(
        df_test["target"].values, df_test["target_pred"].values,
        df_test["target_naive"].values,
    )
    print(f"[XGBoost-only] Backtest 2025 metrics: {metrics}")
    du.save_predictions(
        df_test[["timestamp", "target", "target_pred"]],
        "backtest_2025_xgboost_only.csv",
    )
    pd.DataFrame([metrics]).assign(model=MODEL_NAME).to_csv(
        f"{du.PRED_DIR}/metrics_backtest_xgboost_only.csv", index=False
    )

    # ---- Long-horizon projection 2026-2035 (extrapolation-limit check) ----
    print("[XGBoost-only] Forecasting 2026-2035 projection ...")
    df_fut = du.build_future_calendar()
    df_fut = add_features(df_fut)
    df_fut["target_pred"] = model.predict(df_fut[FEATURE_COLS].values)
    du.save_predictions(
        df_fut[["timestamp", "Año", "mes", "hora", "target_pred"]],
        "projection_2026_2035_xgboost_only.csv",
    )
    print(f"[XGBoost-only] Done. Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
