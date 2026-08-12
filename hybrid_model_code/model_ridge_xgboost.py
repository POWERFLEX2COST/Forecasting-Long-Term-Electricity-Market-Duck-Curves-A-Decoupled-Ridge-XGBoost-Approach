# =============================================================================
# REFERENCE MODEL: Ridge + XGBoost hybrid (the paper's own model), reproduced
# through the SAME harness (data_utils.py) used for all other baselines, so
# that backtest metrics and 2026-2035 stability are directly comparable.
#
# Note: this reproduction trains Ridge+XGBoost on 2019-2024 only (Ridge
# trend fitted on 2019-2024 annual means, XGBoost train 2019-2023 / early
# stopping validation 2024) for BOTH the 2025 backtest AND the 2026-2035
# stability projection, so that every method in the comparison uses an
# identical training cutoff. This differs slightly from the paper's final
# published 2026-2035 projection (Horas_negativas_Ridge_XGBoost.py), which
# retrains on the full 2019-2025 dataset once backtesting is validated —
# that difference is immaterial to the backtest metrics (identical to
# XGBoost_backtexting.py) and only affects the projection by one extra
# training year, which if anything makes this reproduction a slightly more
# conservative (harder) test for the hybrid model.
# =============================================================================

import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import data_utils as du

MODEL_NAME = "Ridge + XGBoost (proposed)"
RIDGE_ALPHA = 1000
MIN_OBS_CELDA = 20
SEED = 42
np.random.seed(SEED)

XGB_FEATURE_COLS = [
    "hora_sin", "hora_cos", "mes_sin", "mes_cos", "semana_sin", "semana_cos",
    "es_fin_semana", "hora_raw", "mes_raw", "hora_x_mes", "hora_x_diasem",
]


def add_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    df = du.add_cyclic_features(df).copy()
    df["hora_x_mes"] = df["hora"] * df["mes"]
    df["hora_x_diasem"] = df["hora"] * df["dia_semana"]
    return df


class ModeloHibrido:
    def __init__(self):
        self.modelos_tendencia = {}
        self.xgb = None

    def _ajustar_tendencia(self, X_anios, y, clave):
        m, h, f = clave
        X_norm = X_anios - du.TRAIN_START_YEAR
        m_cuad = make_pipeline(PolynomialFeatures(2), Ridge(alpha=RIDGE_ALPHA))
        m_cuad.fit(X_norm, y)
        p2025 = m_cuad.predict([[2025 - du.TRAIN_START_YEAR]])[0]
        p2030 = m_cuad.predict([[2030 - du.TRAIN_START_YEAR]])[0]
        if (p2030 > p2025) and (9 <= h <= 17):
            m_lin = make_pipeline(PolynomialFeatures(1), Ridge(alpha=RIDGE_ALPHA))
            m_lin.fit(X_norm, y)
            return m_lin
        return m_cuad

    def fit(self, df_fit: pd.DataFrame):
        df_fit = df_fit.copy()
        df_fit["trend"] = 0.0
        print("  -> Fitting 576 Ridge trend cells ...")
        for m in range(1, 13):
            for h in range(1, 25):
                for f in [0, 1]:
                    clave = (m, h, f)
                    mask = (df_fit["mes"] == m) & (df_fit["hora"] == h) & (df_fit["es_fin_semana"] == f)
                    if mask.sum() < MIN_OBS_CELDA:
                        continue
                    X_lr = df_fit.loc[mask, ["Año"]].values
                    y_lr = df_fit.loc[mask, "target"].values
                    modelo = self._ajustar_tendencia(X_lr, y_lr, clave)
                    self.modelos_tendencia[clave] = modelo
                    df_fit.loc[mask, "trend"] = modelo.predict(X_lr - du.TRAIN_START_YEAR)

        df_fit["trend"] = df_fit["trend"].replace(0.0, 100.0)
        df_fit["residual"] = df_fit["target"] - df_fit["trend"]

        print("  -> Training XGBoost on residuals (train<=2023, val=2024) ...")
        mask_tr = df_fit["Año"] <= 2023
        mask_val = df_fit["Año"] == 2024
        X_train = df_fit.loc[mask_tr, XGB_FEATURE_COLS].values
        y_train = df_fit.loc[mask_tr, "residual"].values
        X_val = df_fit.loc[mask_val, XGB_FEATURE_COLS].values
        y_val = df_fit.loc[mask_val, "residual"].values

        self.xgb = xgb.XGBRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
            tree_method="hist", random_state=SEED, early_stopping_rounds=50,
        )
        self.xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    def predict(self, df_pred: pd.DataFrame) -> np.ndarray:
        df_p = df_pred.copy()
        df_p["trend_pred"] = 100.0
        for m in range(1, 13):
            for h in range(1, 25):
                for f in [0, 1]:
                    clave = (m, h, f)
                    mask = (df_p["mes"] == m) & (df_p["hora"] == h) & (df_p["es_fin_semana"] == f)
                    if clave in self.modelos_tendencia:
                        df_p.loc[mask, "trend_pred"] = self.modelos_tendencia[clave].predict(
                            df_p.loc[mask, ["Año"]].values - du.TRAIN_START_YEAR
                        )
        residual_pred = self.xgb.predict(df_p[XGB_FEATURE_COLS].values)
        return np.clip(df_p["trend_pred"].values + residual_pred, a_min=-50, a_max=None)


def main():
    t0 = time.time()
    df = du.load_data()
    df = add_xgb_features(df)
    df_train, df_test = du.train_test_split(df)

    print(f"[Ridge+XGBoost] Training observations: {len(df_train)}")
    modelo = ModeloHibrido()
    modelo.fit(df_train)
    print(f"[Ridge+XGBoost] Fit done in {time.time() - t0:.1f}s")

    # ---- Backtest 2025 -----------------------------------------------------
    print("[Ridge+XGBoost] Forecasting blind 2025 backtest ...")
    df_test = df_test.copy()
    df_test["target_pred"] = modelo.predict(df_test)

    metrics = du.compute_metrics(
        df_test["target"].values, df_test["target_pred"].values,
        df_test["target_naive"].values,
    )
    print(f"[Ridge+XGBoost] Backtest 2025 metrics: {metrics}")
    du.save_predictions(
        df_test[["timestamp", "target", "target_pred"]],
        "backtest_2025_ridge_xgboost.csv",
    )
    pd.DataFrame([metrics]).assign(model=MODEL_NAME).to_csv(
        f"{du.PRED_DIR}/metrics_backtest_ridge_xgboost.csv", index=False
    )

    # ---- Long-horizon projection 2026-2035 (stability check) --------------
    print("[Ridge+XGBoost] Forecasting 2026-2035 projection (stability check) ...")
    df_fut = du.build_future_calendar()
    df_fut = add_xgb_features(df_fut)
    df_fut["target_pred"] = modelo.predict(df_fut)

    du.save_predictions(
        df_fut[["timestamp", "Año", "mes", "hora", "target_pred"]],
        "projection_2026_2035_ridge_xgboost.csv",
    )
    print(f"[Ridge+XGBoost] Done. Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
