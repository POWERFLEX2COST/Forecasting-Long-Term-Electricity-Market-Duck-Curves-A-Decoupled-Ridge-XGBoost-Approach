# =============================================================================
# ABLATION 1: Ridge trend component ALONE (no XGBoost residual correction)
#
# Purpose: isolate the contribution of the residual-correction stage. This
# uses the exact same 576 cell-wise quadratic/linear Ridge fits (alpha=1000,
# same physical constraint on solar hours 9-17) as the full hybrid model in
# model_ridge_xgboost.py, but the final prediction IS the trend — the
# intra-annual residual is left unmodelled (equivalent to predicting the
# residual as identically zero everywhere).
# =============================================================================

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import data_utils as du

MODEL_NAME = "Ridge only (no residual correction)"
RIDGE_ALPHA = 1000
MIN_OBS_CELDA = 20


class RidgeTrendOnly:
    def __init__(self):
        self.modelos_tendencia = {}

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
        print("  -> Fitting 576 Ridge trend cells (2019-2024) ...")
        for m in range(1, 13):
            for h in range(1, 25):
                for f in [0, 1]:
                    clave = (m, h, f)
                    mask = (df_fit["mes"] == m) & (df_fit["hora"] == h) & (df_fit["es_fin_semana"] == f)
                    if mask.sum() < MIN_OBS_CELDA:
                        continue
                    X_lr = df_fit.loc[mask, ["Año"]].values
                    y_lr = df_fit.loc[mask, "target"].values
                    self.modelos_tendencia[clave] = self._ajustar_tendencia(X_lr, y_lr, clave)

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
        return np.clip(df_p["trend_pred"].values, a_min=-50, a_max=None)


def main():
    t0 = time.time()
    df = du.load_data()
    df_train, df_test = du.train_test_split(df)

    print(f"[Ridge-only] Training observations: {len(df_train)}")
    modelo = RidgeTrendOnly()
    modelo.fit(df_train)
    print(f"[Ridge-only] Fit done in {time.time() - t0:.1f}s")

    # ---- Backtest 2025 -----------------------------------------------------
    print("[Ridge-only] Forecasting blind 2025 backtest ...")
    df_test = df_test.copy()
    df_test["target_pred"] = modelo.predict(df_test)

    metrics = du.compute_metrics(
        df_test["target"].values, df_test["target_pred"].values,
        df_test["target_naive"].values,
    )
    print(f"[Ridge-only] Backtest 2025 metrics: {metrics}")
    du.save_predictions(
        df_test[["timestamp", "target", "target_pred"]],
        "backtest_2025_ridge_only.csv",
    )
    pd.DataFrame([metrics]).assign(model=MODEL_NAME).to_csv(
        f"{du.PRED_DIR}/metrics_backtest_ridge_only.csv", index=False
    )

    # ---- Long-horizon projection 2026-2035 ---------------------------------
    print("[Ridge-only] Forecasting 2026-2035 projection ...")
    df_fut = du.build_future_calendar()
    df_fut["target_pred"] = modelo.predict(df_fut)
    du.save_predictions(
        df_fut[["timestamp", "Año", "mes", "hora", "target_pred"]],
        "projection_2026_2035_ridge_only.csv",
    )
    print(f"[Ridge-only] Done. Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
