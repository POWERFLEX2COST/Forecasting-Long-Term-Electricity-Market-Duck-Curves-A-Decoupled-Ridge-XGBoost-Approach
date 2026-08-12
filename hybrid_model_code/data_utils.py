# =============================================================================
# SHARED DATA/METRICS MODULE FOR FORECASTING-METHOD COMPARISON
# Mirrors exactly the data preparation used in the paper's production scripts
# (Analisis precios PV/XGBoost_backtexting.py, Horas_negativas_Ridge_XGBoost.py):
# target = price as % of monthly mean, train 2019-2024, blind test 2025,
# naive week-ahead benchmark, MAE/RMSE/WAPE/rMAE/rRMSE per Weron (2014) /
# Lago et al. (2021).
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

CSV_PATH = (
    r"C:\Users\lgali\OneDrive\Documents\POWERFLEX2COST\Analisis precios PV"
    r"\Dataset_Modelo_PrecioFV_Porcentaje.csv"
)

TRAIN_START_YEAR = 2019
TRAIN_END_YEAR = 2024     # Ridge/XGB fitted on 2019-2024 for the 2025 blind backtest
TEST_YEAR = 2025
PROJ_START_YEAR = 2026
PROJ_END_YEAR = 2035

PRED_DIR = os.path.join(os.path.dirname(__file__), "predictions")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), "tables")
for _d in (PRED_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(_d, exist_ok=True)


def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load OMIE hourly prices and build the relative-price target exactly as
    in the production Ridge+XGBoost scripts."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df[df["Fecha"].dt.year >= TRAIN_START_YEAR]
    df = df.sort_values(["Fecha", "hora"]).reset_index(drop=True)

    df["mes"] = df["Fecha"].dt.month
    df["dia_semana"] = df["Fecha"].dt.dayofweek
    df["semana_anio"] = df["Fecha"].dt.isocalendar().week.astype(int)
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)

    df["precio_medio_mensual"] = df.groupby(["Año", "mes"])["Precios"].transform("mean")
    df["target"] = df["Precios"] / df["precio_medio_mensual"] * 100

    df["timestamp"] = df["Fecha"] + pd.to_timedelta(df["hora"] - 1, unit="h")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Naive week-ahead benchmark (Lago et al. 2021), merged by timestamp so it
    # is robust to any missing hours / DST irregularities.
    df_naive = df[["timestamp", "target"]].copy()
    df_naive["timestamp"] = df_naive["timestamp"] + pd.Timedelta(weeks=1)
    df_naive = df_naive.rename(columns={"target": "target_naive"})
    df = df.merge(df_naive, on="timestamp", how="left")

    return df


def build_future_calendar(start_year: int = PROJ_START_YEAR,
                           end_year: int = PROJ_END_YEAR) -> pd.DataFrame:
    """Deterministic calendar frame for the projection horizon — no price
    information at all, consistent with the paper's no-recursive-feedback
    design constraint (Sec. 3.2 of main.tex)."""
    rango = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31 23:00:00",
        freq="h",
    )
    df_fut = pd.DataFrame({"timestamp": rango})
    df_fut["Fecha"] = df_fut["timestamp"].dt.date
    df_fut["hora"] = df_fut["timestamp"].dt.hour + 1
    df_fut["Año"] = df_fut["timestamp"].dt.year
    df_fut["mes"] = df_fut["timestamp"].dt.month
    df_fut["dia_semana"] = df_fut["timestamp"].dt.dayofweek
    df_fut["semana_anio"] = df_fut["timestamp"].dt.isocalendar().week.astype(int)
    df_fut["es_fin_semana"] = (df_fut["dia_semana"] >= 5).astype(int)
    return df_fut


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["semana_sin"] = np.sin(2 * np.pi * df["semana_anio"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana_anio"] / 52)
    df["hora_raw"] = df["hora"]
    df["mes_raw"] = df["mes"]
    df["anio_norm"] = df["Año"] - TRAIN_START_YEAR
    return df


def train_test_split(df: pd.DataFrame):
    df_train = df[df["Año"] <= TRAIN_END_YEAR].copy()
    df_test = df[df["Año"] == TEST_YEAR].copy()
    return df_train, df_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_naive: np.ndarray) -> dict:
    """MAE / RMSE / WAPE / rMAE / rRMSE, identical definitions to
    XGBoost_backtexting.py and Eq. (8)-(9) of main.tex."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_naive))
    y_true, y_pred, y_naive = y_true[mask], y_pred[mask], y_naive[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_naive = mean_absolute_error(y_true, y_naive)
    rmse_naive = np.sqrt(mean_squared_error(y_true, y_naive))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape * 100,
        "rMAE": mae / mae_naive,
        "rRMSE": rmse / rmse_naive,
        "n": int(mask.sum()),
    }


def save_predictions(df: pd.DataFrame, filename: str):
    path = os.path.join(PRED_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  -> Saved predictions: {path}")
    return path
