"""Shared helpers for the figure scripts: paths and the projection table in long format."""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def load_projection() -> pd.DataFrame:
    """2019-2035 hourly table (observed 2019-2025, projected 2026-2035) with the column names
    used by the figure scripts: timestamp, Año, anio, mes, hora, porcentaje_sobre_media."""
    p = pd.read_csv(DATA / "spain_electricity_price_pct_monthly_average_2019_2035.csv",
                    parse_dates=["Date and Time"])
    t = p["Date and Time"]
    return pd.DataFrame({"timestamp": t, "Año": t.dt.year, "anio": t.dt.year, "mes": t.dt.month,
                         "hora": t.dt.hour + 1, "porcentaje_sobre_media": p["Price (% of Monthly Average)"]})
