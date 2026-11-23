"""Hourly day-ahead price for 2014-2015 (Spanish zone; REData indicator 600, equal to the OMIE
marginal price) from the public REE API (apidatos.ree.es). OMIE files can only be downloaded from
2018 onwards and the REE API has no data before 2014.
Checks the match with the paper dataset in January-March 2016.
Output: ../data/ree_spot_prices_2014_2015.csv (Fecha, hora, Precios). The file is already in the
repository; this script documents how it was obtained."""
import time
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
URL = "https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"


def download(start, end):
    r = requests.get(URL, params={"start_date": f"{start}T00:00", "end_date": f"{end}T23:59", "time_trunc": "hour"},
                     timeout=60)
    r.raise_for_status()
    it = [i for i in r.json()["included"] if i["id"] == "600"][0]
    return pd.DataFrame(it["attributes"]["values"])[["datetime", "value"]]


def to_hours(v):
    t = pd.to_datetime(v.datetime.str[:19])           # local start time
    return pd.DataFrame({"Fecha": t.dt.normalize(), "hora": t.dt.hour + 1, "Precios": v.value})


parts = []
for start in pd.date_range("2014-01-01", "2016-03-01", freq="MS"):
    end = (start + pd.offsets.MonthEnd(0)).date()
    parts.append(to_hours(download(start.date(), end))); time.sleep(0.5)
d = pd.concat(parts).groupby(["Fecha", "hora"], as_index=False).Precios.mean()

# check against the paper dataset (Jan-Mar 2016)
ref = pd.read_csv(DATA / "spain_hourly_prices_pv_2016_2025.csv", parse_dates=["Fecha"])
c = d.merge(ref[["Fecha", "hora", "Precios"]], on=["Fecha", "hora"], suffixes=("_ree", "_paper"))
print(f"Check 2016: n={len(c)}, MAE={(c.Precios_ree - c.Precios_paper).abs().mean():.4f} EUR/MWh, "
      f"max={(c.Precios_ree - c.Precios_paper).abs().max():.2f}")
d = d[d.Fecha.dt.year <= 2015]
d.to_csv(DATA / "ree_spot_prices_2014_2015.csv", index=False)
print(d.groupby(d.Fecha.dt.year).agg(n=("Precios", "size"), mean=("Precios", "mean"),
                                     le0=("Precios", lambda s: int((s <= 0).sum()))))
