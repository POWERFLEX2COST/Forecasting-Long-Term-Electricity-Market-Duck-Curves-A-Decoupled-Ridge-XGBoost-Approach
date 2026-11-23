# =============================================================================
# NORMALISATION OF THE TARGET AND ERRORS IN EUR/MWh WITH AN EX-ANTE LEVEL (revised version)
#  - error in EUR/MWh with a price level known in advance (ex ante), not the realised one;
#  - sensitivity to the normalisation of the target.
# Normalisations: monthly mean (proposed), daily mean, monthly median, annual mean.
# Each one: target = price / normaliser x 100; same model (cell-wise quadratic Ridge +
# XGBoost); post-processing = floor 0 + rescaling to mean 100 within the normaliser's group
# (except the median, which only gets the floor).
# Tests: 2025 backtest (training 2019-2024) and Jan-Oct 2026 check (training 2019-2025).
# Conversion to EUR/MWh ex ante: level of the same period of the previous year
#   (monthly and daily -> mean of the same month of the previous year; median -> median of the
#    same month; annual -> mean of the previous year).
# EUR/MWh references: weekly naive (t-168 h, uses the real prices of the previous week) and
# annual naive (same weekday 52 weeks earlier, t-364 d), which is the ex-ante reference.
# Output: results/normalisation_eur.csv
# =============================================================================
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "hybrid_model_code"))
import data_utils as du                                  # noqa: E402
from model_ridge_xgboost import add_xgb_features, XGB_FEATURE_COLS   # noqa: E402
from sensitivity import Hybrid                            # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
DAY_THRESHOLD = 5.0   # EUR/MWh: days with a lower mean are not used for training with the daily normalisation


def prices():
    """Hourly prices 2019-2025 (paper dataset) + Jan-Oct 2026 (check)."""
    h = du.load_data()[["Fecha", "hora", "Precios"]]
    r = pd.read_csv(du.CSV_2026_PATH, parse_dates=["fecha"])
    r = r.rename(columns={"fecha": "Fecha", "precio": "Precios"})[["Fecha", "hora", "Precios"]]
    d = pd.concat([h, r]).drop_duplicates(["Fecha", "hora"]).sort_values(["Fecha", "hora"]).reset_index(drop=True)
    d["Año"], d["mes"], d["dia"] = d.Fecha.dt.year, d.Fecha.dt.month, d.Fecha.dt.day
    d["dia_semana"] = d.Fecha.dt.dayofweek
    d["semana_anio"] = d.Fecha.dt.isocalendar().week.astype(int)
    d["es_fin_semana"] = (d.dia_semana >= 5).astype(int)
    d["timestamp"] = d.Fecha + pd.to_timedelta(d.hora - 1, unit="h")
    g = d.groupby
    d["N_mes"] = g(["Año", "mes"]).Precios.transform("mean")
    d["N_dia"] = g(["Año", "mes", "dia"]).Precios.transform("mean")
    d["N_med"] = g(["Año", "mes"]).Precios.transform("median")
    d["N_anio"] = g("Año").Precios.transform("mean")
    # naive forecasts in EUR/MWh
    for name, lag in (("naive168", pd.Timedelta(hours=168)), ("naive364", pd.Timedelta(days=364))):
        s = d[["timestamp", "Precios"]].copy(); s["timestamp"] += lag
        d = d.merge(s.rename(columns={"Precios": name}), on="timestamp", how="left")
    # ex-ante levels (previous year)
    nm = d.groupby(["Año", "mes"]).agg(m=("Precios", "mean"), md=("Precios", "median")).reset_index()
    nm["Año"] += 1
    d = d.merge(nm.rename(columns={"m": "L_mes", "md": "L_med"}), on=["Año", "mes"], how="left")
    na = d.groupby("Año").Precios.mean().rename("L_anio").reset_index(); na["Año"] += 1
    return add_xgb_features(d.merge(na, on="Año", how="left"))


VARIANTS = {  # name: (normaliser, rescaling group, ex-ante level)
    "Monthly mean (proposed)": ("N_mes", ["Año", "mes"], "L_mes"),
    "Daily mean": ("N_dia", ["Año", "mes", "dia"], "L_mes"),
    "Monthly median": ("N_med", None, "L_med"),
    "Annual mean": ("N_anio", ["Año"], "L_anio"),
}


def post(raw, d, group):
    x = pd.Series(np.clip(raw, 0, None), index=d.index)
    return (x / x.groupby([d[c] for c in group]).transform("mean") * 100).values if group else x.values


def evaluate(te, pred, norm, level):
    obs = te.Precios.values
    x_obs = obs / te[norm].values * 100
    x_nv = te.naive168.values / te[norm].values * 100          # weekly naive on the same scale
    ok = ~np.isnan(te.naive168.values) & ~np.isnan(te.naive364.values) & ~np.isnan(te[level].values)
    p_ex = pred * te[level].values / 100                         # EUR/MWh with ex-ante level
    p_or = pred * te[norm].values / 100                          # EUR/MWh with realised level (oracle)
    mae = lambda a: np.abs(obs[ok] - a[ok]).mean()  # noqa: E731
    le0_o, le0_p = obs <= 0, pred <= 1e-9
    return {"rMAE_rel": np.abs(x_obs - pred)[ok].mean() / np.abs(x_obs - x_nv)[ok].mean(),
            "MAE_eur_exante": mae(p_ex), "MAE_eur_oracle": mae(p_or),
            "MAE_naive168": mae(te.naive168.values), "MAE_naive364": mae(te.naive364.values),
            "rMAE_exante_vs_naive364": mae(p_ex) / mae(te.naive364.values),
            "rMAE_exante_vs_naive168": mae(p_ex) / mae(te.naive168.values),
            "le0_obs": int(le0_o.sum()), "le0_pred": int(le0_p.sum()), "le0_hits": int((le0_o & le0_p).sum()),
            "r_profile": pd.Series(obs).groupby([te.mes.values, te.hora.values]).mean().corr(
                pd.Series(p_ex).groupby([te.mes.values, te.hora.values]).mean())}


def main():
    d = prices()
    print("Days 2019-2026 with daily mean <", DAY_THRESHOLD, "EUR/MWh:",
          int((d.groupby(["Año", "mes", "dia"]).N_dia.first() < DAY_THRESHOLD).sum()),
          "| with mean <= 0:", int((d.groupby(["Año", "mes", "dia"]).N_dia.first() <= 0).sum()))
    rows = []
    for name, (norm, group, level) in VARIANTS.items():
        for test, end, year in (("Backtest 2025", 2024, 2025), ("Check Jan-Oct 2026", 2025, 2026)):
            tr = d[d.Año.between(2019, end)].copy()
            if norm == "N_dia":
                tr = tr[tr.N_dia >= DAY_THRESHOLD]
            tr["target"] = tr.Precios / tr[norm] * 100
            mod = Hybrid(2019, 2, False).fit(tr)
            te = d[d.Año == year].copy()
            trend = pd.Series(100.0, index=te.index)
            for k, g in te.groupby(["mes", "hora", "es_fin_semana"]):
                if k in mod.trend:
                    trend.loc[g.index] = mod.trend[k].predict(g[["Año"]].values - 2019)
            raw = trend.values + mod.xgb.predict(te[XGB_FEATURE_COLS].values)
            pred = post(raw, te, group)
            row = {"normalisation": name, "test": test, **evaluate(te, pred, norm, level)}
            if end == 2025:   # projection: hours <= 0 (depends only on the sign of the raw prediction)
                fut = add_xgb_features(du.build_future_calendar())
                tf = pd.Series(100.0, index=fut.index)
                for k, g in fut.groupby(["mes", "hora", "es_fin_semana"]):
                    if k in mod.trend:
                        tf.loc[g.index] = mod.trend[k].predict(g[["Año"]].values - 2019)
                rf = tf.values + mod.xgb.predict(fut[XGB_FEATURE_COLS].values)
                for a in (2030, 2035):
                    row[f"le0_proj_{a}"] = int((rf[fut["Año"].values == a] <= 0).sum())
            rows.append(row)
            print(name, test, "done", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "normalisation_eur.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
