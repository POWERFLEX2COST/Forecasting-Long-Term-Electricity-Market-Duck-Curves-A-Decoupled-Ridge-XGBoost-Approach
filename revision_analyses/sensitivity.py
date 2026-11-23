# =============================================================================
# SENSITIVITY OF THE HYBRID MODEL (revised version) to
#   a) the solar constraint (on / off)
#   b) the training window (start 2019-2022, without 2020, without 2022-23)
#   c) the degree of the trend polynomial (1, 2, 3)
#   d) errors of the 2025 backtest also in EUR/MWh
# For each configuration:
#   - blind 2025 backtest: training <= 2024 (XGBoost: validation = last year)
#   - 2026-2035 projection: training <= 2025 (XGBoost train <= 2024, validation 2025)
#   - out-of-sample check with the observed January-October 2026 prices
# The post-processing (floor at 0 + monthly renormalisation) is the same for all.
# Outputs (results/): sens_backtest.csv, sens_projection.csv, sens_trajectories.csv,
#                     sens_2026.csv, fig_sensitivity_h14_h20.png
# =============================================================================

import sys
import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "hybrid_model_code"))
import data_utils as du                                    # noqa: E402
from model_ridge_xgboost import add_xgb_features, XGB_FEATURE_COLS  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

ALPHA = 1000
CONFIGS = [
    # name, training start, excluded years, degree, solar constraint
    # The base case has no solar constraint; the submitted version appears as a variant.
    ("Base (2019-, deg 2)", 2019, (), 2, False),
    ("With solar constraint (submitted)", 2019, (), 2, True),
    ("Linear trend (deg 1)", 2019, (), 1, False),
    ("Cubic trend (deg 3)", 2019, (), 3, False),
    ("Train from 2020", 2020, (), 2, False),
    ("Train from 2021", 2021, (), 2, False),
    ("Train from 2022", 2022, (), 2, False),
    ("Excl. 2020 (COVID)", 2019, (2020,), 2, False),
    ("Excl. 2022-23 (gas cap)", 2019, (2022, 2023), 2, False),
]


class Hybrid:
    """Configurable version of the hybrid (cell-wise Ridge trend + XGBoost on residuals)."""

    def __init__(self, start, degree, constraint):
        self.start, self.degree, self.constraint = start, degree, constraint
        self.trend, self.xgb = {}, None
        self.n_linear = 0

    def _cell(self, x, y, h):
        m = make_pipeline(PolynomialFeatures(self.degree), Ridge(alpha=ALPHA)).fit(x, y)
        if self.constraint and self.degree >= 2 and 9 <= h <= 17:
            p25, p30 = m.predict([[2025 - self.start], [2030 - self.start]])
            if p30 > p25:
                self.n_linear += 1
                m = make_pipeline(PolynomialFeatures(1), Ridge(alpha=ALPHA)).fit(x, y)
        return m

    def fit(self, d):
        d = d.copy()
        d["trend"] = 100.0
        for (m, h, f), g in d.groupby(["mes", "hora", "es_fin_semana"]):
            if len(g) < 20:
                continue
            x = g[["Año"]].values - self.start
            mod = self._cell(x, g["target"].values, h)
            self.trend[(m, h, f)] = mod
            d.loc[g.index, "trend"] = mod.predict(x)
        d["residual"] = d["target"] - d["trend"]
        last = d["Año"].max()
        tr, va = d["Año"] < last, d["Año"] == last
        self.xgb = xgb.XGBRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="reg:squarederror", tree_method="hist",
            random_state=42, early_stopping_rounds=50)
        self.xgb.fit(d.loc[tr, XGB_FEATURE_COLS].values, d.loc[tr, "residual"].values,
                     eval_set=[(d.loc[va, XGB_FEATURE_COLS].values, d.loc[va, "residual"].values)],
                     verbose=False)
        return self

    def predict(self, d, trend_only=False):
        trend = pd.Series(100.0, index=d.index)
        for (m, h, f), g in d.groupby(["mes", "hora", "es_fin_semana"]):
            if (m, h, f) in self.trend:
                trend.loc[g.index] = self.trend[(m, h, f)].predict(g[["Año"]].values - self.start)
        raw = trend.values if trend_only else trend.values + self.xgb.predict(d[XGB_FEATURE_COLS].values)
        return du.postprocess(raw, d)


def metrics_eur(test, pred_pct):
    """Error in EUR/MWh given the realised monthly level: price = % x monthly mean / 100."""
    p_eur = pred_pct * test["precio_medio_mensual"].values / 100
    real = test["Precios"].values
    naive = test["precio_naive"].values
    ok = ~np.isnan(naive)
    e, en = real[ok] - p_eur[ok], real[ok] - naive[ok]
    return {"MAE_eur": np.abs(e).mean(), "RMSE_eur": np.sqrt((e ** 2).mean()),
            "rMAE_eur": np.abs(e).mean() / np.abs(en).mean(),
            "rRMSE_eur": np.sqrt((e ** 2).mean()) / np.sqrt((en ** 2).mean())}


def main():
    t0 = time.time()
    df = du.load_data()
    nv = df[["timestamp", "Precios"]].copy()
    nv["timestamp"] += pd.Timedelta(weeks=1)
    df = df.merge(nv.drop_duplicates("timestamp").rename(columns={"Precios": "precio_naive"}),
                  on="timestamp", how="left")  # without duplicating rows (DST change)
    df = add_xgb_features(df)
    test = df[df["Año"] == 2025].copy()
    fut = add_xgb_features(du.build_future_calendar())
    # observed 2026 (January-October)
    r26 = pd.read_csv(du.CSV_2026_PATH)
    r26["Fecha"] = pd.to_datetime(r26["fecha"]).dt.date
    r26 = r26[["Fecha", "hora", "x", "x_naive", "precio", "P_mes", "precio_naive"]]

    rows_bt, rows_pr, traj, rows_26 = [], [], [], []
    for name, start, excl, degree, constr in CONFIGS:
        base = df[(df["Año"] >= start) & (~df["Año"].isin(excl))]

        # ---- 2025 backtest
        mod = Hybrid(start, degree, constr).fit(base[base["Año"] <= 2024])
        for var, only in (("hybrid", False), ("ridge_only", True)):
            p = mod.predict(test, trend_only=only)
            m = du.compute_metrics(test["target"].values, p, test["target_naive"].values)
            rows_bt.append({"config": name, "model": var, "n_linear_cells": mod.n_linear,
                            **m, **metrics_eur(test, p),
                            "hours_le0_pred": int((p <= 0).sum()),
                            "hours_le0_real": int((test["target"] <= 0).sum())})

        # ---- 2026-2035 projection
        mod = Hybrid(start, degree, constr).fit(base)
        p = fut.assign(pred=mod.predict(fut))
        c = p[["Fecha", "hora", "pred"]].merge(r26, on=["Fecha", "hora"]).dropna()
        e, en = c.x - c.pred, c.x - c.x_naive
        ee, een = c.precio - c.pred * c.P_mes / 100, c.precio - c.precio_naive
        rows_26.append({"config": name, "n": len(c), "MAE": e.abs().mean(),
                        "rMAE": e.abs().mean() / en.abs().mean(),
                        "rRMSE": np.sqrt((e ** 2).mean()) / np.sqrt((en ** 2).mean()),
                        "MAE_eur": ee.abs().mean(), "rMAE_eur": ee.abs().mean() / een.abs().mean(),
                        "hours_le0_pred": int((c.pred <= 0).sum()), "hours_le0_real": int((c.x <= 0).sum())})
        for a in range(2026, 2036):
            g = p[p["Año"] == a]
            traj.append({"config": name, "year": a,
                         "h14": g[g.hora == 14].pred.mean(), "h20": g[g.hora == 20].pred.mean(),
                         "hours_le0": int((g.pred <= 0).sum())})
        for a in (2026, 2030, 2035):
            g = p[p["Año"] == a]
            rows_pr.append({"config": name, "year": a, "n_linear_cells": mod.n_linear,
                            "h14": g[g.hora == 14].pred.mean(), "h20": g[g.hora == 20].pred.mean(),
                            "valley_9_17": g[g.hora.between(9, 17)].pred.mean(),
                            "max_month_hour": g.groupby(["mes", "hora"]).pred.mean().max(),
                            "hours_le0": int((g.pred <= 0).sum())})
        print(f"{name:34s} done ({time.time() - t0:.0f}s)", flush=True)

    bt, pr, tr = pd.DataFrame(rows_bt), pd.DataFrame(rows_pr), pd.DataFrame(traj)
    bt.to_csv(os.path.join(OUT, "sens_backtest.csv"), index=False)
    pr.to_csv(os.path.join(OUT, "sens_projection.csv"), index=False)
    tr.to_csv(os.path.join(OUT, "sens_trajectories.csv"), index=False)
    c26 = pd.DataFrame(rows_26)
    c26.to_csv(os.path.join(OUT, "sens_2026.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\n=== Observed 2026 check (Jan-Oct)")
    print(c26.round(3).to_string(index=False))
    print("\n=== Backtest 2025")
    print(bt.drop(columns=["n"]).round(3).to_string(index=False))
    print("\n=== Projection")
    print(pr.round(1).to_string(index=False))

    # ---- figure
    hist = df[df["Año"] <= 2025]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for ax, col, tit in zip(axs, ("h14", "h20", "hours_le0"),
                            ("Hour 14:00 (% of monthly mean)", "Hour 20:00 (% of monthly mean)",
                             "Hours with price <= 0 per year")):
        if col == "hours_le0":
            hh = hist.groupby("Año").target.apply(lambda s: (s <= 0).sum())
        else:
            hh = hist[hist.hora == int(col[1:])].groupby("Año").target.mean()
        ax.plot(hh.index, hh.values, "k-o", lw=2, label="Actual")
        for i, (name, *_) in enumerate(CONFIGS):
            s = tr[tr.config == name]
            ax.plot(s.year, s[col], "-" if i == 0 else "--", lw=2.5 if i == 0 else 1.2,
                    color="C3" if i == 0 else f"C{i if i < 3 else i + 1}", label=name)
        ax.set_title(tit, fontweight="bold"); ax.set_xlabel("Year"); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.suptitle("Sensitivity of the 2026-2035 projection (floor 0 % + monthly renormalisation in all cases)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sensitivity_h14_h20.png"), dpi=200)


if __name__ == "__main__":
    main()
