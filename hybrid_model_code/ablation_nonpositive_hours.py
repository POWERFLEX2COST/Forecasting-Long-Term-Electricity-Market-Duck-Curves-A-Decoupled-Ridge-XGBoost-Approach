# =============================================================================
# ABLATION FOCUSED ON HOURS WITH PRICE <= 0: Ridge only vs hybrid (Ridge + XGBoost)
# - Expanding-window backtests: cutoff 2020..2024, each later year up to 2025 (15 cases).
# - Out-of-sample check January-October 2026 (training 2019-2025, as in production).
# - 2026-2035 projection (training 2019-2025).
# For each case: observed and predicted hours <= 0, hour-by-hour hits (precision,
# recall, F1) and error of the monthly distribution (MAE of the monthly counts).
# Outputs: tables/ablation_nonpositive_hours.csv,
#          tables/ablation_nonpositive_hours_projection.csv
# =============================================================================
import os
import numpy as np
import pandas as pd

import data_utils as du
from model_ridge_xgboost import add_xgb_features
from model_ridge_only import RidgeTrendOnly
from horizon_error_analysis import fit_hybrid

EPS = 1e-9


def fit_ridge(df_train):
    m = RidgeTrendOnly()
    m.fit(df_train)
    return m


def scores(obs_le0, pred_le0, month):
    tp = int((obs_le0 & pred_le0).sum())
    n_obs, n_pred = int(obs_le0.sum()), int(pred_le0.sum())
    prec = tp / n_pred if n_pred else np.nan
    rec = tp / n_obs if n_obs else np.nan
    f1 = 2 * prec * rec / (prec + rec) if n_pred and n_obs and (prec + rec) else np.nan
    mo = pd.DataFrame({"m": month, "o": obs_le0, "p": pred_le0}).groupby("m").sum()
    return {"obs": n_obs, "pred": n_pred, "hits": tp, "precision": prec,
            "recall": rec, "F1": f1, "MAE_monthly": float((mo.o - mo.p).abs().mean())}


def main():
    df = du.load_data()
    df_x = add_xgb_features(df)
    rows = []

    # ---- 1) Expanding-window backtests -----------------------------------
    for cutoff in range(2020, 2025):
        tr = df[df["Año"] <= cutoff].copy()
        models = {"Ridge only": fit_ridge(tr), "Hybrid": fit_hybrid(tr)}
        for year in range(cutoff + 1, 2026):
            te = df_x[df_x["Año"] == year].copy()
            obs = (te["target"] <= 0).values
            for name, mod in models.items():
                pred = mod.predict(te) <= EPS
                rows.append({"case": f"{cutoff}->{year}", "cutoff": cutoff, "year": year,
                             "horizon": year - cutoff, "model": name,
                             **scores(obs, pred, te["mes"].values)})
        print("cutoff", cutoff, "done", flush=True)

    # ---- 2) 2026 check and projection (training 2019-2025) ---------------
    tr = df[df["Año"] <= 2025].copy()
    models = {"Ridge only": fit_ridge(tr), "Hybrid": fit_hybrid(tr)}
    fut = add_xgb_features(du.build_future_calendar())
    real = pd.read_csv(du.CSV_2026_PATH, parse_dates=["fecha"])[["fecha", "hora", "precio"]]
    proj = []
    for name, mod in models.items():
        f = fut[["Fecha", "hora", "Año", "mes", "es_fin_semana"]].copy()
        f["pred"] = mod.predict(fut)
        f["fecha"] = pd.to_datetime(f["Fecha"])
        c = f[f["Año"] == 2026].merge(real, on=["fecha", "hora"])
        rows.append({"case": "2025->2026 (Jan-Oct)", "cutoff": 2025, "year": 2026, "horizon": 1,
                     "model": name, **scores((c.precio <= 0).values, (c.pred <= EPS).values, c.mes.values)})
        g = f.assign(le0=f.pred <= EPS).groupby("Año").le0.sum().rename(name)
        proj.append(g)

    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(du.TAB_DIR, "ablation_nonpositive_hours.csv"), index=False, float_format="%.4f")
    P = pd.concat(proj, axis=1)
    P.to_csv(os.path.join(du.TAB_DIR, "ablation_nonpositive_hours_projection.csv"))

    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    print("\nSummary by model (15 backtests):")
    b = T[T.cutoff <= 2024]
    print(b.assign(err=(b.pred - b.obs), abserr=(b.pred - b.obs).abs())
          .groupby("model")[["abserr", "err", "precision", "recall", "F1", "MAE_monthly"]].mean().round(3))
    print("\nProjected hours <= 0:\n", P.to_string())


if __name__ == "__main__":
    main()
