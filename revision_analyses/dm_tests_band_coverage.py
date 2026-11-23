# =============================================================================
# DIEBOLD-MARIANO TESTS, EUR/MWh BACKTEST AND ERROR-BAND COVERAGE (revised version)
#   a) empirical coverage of the 95 % band: leave-one-out over the 15
#      (cutoff, test year) combinations and leave-one-cutoff-out (the rows of
#      the same trained model are removed), plus the observed 2026 check.
#   b) 2025 backtest of all methods also in EUR/MWh.
#   c) Diebold-Mariano test of the hybrid against each alternative
#      (absolute and squared loss; HAC Newey-West, lag 168 h; robustness
#      check with daily mean losses and lag 7 d).
# All alternatives receive the same post-processing (floor 0 + renormalisation).
# Requires: ../hybrid_model_code/predictions/backtest_2025_*.csv (run_comparison.py)
#           ../hybrid_model_code/tables/horizon_error_*.csv (horizon_error_analysis.py)
# Outputs (results/): dm_tests.csv, backtest_eur_all.csv, band_coverage.csv
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
COD = os.path.join(HERE, "..", "hybrid_model_code")
sys.path.insert(0, COD)
import data_utils as du  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

METHODS = {"ridge_xgboost": "Ridge + XGBoost", "ridge_only": "Ridge only", "xgboost_only": "XGBoost only",
           "holtwinters": "Holt-Winters", "sarimax": "SARIMAX", "prophet": "Prophet",
           "lstm": "LSTM", "nbeats": "N-BEATS"}
ALREADY_PP = ("ridge_xgboost", "ridge_only")
OBS_2026 = {"rMAE": 0.881, "rRMSE": 0.805}   # from check_2026.py (results/check_2026_monthly.csv, Jan-Oct)


def dm(d, lag):
    """DM with HAC (Newey-West) variance. d = loss(hybrid) - loss(other); < 0 favours the hybrid."""
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return r.params[0], r.tvalues[0], r.pvalues[0]


def main():
    df = du.load_data()
    nv = df[["timestamp", "Precios"]].copy()
    nv["timestamp"] += pd.Timedelta(weeks=1)
    df = df.merge(nv.drop_duplicates("timestamp").rename(columns={"Precios": "precio_naive"}),
                  on="timestamp", how="left")  # without duplicating rows (DST change)
    base = df[df["Año"] == 2025][["timestamp", "Año", "mes", "target", "target_naive",
                                  "Precios", "precio_naive", "precio_medio_mensual"]]

    pred = {}
    for k in METHODS:
        b = pd.read_csv(f"{du.PRED_DIR}/backtest_2025_{k}.csv", parse_dates=["timestamp"])
        b = base.merge(b[["timestamp", "target_pred"]], on="timestamp", how="inner").reset_index(drop=True)
        p = b.target_pred.values if k in ALREADY_PP else du.postprocess(b.target_pred.values, b)
        pred[k] = b.assign(p=p)

    # ---- b) EUR/MWh
    rows = []
    for k, b in pred.items():
        b = b.dropna(subset=["precio_naive", "target_naive"])
        e = b.Precios - b.p * b.precio_medio_mensual / 100
        en = b.Precios - b.precio_naive
        rows.append({"method": METHODS[k], "MAE_eur": e.abs().mean(), "RMSE_eur": np.sqrt((e ** 2).mean()),
                     "rMAE_eur": e.abs().mean() / en.abs().mean(),
                     "rRMSE_eur": np.sqrt((e ** 2).mean()) / np.sqrt((en ** 2).mean()),
                     "MAE_naive_eur": en.abs().mean(), "RMSE_naive_eur": np.sqrt((en ** 2).mean())})
    eur = pd.DataFrame(rows)
    eur.to_csv(os.path.join(OUT, "backtest_eur_all.csv"), index=False)
    print("=== Backtest 2025 in EUR/MWh (realised monthly level; same post-processing)")
    print(eur.round(3).to_string(index=False))

    # ---- c) Diebold-Mariano
    h = pred["ridge_xgboost"].dropna(subset=["target_naive"])
    rivals = {k: v for k, v in pred.items() if k != "ridge_xgboost"}
    rows = []
    for k, b in list(rivals.items()) + [("naive", None)]:
        o = h[["timestamp"]].merge(b[["timestamp", "p"]], on="timestamp") if b is not None else \
            h[["timestamp"]].assign(p=h.target_naive.values)
        hh = h[h.timestamp.isin(o.timestamp)].reset_index(drop=True)
        o = o.set_index("timestamp").loc[hh.timestamp].reset_index()
        e1, e2 = hh.target.values - hh.p.values, hh.target.values - o.p.values
        for loss, f in (("absolute", np.abs), ("squared", np.square)):
            d = f(e1) - f(e2)
            m, t, pv = dm(d, 168)
            dd = pd.Series(d).groupby(hh.timestamp.dt.date.values).mean().values
            _, td, pvd = dm(dd, 7)
            rows.append({"rival": METHODS.get(k, "Naive (t-168)"), "loss": loss, "mean_diff": m,
                         "DM_hourly_HAC168": t, "p_hourly": pv, "DM_daily_HAC7": td, "p_daily": pvd, "n": len(d)})
    dmt = pd.DataFrame(rows)
    dmt.to_csv(os.path.join(OUT, "dm_tests.csv"), index=False)
    print("\n=== Diebold-Mariano, hybrid vs rival (mean_diff < 0 favours the hybrid)")
    print(dmt.round(4).to_string(index=False))

    # ---- a) band coverage
    he = pd.read_csv(f"{du.TAB_DIR}/horizon_error_analysis.csv")
    rows = []
    for metric in ("MAE", "RMSE", "WAPE", "rMAE", "rRMSE"):
        for scheme in ("leave-one-out", "leave-one-cutoff-out"):
            inside, widths = [], []
            for i, row in he.iterrows():
                tr = he.drop(i) if scheme == "leave-one-out" else he[he.train_end != row.train_end]
                if tr.ratio.nunique() < 2:
                    continue
                ols = sm.OLS(tr[metric], sm.add_constant(tr[["ratio"]])).fit()
                fr = ols.get_prediction(np.array([[1.0, row.ratio]])).summary_frame(alpha=0.05)
                lo, hi = fr.obs_ci_lower.iloc[0], fr.obs_ci_upper.iloc[0]
                inside.append(lo <= row[metric] <= hi); widths.append(hi - lo)
            rows.append({"metric": metric, "scheme": scheme, "n": len(inside),
                         "coverage_95": np.mean(inside), "mean_width": np.mean(widths),
                         "binom_p_vs_0.95": stats.binomtest(int(np.sum(inside)), len(inside), 0.95).pvalue})
    cov = pd.DataFrame(rows)
    band = pd.read_csv(f"{du.TAB_DIR}/horizon_error_projected_uncertainty.csv").set_index("proj_year")
    for metric, obs in OBS_2026.items():
        lo, hi = band.loc[2026, f"{metric}_lo95"], band.loc[2026, f"{metric}_hi95"]
        cov = pd.concat([cov, pd.DataFrame([{"metric": metric, "scheme": "real 2026 (Jan-Oct)", "n": 1,
                                             "coverage_95": float(lo <= obs <= hi), "mean_width": hi - lo,
                                             "obs": obs, "lo95": lo, "hi95": hi}])])
    cov.to_csv(os.path.join(OUT, "band_coverage.csv"), index=False)
    print("\n=== Coverage of the 95 % band")
    print(cov.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
