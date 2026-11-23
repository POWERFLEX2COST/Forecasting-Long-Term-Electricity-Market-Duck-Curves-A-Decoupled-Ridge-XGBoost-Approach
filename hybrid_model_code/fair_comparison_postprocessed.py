# =============================================================================
# FAIR COMPARISON WITH THE ALTERNATIVES (revised version)
#
# The hybrid and Ridge-only outputs are already post-processed (floor at 0 % +
# monthly renormalisation, data_utils.postprocess). Here EXACTLY the same
# post-processing is applied to the outputs of the five alternatives and of
# XGBoost only, so the annual mean is 100 % for every method by construction and
# the long-term comparison becomes a comparison of SHAPE:
#   1. 2025 backtest with and without post-processing (to show that the
#      post-processing does not favour the hybrid only);
#   2. continuity of the 2019-2025 historical trend at hours 14 and 20
#      (Fig. 7), depth of the solar valley and projected hours <= 0.
# Requires the prediction files written by run_comparison.py.
# Outputs: tables/backtest_comparison_v2.csv, tables/stability_v2.csv,
#          figures/fig9_trend_continuity_v2.png
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_utils as du
from run_comparison import MODEL_ORDER, COLOR_MAP

ALTERNATIVES = ["holtwinters", "sarimax", "prophet", "lstm", "nbeats", "xgboost_only"]
ALREADY_POSTPROCESSED = ["ridge_only", "ridge_xgboost"]
LABEL = dict(MODEL_ORDER) | {"xgboost_only": "XGBoost only (ablation)", "ridge_only": "Ridge only (ablation)"}
COLOR = COLOR_MAP | {"xgboost_only": "#8E44AD", "ridge_only": "#1ABC9C"}


def with_calendar(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["Año"], df["mes"], df["hora"] = df.timestamp.dt.year, df.timestamp.dt.month, df.timestamp.dt.hour + 1
    return df.reset_index(drop=True)


def main():
    data = du.load_data()
    naive = data[["timestamp", "target_naive"]]

    # ---- 1. 2025 backtest
    rows = []
    for k in ALTERNATIVES + ALREADY_POSTPROCESSED:
        b = with_calendar(pd.read_csv(f"{du.PRED_DIR}/backtest_2025_{k}.csv")).merge(naive, on="timestamp", how="left")
        y, nv = b.target.values, b.target_naive.values
        variants = {"post-processed": b.target_pred.values} if k in ALREADY_POSTPROCESSED else \
            {"raw": b.target_pred.values, "post-processed": du.postprocess(b.target_pred.values, b)}
        for v, p in variants.items():
            rows.append({"method": LABEL[k], "key": k, "version": v} | du.compute_metrics(y, p, nv))
    bt = pd.DataFrame(rows)
    bt.to_csv(f"{du.TAB_DIR}/backtest_comparison_v2.csv", index=False)
    print("=== Backtest 2025: raw vs post-processed (floor 0 + monthly renormalisation)")
    print(bt.drop(columns="key").round(3).to_string(index=False))

    # ---- 2. 2026-2035 projection
    hist = data[data["Año"] <= 2025]
    h_hist = {hh: hist[hist.hora == hh].groupby("Año").target.mean() for hh in (14, 20)}
    proj, rows = {}, []
    for k in ALTERNATIVES + ALREADY_POSTPROCESSED:
        p = with_calendar(pd.read_csv(f"{du.PRED_DIR}/projection_2026_2035_{k}.csv"))
        raw_mean = p.groupby("Año").target_pred.mean()
        if k not in ALREADY_POSTPROCESSED:
            p["target_pred"] = du.postprocess(p.target_pred.values, p)
        proj[k] = p
        for a in (2026, 2030, 2035):
            g = p[p["Año"] == a]
            rows.append({"method": LABEL[k], "year": a,
                         "annual_mean_raw": raw_mean.get(a, np.nan) if k not in ALREADY_POSTPROCESSED else np.nan,
                         "h14": g[g.hora == 14].target_pred.mean(), "h20": g[g.hora == 20].target_pred.mean(),
                         "valley_10_18": g[g.hora.between(10, 18)].target_pred.mean(),
                         "spring_evening_max": g[g.mes.isin([3, 4, 5, 6])].groupby(["mes", "hora"]).target_pred.mean().max(),
                         "hours_le0": int((g.target_pred <= 0).sum())})
    st = pd.DataFrame(rows)
    st.to_csv(f"{du.TAB_DIR}/stability_v2.csv", index=False)
    print("\n=== Projection after identical post-processing (annual mean = 100 % for all by construction)")
    print(st.round(1).to_string(index=False))
    print(f"\nHistorical 2019 -> 2025: h14 {h_hist[14].iloc[0]:.1f} -> {h_hist[14].iloc[-1]:.1f}; "
          f"h20 {h_hist[20].iloc[0]:.1f} -> {h_hist[20].iloc[-1]:.1f}; hours <= 0 2025: {(hist[hist['Año'] == 2025].target <= 0).sum()}")

    # ---- Fig. 7 (revised)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ax, hh, tit in zip(axs, (14, 20), ("Hour 14:00 (solar cannibalisation)", "Hour 20:00 (evening peak)")):
        ax.plot(h_hist[hh].index, h_hist[hh].values, "k-o", lw=2, label="Actual (2019-2025)")
        for k, p in proj.items():
            s = p[p.hora == hh].groupby("Año").target_pred.mean()
            ax.plot(s.index, s.values, "--o", ms=3, color=COLOR[k], lw=2.2 if k == "ridge_xgboost" else 1.2, label=LABEL[k])
        ax.axhline(0, color="0.6", lw=0.8); ax.set_title(tit, fontweight="bold"); ax.set_xlabel("Year"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("% of monthly mean (annual average)"); axs[0].legend(fontsize=7)
    fig.suptitle("Trend continuity after identical post-processing (floor 0 %, monthly mean renormalised to 100 %)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{du.FIG_DIR}/fig9_trend_continuity_v2.png", dpi=200)


if __name__ == "__main__":
    main()
