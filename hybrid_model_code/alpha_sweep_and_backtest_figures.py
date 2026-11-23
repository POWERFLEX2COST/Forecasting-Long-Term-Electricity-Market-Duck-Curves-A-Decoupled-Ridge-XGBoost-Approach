"""Revised version: sweep of the Ridge penalty alpha, backtest bar chart (all alternatives
post-processed) and Fig. 7 (2025 backtest) with the new post-processing.
Requires the prediction files of run_comparison.py and the table of fair_comparison_postprocessed.py."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_utils as du
import model_ridge_xgboost as mrx

df = mrx.add_xgb_features(du.load_data())
tr, te = du.train_test_split(df)

# ---- alpha sweep (Ridge of the hybrid)
rows = []
for a in [0.1, 1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    mrx.RIDGE_ALPHA = a
    m = mrx.ModeloHibrido(); m.fit(tr)
    rows.append({"alpha": a} | du.compute_metrics(te.target.values, m.predict(te), te.target_naive.values))
mrx.RIDGE_ALPHA = 1000
A = pd.DataFrame(rows); A.to_csv(f"{du.TAB_DIR}/alpha_sweep_v2.csv", index=False)
print(A.round(4).to_string(index=False))
fig, ax = plt.subplots(figsize=(9, 4.5)); ax2 = ax.twinx()
ax.semilogx(A.alpha, A.rRMSE, "o-", color="#2471A3", label="rRMSE"); ax2.semilogx(A.alpha, A.WAPE, "s--", color="#C0392B", label="WAPE")
ax.axvline(1000, color="k", ls=":", lw=1); ax.set_xlabel(r"$\alpha$ (Ridge L2 penalty)"); ax.set_ylabel("rRMSE (vs. naïve)"); ax2.set_ylabel("WAPE (%)")
ax.grid(alpha=0.3); fig.legend(loc="upper center", ncol=2, fontsize=9); fig.tight_layout(); fig.savefig(f"{du.FIG_DIR}/alpha_sweep_v2.png", dpi=200)

# ---- backtest bars, all alternatives post-processed
bt = pd.read_csv(f"{du.TAB_DIR}/backtest_comparison_v2.csv")
bt = bt[(bt.version == "post-processed") & ~bt.key.isin(["xgboost_only", "ridge_only"])]
order = ["holtwinters", "sarimax", "prophet", "lstm", "nbeats", "ridge_xgboost"]
bt = bt.set_index("key").loc[order].reset_index()
from run_comparison import COLOR_MAP, MODEL_ORDER
lab = dict(MODEL_ORDER)
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ax, met in zip(axs, ["rMAE", "rRMSE"]):
    keys = ["naive"] + order; vals = [1.0] + list(bt[met])
    b = ax.bar([lab[k] for k in keys], vals, color=[COLOR_MAP[k] for k in keys])
    ax.axhline(1, color="k", ls="--", lw=1); ax.set_title(f"{met} (vs. naïve)", fontweight="bold"); ax.tick_params(axis="x", rotation=35)
    for r, v in zip(b, vals): ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for t in ax.get_xticklabels(): t.set_horizontalalignment("right")
fig.tight_layout(); fig.savefig(f"{du.FIG_DIR}/comparison_backtest_bars_v2.png", dpi=200)

# ---- Fig. 7: backtest 2025
b = pd.read_csv(f"{du.PRED_DIR}/backtest_2025_ridge_xgboost.csv", parse_dates=["timestamp"])
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(b.timestamp, b.target, color="0.45", lw=0.5, label="Actual prices 2025")
ax.plot(b.timestamp, b.target_pred, color="#E74C3C", lw=0.5, alpha=0.8, label="Blind hybrid prediction")
ax.axhline(0, color="k", ls=":", lw=0.8); ax.set_ylabel("% of monthly mean"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{du.FIG_DIR}/backtesting_8760_v2.png", dpi=200)
print("ok")
