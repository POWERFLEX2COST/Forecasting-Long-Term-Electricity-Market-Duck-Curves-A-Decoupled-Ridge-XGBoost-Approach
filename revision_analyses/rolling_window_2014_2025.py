# =============================================================================
# ROLLING-WINDOW BACKTEST OVER 2014-2025 (paper Section 3.4, Fig. 9)
# The model (no solar constraint, floor 0 + monthly renormalisation) is trained on
# rolling windows of 3 and 4 years from 2014 and forecasts blindly each of the
# following three years (up to 2025). This shows how it behaves across the breaks
# of 2020 (COVID-19), 2021-2023 (gas crisis and Iberian gas-price cap) and the
# massive entry of PV, and whether it recovers accuracy once the new trend settles.
# Same model as sensitivity.py (class Hybrid), with the trend origin at the first
# year of each window.
# Data: ../data/spain_hourly_prices_pv_2016_2025.csv + ../data/ree_spot_prices_2014_2015.csv
#       (2014-2015 from REData, see download_ree_2014_2015.py).
# Outputs (results/): rolling_window.csv, fig_rolling_window.png
# =============================================================================
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "hybrid_model_code"))
import data_utils as du                                    # noqa: E402
START = 2014
du.TRAIN_START_YEAR = START
from model_ridge_xgboost import add_xgb_features           # noqa: E402
from sensitivity import Hybrid                             # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LENGTHS = (3, 4)
MAX_HORIZON = 3


def data():
    # paper dataset (2016-2025) + 2014-2015 from REE, same format
    ref = pd.read_csv(du.CSV_PATH, parse_dates=["Fecha"])[["Fecha", "hora", "Precios"]]
    ree = pd.read_csv(os.path.join(du.DATA_DIR, "ree_spot_prices_2014_2015.csv"), parse_dates=["Fecha"])
    d = pd.concat([ree, ref]).drop_duplicates(["Fecha", "hora"]).sort_values(["Fecha", "hora"])
    d["Año"] = d.Fecha.dt.year
    path = os.path.join(OUT, "_dataset_2014_2025.csv")
    d.to_csv(path, index=False)
    df = du.load_data(path)
    os.remove(path)
    return df


def main():
    df = add_xgb_features(data())
    assert df["Año"].min() == START
    rows = []
    for L in LENGTHS:
        for ini in range(START, 2026 - L):
            end = ini + L - 1
            mod = Hybrid(ini, 2, False).fit(df[df["Año"].between(ini, end)])
            for a in range(end + 1, min(end + MAX_HORIZON, 2025) + 1):
                t = df[df["Año"] == a]
                for var, only in (("hybrid", False), ("ridge_only", True)):
                    p = mod.predict(t, trend_only=only)
                    m = du.compute_metrics(t["target"].values, p, t["target_naive"].values)
                    rows.append({"window_years": L, "train": f"{ini}-{end}", "train_end": end,
                                 "test_year": a, "horizon": a - end, "model": var,
                                 "hours_le0_real": int((t["target"] <= 0).sum()),
                                 "hours_le0_pred": int((p <= 0).sum()), **m})
            print(f"L={L} {ini}-{end} done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "rolling_window.csv"), index=False)
    h = r[r.model == "hybrid"]
    pd.set_option("display.width", 200)
    for L in LENGTHS:
        print(f"\n=== {L}-year window: rMAE of the hybrid (rows: training window; columns: test year)")
        print(h[h.window_years == L].pivot(index="train", columns="test_year", values="rMAE").round(3).to_string())
        print("--- rRMSE")
        print(h[h.window_years == L].pivot(index="train", columns="test_year", values="rRMSE").round(3).to_string())
    print("\n=== Mean rMAE by horizon")
    print(h.groupby(["window_years", "horizon"])[["rMAE", "rRMSE"]].mean().round(3).to_string())

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, L in zip(axs, LENGTHS):
        s = h[h.window_years == L]
        for k, hz in enumerate(range(1, MAX_HORIZON + 1)):
            q = s[s.horizon == hz].sort_values("test_year")
            ax.plot(q.test_year, q.rMAE, "o-", color=["#C0392B", "#E67E22", "#7F8C8D"][k], label=f"{hz}-year horizon")
        for a0, a1, lab in ((2019.6, 2020.4, "COVID-19"), (2021.6, 2023.4, "gas crisis / cap")):
            ax.axvspan(a0, a1, color="0.9", zorder=0)
            ax.text((a0 + a1) / 2, 0.02, lab, ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.set_title(f"Rolling {L}-year training window", fontweight="bold")
        ax.set_xlabel("Test year"); ax.grid(alpha=0.3); ax.set_xticks(range(START + 3, 2026))
    axs[0].set_ylabel("rMAE (vs. naïve)"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_rolling_window.png"), dpi=200)


if __name__ == "__main__":
    main()
