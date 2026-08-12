# =============================================================================
# STABILITY COMPARISON: 2026-2035 long-horizon projection across all methods.
#
# Two diagnostics:
#   1. Annual-mean bias drift: the target is defined so that every month's
#      hours average to exactly 100 (% of monthly mean); therefore any
#      method whose ANNUAL mean prediction drifts far from 100 over the
#      projection is producing an internally inconsistent / unphysical
#      long-horizon forecast, regardless of its 2025 backtest accuracy.
#   2. Structural-trend continuity at two key hours: hour 14:00 (deep solar
#      cannibalisation hour, historically falling) and hour 20:00 (evening
#      peak, historically rising). A method that lacks an explicit
#      structural trend component is expected to flatten/revert toward the
#      2019-2024 seasonal average instead of continuing the observed
#      duck-curve evolution — this is the central empirical test of the
#      paper's design-rationale argument (Sec. 3.2, main.tex).
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import data_utils as du
from run_comparison import MODEL_ORDER, COLOR_MAP

sns.set_theme(style="whitegrid")

PROJ_FILES = {
    "sarimax": "projection_2026_2035_sarimax.csv",
    "holtwinters": "projection_2026_2035_holtwinters.csv",
    "prophet": "projection_2026_2035_prophet.csv",
    "lstm": "projection_2026_2035_lstm.csv",
    "nbeats": "projection_2026_2035_nbeats.csv",
    "ridge_xgboost": "projection_2026_2035_ridge_xgboost.csv",
}
LABELS = {k: v for k, v in MODEL_ORDER}


def annual_bias_table() -> pd.DataFrame:
    rows = []
    for key, fname in PROJ_FILES.items():
        df = pd.read_csv(f"{du.PRED_DIR}/{fname}", parse_dates=["timestamp"])
        g = df.groupby("Año")["target_pred"].agg(["mean", "std", "min", "max"])
        for year in [2026, 2030, 2035]:
            if year in g.index:
                rows.append({
                    "key": key, "label": LABELS[key], "year": year,
                    "annual_mean": g.loc[year, "mean"],
                    "annual_std": g.loc[year, "std"],
                    "min": g.loc[year, "min"],
                    "max": g.loc[year, "max"],
                })
    return pd.DataFrame(rows)


def write_stability_table(df: pd.DataFrame):
    piv_mean = df.pivot(index="label", columns="year", values="annual_mean")
    piv_mean = piv_mean.reindex([lbl for k, lbl in MODEL_ORDER if k in PROJ_FILES])

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Annual-mean bias drift of the projected relative price "
                 r"(\% of monthly mean) for 2026, 2030 and 2035. Because the target is "
                 r"constructed so every month averages to 100\,\%, a method whose annual "
                 r"mean departs substantially from 100\,\% over the projection horizon is "
                 r"producing an internally inconsistent long-term forecast. "
                 r"Source: Own elaboration.}")
    lines.append(r"    \label{tab:stability_bias}")
    lines.append(r"    \begin{tabular}{lccc}")
    lines.append(r"        \toprule")
    lines.append(r"        \textbf{Method} & \textbf{2026 (\%)} & \textbf{2030 (\%)} & \textbf{2035 (\%)} \\")
    lines.append(r"        \midrule")
    for label, row in piv_mean.iterrows():
        bold = label == "Ridge + XGBoost (proposed)"
        vals = [f"{row[y]:.1f}" for y in [2026, 2030, 2035]]
        lab = r"\textbf{" + label + "}" if bold else label
        if bold:
            vals = [r"\textbf{" + v + "}" for v in vals]
        lines.append(f"        {lab} & " + " & ".join(vals) + r" \\")
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    path = f"{du.TAB_DIR}/table_stability_bias.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  -> Saved LaTeX table: {path}")
    df.to_csv(f"{du.TAB_DIR}/stability_bias.csv", index=False)


def key_hour_series(hour: int) -> pd.DataFrame:
    """Annual mean at a fixed clock hour, historical (2019-2025) + each
    method's projection (2026-2035)."""
    df_hist = du.load_data()
    df_hist = df_hist[df_hist["timestamp"].dt.hour == hour]
    hist_annual = df_hist.groupby("Año")["target"].mean()

    series = {"Actual (2019-2025)": hist_annual}
    for key, fname in PROJ_FILES.items():
        df = pd.read_csv(f"{du.PRED_DIR}/{fname}", parse_dates=["timestamp"])
        df = df[df["timestamp"].dt.hour == hour]
        series[LABELS[key]] = df.groupby("Año")["target_pred"].mean()

    out = pd.DataFrame(series)
    return out


def plot_key_hours():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, hour, title in zip(axes, [14, 20],
                                ["Hour 14:00 (solar cannibalisation)",
                                 "Hour 20:00 (evening peak)"]):
        df = key_hour_series(hour)
        for col in df.columns:
            if col == "Actual (2019-2025)":
                ax.plot(df.index, df[col], color="black", lw=2.2, marker="o",
                        markersize=4, label=col, zorder=5)
                continue
            key = [k for k, lbl in MODEL_ORDER if lbl == col][0]
            proj = df[col].dropna()
            ax.plot(proj.index, proj.values, color=COLOR_MAP[key], lw=1.6,
                     linestyle="--", marker="o", markersize=3, label=col, alpha=0.9)
        ax.axvline(2025.5, color="grey", lw=1, linestyle=":", alpha=0.7)
        ax.axhline(100, color="grey", lw=0.8, linestyle="-", alpha=0.4)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("% of monthly mean (annual average)")
    axes[0].legend(loc="upper left", fontsize=8, ncol=1)
    fig.suptitle("Long-horizon structural-trend continuity: historical (2019-2025, solid) "
                 "vs. each method's projection (2026-2035, dashed)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = f"{du.FIG_DIR}/comparison_projection_key_hours.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved figure: {path}")
    plt.close(fig)


def plot_annual_mean_drift(df_bias: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 6))
    for key, fname in PROJ_FILES.items():
        df = pd.read_csv(f"{du.PRED_DIR}/{fname}", parse_dates=["timestamp"])
        annual = df.groupby("Año")["target_pred"].mean()
        ax.plot(annual.index, annual.values, color=COLOR_MAP[key], lw=1.8,
                marker="o", markersize=4, label=LABELS[key])
    ax.axhline(100, color="black", lw=1, linestyle="--", alpha=0.7,
               label="Target design mean (100%)")
    ax.set_xlabel("Projection year")
    ax.set_ylabel("Annual mean of predicted % of monthly mean")
    ax.set_title("Annual-mean bias drift over the 2026-2035 projection horizon "
                 "(should stay near 100% by construction)", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path = f"{du.FIG_DIR}/comparison_annual_bias_drift.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved figure: {path}")
    plt.close(fig)


def main():
    print("[Stability] Building annual bias table (2026/2030/2035) ...")
    df_bias = annual_bias_table()
    print(df_bias.to_string(index=False))
    write_stability_table(df_bias)

    print("[Stability] Plotting key-hour structural trend continuity ...")
    plot_key_hours()

    print("[Stability] Plotting annual bias drift across full horizon ...")
    plot_annual_mean_drift(df_bias)

    print("[Stability] Done.")


if __name__ == "__main__":
    main()
