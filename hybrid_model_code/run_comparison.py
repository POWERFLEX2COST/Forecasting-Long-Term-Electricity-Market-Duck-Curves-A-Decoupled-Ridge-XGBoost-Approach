# =============================================================================
# AGGREGATE: 2025 backtest comparison across all forecasting methods.
# Produces: tables/table_backtest_comparison.tex (+.csv) and
#           figures/comparison_backtest_bars.png,
#           figures/comparison_backtest_panels.png
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import data_utils as du

sns.set_theme(style="whitegrid")

# Fixed categorical order/colors (consistent across all comparison figures)
MODEL_ORDER = [
    ("naive", "Naïve (week-ahead)"),
    ("holtwinters", "Holt-Winters (ETS)"),
    ("sarimax", "SARIMAX (Fourier)"),
    ("prophet", "Prophet"),
    ("lstm", "LSTM"),
    ("nbeats", "N-BEATS (calendar)"),
    ("ridge_xgboost", "Ridge + XGBoost (proposed)"),
]
PALETTE = sns.color_palette("colorblind", n_colors=len(MODEL_ORDER) - 1)
COLOR_MAP = {key: PALETTE[i] for i, (key, _) in enumerate(MODEL_ORDER[:-1])}
COLOR_MAP["ridge_xgboost"] = "#C0392B"  # distinct red — matches existing
                                         # backtesting_8760 figure convention
                                         # for the proposed model
COLOR_MAP["naive"] = "#7f7f7f"


def load_all_metrics() -> pd.DataFrame:
    rows = []
    for key, label in MODEL_ORDER:
        if key == "naive":
            continue
        path = f"{du.PRED_DIR}/metrics_backtest_{key}.csv"
        try:
            m = pd.read_csv(path)
        except FileNotFoundError:
            print(f"  [!] Missing metrics file for {key}: {path}")
            continue
        m["key"] = key
        m["label"] = label
        rows.append(m)

    # Naive reference row (rMAE = rRMSE = 1 by construction)
    df = du.load_data()
    _, df_test = du.train_test_split(df)
    naive_metrics = du.compute_metrics(
        df_test["target"].values, df_test["target_naive"].values,
        df_test["target_naive"].values,
    )
    naive_row = pd.DataFrame([{**naive_metrics, "model": "Naïve (week-ahead)",
                                "key": "naive", "label": "Naïve (week-ahead)"}])
    rows.append(naive_row)

    out = pd.concat(rows, ignore_index=True)
    order_keys = [k for k, _ in MODEL_ORDER]
    out["order"] = out["key"].map({k: i for i, k in enumerate(order_keys)})
    out = out.sort_values("order").reset_index(drop=True)
    return out[["label", "MAE", "RMSE", "WAPE", "rMAE", "rRMSE", "n", "key"]]


def write_latex_table(df: pd.DataFrame):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Backtesting metrics (year 2025) for the proposed hybrid model "
                 r"against classical statistical, machine-learning and deep-learning long-term "
                 r"forecasting baselines. Source: Own elaboration based on OMIE~\cite{omie2025}.}")
    lines.append(r"    \label{tab:method_comparison}")
    lines.append(r"    \begin{tabular}{lccccc}")
    lines.append(r"        \toprule")
    lines.append(r"        \textbf{Method} & \textbf{MAE (\%)} & \textbf{RMSE (\%)} & "
                 r"\textbf{WAPE (\%)} & \textbf{rMAE} & \textbf{rRMSE} \\")
    lines.append(r"        \midrule")
    for _, row in df.iterrows():
        bold = row["key"] == "ridge_xgboost"
        label = row["label"]
        vals = [f"{row['MAE']:.2f}", f"{row['RMSE']:.2f}", f"{row['WAPE']:.2f}",
                f"{row['rMAE']:.3f}", f"{row['rRMSE']:.3f}"]
        if bold:
            label = r"\textbf{" + label + "}"
            vals = [r"\textbf{" + v + "}" for v in vals]
        lines.append(f"        {label} & " + " & ".join(vals) + r" \\")
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    path = f"{du.TAB_DIR}/table_backtest_comparison.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  -> Saved LaTeX table: {path}")


def plot_bars(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, metric, title in zip(axes, ["rMAE", "rRMSE"],
                                  ["rMAE (vs. naïve)", "rRMSE (vs. naïve)"]):
        colors = [COLOR_MAP[k] for k in df["key"]]
        bars = ax.bar(df["label"], df[metric], color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(1.0, color="black", lw=1, linestyle="--", alpha=0.6)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis="x", rotation=35, labelsize=10)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    fig.suptitle("2025 blind backtest: relative error vs. naïve week-ahead benchmark "
                 "(lower is better; dashed line = naïve)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = f"{du.FIG_DIR}/comparison_backtest_bars.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved figure: {path}")
    plt.close(fig)


def plot_panels():
    df_actual = du.load_data()
    _, df_test_actual = du.train_test_split(df_actual)
    df_test_actual = df_test_actual.set_index("timestamp")

    files = {
        "sarimax": "backtest_2025_sarimax.csv",
        "holtwinters": "backtest_2025_holtwinters.csv",
        "prophet": "backtest_2025_prophet.csv",
        "lstm": "backtest_2025_lstm.csv",
        "nbeats": "backtest_2025_nbeats.csv",
        "ridge_xgboost": "backtest_2025_ridge_xgboost.csv",
    }
    labels = {k: v for k, v in MODEL_ORDER}

    fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (key, fname) in zip(axes, files.items()):
        pred = pd.read_csv(f"{du.PRED_DIR}/{fname}", parse_dates=["timestamp"]).set_index("timestamp")
        ax.plot(df_test_actual.index, df_test_actual["target"], color="#999999",
                lw=0.7, alpha=0.8, label="Actual 2025")
        ax.plot(pred.index, pred["target_pred"], color=COLOR_MAP[key], lw=0.8,
                alpha=0.9, label=labels[key])
        ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
        ax.set_title(labels[key], fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)

    fig.supylabel("% of monthly mean", fontsize=13)
    fig.suptitle("Blind 2025 backtest: actual prices vs. each method's forecast", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"{du.FIG_DIR}/comparison_backtest_panels.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  -> Saved figure: {path}")
    plt.close(fig)


def main():
    print("[Comparison] Loading all backtest metrics ...")
    df = load_all_metrics()
    print(df.to_string(index=False))
    df.to_csv(f"{du.TAB_DIR}/backtest_comparison.csv", index=False)

    print("[Comparison] Writing LaTeX table ...")
    write_latex_table(df)

    print("[Comparison] Plotting bar comparison ...")
    plot_bars(df)

    print("[Comparison] Plotting per-method panels ...")
    plot_panels()

    print("[Comparison] Done.")


if __name__ == "__main__":
    main()
