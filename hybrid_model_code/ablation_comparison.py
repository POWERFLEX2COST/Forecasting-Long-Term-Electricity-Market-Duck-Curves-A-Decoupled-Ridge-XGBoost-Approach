# =============================================================================
# ABLATION AGGREGATION: Ridge-only vs XGBoost-only vs full Ridge+XGBoost hybrid
# Produces: tables/table_ablation.tex (+.csv),
#           figures/ablation_backtest_bars.png,
#           figures/ablation_annual_mean_drift.png
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import data_utils as du

sns.set_theme(style="whitegrid")

ORDER = [
    ("naive", "Naïve (week-ahead)"),
    ("xgboost_only", "XGBoost only (no Ridge trend)"),
    ("ridge_only", "Ridge only (no residual correction)"),
    ("ridge_xgboost", "Ridge + XGBoost (full hybrid)"),
]
COLORS = {
    "naive": "#7f7f7f",
    "xgboost_only": "#8172B2",
    "ridge_only": "#64B5A6",
    "ridge_xgboost": "#C0392B",
}


def load_metrics() -> pd.DataFrame:
    rows = []
    for key, label in ORDER:
        if key == "naive":
            df = du.load_data()
            _, df_test = du.train_test_split(df)
            m = du.compute_metrics(
                df_test["target"].values, df_test["target_naive"].values,
                df_test["target_naive"].values,
            )
            rows.append({**m, "key": key, "label": label})
            continue
        m = pd.read_csv(f"{du.PRED_DIR}/metrics_backtest_{key}.csv").iloc[0].to_dict()
        rows.append({**m, "key": key, "label": label})
    out = pd.DataFrame(rows)
    order_map = {k: i for i, (k, _) in enumerate(ORDER)}
    out["order"] = out["key"].map(order_map)
    return out.sort_values("order").reset_index(drop=True)[
        ["label", "MAE", "RMSE", "WAPE", "rMAE", "rRMSE", "key"]
    ]


def write_latex_table(df: pd.DataFrame):
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Ablation study (2025 backtest): contribution of the Ridge trend "
        r"component and the XGBoost residual-correction component to the full hybrid "
        r"model. Source: Own elaboration based on OMIE~\cite{omie2025}.}",
        r"    \label{tab:ablation}",
        r"    \begin{tabular}{lccccc}",
        r"        \toprule",
        r"        \textbf{Configuration} & \textbf{MAE (\%)} & \textbf{RMSE (\%)} & "
        r"\textbf{WAPE (\%)} & \textbf{rMAE} & \textbf{rRMSE} \\",
        r"        \midrule",
    ]
    for _, row in df.iterrows():
        bold = row["key"] == "ridge_xgboost"
        label = row["label"]
        vals = [f"{row['MAE']:.2f}", f"{row['RMSE']:.2f}", f"{row['WAPE']:.2f}",
                f"{row['rMAE']:.3f}", f"{row['rRMSE']:.3f}"]
        if bold:
            label = r"\textbf{" + label + "}"
            vals = [r"\textbf{" + v + "}" for v in vals]
        lines.append(f"        {label} & " + " & ".join(vals) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    path = f"{du.TAB_DIR}/table_ablation.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  -> Saved LaTeX table: {path}")


def plot_bars(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, metric, title in zip(axes, ["rMAE", "rRMSE"],
                                  ["rMAE (vs. naïve)", "rRMSE (vs. naïve)"]):
        colors = [COLORS[k] for k in df["key"]]
        bars = ax.bar(df["label"], df[metric], color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(1.0, color="black", lw=1, linestyle="--", alpha=0.6)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Ablation: 2025 blind backtest (lower is better; dashed = naïve)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = f"{du.FIG_DIR}/ablation_backtest_bars.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved figure: {path}")
    plt.close(fig)


def plot_annual_drift():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key, label in ORDER:
        if key == "naive":
            continue
        df = pd.read_csv(f"{du.PRED_DIR}/projection_2026_2035_{key}.csv", parse_dates=["timestamp"])
        annual = df.groupby("Año")["target_pred"].mean()
        ax.plot(annual.index, annual.values, color=COLORS[key], lw=1.8,
                marker="o", markersize=4, label=label)
    ax.axhline(100, color="black", lw=1, linestyle="--", alpha=0.7, label="Target design mean (100%)")
    ax.set_xlabel("Projection year")
    ax.set_ylabel("Annual mean of predicted % of monthly mean")
    ax.set_title("Ablation: annual-mean drift over 2026-2035 — XGBoost-only stays\n"
                 "flat (trees cannot extrapolate beyond the training year range), while\n"
                 "Ridge-only and the full hybrid both continue evolving", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path = f"{du.FIG_DIR}/ablation_annual_mean_drift.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved figure: {path}")
    plt.close(fig)


def main():
    df = load_metrics()
    print(df.to_string(index=False))
    df.to_csv(f"{du.TAB_DIR}/ablation_comparison.csv", index=False)
    write_latex_table(df)
    plot_bars(df)
    plot_annual_drift()
    print("[Ablation] Done.")


if __name__ == "__main__":
    main()
