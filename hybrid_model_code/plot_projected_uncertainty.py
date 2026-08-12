# =============================================================================
# FIGURE: expected error band for the real 2026-2035 hybrid-model projection.
#
# Reads tables/horizon_error_projected_uncertainty.csv (produced by
# horizon_error_analysis.py: a linear fit of metric ~ ratio over the 15
# historical expanding-window backtest points, evaluated at the ratios the
# real 2026-2035 projection needs — train 2019-2025, horizon 1-10 years).
#
# Two panels: rMAE and rRMSE (the paper's headline metrics, normalised
# against the naive week-ahead benchmark = 1.0), each showing the expected
# value line plus its 95% band, so it is visible in which years the hybrid's
# expected relative error may approach or exceed the naive benchmark.
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import data_utils as du

sns.set_theme(style="whitegrid")

COLOR = "#C0392B"  # matches "ridge_xgboost" color used in the other comparison figures


def main():
    df = pd.read_csv(f"{du.TAB_DIR}/horizon_error_projected_uncertainty.csv")

    y_lo = min(df["rMAE_lo95"].min(), df["rRMSE_lo95"].min()) - 0.02
    y_hi = max(df["rMAE_hi95"].max(), df["rRMSE_hi95"].max(), 1.0) + 0.02

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, metric in zip(axes, ["rMAE", "rRMSE"]):
        ax.fill_between(df["proj_year"], df[f"{metric}_lo95"], df[f"{metric}_hi95"],
                         color=COLOR, alpha=0.18, label="95% prediction interval (OLS)")
        ax.plot(df["proj_year"], df[f"{metric}_expected"], marker="o", color=COLOR,
                 lw=2, label="Expected")
        ax.axhline(1.0, color="black", lw=1.2, linestyle="--", alpha=0.7,
                   label="Naïve benchmark = 1.0")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel("Projection year")
        ax.set_xticks(df["proj_year"])
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Error relative to naïve benchmark")
    axes[0].legend(fontsize=9, loc="upper left")

    fig.suptitle("Expected error of the Ridge+XGBoost hybrid, 2026-2035 projection\n"
                 "(extrapolated from the horizon-vs-error relationship, 2019-2025 backtest)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = f"{du.FIG_DIR}/projected_uncertainty_2026_2035.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
