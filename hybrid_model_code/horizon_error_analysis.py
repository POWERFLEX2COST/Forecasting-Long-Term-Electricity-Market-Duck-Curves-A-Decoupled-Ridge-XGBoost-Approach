# =============================================================================
# HORIZON-VS-ERROR ANALYSIS for the Ridge+XGBoost hybrid model.
#
# Goal: quantify how the hybrid's forecast error grows as the gap between
# the training cutoff and the target year widens, using historical data
# where ground truth exists (2019-2025), as an empirical proxy for the
# uncertainty of the real 2026-2035 projection (train cutoff 2025, horizon
# up to 10 years, where no ground truth is available).
#
# For each training cutoff year (2020-2024), the hybrid is fit on
# 2019-cutoff and then forecasts EVERY subsequent year individually up to
# 2025 (not aggregated as in rolling_ablation.py), so each row has a
# well-defined:
#   n_train_years = cutoff - 2019 + 1
#   horizon        = test_year - cutoff
#   ratio          = horizon / n_train_years
# Metrics (MAE/RMSE/WAPE/rMAE/rRMSE) are correlated (Pearson + Spearman)
# against horizon, ratio and n_train_years to see which one best explains
# error growth. A linear fit of each metric vs. ratio is then used to
# extrapolate an expected-error band for the real 2026-2035 projection
# (train 2019-2025, n_train_years=7): all required ratios (1/7 .. 10/7,
# i.e. 0.14-1.43) fall INSIDE the observed range (0.17-2.5), so this is
# interpolation, not extrapolation beyond the fitted data.
# =============================================================================

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

import data_utils as du
from model_ridge_xgboost import ModeloHibrido, add_xgb_features, XGB_FEATURE_COLS

sns.set_theme(style="whitegrid")

TRAIN_ENDS = [2020, 2021, 2022, 2023, 2024]
METRICS = ["MAE", "RMSE", "WAPE", "rMAE", "rRMSE"]
REAL_N_TRAIN_YEARS = 2025 - du.TRAIN_START_YEAR + 1  # = 7, the real production cutoff


def xgb_val_split(df_train_full: pd.DataFrame):
    """Mirrors rolling_ablation.py: hold out the last training year as
    validation; fall back to an 85/15 chronological split if only one
    training year is available."""
    years = sorted(df_train_full["Año"].unique())
    if len(years) >= 2:
        val_year = years[-1]
        train_mask = df_train_full["Año"] < val_year
        val_mask = df_train_full["Año"] == val_year
    else:
        df_sorted = df_train_full.sort_values("timestamp")
        n = len(df_sorted)
        cut = int(n * 0.85)
        cut_ts = df_sorted["timestamp"].iloc[cut]
        train_mask = df_train_full["timestamp"] < cut_ts
        val_mask = df_train_full["timestamp"] >= cut_ts
    return train_mask, val_mask


def fit_hybrid(df_train_full: pd.DataFrame) -> ModeloHibrido:
    """Reproduces ModeloHibrido.fit() but with a configurable train/val
    split (the class itself hardcodes <=2023/==2024) — same pattern used
    in rolling_ablation.py for the ablation robustness check."""
    hybrid = ModeloHibrido()
    df_fit = add_xgb_features(df_train_full).copy()
    df_fit["trend"] = 0.0
    for m in range(1, 13):
        for h in range(1, 25):
            for f in [0, 1]:
                clave = (m, h, f)
                mask = (df_fit["mes"] == m) & (df_fit["hora"] == h) & (df_fit["es_fin_semana"] == f)
                if mask.sum() < 20:
                    continue
                X_lr = df_fit.loc[mask, ["Año"]].values
                y_lr = df_fit.loc[mask, "target"].values
                modelo = hybrid._ajustar_tendencia(X_lr, y_lr, clave)
                hybrid.modelos_tendencia[clave] = modelo
                df_fit.loc[mask, "trend"] = modelo.predict(X_lr - du.TRAIN_START_YEAR)
    df_fit["trend"] = df_fit["trend"].replace(0.0, 100.0)
    df_fit["residual"] = df_fit["target"] - df_fit["trend"]

    tr_mask, va_mask = xgb_val_split(df_fit)
    hybrid.xgb = xgb.XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
        tree_method="hist", random_state=42, early_stopping_rounds=50,
    )
    hybrid.xgb.fit(
        df_fit.loc[tr_mask, XGB_FEATURE_COLS].values, df_fit.loc[tr_mask, "residual"].values,
        eval_set=[(df_fit.loc[va_mask, XGB_FEATURE_COLS].values, df_fit.loc[va_mask, "residual"].values)],
        verbose=False,
    )
    return hybrid


def main():
    df = du.load_data()
    all_results = []

    for train_end in TRAIN_ENDS:
        n_train_years = train_end - du.TRAIN_START_YEAR + 1
        df_train_full = df[(df["Año"] >= du.TRAIN_START_YEAR) & (df["Año"] <= train_end)].copy()
        print(f"\n=== Train 2019-{train_end} ({n_train_years} yr) ===")
        hybrid = fit_hybrid(df_train_full)

        for test_year in range(train_end + 1, 2026):
            df_test = df[df["Año"] == test_year].copy()
            df_test_hyb = add_xgb_features(df_test)
            pred = hybrid.predict(df_test_hyb)
            m = du.compute_metrics(df_test["target"].values, pred, df_test["target_naive"].values)
            horizon = test_year - train_end
            row = {
                "train_end": train_end, "n_train_years": n_train_years,
                "test_year": test_year, "horizon": horizon,
                "ratio": horizon / n_train_years, **m,
            }
            all_results.append(row)
            print(f"  -> train_end {train_end}, +{horizon}y (test {test_year}): "
                  f"MAE={m['MAE']:.2f} rMAE={m['rMAE']:.3f} rRMSE={m['rRMSE']:.3f}")

    out = pd.DataFrame(all_results)
    out.to_csv(f"{du.TAB_DIR}/horizon_error_analysis.csv", index=False)
    print(f"\nSaved: {du.TAB_DIR}/horizon_error_analysis.csv")

    # ---- Correlation analysis: which driver best explains error growth? ----
    print("\n=== Correlation of error metrics vs. horizon / ratio / n_train_years ===")
    corr_rows = []
    for driver in ["horizon", "ratio", "n_train_years"]:
        for metric in METRICS:
            pear_r, pear_p = stats.pearsonr(out[driver], out[metric])
            spear_r, spear_p = stats.spearmanr(out[driver], out[metric])
            corr_rows.append({
                "driver": driver, "metric": metric,
                "pearson_r": pear_r, "pearson_p": pear_p,
                "spearman_r": spear_r, "spearman_p": spear_p,
            })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(f"{du.TAB_DIR}/horizon_error_correlations.csv", index=False)
    print(corr_df.to_string(index=False))
    print(f"\nSaved: {du.TAB_DIR}/horizon_error_correlations.csv")

    # ---- Figure: metric vs ratio, one line per training-cutoff -------------
    palette = sns.color_palette("viridis", n_colors=len(TRAIN_ENDS))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.6))
    for ax, metric in zip(axes, METRICS):
        for i, train_end in enumerate(TRAIN_ENDS):
            sub = out[out["train_end"] == train_end].sort_values("ratio")
            if sub.empty:
                continue
            ax.plot(sub["ratio"], sub[metric], marker="o", color=palette[i], lw=1.5,
                     label=f"train 2019-{train_end}")
        if metric in ("rMAE", "rRMSE"):
            ax.axhline(1.0, color="black", lw=1, linestyle="--", alpha=0.5)
        r = corr_df[(corr_df["driver"] == "ratio") & (corr_df["metric"] == metric)].iloc[0]
        ax.set_title(f"{metric}  (Pearson r={r['pearson_r']:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Ratio = horizon / n_train_years")
    axes[0].set_ylabel("Error")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Hybrid model: error vs. extrapolation ratio (expanding-window backtest, 2019-2025)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = f"{du.FIG_DIR}/horizon_error_ratio.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

    # ---- Figure: metric vs horizon, one line per training-cutoff -----------
    fig2, axes2 = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.6))
    for ax, metric in zip(axes2, METRICS):
        for i, train_end in enumerate(TRAIN_ENDS):
            sub = out[out["train_end"] == train_end].sort_values("horizon")
            if sub.empty:
                continue
            ax.plot(sub["horizon"], sub[metric], marker="o", color=palette[i], lw=1.5,
                     label=f"train 2019-{train_end}")
        if metric in ("rMAE", "rRMSE"):
            ax.axhline(1.0, color="black", lw=1, linestyle="--", alpha=0.5)
        r = corr_df[(corr_df["driver"] == "horizon") & (corr_df["metric"] == metric)].iloc[0]
        ax.set_title(f"{metric}  (Pearson r={r['pearson_r']:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Horizon (years ahead)")
    axes2[0].set_ylabel("Error")
    axes2[-1].legend(fontsize=8, loc="best")
    fig2.suptitle("Hybrid model: error vs. forecast horizon (expanding-window backtest, 2019-2025)",
                  fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    path2 = f"{du.FIG_DIR}/horizon_error_horizon.png"
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    print(f"Saved: {path2}")

    # ---- Extrapolated uncertainty band for the REAL 2026-2035 projection ---
    # OLS metric ~ ratio fit via statsmodels using the 15 historical
    # (ratio, metric) pairs, then a full 95% PREDICTION interval (not just a
    # constant residual-std band) evaluated at the ratios the real projection
    # actually needs (n_train_years=7, horizon=1..10 -> ratio=0.14..1.43,
    # which falls inside the observed ratio range 0.17-2.5 above, so this is
    # interpolation). The prediction interval accounts for BOTH residual
    # variance AND parameter (slope/intercept) uncertainty, so it widens the
    # further a given ratio sits from the mean of the observed ratios, and
    # uses a t-distribution (df=n-2=13) instead of a normal approximation —
    # appropriate given the small sample (n=15, drawn from only 5 independent
    # trained models).
    print("\n=== Extrapolated expected-error band for the real 2026-2035 projection ===")
    proj_years = list(range(2026, 2036))
    horizons = [y - 2025 for y in proj_years]
    ratios = [h / REAL_N_TRAIN_YEARS for h in horizons]
    proj_df = pd.DataFrame({"proj_year": proj_years, "horizon": horizons, "ratio": ratios})

    X_train = sm.add_constant(out[["ratio"]])
    X_new = sm.add_constant(proj_df[["ratio"]], has_constant="add")
    ols_summaries = {}
    for metric in METRICS:
        ols_model = sm.OLS(out[metric], X_train).fit()
        ols_summaries[metric] = ols_model
        pred = ols_model.get_prediction(X_new).summary_frame(alpha=0.05)
        proj_df[f"{metric}_expected"] = pred["mean"].values
        proj_df[f"{metric}_lo95"] = pred["obs_ci_lower"].values
        proj_df[f"{metric}_hi95"] = pred["obs_ci_upper"].values
        print(f"  {metric}: R^2={ols_model.rsquared:.3f}, "
              f"slope p-value={ols_model.pvalues['ratio']:.4f}")

    proj_df.to_csv(f"{du.TAB_DIR}/horizon_error_projected_uncertainty.csv", index=False)
    print()
    print(proj_df[["proj_year", "ratio", "MAE_expected", "MAE_lo95", "MAE_hi95",
                    "rMAE_expected", "rMAE_lo95", "rMAE_hi95"]].to_string(index=False))
    print(f"\nSaved: {du.TAB_DIR}/horizon_error_projected_uncertainty.csv")
    print("\nNote: full OLS prediction interval (statsmodels get_prediction, "
          "obs_ci_lower/upper) — accounts for parameter uncertainty (widens away "
          "from the mean observed ratio) and uses a t-distribution given the "
          "small sample (n=15). Still assumes the ratio-error relationship stays "
          "linear and homoscedastic; the 15 points are not fully independent "
          "(drawn from only 5 trained models).")


if __name__ == "__main__":
    main()
