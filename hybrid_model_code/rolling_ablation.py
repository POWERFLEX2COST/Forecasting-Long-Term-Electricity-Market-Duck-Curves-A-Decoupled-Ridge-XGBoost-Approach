# =============================================================================
# ROBUSTNESS CHECK: expanding-window ablation backtest, anchored at 2025.
#
# The single-year (2025-only) ablation backtest showed Ridge-only and the
# full hybrid within noise of each other in aggregate MAE/RMSE. Before
# writing any claim about the marginal value of the XGBoost residual
# correction into the paper, we re-run the SAME three configurations
# (Ridge-only, XGBoost-only, full hybrid) across five held-out TEST windows
# that all end in 2025 but expand backwards, shrinking the training set:
#
#   test window        train window
#   [2024, 2025]        2019-2023  (5 train years)
#   [2023, 2025]        2019-2022  (4 train years)
#   [2022, 2025]        2019-2021  (3 train years)
#   [2021, 2025]        2019-2020  (2 train years)
#   [2020, 2025]        2019       (1 train year — degenerate for the Ridge
#                                    per-cell year-trend: with a single
#                                    training year the trend regression has
#                                    zero variance in the year regressor and
#                                    collapses to a flat cell mean, which is
#                                    the correct/expected fallback, not a bug)
#
# XGBoost early-stopping validation: the last available training year is
# held out as validation (mirroring the paper's train<=N-1/val=N protocol);
# if only one training year exists, a chronological 85/15 split within that
# year is used instead, since no second year is available.
# =============================================================================

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

import data_utils as du
from model_ridge_only import RidgeTrendOnly
from model_ridge_xgboost import ModeloHibrido, add_xgb_features, XGB_FEATURE_COLS
from model_xgboost_only import add_features as add_xgb_only_features, FEATURE_COLS as XGB_ONLY_FEATURE_COLS

sns.set_theme(style="whitegrid")

TEST_STARTS = [2024, 2023, 2022, 2021, 2020]
COLORS = {
    "xgboost_only": "#8172B2",
    "ridge_only": "#64B5A6",
    "ridge_xgboost": "#C0392B",
}
LABELS = {
    "xgboost_only": "XGBoost only",
    "ridge_only": "Ridge only",
    "ridge_xgboost": "Ridge + XGBoost (hybrid)",
}


def xgb_val_split(df_train_full: pd.DataFrame):
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


def run_window(df: pd.DataFrame, test_start: int):
    train_end = test_start - 1
    df_train_full = df[(df["Año"] >= du.TRAIN_START_YEAR) & (df["Año"] <= train_end)].copy()
    df_test = df[(df["Año"] >= test_start) & (df["Año"] <= 2025)].copy()
    n_train_years = train_end - du.TRAIN_START_YEAR + 1

    results = []

    # ---- Ridge only ---------------------------------------------------
    ridge_model = RidgeTrendOnly()
    ridge_model.fit(df_train_full)
    pred_ridge = ridge_model.predict(df_test)
    m = du.compute_metrics(df_test["target"].values, pred_ridge, df_test["target_naive"].values)
    results.append({"test_start": test_start, "n_train_years": n_train_years,
                     "key": "ridge_only", **m})

    # ---- XGBoost only (calendar features + year) -----------------------
    df_train_feat = add_xgb_only_features(df_train_full)
    df_test_feat = add_xgb_only_features(df_test)
    tr_mask, va_mask = xgb_val_split(df_train_feat)
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
        tree_method="hist", random_state=42, early_stopping_rounds=50,
    )
    xgb_model.fit(
        df_train_feat.loc[tr_mask, XGB_ONLY_FEATURE_COLS].values,
        df_train_feat.loc[tr_mask, "target"].values,
        eval_set=[(df_train_feat.loc[va_mask, XGB_ONLY_FEATURE_COLS].values,
                   df_train_feat.loc[va_mask, "target"].values)],
        verbose=False,
    )
    pred_xgb = xgb_model.predict(df_test_feat[XGB_ONLY_FEATURE_COLS].values)
    m = du.compute_metrics(df_test["target"].values, pred_xgb, df_test["target_naive"].values)
    results.append({"test_start": test_start, "n_train_years": n_train_years,
                     "key": "xgboost_only", **m})

    # ---- Full hybrid ----------------------------------------------------
    df_train_hyb = add_xgb_features(df_train_full)
    df_test_hyb = add_xgb_features(df_test)
    hybrid = ModeloHibrido()
    # ModeloHibrido.fit() internally splits train<=N-1 / val=N using its own
    # logic (mask_tr / mask_val on Año); reproduce the same variable-window
    # behaviour by monkey-patching via direct fit on df_train_hyb (it already
    # uses "Año<=2023"/"Año==2024" hardcoded — for windows other than the
    # canonical one we replicate its logic here instead of calling .fit()).
    hybrid.modelos_tendencia = {}
    df_fit = df_train_hyb.copy()
    df_fit["trend"] = 0.0
    for mth in range(1, 13):
        for h in range(1, 25):
            for f in [0, 1]:
                clave = (mth, h, f)
                mask = (df_fit["mes"] == mth) & (df_fit["hora"] == h) & (df_fit["es_fin_semana"] == f)
                if mask.sum() < 20:
                    continue
                X_lr = df_fit.loc[mask, ["Año"]].values
                y_lr = df_fit.loc[mask, "target"].values
                modelo = hybrid._ajustar_tendencia(X_lr, y_lr, clave)
                hybrid.modelos_tendencia[clave] = modelo
                df_fit.loc[mask, "trend"] = modelo.predict(X_lr - du.TRAIN_START_YEAR)
    df_fit["trend"] = df_fit["trend"].replace(0.0, 100.0)
    df_fit["residual"] = df_fit["target"] - df_fit["trend"]

    tr_mask2, va_mask2 = xgb_val_split(df_fit)
    hybrid.xgb = xgb.XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
        tree_method="hist", random_state=42, early_stopping_rounds=50,
    )
    hybrid.xgb.fit(
        df_fit.loc[tr_mask2, XGB_FEATURE_COLS].values, df_fit.loc[tr_mask2, "residual"].values,
        eval_set=[(df_fit.loc[va_mask2, XGB_FEATURE_COLS].values, df_fit.loc[va_mask2, "residual"].values)],
        verbose=False,
    )
    pred_hybrid = hybrid.predict(df_test_hyb)
    m = du.compute_metrics(df_test["target"].values, pred_hybrid, df_test["target_naive"].values)
    results.append({"test_start": test_start, "n_train_years": n_train_years,
                     "key": "ridge_xgboost", **m})

    return results


def main():
    df = du.load_data()
    all_results = []
    for test_start in TEST_STARTS:
        print(f"\n=== Test window [{test_start}, 2025]  (train 2019-{test_start-1}) ===")
        res = run_window(df, test_start)
        for r in res:
            print(f"  {r['key']:15s} MAE={r['MAE']:.2f}  rMAE={r['rMAE']:.3f}  rRMSE={r['rRMSE']:.3f}")
        all_results.extend(res)

    out = pd.DataFrame(all_results)
    out["label"] = out["key"].map(LABELS)
    out["test_window"] = out["test_start"].apply(lambda y: f"{y}-2025")
    out.to_csv(f"{du.TAB_DIR}/rolling_ablation.csv", index=False)
    print(f"\nSaved: {du.TAB_DIR}/rolling_ablation.csv")

    # ---- LaTeX table (rMAE only, compact) ----------------------------------
    piv = out.pivot(index="label", columns="test_window", values="rMAE")
    piv = piv.reindex([LABELS[k] for k in ["ridge_only", "xgboost_only", "ridge_xgboost"]])
    cols = [f"{y}-2025" for y in TEST_STARTS]
    piv = piv[cols]
    lines = [
        r"\begin{table}[H]", r"    \centering",
        r"    \caption{Robustness check: rMAE (vs.\ naïve) of the ablation configurations "
        r"across five expanding test windows anchored at 2025. Source: Own elaboration.}",
        r"    \label{tab:rolling_ablation}",
        r"    \begin{tabular}{l" + "c" * len(cols) + "}",
        r"        \toprule",
        r"        \textbf{Configuration} & " + " & ".join([r"\textbf{" + c + "}" for c in cols]) + r" \\",
        r"        \midrule",
    ]
    for label, row in piv.iterrows():
        vals = [f"{row[c]:.3f}" for c in cols]
        lines.append(f"        {label} & " + " & ".join(vals) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    with open(f"{du.TAB_DIR}/table_rolling_ablation.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {du.TAB_DIR}/table_rolling_ablation.tex")

    # ---- Figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric in zip(axes, ["rMAE", "rRMSE"]):
        for key in ["xgboost_only", "ridge_only", "ridge_xgboost"]:
            sub = out[out["key"] == key].sort_values("test_start", ascending=False)
            ax.plot(sub["test_window"], sub[metric], marker="o", color=COLORS[key],
                     lw=1.8, label=LABELS[key])
        ax.axhline(1.0, color="black", lw=1, linestyle="--", alpha=0.6)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel("Test window (train ends the year before window start)")
        ax.tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle("Ablation robustness: expanding test windows anchored at 2025", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = f"{du.FIG_DIR}/rolling_ablation.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
