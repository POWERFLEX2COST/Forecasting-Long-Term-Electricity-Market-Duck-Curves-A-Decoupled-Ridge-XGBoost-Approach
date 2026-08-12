# =============================================================================
# STRATIFIED ERROR ANALYSIS: hybrid vs. Ridge-only, sliced by the two
# subsets that matter for the paper's actual research questions (negative-
# price hours and extreme peaks), since the aggregate 2025 ablation backtest
# (table_ablation.tex) shows the two configurations statistically tied.
# =============================================================================

import numpy as np
import pandas as pd

import data_utils as du


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    hyb = pd.read_csv(f"{du.PRED_DIR}/backtest_2025_ridge_xgboost.csv")
    rid = pd.read_csv(f"{du.PRED_DIR}/backtest_2025_ridge_only.csv")
    df = hyb.merge(rid, on="timestamp", suffixes=("_hyb", "_ridge"))
    assert (df["target_hyb"] == df["target_ridge"]).all()

    y = df["target_hyb"].values
    p_hyb = df["target_pred_hyb"].values
    p_ridge = df["target_pred_ridge"].values

    neg_mask = y <= 0
    p95 = np.percentile(y, 95)
    ext_mask = y > p95

    rows = [
        {"subset": "Negative-price hours (target<=0)", "n": int(neg_mask.sum()),
         "MAE_hybrid": mae(y[neg_mask], p_hyb[neg_mask]),
         "MAE_ridge_only": mae(y[neg_mask], p_ridge[neg_mask])},
        {"subset": f"Extreme peaks (>p95={p95:.1f})", "n": int(ext_mask.sum()),
         "MAE_hybrid": mae(y[ext_mask], p_hyb[ext_mask]),
         "MAE_ridge_only": mae(y[ext_mask], p_ridge[ext_mask])},
        {"subset": "All hours (signed bias, pred-actual)", "n": len(y),
         "MAE_hybrid": float(np.mean(p_hyb - y)),
         "MAE_ridge_only": float(np.mean(p_ridge - y))},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(f"{du.TAB_DIR}/stratified_error_analysis.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {du.TAB_DIR}/stratified_error_analysis.csv")


if __name__ == "__main__":
    main()
