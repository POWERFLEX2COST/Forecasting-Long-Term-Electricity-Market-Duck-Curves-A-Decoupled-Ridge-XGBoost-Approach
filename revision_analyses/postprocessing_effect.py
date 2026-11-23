# =============================================================================
# EFFECT OF THE POST-PROCESSING (paper Section 2.4.3, Table 5)
# The hybrid (no solar constraint) is fitted once and its raw output is
# post-processed in four ways:
#   lower bound -50 % | lower bound 0 % | -50 % + monthly renormalisation |
#   0 % + monthly renormalisation (final specification).
# Metrics: rMAE in the 2025 backtest (training 2019-2024) and in the
# January-October 2026 check (training 2019-2025); projected annual mean, mean at
# 20:00 and maximum month-hour mean in 2035 (% of monthly mean); hours <= 0 in 2030.
# Output: results/postprocessing_effect.csv
# =============================================================================
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "hybrid_model_code"))
import data_utils as du                                    # noqa: E402
from model_ridge_xgboost import add_xgb_features           # noqa: E402
from horizon_error_analysis import fit_hybrid              # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
VARIANTS = {"Lower bound -50 %": (-50.0, False), "Lower bound 0 %": (0.0, False),
            "Bound -50 % + renormalisation": (-50.0, True), "Bound 0 % + renormalisation": (0.0, True)}


def raw_predict(model, d):
    """Hybrid output before any post-processing."""
    floor, renorm = du.PRED_FLOOR, du.RENORMALISE
    du.PRED_FLOOR, du.RENORMALISE = -np.inf, False
    try:
        return model.predict(d)
    finally:
        du.PRED_FLOOR, du.RENORMALISE = floor, renorm


def post(raw, d, floor, renorm):
    out = pd.Series(np.clip(raw, floor, None), index=d.index)
    if renorm:
        out = out / out.groupby([d["Año"], d["mes"]]).transform("mean") * 100
    return out.values


def main():
    assert not du.SOLAR_CONSTRAINT
    df = du.load_data()
    dfx = add_xgb_features(df)
    test = dfx[dfx["Año"] == 2025].copy()
    raw_bt = raw_predict(fit_hybrid(df[df["Año"] <= 2024].copy()), test)

    fut = add_xgb_features(du.build_future_calendar())
    raw_pr = raw_predict(fit_hybrid(df[df["Año"] <= 2025].copy()), fut)
    r26 = pd.read_csv(du.CSV_2026_PATH)
    r26["Fecha"] = pd.to_datetime(r26["fecha"]).dt.date

    rows = []
    for name, (floor, renorm) in VARIANTS.items():
        bt = du.compute_metrics(test["target"].values, post(raw_bt, test, floor, renorm), test["target_naive"].values)
        p = fut[["Fecha", "hora", "Año", "mes"]].assign(pred=post(raw_pr, fut, floor, renorm))
        c = p.merge(r26[["Fecha", "hora", "x", "x_naive"]], on=["Fecha", "hora"]).dropna()
        g35, g30 = p[p["Año"] == 2035], p[p["Año"] == 2030]
        rows.append({"post_processing": name, "rMAE_2025": bt["rMAE"],
                     "rMAE_2026": (c.x - c.pred).abs().mean() / (c.x - c.x_naive).abs().mean(),
                     "mean_2035": g35.pred.mean(), "h20_2035": g35[g35.hora == 20].pred.mean(),
                     "max_month_hour_2035": g35.groupby(["mes", "hora"]).pred.mean().max(),
                     "hours_le0_2030": int((g30.pred <= 0).sum())})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "postprocessing_effect.csv"), index=False)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
