# =============================================================================
# DIAGNOSTIC: overlay the OLD simple-normal band (point +/- 1.96*resid_std)
# against the NEW statsmodels OLS prediction interval, for rMAE and rRMSE,
# to make visually explicit how much the switch actually changed the band.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

import data_utils as du

sns.set_theme(style="whitegrid")

REAL_N_TRAIN_YEARS = 2025 - du.TRAIN_START_YEAR + 1

out = pd.read_csv(f"{du.TAB_DIR}/horizon_error_analysis.csv")

proj_years = list(range(2026, 2036))
horizons = [y - 2025 for y in proj_years]
ratios = [h / REAL_N_TRAIN_YEARS for h in horizons]
proj_df = pd.DataFrame({"proj_year": proj_years, "ratio": ratios})

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
for ax, metric in zip(axes, ["rMAE", "rRMSE"]):
    # OLD: constant-width band from scipy.linregress residual std
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(out["ratio"], out[metric])
    fitted = intercept + slope * out["ratio"]
    resid_std = float(np.std(out[metric] - fitted, ddof=2))
    old_point = intercept + slope * proj_df["ratio"]
    old_lo = old_point - 1.96 * resid_std
    old_hi = old_point + 1.96 * resid_std

    # NEW: statsmodels OLS full prediction interval
    X_train = sm.add_constant(out[["ratio"]])
    X_new = sm.add_constant(proj_df[["ratio"]], has_constant="add")
    ols_model = sm.OLS(out[metric], X_train).fit()
    pred = ols_model.get_prediction(X_new).summary_frame(alpha=0.05)

    ax.fill_between(proj_df["proj_year"], old_lo, old_hi, color="#8172B2", alpha=0.25,
                     label="OLD: normal band (const. width)")
    ax.fill_between(proj_df["proj_year"], pred["obs_ci_lower"], pred["obs_ci_upper"],
                     color="#C0392B", alpha=0.25, label="NEW: OLS prediction interval")
    ax.plot(proj_df["proj_year"], pred["mean"], color="#C0392B", lw=2, marker="o")
    ax.axhline(1.0, color="black", lw=1, linestyle="--", alpha=0.6)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_xlabel("Projection year")
    ax.tick_params(axis="x", rotation=45)

    width_old = (old_hi - old_lo)
    width_new = (pred["obs_ci_upper"] - pred["obs_ci_lower"]).values
    print(f"{metric}: band width 2026 old={width_old.iloc[0]:.3f} new={width_new[0]:.3f} "
          f"(+{100*(width_new[0]/width_old.iloc[0]-1):.0f}%);  "
          f"2035 old={width_old.iloc[-1]:.3f} new={width_new[-1]:.3f} "
          f"(+{100*(width_new[-1]/width_old.iloc[-1]-1):.0f}%)")

axes[0].set_ylabel("Error relative to naive benchmark")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Old constant-width normal band vs. new OLS prediction interval (overlaid)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
path = f"{du.FIG_DIR}/band_method_comparison.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
print(f"\nSaved: {path}")
