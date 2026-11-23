# =============================================================================
# FINAL 2026-2035 PROJECTION (revised version, the one reported in the paper)
#
# Ridge trend cells fitted on 2019-2025, XGBoost on the residuals (train <= 2024,
# early-stopping validation 2025), then floor at 0 % + monthly renormalisation
# (data_utils.postprocess). No solar constraint (data_utils.SOLAR_CONSTRAINT).
#
# Output (in ../data):
#   spain_electricity_price_pct_monthly_average_2019_2035.xlsx / .csv
#     Date and Time, Day Type (Working Day / Holiday = weekend or Spanish
#     national public holiday), Historical/Projected, Price (% of Monthly Average).
#     2019-2025 = observed values; 2026-2035 = projection.
# Expected hours with price <= 0: 913 (2026), 2524 (2030), 3116 (2035).
# =============================================================================
import os
import holidays
import pandas as pd

import data_utils as du
from model_ridge_xgboost import add_xgb_features
from horizon_error_analysis import fit_hybrid

OUT_NAME = "spain_electricity_price_pct_monthly_average_2019_2035"
VALUE = "Price (% of Monthly Average)"

assert not du.SOLAR_CONSTRAINT
df = du.load_data()
model = fit_hybrid(df[df["Año"] <= 2025].copy())
fut = add_xgb_features(du.build_future_calendar())
fut["target_pred"] = model.predict(fut)

hist = pd.DataFrame({"Date and Time": df["timestamp"], "Historical/Projected": "Historical",
                     VALUE: df["target"]})
proj = pd.DataFrame({"Date and Time": fut["timestamp"], "Historical/Projected": "Projected",
                     VALUE: fut["target_pred"]})
# drop_duplicates removes exact duplicate rows of the input; the two distinct
# hours of the October DST change are both kept.
out = pd.concat([hist, proj]).drop_duplicates().sort_values("Date and Time", kind="stable")

dates = out["Date and Time"].dt.date
es_holidays = holidays.Spain(years=range(2019, 2036))
is_holiday = (out["Date and Time"].dt.dayofweek >= 5) | dates.map(lambda d: d in es_holidays)
out.insert(1, "Day Type", is_holiday.map({True: "Holiday", False: "Working Day"}))
out = out.reset_index(drop=True)

out.to_csv(os.path.join(du.DATA_DIR, OUT_NAME + ".csv"), index=False)
out.to_excel(os.path.join(du.DATA_DIR, OUT_NAME + ".xlsx"), index=False,
             sheet_name="Price pct of Monthly Avg")

year = out["Date and Time"].dt.year
hour = out["Date and Time"].dt.hour + 1
print("Hours <= 0 per year:\n", out[VALUE].le(0).groupby(year).sum().to_string())
for h in (14, 20):
    s = out[hour == h].groupby(year[hour == h])[VALUE].mean()
    print(f"hour {h}:", s.loc[[2025, 2026, 2030, 2035]].round(1).to_dict())
print("Annual mean 2035:", round(out.loc[year == 2035, VALUE].mean(), 2))
