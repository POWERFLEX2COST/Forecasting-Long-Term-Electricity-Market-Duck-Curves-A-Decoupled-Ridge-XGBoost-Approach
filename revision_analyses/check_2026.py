# =============================================================================
# OUT-OF-SAMPLE CHECK OF THE 2026 PROJECTION WITH OBSERVED PRICES (Jan-Oct 2026)
# Projection: ../data/spain_electricity_price_pct_monthly_average_2019_2035.csv
#             (written by ../hybrid_model_code/final_projection.py).
# Observed: ../data/observed_prices_2026_jan_oct.csv (OMIE Jan-Sep, Ember Oct).
#   x = price as % of the realised monthly mean; naive = value at t-168 h.
# Metrics per month and for Jan-Oct: MAE, rMAE and rRMSE vs the weekly naive,
# MAE in EUR/MWh with the realised monthly level, hours <= 0, profile correlation.
# Outputs (results/): check_2026_monthly.csv, fig_profiles_2026.png, fig_summary_2026.png
# =============================================================================
import calendar
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

d = pd.read_csv(DATA / "observed_prices_2026_jan_oct.csv", parse_dates=["fecha"])
p = pd.read_csv(DATA / "spain_electricity_price_pct_monthly_average_2019_2035.csv", parse_dates=["Date and Time"])
p = p[p["Date and Time"].dt.year == 2026]
p = pd.DataFrame({"fecha": p["Date and Time"].dt.normalize(), "hora": p["Date and Time"].dt.hour + 1,
                  "xp": p["Price (% of Monthly Average)"]})
d = d.merge(p, on=["fecha", "hora"])
d["Pp"] = d.xp / 100 * d.P_mes
rows = []
for lab, g in [("Jan-Oct", d)] + [(calendar.month_abbr[m], g) for m, g in d.groupby("mes")]:
    mae = np.abs(g.x - g.xp).mean(); maen = np.abs(g.x - g.x_naive).mean()
    rows.append({"period": lab, "P_mean": g.precio.mean(), "MAE_pct": mae, "rMAE": mae / maen,
                 "rRMSE": np.sqrt(((g.x - g.xp) ** 2).mean()) / np.sqrt(((g.x - g.x_naive) ** 2).mean()),
                 "MAE_eur": np.abs(g.precio - g.Pp).mean(), "MAE_eur_naive": np.abs(g.precio - g.precio_naive).mean(),
                 "le0_obs": int((g.precio <= 0).sum()), "le0_proj": int((g.xp <= 0).sum()),
                 "r_profile": g.groupby("hora").x.mean().corr(g.groupby("hora").xp.mean())})
T = pd.DataFrame(rows); T.to_csv(OUT / "check_2026_monthly.csv", index=False)
print(T.round(3).to_string(index=False))

fig, axs = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
for ax, (m, g) in zip(axs.flat, d.groupby("mes")):
    ax.plot(range(1, 25), g.groupby("hora").x.mean(), color="0.25", lw=1.8, label="Observed 2026")
    ax.plot(range(1, 25), g.groupby("hora").xp.mean(), color="#C0392B", lw=1.6, ls="--", label="Projected 2026")
    ax.axhline(100, color="0.7", lw=0.8, ls=":"); ax.axhline(0, color="0.7", lw=0.8)
    ax.set_title(calendar.month_name[m], fontsize=10); ax.grid(alpha=0.3); ax.set_xticks([1, 7, 13, 19, 24])
for ax in axs[:, 0]: ax.set_ylabel("% of monthly mean")
for ax in axs[1]: ax.set_xlabel("Hour of the day")
axs[0, 0].legend(fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig(OUT / "fig_profiles_2026.png", dpi=200)

M = T[T.period != "Jan-Oct"]; x = np.arange(1, len(M) + 1); w = 0.38
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
a1.bar(x - w / 2, M.le0_obs, w, color="0.35", label="Observed"); a1.bar(x + w / 2, M.le0_proj, w, color="#C0392B", label="Projected")
a1.set_xticks(x, M.period); a1.set_ylabel("Hours with price ≤ 0"); a1.legend(); a1.grid(axis="y", alpha=0.3)
a2b = a2.twinx(); a2b.bar(x, M.P_mean, color="0.85"); a2b.set_ylabel("Observed monthly mean (€/MWh)")
a2.plot(x, M.rMAE, "o-", color="#C0392B", label="rMAE (% of monthly mean)")
a2.plot(x, M.MAE_eur / M.MAE_eur_naive, "s--", color="#2471A3", label="rMAE (€/MWh, realised monthly level)")
a2.axhline(1, color="k", ls="--", lw=1); a2.set_xticks(x, M.period); a2.set_ylabel("Relative error vs naïve")
a2.set_zorder(a2b.get_zorder() + 1); a2.patch.set_visible(False); a2.legend(fontsize=8, loc="upper left"); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_summary_2026.png", dpi=200)
