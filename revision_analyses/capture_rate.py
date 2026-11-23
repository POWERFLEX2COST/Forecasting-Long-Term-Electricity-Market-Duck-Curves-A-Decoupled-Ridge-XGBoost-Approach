"""SOLAR CAPTURE RATE: observed (OMIE x REE) and implied by the projection (paper Section 3.9, Fig. 17).

CR = sum(P*G) / (sum(G) * mean(P)), annual.
- Observed: real P (EUR/MWh), G measured PV generation (REE).
- Projection: P = x (projection, % of monthly mean) * monthly level; G = 2025 PV profile by
  (month, hour). Monthly level variants: (a) flat (all months equal); (b) mean 2023-2025
  seasonality of the monthly price (monthly mean / annual mean).
Outputs (results/): capture_rate.csv, fig_capture_rate_2030.png
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

hist = pd.read_csv(DATA / "spain_hourly_prices_pv_2016_2025.csv", parse_dates=["Fecha"])
hist["a"] = hist.Fecha.dt.year; hist["m"] = hist.Fecha.dt.month
hist = hist[hist.a.between(2016, 2025)]


def cr(p, g):
    return (p * g).sum() / g.sum() / p.mean()


rows = []
for a, s in hist.groupby("a"):
    x = s.Precios / s.groupby("m").Precios.transform("mean")   # relative price, flat monthly level
    rows.append({"year": a, "CR_real": cr(s.Precios, s.Produccion_FV_MWh),
                 "CR_flat_level": cr(x, s.Produccion_FV_MWh),
                 "hours_le0": int((s.Precios <= 0).sum())})
h = pd.DataFrame(rows)

# 2025 PV profile and price seasonality
g25 = hist[hist.a == 2025].groupby(["m", "hora"]).Produccion_FV_MWh.mean().rename("G")
sea = hist[hist.a.between(2023, 2025)].groupby(["a", "m"]).Precios.mean()
sea = (sea / sea.groupby("a").transform("mean")).groupby("m").mean().rename("level")

pr = pd.read_csv(DATA / "spain_electricity_price_pct_monthly_average_2019_2035.csv", parse_dates=["Date and Time"])
t = pr["Date and Time"]
pr = pd.DataFrame({"a": t.dt.year, "m": t.dt.month, "hora": t.dt.hour + 1, "x": pr["Price (% of Monthly Average)"]})
pr = pr.join(g25, on=["m", "hora"]).join(sea, on="m")
rows = []
for a, s in pr.groupby("a"):
    rows.append({"year": a, "CR_proj_flat_level": cr(s.x, s.G),
                 "CR_proj_seasonal_level": cr(s.x * s.level, s.G),
                 "hours_le0_proj": int((s.x <= 1e-9).sum()),
                 "x_10_18h": s.x[s.hora.between(10, 18)].mean()})
p = pd.DataFrame(rows)
out = h.merge(p, on="year", how="outer").sort_values("year")
out.to_csv(OUT / "capture_rate.csv", index=False, float_format="%.4f")
print(out.round(3).to_string(index=False))

# ---- figure (Fig. 17)
h = out[out.year <= 2025]; p = out[out.year >= 2025]
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(h.year, h.CR_real * 100, "k-o", lw=2, label="Observed (OMIE prices, REE PV output)")
ax.plot(p.year, p.CR_proj_seasonal_level * 100, "--o", color="#C0392B", lw=2,
        label="This paper, BAU projection (2025 PV profile)")
ax.fill_between(p.year, p.CR_proj_flat_level * 100, p.CR_proj_seasonal_level * 100, color="#C0392B", alpha=.15)
# published references (see the paper, Table 14)
ax.plot([2030], [20], "v", ms=11, color="#2471A3", label="S&P Global via EIFO (2026): below 20% (2030), ~16% (2035)")
ax.plot([2035], [16], "s", ms=9, color="#2471A3")
ax.plot([2030, 2035], [60, 45], "D", ms=9, color="#7D3C98", label="Maguire, Reuters (2024): ~60% (2030), ~45% (2035)")
ax.plot([2025, 2030], [50, 50], "-", lw=5, color="#28B463", alpha=.6, label="G-advisory (2024): below 50%, 2025–2030")
ax.axvline(2025.5, color="grey", ls=":", lw=1)
ax.set_xlabel("Year", fontsize=12); ax.set_ylabel("Solar capture rate (%)", fontsize=12)
ax.set_ylim(0, 110); ax.set_xticks(range(2016, 2036, 2)); ax.grid(alpha=.3)
ax.legend(fontsize=9, loc="lower left", frameon=True)
fig.tight_layout()
fig.savefig(OUT / "fig_capture_rate_2030.png", dpi=300)
