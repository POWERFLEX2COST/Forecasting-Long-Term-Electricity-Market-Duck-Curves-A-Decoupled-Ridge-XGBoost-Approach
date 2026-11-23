"""Figs. 11-13 of the paper, from the final projection (data/spain_electricity_price_pct_monthly_average_2019_2035.csv).
- Fig. 11: annual mean relative price at 14:00 and 20:00, 2019-2035.
- Fig. 12: mean hourly profile for 2019, 2025 and 2035 (2026-2034 in light red).
- Fig. 13: mean hourly profiles for each month: 2019, 2025, 2030, 2035.
"""
import calendar
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_projection, OUT  # noqa: E402
import matplotlib.pyplot as plt

p = load_projection()
V = "porcentaje_sobre_media"

# Fig. 11: hours 14 and 20
fig, ax = plt.subplots(figsize=(11, 3.6))
for hh, col, lab in ((14, "#E1A100", "Solar hour (14:00)"), (20, "#5B2C83", "Peak demand (20:00)")):
    s = p[p.hora == hh].groupby("anio")[V].mean()
    ax.plot(s.index, s.values, "o-", color=col, label=lab)
ax.axvline(2025.5, color="k", ls="--", lw=1); ax.axhline(0, color="0.6", lw=0.8)
ax.set_ylabel("% of monthly mean"); ax.set_xticks(range(2019, 2036)); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "fig10_h14_h20_paper.png", dpi=200)

# Fig. 12: mean annual profile
fig, ax = plt.subplots(figsize=(11, 3.8))
for a in range(2026, 2035):
    ax.plot(range(1, 25), p[p.anio == a].groupby("hora")[V].mean(), color="#C0392B", alpha=0.12 + 0.05 * (a - 2026), lw=0.8)
ax.plot(range(1, 25), p[p.anio == 2019].groupby("hora")[V].mean(), color="#2471A3", lw=2, label="Year 2019")
ax.plot(range(1, 25), p[p.anio == 2025].groupby("hora")[V].mean(), "k--", lw=2, label="2025 (transition)")
ax.plot(range(1, 25), p[p.anio == 2035].groupby("hora")[V].mean(), color="#C0392B", lw=2.2, label="Year 2035")
ax.set_xticks(range(1, 25)); ax.set_xlabel("Hour of the day"); ax.set_ylabel("% of monthly mean"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig11_perfil_anual_paper.png", dpi=200)

# Fig. 13: 12 months
fig, axs = plt.subplots(4, 3, figsize=(13, 13), sharex=True)
style = {2019: dict(color="0.5", ls=":"), 2025: dict(color="#2471A3", ls="-"),
         2030: dict(color="#E67E22", ls="--"), 2035: dict(color="#C0392B", ls="-.")}
for ax, m in zip(axs.flat, range(1, 13)):
    for a, kw in style.items():
        ax.plot(range(1, 25), p[(p.anio == a) & (p.mes == m)].groupby("hora")[V].mean(), lw=1.8, label=str(a), **kw)
    ax.set_title(calendar.month_name[m], fontsize=10, fontweight="bold"); ax.grid(alpha=0.3); ax.set_xticks([1, 5, 9, 13, 17, 21])
    ax.set_ylim(-40, 600)
for ax in axs[:, 0]: ax.set_ylabel("% of monthly mean")
for ax in axs[-1]: ax.set_xlabel("Hour of the day")
axs[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig12_perfiles_mensuales_paper.png", dpi=170)
print("ok")
