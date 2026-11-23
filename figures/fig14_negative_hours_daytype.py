"""
Fig. 14 of the paper: annual hours with price <= 0 by day type (working day vs weekend/holiday).
Stacked bar chart: annual negative-price hours split by Working Day vs Holiday/Weekend.
Blue (Holiday/Weekend) at bottom, Red (Working Day) on top.
Labels at each segment boundary (inside if large enough, above if small),
plus total above each bar.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_projection, OUT  # noqa: E402
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import holidays

FIG_OUT = OUT / "negative_hours_daytype_comparison.png"

YEARS_ALL    = list(range(2019, 2036))
HOURS_IN_YEAR = {y: 8784 if y in (2024, 2028, 2032) else 8760 for y in YEARS_ALL}
COLOR_HOL    = '#5C8BE0'   # blue  – Holiday / Weekend
COLOR_WRK    = '#E05C5C'   # red   – Working Day
HIST_COLOUR  = '#cccccc'   # light shade for historical bars
MIN_INNER    = 90          # minimum segment height (h) to place label inside

# ── Load & classify ────────────────────────────────────────────────────────────
df = load_projection()
df['Año'] = df['Año'].astype(int)

es_hols: dict = {}
for y in sorted(df['Año'].unique()):
    es_hols.update(holidays.Spain(years=int(y)))

df['date']       = df['timestamp'].dt.date
df['dow']        = df['timestamp'].dt.dayofweek
df['is_holiday'] = df['date'].isin(es_hols)
df['is_weekend'] = df['dow'] >= 5
df['day_type']   = np.where(
    (~df['is_holiday']) & (~df['is_weekend']),
    'Working Day', 'Holiday / Weekend',
)

df_neg = df[df['porcentaje_sobre_media'] <= 0]

counts = (
    df_neg.groupby(['Año', 'day_type']).size()
          .unstack(fill_value=0)
          .reindex(YEARS_ALL, fill_value=0)
)
for col in ('Working Day', 'Holiday / Weekend'):
    if col not in counts.columns:
        counts[col] = 0

hol_h = counts['Holiday / Weekend'].values.astype(float)
wrk_h = counts['Working Day'].values.astype(float)
tot_h = hol_h + wrk_h
yr_h  = np.array([HOURS_IN_YEAR[y] for y in YEARS_ALL], dtype=float)

hol_pct = hol_h / yr_h * 100
wrk_pct = wrk_h / yr_h * 100
tot_pct = tot_h / yr_h * 100

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(20, 8))
x     = np.arange(len(YEARS_ALL))
width = 0.60

# Stacked bars
ax.bar(x, hol_h, width, color=COLOR_HOL, label='Holiday / Weekend',
       edgecolor='white', linewidth=0.4)
ax.bar(x, wrk_h, width, bottom=hol_h, color=COLOR_WRK, label='Working Day',
       edgecolor='white', linewidth=0.4)

# ── Annotations ────────────────────────────────────────────────────────────────
for i, (hh, wh, th, hp, wp, tp, yr) in enumerate(
        zip(hol_h, wrk_h, tot_h, hol_pct, wrk_pct, tot_pct, YEARS_ALL)):

    hh, wh, th = int(hh), int(wh), int(th)

    # — Holiday / Weekend segment —
    if hh >= MIN_INNER:
        # label inside blue bar, vertically centred
        ax.text(x[i], hh / 2,
                f'{hh}\n({hp:.1f}%)',
                ha='center', va='center',
                fontsize=11, color='black', fontweight='bold')
    elif hh > 0:
        # too small for inside label: annotate to the right of bar top
        ax.annotate(
            f'{hh} ({hp:.1f}%)',
            xy=(x[i] + width / 2, hh),
            xytext=(x[i] + width / 2 + 0.05, hh + max(tot_h) * 0.03),
            fontsize=10, color=COLOR_HOL, fontweight='bold',
            ha='left', va='bottom',
            arrowprops=dict(arrowstyle='-', color=COLOR_HOL, lw=0.8),
        )

    # — Working Day segment —
    if wh >= MIN_INNER:
        # label inside red bar, vertically centred
        ax.text(x[i], hh + wh / 2,
                f'{wh}\n({wp:.1f}%)',
                ha='center', va='center',
                fontsize=11, color='black', fontweight='bold')
    elif wh > 0:
        ax.annotate(
            f'{wh} ({wp:.1f}%)',
            xy=(x[i] + width / 2, th),
            xytext=(x[i] + width / 2 + 0.05, th + max(tot_h) * 0.05),
            fontsize=10, color=COLOR_WRK, fontweight='bold',
            ha='left', va='bottom',
            arrowprops=dict(arrowstyle='-', color=COLOR_WRK, lw=0.8),
        )

    # — Total above bar —
    if th > 0:
        ax.text(x[i], th + max(tot_h) * 0.015,
                f'{th}\n({tp:.1f}%)',
                ha='center', va='bottom',
                fontsize=11, color='#222222', fontweight='bold')

# Dashed separator: historical vs projection
sep = years_all_idx = YEARS_ALL.index(2026) - 0.5
ax.axvline(sep, color='black', linestyle='--', linewidth=1.5,
           label='End of Historical / Start of Projection')

ax.set_xticks(x)
ax.set_xticklabels(YEARS_ALL, fontsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.set_xlabel('Year', fontsize=15)
ax.set_ylabel('Number of Hours with Price ≤ 0', fontsize=15)
ax.set_ylim(0, max(tot_h) * 1.22)
ax.yaxis.set_major_locator(mticker.MultipleLocator(500))
ax.legend(fontsize=13, loc='upper left', framealpha=0.9)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
fig.savefig(FIG_OUT, dpi=200, bbox_inches='tight')
print(f"Saved: {FIG_OUT}")
