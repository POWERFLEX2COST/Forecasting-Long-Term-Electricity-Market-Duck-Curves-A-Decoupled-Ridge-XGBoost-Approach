"""
Fig. 15 of the paper: monthly heatmap of hours with price <= 0, 2019-2035.
Monthly heatmap of negative-price hours (2019-2035).
Vertical dashed line between 2025 and 2026 separates historical from projected.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_projection, OUT  # noqa: E402
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIG_OUT = OUT / "estacionalidad_canibalizacion.png"

YEARS_ALL   = list(range(2019, 2036))
MONTHS_ALL  = list(range(1, 13))
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
HOURS_IN_YEAR = {y: 8784 if y in (2024, 2028, 2032) else 8760 for y in YEARS_ALL}

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_projection()
df['Año'] = df['Año'].astype(int)
df['mes'] = df['mes'].astype(int)

df_neg = df[df['porcentaje_sobre_media'] <= 0]

# Total hours per (year, month) for percentage calculation
total_hrs = (df.groupby(['Año', 'mes']).size()
               .reset_index(name='total'))
neg_hrs = (df_neg.groupby(['Año', 'mes']).size()
                 .reset_index(name='neg'))

idx = pd.MultiIndex.from_product([YEARS_ALL, MONTHS_ALL], names=['Año', 'mes'])
total_hrs = (total_hrs.set_index(['Año', 'mes']).reindex(idx, fill_value=0).reset_index())
neg_hrs   = (neg_hrs.set_index(['Año', 'mes']).reindex(idx, fill_value=0).reset_index())

merged = neg_hrs.merge(total_hrs, on=['Año', 'mes'])
merged['pct'] = np.where(
    merged['total'] > 0,
    merged['neg'] / merged['total'] * 100, 0.0
)

pivot_counts = merged.pivot(index='mes', columns='Año', values='neg')
pivot_pct    = merged.pivot(index='mes', columns='Año', values='pct')

# Build annotation matrix
annot = pd.DataFrame(index=MONTHS_ALL, columns=YEARS_ALL, dtype=object)
for m in MONTHS_ALL:
    for y in YEARS_ALL:
        c = int(pivot_counts.loc[m, y])
        p = pivot_pct.loc[m, y]
        annot.loc[m, y] = '0' if c == 0 else f'{c}\n({p:.1f}%)'

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

fig, ax = plt.subplots(figsize=(22, 10))

sns.heatmap(
    pivot_counts,
    ax=ax,
    cmap='Blues',
    annot=annot,
    fmt='',
    annot_kws={'size': 12, 'fontweight': 'bold'},
    linewidths=0.4,
    linecolor='#dddddd',
    vmin=0,
    vmax=int(pivot_counts.values.max()),
    cbar_kws={'label': 'No. of Hours with Price ≤ 0', 'shrink': 0.85},
    xticklabels=YEARS_ALL,
    yticklabels=MONTH_NAMES,
)

# Vertical dashed line between 2025 (col index 6) and 2026 (col index 7)
sep_x = YEARS_ALL.index(2026)   # = 7
ax.axvline(sep_x, color='black', linestyle='--', linewidth=2.0)

# Labels above the separator line (outside axes, in figure coords)
# Convert data x-coordinate to axes fraction
x_frac = sep_x / len(YEARS_ALL)
ax.text(x_frac - 0.02, 1.025, '← Historical',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=15, fontstyle='italic', color='#333333')
ax.text(x_frac + 0.02, 1.025, 'Projected →',
        transform=ax.transAxes, ha='left', va='bottom',
        fontsize=15, fontstyle='italic', color='#333333')

ax.set_xlabel('Year', fontsize=16)
ax.set_ylabel('Month of the Year', fontsize=16)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14, rotation=0)

# Colorbar font sizes
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=13)
cbar.set_label('No. of Hours with Price ≤ 0', fontsize=15)

plt.tight_layout()
fig.savefig(FIG_OUT, dpi=200, bbox_inches='tight')
print(f"Saved: {FIG_OUT}")
