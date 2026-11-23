"""
Fig. 16 of the paper: hours with price <= 0 by hour and month for 2023, 2025, 2030 and 2035.
Combined intraday/seasonal heatmap of hours with price <= 0 for four
representative years (2023, 2025, 2030, 2035), (the five
detailed figures are in the appendix, see figA1_A5_negative_hours_heatmaps.py).
Laid out as a 2x2 grid so each panel renders large and legible at
\\textwidth in the paper.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_projection, OUT  # noqa: E402
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIG_OUT = OUT / "evolucion_h_negativas_2023_2025_2030_2035.png"

YEARS = [2023, 2025, 2030, 2035]   # row-major: (2023,2025) top row, (2030,2035) bottom row
HOURS_DESC = list(range(24, 0, -1))
MONTHS_ALL = list(range(1, 13))
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# ── Load data ────────────────────────────────────────────────────────────────
df = load_projection()
df['Año'] = df['Año'].astype(int)
df['mes'] = df['mes'].astype(int)
df['hora'] = df['hora'].astype(int)

pivots = {}
for y in YEARS:
    sub = df[(df['Año'] == y) & (df['porcentaje_sobre_media'] <= 0)]
    counts = (sub.groupby(['hora', 'mes']).size()
                 .reset_index(name='neg'))
    idx = pd.MultiIndex.from_product([HOURS_DESC, MONTHS_ALL], names=['hora', 'mes'])
    counts = (counts.set_index(['hora', 'mes']).reindex(idx, fill_value=0).reset_index())
    pivots[y] = counts.pivot(index='hora', columns='mes', values='neg').loc[HOURS_DESC, MONTHS_ALL]

vmax = max(p.values.max() for p in pivots.values())

# ── Plot (2 rows x 2 cols) ────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'serif', 'font.size': 13})

fig, axes = plt.subplots(2, 2, figsize=(15.5, 16.5))
cbar_ax = fig.add_axes([0.91, 0.15, 0.016, 0.7])

for i, (y, ax) in enumerate(zip(YEARS, axes.flat)):
    pivot = pivots[y]
    is_last = (i == len(YEARS) - 1)
    sns.heatmap(
        pivot, ax=ax, cmap='YlOrRd', vmin=0, vmax=vmax,
        annot=True, fmt='d', annot_kws={'size': 15},
        linewidths=0.5, linecolor='white',
        xticklabels=MONTH_NAMES, yticklabels=HOURS_DESC,
        cbar=is_last, cbar_ax=cbar_ax if is_last else None,
    )
    # NOTE: no set_box_aspect() here on purpose -- forcing a fixed box
    # aspect makes matplotlib shrink each axes to fit within its grid
    # slot, leaving large uncontrolled blank margins between rows/cols.
    # Instead the panel shape is fixed directly via figsize + wspace/hspace
    # below, so each axes fills its slot exactly with no dead space.
    label = 'Historical' if y <= 2025 else 'Projected'
    ax.set_title(f'{label} Year {y}', fontsize=18, fontweight='bold')
    ax.set_xlabel('Month', fontsize=15)
    ax.set_ylabel('Hour of the day', fontsize=15)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=12, rotation=0)

    # Contrast-aware annotation colour (white on dark cells, black on light cells)
    for text_obj, val in zip(ax.texts, pivot.values.flatten()):
        text_obj.set_color('white' if val > vmax * 0.55 else 'black')

cbar_ax.set_ylabel('No. of Negative Hours', fontsize=16)
cbar_ax.tick_params(labelsize=14)

fig.subplots_adjust(left=0.05, right=0.895, top=0.96, bottom=0.04, wspace=0.12, hspace=0.15)
fig.savefig(FIG_OUT, dpi=260, bbox_inches='tight')
print(f"Saved: {FIG_OUT}")
