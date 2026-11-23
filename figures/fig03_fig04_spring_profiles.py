"""Figs. 3 and 4 of the paper: historical (2019-2025) mean hourly price profiles (% of monthly mean).
- Fig. 3 (mes_3_4_5_6.png): spring months March-June, all day types.
- Fig. 4 (spring_daytype_profile.png): April and May, working days vs weekends and national
  public holidays. This descriptive split groups weekends with national public holidays, whereas
  the model's day type f treats weekday holidays as working days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, OUT  # noqa: E402
import holidays
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

YEARS = list(range(2019, 2026))
COLORS = cm.plasma(np.linspace(0.10, 0.85, len(YEARS)))
MONTH_NAMES = {3: 'March', 4: 'April', 5: 'May', 6: 'June'}

df = pd.read_csv(DATA / "spain_hourly_prices_pv_2016_2025.csv")
df['Fecha'] = pd.to_datetime(df['Fecha'])
df['year'] = df['Fecha'].dt.year
df['month'] = df['Fecha'].dt.month
# x = P / monthly mean * 100 (paper Eq. 1)
df['pct_monthly'] = df['Precios'] / df.groupby(['year', 'month'])['Precios'].transform('mean') * 100

es_holidays = holidays.ES(years=df['year'].unique())
is_hol = df['Fecha'].dt.date.apply(lambda d: d in es_holidays)
is_wknd = df['Fecha'].dt.dayofweek >= 5
df['daytype'] = np.where(is_hol | is_wknd, 'Weekend / Public Holiday', 'Working Day')
df = df[df['year'].isin(YEARS) & df['month'].isin(MONTH_NAMES)]

# ---- Fig. 3: spring months, all day types
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, month in zip(axes.flatten(), MONTH_NAMES):
    df_m = df[df['month'] == month]
    for i, year in enumerate(YEARS):
        hourly = df_m[df_m['year'] == year].groupby('hora')['pct_monthly'].mean()
        ax.plot(hourly.index, hourly.values, color=COLORS[i], label=str(year), linewidth=1.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.25)
    ax.set_title(f"{MONTH_NAMES[month]} (Spain)", fontsize=13, fontweight='bold')
    ax.set_xlabel('Hour of the Day', fontsize=10)
    ax.set_ylabel('Price (% of Monthly Mean)', fontsize=10)
    ax.set_xticks(range(1, 25, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='upper left')
fig.tight_layout()
fig.savefig(OUT / "mes_3_4_5_6.png", dpi=200, bbox_inches='tight')

# ---- Fig. 4: April and May by day type
DAYTYPES = ['Working Day', 'Weekend / Public Holiday']
COL_TITLES = ['Working Days', 'Weekends & Public Holidays']
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey='row')
for row, month in enumerate((4, 5)):
    df_m = df[df['month'] == month]
    for col, dtype in enumerate(DAYTYPES):
        ax = axes[row, col]
        df_md = df_m[df_m['daytype'] == dtype]
        for i, year in enumerate(YEARS):
            hourly = df_md[df_md['year'] == year].groupby('hora')['pct_monthly'].mean()
            ax.plot(hourly.index, hourly.values, color=COLORS[i], label=str(year), linewidth=1.8)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.25)
        ax.set_title(f"{MONTH_NAMES[month]} (Spain) — {COL_TITLES[col]}", fontsize=11, fontweight='bold')
        ax.set_xticks(range(1, 25, 2))
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Hour of the Day', fontsize=10)
        if col == 0:
            ax.set_ylabel('Price (% of Monthly Mean)', fontsize=10)
        ax.legend(fontsize=8, ncol=2, loc='upper left')
fig.tight_layout()
fig.savefig(OUT / "spring_daytype_profile.png", dpi=200, bbox_inches='tight')
print("ok")
