"""Figs. A.1-A.5 of the paper: hours with price <= 0 by hour of the day and month, year by year
(2022-2035), in groups of three years."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_projection, OUT  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
GROUPS = ([2022, 2023, 2024], [2025, 2026, 2027], [2028, 2029, 2030], [2031, 2032, 2033], [2034, 2035])


def plot_group(df, years):
    df_neg = df[df['porcentaje_sobre_media'] <= 0]
    sns.set_theme(style="whitegrid")
    detail = df_neg.groupby(['Año', 'mes', 'hora']).size().reset_index(name='neg_hours')
    full = pd.MultiIndex.from_product([range(2019, 2036), range(1, 13), range(1, 25)], names=['Año', 'mes', 'hora'])
    detail = detail.set_index(['Año', 'mes', 'hora']).reindex(full, fill_value=0).reset_index()

    # always 3 heatmap slots + colorbar (3rd slot left blank if only 2 years)
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.1)
    ax0 = fig.add_subplot(gs[0])
    axes = [ax0, fig.add_subplot(gs[1], sharey=ax0), fig.add_subplot(gs[2], sharey=ax0)]
    cbar_ax = fig.add_subplot(gs[3])
    vmax = max(detail[detail['Año'] == y]['neg_hours'].max() for y in years)

    for i in range(3):
        if i < len(years):
            year = years[i]
            pivot = detail[detail['Año'] == year].pivot(index='hora', columns='mes', values='neg_hours')
            sns.heatmap(pivot, ax=axes[i], cmap='YlOrRd', annot=True, fmt='d', annot_kws={'size': 10},
                        vmin=0, vmax=vmax, cbar=True, cbar_ax=cbar_ax,
                        cbar_kws={'label': 'No. of Negative Hours'}, xticklabels=MONTH_NAMES)
            label = 'Projected' if year >= 2026 else 'Historical'
            axes[i].set_title(f'{label} Year {year}', fontsize=16, fontweight='bold')
            axes[i].set_xlabel('Month', fontsize=15)
            axes[i].tick_params(axis='both', labelsize=12)
            if i == 0:
                axes[i].set_ylabel('Hour of the day', fontsize=15)
            else:
                axes[i].set_ylabel('')
                axes[i].tick_params(labelleft=False)
        else:
            axes[i].axis('off')
    axes[0].invert_yaxis()
    cbar_ax.tick_params(labelsize=12)
    cbar_ax.yaxis.label.set_size(15)

    suffix = '_'.join(str(y)[-2:] for y in years)
    fig.tight_layout()
    fig.savefig(OUT / f'evolucion_h_negativas_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


df = load_projection()
for g in GROUPS:
    plot_group(df, g)
print("ok")
