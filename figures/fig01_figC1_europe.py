"""
Figs. 1 and C.1 of the paper: Spain in the European context (Ember data, 31 markets).

  Fig. 1  : negative_hours_europe.png
            Choropleth maps - annual hours with price <= 0, 2022-2025, 31 countries
  Fig. C.1: spring_profile_europe.png
            Mean hourly relative price profile (% of monthly mean), spring 2025
            (March-June), 31 countries

Data source: Ember - European Wholesale Electricity Price Data (hourly), CC BY 4.0
  https://ember-energy.org/data/european-wholesale-electricity-price-data/
The file is not redistributed here (40 MB). Download the hourly zip and save it as
  data/external/european_wholesale_electricity_price_data_hourly.zip
Country borders: Natural Earth 1:110m (downloaded on first run). Requires geopandas.
"""

import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBER_ZIP   = os.path.join(REPO, "data", "external",
                           "european_wholesale_electricity_price_data_hourly.zip")
FIGURES_DIR = os.path.join(REPO, "figures", "output")
if not os.path.exists(EMBER_ZIP):
    raise SystemExit(f"Ember data not found: {EMBER_ZIP} (see the docstring for the download link)")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

# ---------------------------------------------------------------------------
# Load Ember data
# ---------------------------------------------------------------------------
print("Loading Ember European wholesale electricity price data ...")
with zipfile.ZipFile(EMBER_ZIP) as z:
    with z.open("all_countries.csv") as f:
        raw = pd.read_csv(
            f,
            usecols=["Country", "Datetime (UTC)", "Price (EUR/MWhe)"],
            parse_dates=["Datetime (UTC)"],
        )

raw = raw.rename(columns={
    "Datetime (UTC)":  "Datetime",
    "Price (EUR/MWhe)": "Price_EUR_MWh",
})
raw["Date"]  = raw["Datetime"].dt.normalize()
raw["Hour"]  = raw["Datetime"].dt.hour + 1   # 0-based UTC hour → 1–24
raw["Year"]  = raw["Datetime"].dt.year
raw["Month"] = raw["Datetime"].dt.month

# % of monthly mean per country (used in Figure B)
monthly_mean = raw.groupby(["Country", "Year", "Month"])["Price_EUR_MWh"].transform("mean")
raw["Price_pct_of_monthly_mean"] = (raw["Price_EUR_MWh"] / monthly_mean * 100).round(4)

all_data = raw.copy()
countries_list = sorted(all_data["Country"].unique())
print(f"  {len(all_data):,} rows | {len(countries_list)} countries")
print(f"  Date range: {all_data['Datetime'].min().date()} to {all_data['Datetime'].max().date()}\n")

# ---------------------------------------------------------------------------
# Negative hours (price ≤ 0) per country per year
# ---------------------------------------------------------------------------
print("Counting hours with price <= 0 ...")
neg_table: dict[str, dict[int, int]] = {}
neg_df = all_data[all_data["Price_EUR_MWh"] <= 0]
for country in countries_list:
    grp = neg_df[neg_df["Country"] == country].groupby("Year").size()
    neg_table[country] = grp.to_dict()
    for yr in range(2019, 2027):
        neg_table[country].setdefault(yr, 0)

years_all = list(range(2019, 2026))
header = f"{'Country':<22}" + "".join(f"{y:>6}" for y in years_all)
print(header)
print("-" * len(header))
for name in sorted(neg_table, key=lambda n: neg_table[n].get(2025, 0), reverse=True):
    row = f"{name:<22}" + "".join(f"{neg_table[name].get(y, 0):>6}" for y in years_all)
    print(row)
print()


# ---------------------------------------------------------------------------
# FIGURE A  –  Choropleth maps: annual hours with price ≤ 0, 2022-2025
# ---------------------------------------------------------------------------
YEARS_CHORO  = [2022, 2023, 2024, 2025]
HOURS_IN_YEAR = {2022: 8760, 2023: 8760, 2024: 8784, 2025: 8760}

NE_URL   = ("https://naciscdn.org/naturalearth/110m/cultural/"
            "ne_110m_admin_0_countries.zip")
NE_CACHE = os.path.join(REPO, "data", "external", "ne_110m_admin_0_countries.gpkg")
print("Loading Natural Earth country geometries ...")
if os.path.exists(NE_CACHE):
    world = gpd.read_file(NE_CACHE)
else:
    world = gpd.read_file(NE_URL)
    os.makedirs(os.path.dirname(NE_CACHE), exist_ok=True)
    world.to_file(NE_CACHE, driver="GPKG")

# Ember name → Natural Earth NAME field (only entries that differ)
NAME_OVERRIDE = {
    "United Kingdom": "United Kingdom",
    "North Macedonia": "North Macedonia",
}
NAME_MAP = {c: NAME_OVERRIDE.get(c, c) for c in countries_list}

# Manual label anchor positions (lon, lat) – spread to avoid overlap
LABEL_POS = {
    # Iberian Peninsula
    "Spain":           ( -3.5,  40.0),
    "Portugal":        ( -8.0,  39.5),
    # British Isles
    "United Kingdom":  ( -2.0,  53.5),
    "Ireland":         ( -8.5,  53.0),
    # Western / Nordic
    "France":          (  2.5,  46.5),
    "Germany":         (  9.5,  50.8),
    "Norway":          (  8.5,  65.0),
    "Sweden":          ( 17.0,  62.5),
    "Denmark":         ( 10.5,  56.2),
    "Finland":         ( 27.5,  63.5),
    # Benelux
    "Netherlands":     (  5.3,  52.4),
    "Belgium":         (  4.0,  50.5),
    "Luxembourg":      (  6.5,  49.7),
    # Central Europe
    "Poland":          ( 19.5,  52.0),
    "Czechia":         ( 15.5,  49.8),
    "Slovakia":        ( 20.0,  48.9),
    "Hungary":         ( 18.5,  47.0),
    "Austria":         ( 12.5,  47.5),
    "Switzerland":     (  7.8,  46.7),
    # SE Central
    "Slovenia":        ( 15.5,  46.2),
    "Croatia":         ( 17.5,  45.0),
    # Eastern / Balkan
    "Romania":         ( 24.5,  46.0),
    "Bulgaria":        ( 25.2,  43.0),
    "Serbia":          ( 21.0,  44.2),
    "Montenegro":      ( 19.2,  43.0),
    "North Macedonia": ( 22.5,  41.3),
    # Baltic
    "Estonia":         ( 26.5,  59.0),
    "Latvia":          ( 25.5,  57.2),
    "Lithuania":       ( 23.5,  55.7),
    # Mediterranean
    "Italy":           ( 12.5,  42.5),
    "Greece":          ( 22.0,  39.5),
}

# Countries displayed with full two-line label: count + (pct%)
FULL_LABEL = {"Spain", "France", "Germany", "Italy", "Poland", "United Kingdom",
              "Romania", "Norway", "Sweden", "Finland", "Portugal", "Ireland",
              "Greece", "Bulgaria", "Denmark", "Netherlands", "Belgium"}
# Countries shown with count only (polygon too small for full text)
COUNT_ONLY = {"Montenegro", "North Macedonia", "Luxembourg"}
# All others: single-line  "count (pct%)"

def make_label(country: str, count: int, hours_total: int) -> str:
    if count == 0:
        return "0"
    pct = count / hours_total * 100
    if country in COUNT_ONLY:
        return str(count)
    if country in FULL_LABEL:
        return f"{count}\n({pct:.1f}%)"
    return f"{count} ({pct:.1f}%)"

def label_fs(country: str) -> float:
    if country in FULL_LABEL:
        return 7.5
    if country in COUNT_ONLY:
        return 6.0
    return 6.5

EUROPE_XLIM = (-12, 35)
EUROPE_YLIM = ( 34, 72)

WHITE_LABELS = {
    2024: {"Spain", "Finland"},
    2025: {"Spain", "France"},
}

vmax = max(neg_table[c].get(y, 0) for c in neg_table for y in YEARS_CHORO)
norm = Normalize(vmin=0, vmax=vmax)
cmap = plt.cm.YlOrRd

# Axes are placed directly (not via subplots/subplots_adjust): geopandas'
# .plot() here fills the given axes box exactly (no equal-aspect
# letterboxing), so laying out adjacent rectangles edge-to-edge makes the
# maps touch the dividing lines with zero gap, by construction.
FIG_W, FIG_H = 16, 11
LEFT, RIGHT  = 0.02, 0.87   # RIGHT leaves room for the colorbar
TOP, BOTTOM  = 0.98, 0.02   # year labels now sit inside the maps, not above

col_w = (RIGHT - LEFT) / 2
row_h = (TOP - BOTTOM) / 2

POS = {
    2022: [LEFT,         TOP - row_h, col_w, row_h],
    2023: [LEFT + col_w, TOP - row_h, col_w, row_h],
    2024: [LEFT,         BOTTOM,      col_w, row_h],
    2025: [LEFT + col_w, BOTTOM,      col_w, row_h],
}

fig = plt.figure(figsize=(FIG_W, FIG_H))
axes_by_year = {y: fig.add_axes(POS[y]) for y in YEARS_CHORO}
axes = [axes_by_year[y] for y in YEARS_CHORO]

x_mid, y_mid = LEFT + col_w, BOTTOM + row_h
x_left, x_right = LEFT, RIGHT
y_top, y_bottom = TOP, BOTTOM

for ax, year in zip(axes, YEARS_CHORO):
    hours_total = HOURS_IN_YEAR[year]

    ne_name_to_val = {NAME_MAP[c]: neg_table[c].get(year, 0) for c in countries_list}

    world_copy = world.copy()
    world_copy["neg_hours"] = world_copy["NAME"].map(ne_name_to_val)

    europe = world_copy.cx[EUROPE_XLIM[0]:EUROPE_XLIM[1],
                            EUROPE_YLIM[0]:EUROPE_YLIM[1]].copy()

    no_data  = europe[europe["neg_hours"].isna()]
    has_data = europe[~europe["neg_hours"].isna()].copy()

    no_data.plot( ax=ax, color="#d4d4d4", edgecolor="white", linewidth=0.5)
    has_data.plot(ax=ax, column="neg_hours", cmap=cmap, norm=norm,
                  edgecolor="white", linewidth=0.5)

    for country, (lon, lat) in LABEL_POS.items():
        count = neg_table[country].get(year, 0)
        label = make_label(country, count, hours_total)
        txt_color = "white" if country in WHITE_LABELS.get(year, set()) else "black"
        ax.text(lon, lat, label, ha="center", va="center",
                fontsize=label_fs(country), color=txt_color, fontweight="bold",
                transform=ax.transData)

    ax.set_xlim(EUROPE_XLIM)
    ax.set_ylim(EUROPE_YLIM)
    ax.axis("off")

    # Year label in a small boxed tag, top-left corner of each map.
    ax.text(0.03, 0.97, str(year), transform=ax.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="#555555", linewidth=0.9,
                      alpha=0.92, boxstyle="round,pad=0.3"))

divider_kwargs = dict(color="#555555", linewidth=1.3, transform=fig.transFigure,
                       zorder=10, clip_on=False, solid_capstyle="butt")
fig.add_artist(Line2D([x_mid, x_mid], [y_bottom, y_top], **divider_kwargs))
fig.add_artist(Line2D([x_left, x_right], [y_mid, y_mid], **divider_kwargs))

cax = fig.add_axes([RIGHT + 0.02, BOTTOM, 0.022, TOP - BOTTOM])
sm  = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
cbar.set_label("Annual hours with price $\\leq$ 0", fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)

out_a = os.path.join(FIGURES_DIR, "negative_hours_europe.png")
fig.savefig(out_a, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_a}")


# ---------------------------------------------------------------------------
# FIGURE B  –  Spring intraday profile comparison, 2025
# ---------------------------------------------------------------------------
SPRING_MONTHS = [3, 4, 5, 6]
YEAR_PROFILE  = 2025

PROFILE_HIGHLIGHTS = {
    "Spain":   ("#c0392b", "-",              2.8, 10),
    "Italy":   ("#e67e22", "--",             1.8,  8),
    "Germany": ("#1a72b8", "-.",             1.8,  6),
    "France":  ("#8e44ad", (0, (3, 1, 1, 1)), 1.5,  5),
    "Greece":  ("#27ae60", ":",              1.8,  7),
}
GREY_LINE = dict(color="#cccccc", lw=0.9, alpha=0.65, zorder=1)

spring_data = all_data[
    (all_data["Year"] == YEAR_PROFILE) &
    (all_data["Month"].isin(SPRING_MONTHS))
].copy()

fig, ax = plt.subplots(figsize=(10, 6))

for name in countries_list:
    if name in PROFILE_HIGHLIGHTS:
        continue
    sub = spring_data[spring_data["Country"] == name]
    if sub.empty:
        continue
    profile = sub.groupby("Hour")["Price_pct_of_monthly_mean"].mean()
    ax.plot(profile.index, profile.values, **GREY_LINE)

for name, (color, ls, lw, zo) in PROFILE_HIGHLIGHTS.items():
    sub = spring_data[spring_data["Country"] == name]
    if sub.empty:
        continue
    profile = sub.groupby("Hour")["Price_pct_of_monthly_mean"].mean()
    ax.plot(profile.index, profile.values,
            color=color, linestyle=ls, linewidth=lw, zorder=zo, label=name)

ax.axhline(100, color="black", lw=0.8, ls=":", alpha=0.5)
ax.axhline(0,   color="black", lw=0.8, ls="-", alpha=0.3)

grey_patch = plt.Line2D([0], [0], color="#cccccc", lw=1.0, label="Other countries")
handles, labels_leg = ax.get_legend_handles_labels()
handles.append(grey_patch); labels_leg.append("Other countries")
ax.legend(handles, labels_leg, fontsize=10, loc="upper left", framealpha=0.9)

ax.set_xlabel("Hour of day (UTC)", fontsize=11)
ax.set_ylabel("Price (\\% of monthly mean)", fontsize=11)
ax.set_xticks(range(1, 25))
ax.set_xticklabels([str(h) for h in range(1, 25)], fontsize=8)
ax.grid(linestyle="--", alpha=0.3, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
out_b = os.path.join(FIGURES_DIR, "spring_profile_europe.png")
fig.savefig(out_b, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_b}")

print("\nAll done.")
