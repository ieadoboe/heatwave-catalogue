"""Every figure in the catalogue paper, from the results tables and ERA5.

    python paper/figures.py [workdir]        (default workdir: hwcat-data)

Writes paper/figures/*.pdf, then the flat copies the manuscript includes,
paper/fig1.pdf to fig4.pdf (AMS wants flat names with no underscores). The
sections below are the cells of the former figures notebook, in order; they
share variables, so they run top to bottom.
"""
import sys
from pathlib import Path

WORKDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("hwcat-data")

# ----------------------------------------------------------------------
# setup: events table
# ----------------------------------------------------------------------
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, Normalize

YEAR = 2026
PAPER = Path(__file__).resolve().parent
RESULTS, PLOTS = PAPER / "results", PAPER / "figures"
PLOTS.mkdir(parents=True, exist_ok=True)

# hwcat.plots calls mpl_apply() on import: the project rcParams come from
# there. Do not set rcParams here.
from hwcat import detect as D, plots as P  # noqa: E402

events = pd.read_csv(RESULTS / f"heatwave_events_europe_{YEAR}.csv",
                     parse_dates=["start", "peak", "end"])
sev = events["severity_C_km2_day"]
norm = LogNorm(vmin=sev.min(), vmax=sev.max())
cmap = plt.get_cmap("inferno_r")

print(f"{len(events)} events, {events.start.min():%d %b} to {events.end.max():%d %b}")

# ----------------------------------------------------------------------
# centroid countries
# ----------------------------------------------------------------------
# Country of the CENTROID, read off the coordinates by hand: there is no
# country dataset in the repo and no offline geocoder in the .venv, and a
# lookup this size beats adding a dependency for one column. Grouped by
# country so the per-country counts are readable.
#
# It labels the balance point, not the extent. Events 1 to 4 each stretch more
# than 1,000 km from centroid to hottest cell and span several countries, so
# read this column as "where the event sat", never as "the country's heat wave".
CENTROID_COUNTRY = {rank: country for country, ranks in {
    "Russia":   (7, 8, 11, 18, 19, 21, 24, 28, 30, 38),
    "France":   (2, 3, 4, 6, 34),
    "UK":       (10, 12, 14, 17, 35),
    "Algeria":  (9, 15, 32),
    "Spain":    (13, 33, 37),
    "Turkey":   (25, 27, 31),
    "Ireland":  (20, 22),
    "Sweden":   (16, 36),
    "Bulgaria": (39,),
    "Germany":  (1,),
    "Greece":   (23,),
    "Italy":    (5,),
    "Norway":   (29,),
    "Portugal": (26,),
}.items() for rank in ranks}

glance = events.assign(country=events["rank"].map(CENTROID_COUNTRY))[
    ["rank", "country", "zone", "koppen", "duration_days", "severity_C_km2_day", "mean_anom_C", "peak_tmax_C",
     "peak_tmax_lat", "peak_tmax_lon", "max_anom_C", "mean_anom_C"]]

assert glance.country.notna().all(), "a rank is missing from CENTROID_COUNTRY"
# print(glance.to_string(index=False))

# ----------------------------------------------------------------------
# detection on the 2026 grid
# ----------------------------------------------------------------------
# The threshold the 2026 catalogue was made with (April-October, 15 days,
# full 1991-2020 baseline). Built once into the working folder if missing.
target = D.open_daily_tmax([WORKDIR / "era5" / f"europe_tmax_{YEAR}.nc"])
THRESH = WORKDIR / "thresholds" / "heatwave_threshold_europe_1991_2020_p90_m04-10.nc"
if not THRESH.exists():
    from hwcat.catalogue import run
    from hwcat.workdir import Workdir
    run(YEAR, Workdir(WORKDIR), plot=False)
thresh = xr.open_dataarray(THRESH).load()
land = D.land_mask_for(target["latitude"].values, target["longitude"].values)
det = D.detect(target, thresh, land_mask=land,
               min_days=3, min_area_km2=50_000.0)
hw_days = (det.labels > 0).sum(axis=0)

print(f"{det.n_events} blobs before the area filter; "
      f"busiest cell spent {hw_days.max()} days in an event")

# ----------------------------------------------------------------------
# fig1: heat-wave days
# ----------------------------------------------------------------------
# top_n=0 draws the shaded map with no annotations. The paper's numbering
# and key are added below so all 32 events appear, not the default 8.
fig, ax = P.plot_europe_heatwaves(hw_days, det.lat, det.lon, events=events,
                                  top_n=0, year=YEAR)

# ponytail: greedy nudge, O(n^2) over 32 points. Swap for a real label solver
# only if the catalogue grows past a few hundred events.
placed = []
for _, e in events.iterrows():
    x, y = e.centroid_lon, e.centroid_lat
    while any(abs(x - px) < 1.7 and abs(y - py) < 1.3 for px, py in placed):
        y += 1.3
    placed.append((x, y))
    if y != e.centroid_lat:          # leader line back to the true centroid
        ax.plot([e.centroid_lon, x], [e.centroid_lat, y], color="black",
                linewidth=0.4, zorder=4)
    ax.scatter([e.centroid_lon], [e.centroid_lat], s=4, c="black", zorder=5)
    ax.annotate(str(int(e["rank"])), (x, y), fontsize=5.5, zorder=6,
                ha="center", va="center", color="black",
                bbox=dict(boxstyle="circle,pad=0.13", facecolor="white",
                          edgecolor="black", linewidth=0.5))

ax.set_title(f"European heat waves {YEAR}: days above the calendar-day "
             f"90th percentile\n(CTX90pct, 1991--2020 baseline; all "
             f"{len(events)} events numbered by severity rank)")

# Event key, parked in the empty Atlantic rather than outside the axes, which
# would otherwise leave a third of the canvas white.
key = [f"{int(e['rank']):2d}. {e.start:%d %b}--{e.end:%d %b} {e.koppen}"
       for _, e in events.iterrows()]
ax.text(-24.3, 50.2, "\n".join(key), fontsize=5.0, va="top", ha="left",
        linespacing=1.5, zorder=7)

fig.savefig(PLOTS / f"europe_{YEAR}_fig1_map.pdf", bbox_inches="tight")

# ----------------------------------------------------------------------
# fig1b: with borders (paper Fig. 1)
# ----------------------------------------------------------------------
import json
import urllib.request

BORDERS = Path(P.__file__).parent / "data" / "borders_europe_ne50m.json"
if not BORDERS.exists():          # one-off fetch, ~3 MB down, 318 KB kept
    SRC = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_50m_admin_0_countries.geojson")
    W, S, E, N = -25.0, 35.0, 45.0, 72.0
    with urllib.request.urlopen(SRC, timeout=60) as r:
        world = json.load(r)
    rings = []
    for feat in world["features"]:
        geom = feat["geometry"]
        raw = (geom["coordinates"] if geom["type"] == "Polygon"
               else [r for poly in geom["coordinates"] for r in poly])
        for ring in raw:
            xs, ys = [p[0] for p in ring], [p[1] for p in ring]
            if max(xs) < W or min(xs) > E or max(ys) < S or min(ys) > N:
                continue          # ring lies wholly outside the domain
            rings.append([[round(x, 2), round(y, 2)] for x, y in ring])
    BORDERS.write_text(json.dumps(rings, separators=(",", ":")))
    print(f"fetched {len(rings)} rings -> {BORDERS}")

rings = json.loads(BORDERS.read_text())

fig, ax = P.plot_europe_heatwaves(hw_days, det.lat, det.lon, events=events,
                                  top_n=0, year=YEAR)
for ring in rings:
    ax.plot([p[0] for p in ring], [p[1] for p in ring],
            color="#333333", linewidth=0.35, zorder=3)

# Same numbering as Figure 1, so the two read together.
placed = []
for _, e in events.iterrows():
    x, y = e.centroid_lon, e.centroid_lat
    while any(abs(x - px) < 1.7 and abs(y - py) < 1.3 for px, py in placed):
        y += 1.3
    placed.append((x, y))
    if y != e.centroid_lat:
        ax.plot([e.centroid_lon, x], [e.centroid_lat, y], color="black",
                linewidth=0.4, zorder=4)
    ax.scatter([e.centroid_lon], [e.centroid_lat], s=4, c="black", zorder=5)
    ax.annotate(str(int(e["rank"])), (x, y), fontsize=5.5, zorder=6,
                ha="center", va="center", color="black",
                bbox=dict(boxstyle="circle,pad=0.13", facecolor="white",
                          edgecolor="black", linewidth=0.5))

ax.set_xlim(det.lon.min(), det.lon.max())
ax.set_ylim(det.lat.min(), det.lat.max())
ax.set_title(f"European heat waves {YEAR} over national borders\n"
             f"(days above the calendar-day 90th percentile; all "
             f"{len(events)} events numbered by severity rank)")

fig.savefig(
    PLOTS / f"europe_{YEAR}_fig1b_map_borders.pdf", bbox_inches="tight")

# ----------------------------------------------------------------------
# fig2: timeline (paper Fig. 2)
# ----------------------------------------------------------------------
season = events.sort_values("start").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7.2, 6.4))
for y, (_, e) in enumerate(season.iterrows()):
    ax.barh(y, e.end - e.start + pd.Timedelta(days=1), left=e.start,
            height=0.62, color=cmap(norm(e.severity_C_km2_day)),
            edgecolor="black", linewidth=0.4)
    ax.plot(e.peak, y, marker="|", color="white", ms=5, mew=0.9, zorder=4)

ax.set_yticks(range(len(season)))
ax.set_yticklabels([f"{int(e['rank'])}. {e.koppen}"
                    for _, e in season.iterrows()], fontsize=6.5)
ax.invert_yaxis()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.grid(axis="x", alpha=0.3, linewidth=0.6)
ax.set(xlabel=f"{YEAR}", ylabel="severity rank and Koppen class",
       title=f"When the {len(events)} events ran\n"
       f"(bar spans start to end, tick marks the peak day)")
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
             label=r"severity ($^\circ$C km$^2$ day)", pad=0.02, shrink=0.85)

fig.savefig(PLOTS / f"europe_{YEAR}_fig2_timeline.pdf", bbox_inches="tight")

# ----------------------------------------------------------------------
# fig3: severity
# ----------------------------------------------------------------------
cum = 100 * sev.cumsum() / sev.sum()

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.bar(events["rank"], sev, color=cmap(norm(sev)), edgecolor="black",
       linewidth=0.4, width=0.75)
ax.set_yscale("log")
ax.set(xlabel="severity rank", ylabel=r"severity ($^\circ$C km$^2$ day)",
       title=f"Severity is concentrated: {len(events)} events, "
       f"three carry {cum.iloc[2]:.0f}\\% of the year")

ax2 = ax.twinx()
ax2.plot(events["rank"], cum, color="#1f6feb", marker="o", ms=2.6, lw=1.3)
ax2.set_ylabel(r"cumulative share of total severity (\%)", color="#1f6feb")
ax2.tick_params(axis="y", colors="#1f6feb")
ax2.set_ylim(0, 105)
ax2.grid(False)
for k in (3, 8):
    ax2.annotate(f"top {k}: {cum.iloc[k - 1]:.0f}\\%", (k, cum.iloc[k - 1]),
                 xytext=(10, -4), textcoords="offset points", fontsize=7.5,
                 color="#1f6feb", bbox=dict(facecolor="white", alpha=0.85,
                                            edgecolor="none", pad=1.2))

fig.savefig(PLOTS / f"europe_{YEAR}_fig3_severity.pdf", bbox_inches="tight")

# ----------------------------------------------------------------------
# fig4: anomaly vs Tmax
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 5.2))
sc = ax.scatter(events.peak_tmax_C, events.max_anom_C,
                s=18 + 170 * events.peak_area_km2 / events.peak_area_km2.max(),
                c=events.centroid_lat, cmap="viridis", norm=Normalize(35, 70),
                edgecolor="black", linewidth=0.5, alpha=0.9)
fig.colorbar(sc, ax=ax, label=r"centroid latitude ($^\circ$N)", pad=0.02)

# Every point carries its rank. ponytail: greedy nudge, O(n^2) over 32
# points, same as Figure 1; a real label solver is not worth it at this size.
sizes = 18 + 170 * events.peak_area_km2 / events.peak_area_km2.max()
placed = []
for (_, e), s in zip(events.iterrows(), sizes):
    dx = 0.22 + 0.05 * np.sqrt(s)           # clear the marker, which varies
    x, y = e.peak_tmax_C + dx, e.max_anom_C
    while any(abs(x - px) < 1.6 and abs(y - py) < 0.40 for px, py in placed):
        y += 0.40
    placed.append((x, y))
    if abs(y - e.max_anom_C) > 1e-9:
        ax.plot([e.peak_tmax_C, x], [e.max_anom_C, y], color="0.45",
                linewidth=0.4, zorder=2)
    ax.annotate(int(e["rank"]), (x, y), fontsize=6.5, va="center",
                ha="left", zorder=6)

ax.grid(alpha=0.3, linewidth=0.6)
ax.set(xlabel=r"peak daily maximum temperature ($^\circ$C)",
       ylabel=r"largest exceedance above the local threshold ($^\circ$C)",
       title="Hot in absolute terms is not the same as extreme for the place\n"
             "(marker area scales with peak extent; numbers are severity ranks)")

fig.savefig(PLOTS / f"europe_{YEAR}_fig4_anomaly_vs_tmax.pdf",
            bbox_inches="tight")

# ----------------------------------------------------------------------
# fig5: Koppen classes
# ----------------------------------------------------------------------
SHORT = {"Cfb": "oceanic", "Dfb": "humid continental",
         "Csa": "Mediterranean, hot", "Dfc": "subarctic",
         "Csb": "Mediterranean, mild", "Cfa": "humid subtropical",
         "Dsb": "continental, dry summer", "BSk": "cold semi-arid",
         "Dfa": "humid continental, hot summer"}
ZONE_COLOR = {"cold (continental)": "#4C72B0", "temperate": "#55A868",
              "arid": "#DD8452"}

by_class = (events.groupby(["koppen", "zone"])
            .agg(n=("rank", "size"), severity=("severity_C_km2_day", "sum"))
            .reset_index().sort_values("n"))
by_class["share"] = 100 * by_class.severity / sev.sum()
colors = [ZONE_COLOR[z] for z in by_class.zone]
y = np.arange(len(by_class))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=True)
a1.barh(y, by_class.n, color=colors, edgecolor="black", linewidth=0.4)
a1.set(xlabel="events", xticks=range(0, 11, 2))
for yi, n in zip(y, by_class.n):
    a1.text(n + 0.15, yi, str(n), va="center", fontsize=7.5)

a2.barh(y, by_class.share, color=colors, edgecolor="black", linewidth=0.4)
a2.set(xlabel=r"share of total severity (\%)")
for yi, s in zip(y, by_class.share):
    a2.text(s + 1.0, yi, f"{s:.1f}", va="center", fontsize=7.5)

a1.set_yticks(y)
a1.set_yticklabels([f"{k}  {SHORT[k]}" for k in by_class.koppen], fontsize=8)
for ax_ in (a1, a2):
    ax_.grid(axis="x", alpha=0.3, linewidth=0.6)
    ax_.set_axisbelow(True)
    ax_.margins(x=0.12)

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black",
                         linewidth=0.4) for c in ZONE_COLOR.values()]
fig.legend(handles, ZONE_COLOR.keys(), loc="lower center", ncol=3,
           frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Koppen classes in the 2026 catalogue, counted two ways", y=1.0)

fig.savefig(PLOTS / f"europe_{YEAR}_fig5_koppen.pdf", bbox_inches="tight")

# ----------------------------------------------------------------------
# fig6: footprints (paper Fig. 3)
# ----------------------------------------------------------------------
TOP = 8
top = events.head(TOP)
lat, lon = det.lat, det.lon
extent = [lon.min(), lon.max(), lat.min(), lat.max()]
vmax = max(int((det.labels == eid).sum(axis=0).max()) for eid in top.event_id)

fig, axes = plt.subplots(4, 2, figsize=(7.4, 11.0), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.3, wspace=0.08)
for ax, (_, e) in zip(axes.flat, top.iterrows()):
    days = (det.labels == int(e.event_id)).sum(axis=0)
    im = ax.imshow(np.where(days > 0, days, np.nan), origin="upper",
                   cmap="inferno_r", vmin=1, vmax=vmax, extent=extent,
                   interpolation="nearest", aspect="auto", zorder=2)
    for ring in rings:
        ax.plot([p[0] for p in ring], [p[1] for p in ring],
                color="#333333", linewidth=0.3, zorder=3)
    ax.plot(e.peak_tmax_lon, e.peak_tmax_lat, marker="*", ms=8,
            mfc="#1f6feb", mec="white", mew=0.6, ls="none", zorder=5)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.tick_params(labelsize=7)
    ax.set_title(f"{int(e['rank'])}. {e.start:%d %b} to {e.end:%d %b}, "
                 f"{e.koppen}\n{e.duration_days} d, "
                 f"{e.peak_area_km2 / 1e6:.2f} Mkm$^2$, "
                 f"{e.peak_tmax_C:.1f}$^\\circ$C", fontsize=7.5)

fig.colorbar(im, ax=axes, label="days inside the event", shrink=0.5,
             pad=0.02)
fig.suptitle(f"Footprints of the {TOP} most severe events of {YEAR}\n"
             "(star marks the hottest cell; all panels share the Figure 1 "
             "domain)", y=0.95)

fig.savefig(PLOTS / f"europe_{YEAR}_fig6_footprints.pdf", bbox_inches="tight")

# ----------------------------------------------------------------------
# sensitivity: ranks
# ----------------------------------------------------------------------
SENS_YEARS = (2003, 2022, 2026)
SETTINGS = [(25_000, 0.90), (50_000, 0.90), (100_000, 0.90),
            (25_000, 0.95), (50_000, 0.95), (100_000, 0.95)]
XPOS = np.array([0, 1, 2, 3.5, 4.5, 5.5])
N_REF = 6
SHAPE = {2003: "one August wave", 2022: "two separated centres",
         2026: "three waves over the same ground"}

sens = {y: pd.read_csv(RESULTS / f"catalogue_sensitivity_europe_{y}.csv")
        for y in SENS_YEARS}
ranks_cmap = plt.cm.viridis(np.linspace(0.0, 0.82, N_REF))

fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.0), sharex=True)
fig.subplots_adjust(hspace=0.28)

for ax, year in zip(axes, SENS_YEARS):
    df = sens[year]
    ax.axvspan(XPOS[1] - 0.42, XPOS[1] + 0.42, color="0.88", zorder=0)

    # two reference events landing on one detected event = a merge
    merged = set()
    for (a, p), g in df.groupby(["min_area_km2", "percentile"]):
        for r in g.loc[g.duplicated("rank", keep=False), "rank"]:
            merged.add((a, p, int(r)))

    starts = {}
    for k in range(1, N_REF + 1):
        row = df[df.ref_rank == k].set_index(["min_area_km2", "percentile"])
        ys = np.array([float(row.loc[s, "rank"]) for s in SETTINGS])
        # The two percentile groups are separate experiments, not a sequence,
        # so the trace breaks between them rather than implying a transition.
        for sl in (slice(0, 3), slice(3, 6)):
            ax.plot(XPOS[sl], ys[sl], color=ranks_cmap[k - 1], lw=1.3,
                    marker="o", ms=4.2, mec="white", mew=0.6, zorder=3)
        starts.setdefault(ys[0], []).append(k)
        for s, x, y in zip(SETTINGS, XPOS, ys):
            if (s[0], s[1], int(y)) in merged:
                ax.plot(x, y, marker="o", ms=9, mfc="none", mec="#c1121f",
                        mew=1.1, zorder=5)

    # One label per starting rank: merged events share a point and would
    # otherwise print on top of each other.
    for y0, ks in starts.items():
        ax.annotate(",".join(str(k) for k in sorted(ks)), (XPOS[0], y0),
                    xytext=(-8, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=8, color=ranks_cmap[min(ks) - 1],
                    zorder=4)

    trans = ax.get_xaxis_transform()
    for s, x in zip(SETTINGS, XPOS):
        n = int(df[(df.min_area_km2 == s[0])
                   & (df.percentile == s[1])].n_events.iloc[0])
        ax.text(x, 1.03, str(n), transform=trans, ha="center", va="bottom",
                fontsize=8, color="0.25")
    ax.text(-0.075, 1.03, "events:", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8, color="0.25")

    ax.set_ylim(13.2, 0.1)
    ax.set_yticks([1, 3, 6, 9, 12])
    ax.set_ylabel("rank")
    ax.grid(axis="y", alpha=0.3, lw=0.6)
    ax.set_title(f"{year}: {SHAPE[year]}", fontsize=9.5, loc="left", pad=16)

axes[-1].set_xticks(XPOS)
axes[-1].set_xticklabels(["25k", "50k", "100k"] * 2, fontsize=9)
axes[-1].set_xlim(-0.75, 6.1)
for x, lab in ((1.0, "90th percentile"), (4.5, "95th percentile")):
    axes[-1].text(x, -0.20, lab, transform=axes[-1].get_xaxis_transform(),
                  ha="center", va="top", fontsize=9)
axes[-1].text(2.75, -0.34, r"minimum daily patch area (km$^2$)",
              transform=axes[-1].get_xaxis_transform(), ha="center", va="top",
              fontsize=10)

fig.suptitle("Where the six most severe events of each season land under other "
             "detector settings\n"
             "(shaded column is the reference catalogue; red ring marks two "
             "reference events detected as one)", fontsize=10, y=0.975)

fig.savefig(PLOTS / "europe_sensitivity_ranks.pdf")
print("wrote", PLOTS / "europe_sensitivity_ranks.pdf")

# ----------------------------------------------------------------------
# sensitivity: rank changes (paper Fig. 4)
# ----------------------------------------------------------------------
from matplotlib.patches import Rectangle

AREAS = (25_000, 50_000, 100_000)
PCTS = (0.90, 0.95)
VMAX = 6

fig, axes = plt.subplots(3, 2, figsize=(6.6, 6.2))
fig.subplots_adjust(hspace=0.75, wspace=0.13)
cmap = plt.get_cmap("YlOrRd").copy()
cmap.set_bad("white")              # unchanged really is blank

for row, year in enumerate(SENS_YEARS):
    df = sens[year]
    for col, pct in enumerate(PCTS):
        ax = axes[row, col]
        delta = np.zeros((N_REF, len(AREAS)))
        # Reference events that this setting detects as ONE event, grouped so
        # they can be tied together rather than outlined as unrelated cells.
        groups = []
        for j, area in enumerate(AREAS):
            g = df[(df.min_area_km2 == area) & (df.percentile == pct)]
            for _, r in g.iterrows():
                # Signed so it agrees with the page: rank 1 is at the
                # top, so positive means the event moved UP the
                # ranking and negative means it fell.
                delta[int(r.ref_rank) - 1, j] = int(r.ref_rank) - int(r["rank"])
            for _, tied in g[g.duplicated("rank", keep=False)].groupby("rank"):
                groups.append((j, sorted(int(k) - 1 for k in tied.ref_rank)))

        im = ax.imshow(np.ma.masked_where(delta == 0, np.abs(delta)),
                       cmap=cmap, vmin=0, vmax=VMAX, aspect="auto")
        for i in range(N_REF):
            for j in range(len(AREAS)):
                d = int(delta[i, j])
                if d:
                    ax.text(j, i, f"{d:+d}", ha="center", va="center",
                            fontsize=8, color="black" if abs(d) < 4 else "white")
        for j, rows in groups:
            for i in rows:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor="#1f6feb", lw=1.6, zorder=4))
            # The tie bar is what says "one event", which two loose outlines
            # do not: a reader asked what the blue meant.
            x = j - 0.42
            ax.plot([x, x], [min(rows) - .5, max(rows) + .5], color="#1f6feb",
                    lw=1.6, zorder=5, solid_capstyle="butt")
            ax.text(j - 0.34, np.mean(rows), "one\nevent", fontsize=6.5,
                    color="#1f6feb", ha="left", va="center", zorder=5)

        for j in range(1, len(AREAS)):
            ax.axvline(j - 0.5, color="white", lw=1.4)
        for i in range(1, N_REF):
            ax.axhline(i - 0.5, color="white", lw=1.4)

        counts = [int(df[(df.min_area_km2 == a)
                         & (df.percentile == pct)].n_events.iloc[0])
                  for a in AREAS]
        ax.set_xticks(range(len(AREAS)))
        ax.set_xticklabels(counts, fontsize=8.5)
        ax.tick_params(axis="both", length=0, pad=2)
        ax.set_yticks(range(N_REF))
        ax.set_yticklabels(range(1, N_REF + 1) if col == 0 else [],
                           fontsize=8.5)
        if col == 0:
            ax.set_ylabel("severity rank", fontsize=9)
        if row == len(SENS_YEARS) - 1:
            ax.set_xlabel("events in the whole season", fontsize=8.5,
                          labelpad=2)

        top = ax.secondary_xaxis("top")
        top.set_xticks(range(len(AREAS)))
        top.set_xticklabels([f"{a // 1000}k" for a in AREAS], fontsize=8.5)
        top.tick_params(length=0, pad=2)
        # Column heading on the top row only: the percentile is now a
        # structural axis of the grid, not something the caption has to say.
        # Placed in axes fractions above the year label, which sits at ~1.26.
        if row == 0:
            ax.text(0.5, 1.52, f"{int(pct * 100)}th percentile", fontsize=10,
                    transform=ax.transAxes, ha="center", va="bottom")

    # Year heading, left-aligned over the left panel so it reads as a row label
    axes[row, 0].text(-0.5, -1.55, f"{year}: {SHAPE[year]}", fontsize=9.5,
                      ha="left", va="bottom", fontweight="normal")

cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
cb.set_label("places moved", fontsize=9)   # magnitude; the
# printed sign carries direction, because the claim is that an
# event moved at all, not which way.
fig.suptitle("2003 and 2022 reorder only their 4th to 6th events, and only at "
             "the 95th percentile.\nIn 2026 the reordering reaches rank 2 and "
             "happens at both.\n"
             "(blank = unchanged; a positive number means the event moved UP "
             "the ranking;\nblue ties reference events that this setting "
             "detects as a single event;\ncolumn heads are the minimum patch "
             "area)",
             fontsize=10, y=1.13)

fig.savefig(PLOTS / "europe_sensitivity_rank_changes.pdf",
            bbox_inches="tight")
print("wrote", PLOTS / "europe_sensitivity_rank_changes.pdf")


# ----------------------------------------------------------------------
# flat copies for the manuscript
# ----------------------------------------------------------------------
import shutil  # noqa: E402

for flat, src in {"fig1": f"europe_{YEAR}_fig1b_map_borders.pdf",
                  "fig2": f"europe_{YEAR}_fig2_timeline.pdf",
                  "fig3": f"europe_{YEAR}_fig6_footprints.pdf",
                  "fig4": "europe_sensitivity_rank_changes.pdf"}.items():
    shutil.copyfile(PLOTS / src, PAPER / f"{flat}.pdf")
print("wrote paper/fig1.pdf to fig4.pdf")
