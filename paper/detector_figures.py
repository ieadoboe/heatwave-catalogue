"""Figures that explain the detector, one step at a time, on real 2026 data.

    python paper/detector_figures.py [workdir]     (default workdir: hwcat-data)

Writes to paper/figures/:

    detector_pipeline         all six steps, equations above real panels
    detector_threshold_berlin step 1: how the bar for 27 June is set in Berlin
    detector_step_area        step 3: 20 June, small hot patches deleted
    detector_step_persist     step 4: Warsaw, short spells dropped
    detector_step_label       step 5: event 1 every four days

The panels run the hwcat.detect functions in the order detect() runs them,
and two asserts tie the figures to the code: the step-by-step masks must equal
detect()'s, and the Berlin bar recomputed by hand must equal the cached
threshold. Needs the 1991-2026 ERA5 files and the default threshold in the
working folder (hwcat download, then hwcat detect --year 2026).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Patch,
                                Rectangle)
from matplotlib.transforms import Bbox
from scipy import ndimage

from hwcat import detect as D
from hwcat.catalogue import threshold_name
from hwcat.plots import MPL_CONFIG, mpl_apply
from hwcat.workdir import Workdir

WORK = Workdir(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("hwcat-data"))
PLOTS = Path(__file__).resolve().parent / "figures"
YEAR = 2026


def threshold_path() -> Path:
    """The default threshold (90th, 15-day, 1991-2020) for April-October."""
    return WORK.thresholds / threshold_name(
        WORK.region, 1991, 2020, 0.90, "m04-10", 15, list(range(1991, 2021)))


# ----------------------------------------------------------------------
# shared: settings, colours and the detector run
# ----------------------------------------------------------------------
# The day panels 2-4 share. Chosen because every filter visibly removes
# something on it: 7243 hot land cells, 492 dropped as small patches, then
# 2638 dropped as spells shorter than three days.
DAY = "2026-06-20"
CITY = ("Berlin", 52.5, 13.4)     # inside the event, recognisable to anyone
EXTENT = (-11.0, 41.0, 35.0, 63.0)

HOT = "#C2511A"        # kept
DROP = "#A9A39A"       # removed at this step
LAND = "#EEEAE3"       # land, not hot
SEA = "#FFFFFF"
TUNE = "#1D4E89"       # tunable setting
FIXED = "#6B6B6B"      # fixed choice
INK = "#1B1F2A"
MIN_AREA = 50_000.0
MIN_DAYS = 3


MUTED = "#6B6B6B"
BG = "#F7F4EE"         # the deck's slide background

# step 1 figure
DATE = "06-27"          # peak day of 2026's top-ranked event
TARGET_YEAR = 2026
BASELINE = range(1991, 2021)
HALF = 7                # 15-day window
SHOW = 10               # days shown either side, so the cut-off is visible
Q = 0.90




# steps 3-5 figures
STEP4_CITY = ("Warsaw", 52.2, 21.0)
SPAN = ("2026-06-05", "2026-07-05")
LABEL_DAYS = ["2026-06-12", "2026-06-16", "2026-06-20",
              "2026-06-24", "2026-06-28", "2026-07-02"]
BG = "#F7F4EE"         # the deck's slide background



def run_detector():
    """Each intermediate mask of `detect`, in its order, plus the ranking."""
    target = D.open_daily_tmax([WORK.era5_file(YEAR)])
    threshold = xr.open_dataarray(threshold_path()).load()
    lat = target["latitude"].values
    lon = target["longitude"].values
    land = D.land_mask_for(lat, lon)
    area = D.cell_area_km2(lat, lon)

    exc = D.exceedance(target, threshold).values & land[None]
    coherent = D.spatially_coherent(exc, area, MIN_AREA)
    persistent = D.sustained(coherent, min_days=MIN_DAYS)

    det = D.detect(target, threshold, land_mask=land, min_days=MIN_DAYS,
                   min_area_km2=MIN_AREA)
    assert (det.mask == persistent).all(), "steps no longer match detect()"
    events = D.in_season(D.event_table(det, target, MIN_AREA))
    thresh = threshold.sel(dayofyear=target["time"].dt.dayofyear).values
    return dict(target=target, thresh=thresh, lat=lat, lon=lon, land=land,
                exc=exc, coherent=coherent, persistent=persistent,
                labels=det.labels, time=det.time, events=events)


def show_map(ax, state, colours, r, title):
    lon, lat = r["lon"], r["lat"]
    extent = [lon[0] - 0.125, lon[-1] + 0.125, lat[-1] - 0.125,
              lat[0] + 0.125]
    ax.imshow(state, cmap=ListedColormap(colours), extent=extent,
              interpolation="nearest", vmin=0, vmax=len(colours) - 1)
    ax.contour(lon, lat, r["land"].astype(float), levels=[0.5], colors=INK,
               linewidths=0.25)
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])
    ax.set_aspect(1 / np.cos(np.deg2rad(50)), adjustable="box", anchor="N")
    ax.set_xticks([0, 20, 40], [r"$0^\circ$", r"$20^\circ$E", r"$40^\circ$E"])
    ax.set_yticks([40, 50, 60], [])
    ax.plot(CITY[2], CITY[1], "o", ms=2.6, mfc="white", mec=INK, mew=0.6)
    ax.set_title(title, fontsize=6.6, color=FIXED, pad=3)



# ----------------------------------------------------------------------
# all six steps
# ----------------------------------------------------------------------
def pipeline_figure(r):
    """All six steps: equations on top, real 2026 panels below."""
    mpl_apply({**MPL_CONFIG, "font.size": 7.5, "axes.linewidth": 0.6})
    day = int(np.flatnonzero(r["time"] == pd.Timestamp(DAY))[0])
    top = r["events"].iloc[0]
    eid = int(top["event_id"])

    fig = plt.figure(figsize=(7.2, 3.9))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)

    n = 6
    x0, gap = 0.012, 0.014
    w = (1 - 2 * x0 - (n - 1) * gap) / n
    xs = [x0 + i * (w + gap) for i in range(n)]

    steps = [
        ("Threshold",
         r"$T_q$: $q$-quantile of"
         "\n"
         r"$T_{\max}$, $\pm\lfloor w/2\rfloor$ days"),
        ("Exceedance",
         r"$T_{\max} > T_q$"
         "\n"
         r"on land cells"),
        ("Area filter",
         r"each day, drop"
         "\n"
         r"patches $< A_{\min}$"),
        ("Persistence",
         r"keep runs of"
         "\n"
         r"$\ge D_{\min}$ days"),
        ("Labelling",
         r"connect cells in"
         "\n"
         r"space and time"),
        ("Ranking",
         r"$S=\sum\,(T_{\max}-T_q)\,a_i$"
         "\n"
         r"over each event"),
    ]
    settings = [
        [(r"$q = 0.90$", TUNE), (r"$w = 15$ d", TUNE),
         (r"1991--2020", FIXED)],
        [(r"K\"oppen land mask", FIXED), (r"ERA5, $0.25^\circ$", FIXED)],
        [(r"$A_{\min} = 5\times10^4$ km$^2$", TUNE),
         (r"8-connectivity", FIXED)],
        [(r"$D_{\min} = 3$ d", TUNE)],
        [(r"8 + 2 neighbours", FIXED), (r"peak area $\ge A_{\min}$", FIXED)],
        [(r"rank by $S$", FIXED), (r"touches May--Sep", FIXED)],
    ]

    # Inputs above the first two steps.
    yin, hin = 0.905, 0.075
    inputs = [r"$T_{\max}$, 1991--2020",
              r"$T_{\max}$, 2026"]
    for i, text in enumerate(inputs):
        canvas.add_patch(FancyBboxPatch(
            (xs[i], yin), w, hin, boxstyle="round,pad=0,rounding_size=0.008",
            fc="white", ec=INK, lw=0.6, ls=(0, (3, 2))))
        canvas.text(xs[i] + w / 2, yin + hin / 2, text, ha="center",
                    va="center", fontsize=5.8, color=INK)

    yh, hh = 0.665, 0.205
    for i, (title, eq) in enumerate(steps):
        canvas.add_patch(FancyBboxPatch(
            (xs[i], yh), w, hh, boxstyle="round,pad=0,rounding_size=0.01",
            fc="#FBF6EF", ec=HOT, lw=0.9))
        canvas.text(xs[i] + 0.01, yh + hh - 0.035,
                    rf"\textbf{{{i + 1}}}\enspace\textbf{{{title}}}",
                    ha="left", va="top", fontsize=6.8, color=INK)
        canvas.text(xs[i] + w / 2, yh + 0.07, eq, ha="center", va="center",
                    fontsize=5.8, color=INK, linespacing=1.6)
        if i < n - 1:
            canvas.add_patch(FancyArrowPatch(
                (xs[i] + w + 0.002, yh + hh / 2),
                (xs[i + 1] - 0.002, yh + hh / 2),
                arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK))
    for i in range(2):
        canvas.add_patch(FancyArrowPatch(
            (xs[i] + w / 2, yin), (xs[i] + w / 2, yh + hh),
            arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK))

    yp, hp = 0.215, 0.39
    axes = [fig.add_axes([xs[i] + 0.006, yp, w - 0.012, hp])
            for i in range(n)]

    # 1. One city through the summer against its own threshold.
    lat, lon = r["lat"], r["lon"]
    ci = int(np.abs(lat - CITY[1]).argmin())
    cj = int(np.abs(lon - CITY[2]).argmin())
    t = r["time"]
    season = (t >= "2026-06-01") & (t <= "2026-07-31")
    tt = t[season]
    tmax = r["target"].values[season, ci, cj]
    tq = r["thresh"][season, ci, cj]
    ax = axes[0]
    ax.axvspan(pd.Timestamp(top["start"]), pd.Timestamp(top["end"]),
               color=HOT, alpha=0.08, lw=0)
    ax.plot(tt, tmax, color=INK, lw=0.7, label=r"$T_{\max}$")
    ax.plot(tt, tq, color=HOT, lw=1.1, label=r"$T_q$")
    ax.fill_between(tt, tq, tmax, where=tmax > tq, color=HOT, alpha=0.45,
                    lw=0, interpolate=True)
    ax.axvline(pd.Timestamp(DAY), color=FIXED, lw=0.5, ls=(0, (2, 1.5)))
    ax.set_xticks(pd.to_datetime(["2026-06-01", "2026-07-01", "2026-08-01"]),
                  ["Jun", "Jul", "Aug"])
    ax.set_ylabel(r"$T$ ($^\circ$C)", labelpad=1, fontsize=6.5)
    ax.set_ylim(15, 47)                  # headroom for the legend
    ax.set_yticks([20, 30, 40])
    ax.legend(loc="upper left", frameon=False, fontsize=6, ncol=2,
              handlelength=1.2, columnspacing=0.8, borderaxespad=0.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{CITY[0]}, summer 2026", fontsize=6.6, color=FIXED, pad=3)

    land = r["land"]
    exc, coh, per = r["exc"][day], r["coherent"][day], r["persistent"][day]
    label_day = pd.Timestamp(DAY).strftime("%-d %b")

    # 2-4. The same day after each filter: sea 0, land 1, removed 2, kept 3.
    base = np.where(land, 1, 0)
    show_map(axes[1], np.where(exc, 3, base), [SEA, LAND, DROP, HOT], r,
             f"{label_day}: hot cells")
    show_map(axes[2], np.where(coh, 3, np.where(exc, 2, base)),
             [SEA, LAND, DROP, HOT], r, f"{label_day}: big patches")
    show_map(axes[3], np.where(per, 3, np.where(coh, 2, base)),
             [SEA, LAND, DROP, HOT], r, rf"{label_day}: $\ge$3 days")

    # 5. The top-ranked event's whole footprint, shaded by the day each cell
    # first joined it, so the days stitched into one event are visible.
    in_event = r["labels"] == eid
    first = np.where(in_event.any(0), in_event.argmax(0), -1)
    d0 = int(np.flatnonzero(in_event.any((1, 2)))[0])
    ndays = int(top["duration_days"])
    shades = plt.get_cmap("YlOrRd")(np.linspace(0.25, 0.95, ndays))
    colours = [SEA, LAND] + [tuple(c) for c in shades]
    state = np.where(first >= 0, 2 + first - d0, base)
    show_map(axes[4], state, colours, r, r"event \#1 footprint")
    start = pd.Timestamp(top["start"]).strftime("%-d %b")
    end = pd.Timestamp(top["end"]).strftime("%-d %b")
    axes[4].text(0.03, 0.97,
                 rf"{start}--{end}",
                 transform=axes[4].transAxes, ha="left", va="top",
                 fontsize=5.6, color=INK, linespacing=1.4,
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))

    # 6. The season's most severe events, the catalogue's ranking itself.
    ranked = r["events"].head(5).iloc[::-1]
    scale = 10 ** int(np.floor(np.log10(ranked["severity_C_km2_day"].max())))
    ax = axes[5]
    y = np.arange(len(ranked))
    ax.barh(y, ranked["severity_C_km2_day"] / scale, height=0.62,
            color=[HOT if k == 1 else DROP for k in ranked["rank"]])
    for yi, (_, e) in zip(y, ranked.iterrows()):
        a_, b_ = (pd.Timestamp(e[c]).strftime("%-d %b") for c in ("start", "end"))
        ax.text(e["severity_C_km2_day"] / scale * 1.03, yi, f"{a_}--{b_}",
                va="center", fontsize=5.2, color=INK)
    ax.set_yticks(y, [rf"\#{k}" for k in ranked["rank"]])
    ax.set_xlim(0, ranked["severity_C_km2_day"].max() / scale * 1.9)
    ax.set_xlabel(rf"$S$ ($10^{{{int(np.log10(scale))}}}$ $^\circ$C km$^2$ d)",
                  fontsize=5.8, labelpad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("2026, most severe", fontsize=6.6, color=FIXED, pad=3)

    for ax in axes:
        ax.tick_params(labelsize=6, length=2.5, pad=1.5)
    for ax in axes[1:5]:
        for s in ax.spines.values():
            s.set_linewidth(0.5)

    # The maps shrink to keep their aspect; give the charts the same box.
    fig.canvas.draw()
    box = axes[1].get_position()
    p0 = axes[0].get_position()
    axes[0].set_position([p0.x0 + 0.014, box.y0, p0.width - 0.014, box.height])
    p5 = axes[5].get_position()
    axes[5].set_position([p5.x0 + 0.008, box.y0, p5.width - 0.008, box.height])

    # Shading key for panel 5: the day each cell joined the event.
    cax = fig.add_axes([xs[4] + 0.02, box.y0 - 0.085, w - 0.04, 0.014])
    cax.imshow(shades[None, :, :3], aspect="auto")
    cax.set_xticks([0, ndays - 1], [start, end])
    cax.set_yticks([])
    cax.tick_params(labelsize=5.6, length=1.5, pad=1)
    cax.set_title("day each cell joined", fontsize=5.6, color=FIXED, pad=1.5)

    # Settings under each panel: blue is a free choice, grey is fixed.
    for i, items in enumerate(settings):
        y = box.y0 - 0.15
        for text, colour in items:
            canvas.text(xs[i] + w / 2, y, text, ha="center", va="center",
                        fontsize=6.0, color=colour)
            y -= 0.035

    canvas.plot([x0, 1 - x0], [box.y0 - 0.12] * 2, color=DROP, lw=0.4)
    legend = [(HOT, "hot, kept"), (DROP, "removed at this step"),
              (LAND, rf"land, not hot \enspace $\circ$ {CITY[0]}")]
    lx = x0
    ly = box.y0 - 0.15 - 3 * 0.035
    for colour, text in legend:
        canvas.add_patch(plt.Rectangle((lx, ly - 0.01), 0.011, 0.02, fc=colour,
                                       ec=INK if colour == LAND else "none",
                                       lw=0.3))
        canvas.text(lx + 0.016, ly, text, va="center", fontsize=6.4,
                    color=INK)
        lx += 0.16
    canvas.text(1 - x0, ly, "fixed choice", ha="right", va="center",
                fontsize=6.4, color=FIXED)
    canvas.text(1 - x0 - 0.115, ly, "free setting", ha="right",
                va="center", fontsize=6.4, color=TUNE)

        # The full-figure canvas defeats bbox="tight", so crop below the legend.
    width, height = fig.get_size_inches()
    crop = Bbox.from_extents(-0.2, (ly - 0.03) * height, width + 0.1, height)
    fig.savefig(PLOTS / "detector_pipeline.pdf", bbox_inches=crop)
    fig.savefig(PLOTS / "detector_pipeline.png", dpi=300, facecolor="white",
                bbox_inches=crop)
    print("saved detector_pipeline")


# ----------------------------------------------------------------------
# step 1: the bar
# ----------------------------------------------------------------------
def cell_series(year):
    """Daily maximum at the city's grid cell for one year, degrees C."""
    da = D.open_daily_tmax([WORK.era5_file(year)])
    return da.sel(latitude=CITY[1], longitude=CITY[2], method="nearest")


def threshold_figure():
    """How the step-1 bar is set: Berlin, 27 June, 1991-2020."""
    mpl_apply({**MPL_CONFIG, "font.size": 9, "axes.linewidth": 0.6})

    # Offsets are in day of year, exactly as calendar_day_percentile pools
    # them, so in leap years the window starts a calendar day earlier.
    target_doy = pd.Timestamp(f"2001-{DATE}").dayofyear
    offsets = np.arange(-SHOW, SHOW + 1)
    grid = np.full((len(BASELINE), offsets.size), np.nan)
    for i, year in enumerate(BASELINE):
        s = cell_series(year)
        doy = s["time"].dt.dayofyear.values
        for j, off in enumerate(offsets):
            hit = doy == target_doy + off
            if hit.any():
                grid[i, j] = float(s.values[hit][0])

    inside = np.abs(offsets) <= HALF
    pooled = grid[:, inside].ravel()
    bar = float(np.quantile(pooled, Q))

    cached = xr.open_dataarray(threshold_path()).sel(
        dayofyear=target_doy, latitude=CITY[1], longitude=CITY[2],
        method="nearest")
    assert abs(bar - float(cached)) < 0.01, (bar, float(cached))

    now = cell_series(TARGET_YEAR)
    today = float(now.sel(time=f"{TARGET_YEAR}-{DATE}"))

    fig, (ax, hx) = plt.subplots(
        1, 2, figsize=(5.6, 3.3), width_ratios=[1.45, 1],
        gridspec_kw=dict(wspace=0.32))

    # Left: the calendar of past days.
    vmin, vmax = np.nanpercentile(grid, [2, 98])
    cmap = plt.get_cmap("YlOrRd")
    extent = [offsets[0] - 0.5, offsets[-1] + 0.5, len(BASELINE) - 0.5, -0.5]
    # Days outside the window in grey, so only the pooled ones carry colour.
    ax.imshow(grid, cmap=ListedColormap(["#D8D2C8"]), extent=extent,
              aspect="auto", interpolation="nearest")
    ax.imshow(np.where(inside[None, :], grid, np.nan), cmap=cmap, vmin=vmin,
              vmax=vmax, extent=extent, aspect="auto",
              interpolation="nearest")
    ax.add_patch(Rectangle((-HALF - 0.5, -0.5), 2 * HALF + 1, len(BASELINE),
                           fc="none", ec=TUNE, lw=1.4))
    ax.add_patch(Rectangle((-0.5, -0.5), 1, len(BASELINE), fc="none",
                           ec=INK, lw=1.0))
    ticks = [-HALF, 0, HALF]
    labels = [(pd.Timestamp(f"2001-{DATE}") + pd.Timedelta(days=int(o)))
              .strftime("%-d %b") for o in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_yticks([0, 9, 19, 29], ["1991", "2000", "2010", "2020"])
    ax.tick_params(length=2.5, pad=2, labelsize=8)
    ax.set_title(r"each square: one past day's high", fontsize=8.5,
                 color=MUTED, pad=4)
    ax.text(0.5, -0.17, rf"window: {2 * HALF + 1} days $\times$ "
            rf"{len(BASELINE)} years = {pooled.size} days",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color=TUNE)

    # Right: the pooled days, and where the bar falls.
    bins = np.arange(np.floor(pooled.min()), np.ceil(pooled.max()) + 1, 1.0)
    counts, edges = np.histogram(pooled, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # Same colours as the grid; days under the bar faded.
    colours = cmap(np.clip((centres - vmin) / (vmax - vmin), 0, 1))
    colours[centres <= bar, 3] = 0.35
    hx.bar(centres, counts, width=0.92, color=colours)
    hx.axvline(bar, color=TUNE, lw=1.4)
    top = counts.max()
    hx.text(bar - 0.4, top * 1.02, rf"bar {bar:.1f}$^\circ$C", ha="right",
            va="bottom", fontsize=8.5, color=TUNE)
    hx.text(bar + 0.4, top * 0.55, "hottest\n10\\%", ha="left",
            va="center", fontsize=8, color=HOT)
    hx.plot([today], [top * 0.1], marker="v", ms=6, color=INK)
    hx.text(today, top * 0.2, f"{TARGET_YEAR}\n"
            rf"{today:.1f}$^\circ$C", ha="center", va="bottom",
            fontsize=8, color=INK)
    hx.set_ylim(0, top * 1.18)
    hx.set_xlabel(r"daily high ($^\circ$C)", fontsize=8.5, labelpad=2)
    hx.set_ylabel("days", fontsize=8.5, labelpad=2)
    hx.tick_params(length=2.5, pad=2, labelsize=8)
    for s in ("top", "right"):
        hx.spines[s].set_visible(False)
    hx.set_title(rf"the {pooled.size} days, pooled", fontsize=8.5,
                 color=MUTED, pad=4)

    for a in (ax, hx):
        a.set_facecolor("none")
    fig.suptitle(rf"{CITY[0]}, bar for "
                 rf"{pd.Timestamp('2001-' + DATE).strftime('%-d %B')}",
                 fontsize=10, x=0.12, ha="left", y=1.0)

    fig.savefig(PLOTS / "detector_threshold_berlin.pdf")
    fig.savefig(PLOTS / "detector_threshold_berlin.png", dpi=300, facecolor="#F7F4EE")
    print(f"bar {bar:.2f} C from {pooled.size} days; {TARGET_YEAR}: "
          f"{today:.2f} C")


# ----------------------------------------------------------------------
# steps 3, 4 and 5
# ----------------------------------------------------------------------
def draw_map(ax, state, colours, r):
    lon, lat = r["lon"], r["lat"]
    extent = [lon[0] - 0.125, lon[-1] + 0.125, lat[-1] - 0.125,
              lat[0] + 0.125]
    ax.imshow(state, cmap=ListedColormap(colours), extent=extent,
              interpolation="nearest", vmin=0, vmax=len(colours) - 1)
    ax.contour(lon, lat, r["land"].astype(float), levels=[0.5], colors=INK,
               linewidths=0.3)
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])
    ax.set_aspect(1 / np.cos(np.deg2rad(50)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS / f"{name}.{ext}", dpi=300, facecolor=BG)
    print(f"saved {name}")


def area_figure(r):
    mpl_apply()
    k = int(np.flatnonzero(r["time"] == pd.Timestamp(DAY))[0])
    area = D.cell_area_km2(r["lat"], r["lon"])
    labels, n = ndimage.label(r["exc"][k],
                              structure=ndimage.generate_binary_structure(2, 2))
    sizes = ndimage.sum_labels(area, labels, np.arange(1, n + 1))
    kept = np.flatnonzero(sizes >= MIN_AREA) + 1

    state = np.where(r["land"], 1, 0)
    state = np.where(r["exc"][k], 2, state)
    state = np.where(np.isin(labels, kept), 3, state)

    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    draw_map(ax, state, [SEA, LAND, "#8E877C", HOT], r)
    # Name only the largest patch; the small kept ones sit on the edges.
    biggest = kept[np.argmax(sizes[kept - 1])]
    cy, cx = ndimage.center_of_mass(labels == biggest)
    ax.text(r["lon"][int(round(cx))], r["lat"][int(round(cy))],
            f"{sizes[biggest - 1] / 1e6:.1f} million km$^2$", ha="center",
            va="center", fontsize=8.5, color=INK,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    ax.set_title(rf"20 June 2026: {n} hot patches, {n - kept.size} deleted, "
                 rf"{kept.size} kept", fontsize=10, color=INK, pad=5)
    ax.legend(handles=[Patch(fc=HOT, label=r"kept, $\ge$ 50,000 km$^2$"),
                       Patch(fc="#8E877C", label="deleted, too small")],
              loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              frameon=False, fontsize=8.5, handlelength=1.2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
    save(fig, "detector_step_area")


def persist_figure(r):
    t = r["time"]
    i = int(np.abs(r["lat"] - STEP4_CITY[1]).argmin())
    j = int(np.abs(r["lon"] - STEP4_CITY[2]).argmin())
    sel = (t >= SPAN[0]) & (t <= SPAN[1])
    days = t[sel]
    tmax = r["target"].values[sel, i, j]
    bar = r["thresh"][sel, i, j]
    coh = r["coherent"][sel, i, j]
    per = r["persistent"][sel, i, j]

    colours = np.where(per, HOT, np.where(coh, "#8E877C", "#DDD7CD"))
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    base = 10.0
    ax.bar(days, tmax - base, bottom=base, width=0.8, color=colours)
    ax.step(days, bar, where="mid", color=TUNE, lw=1.6)

    # Label each spell above the bar by its length and fate.
    runs, start = [], None
    for d in range(len(days) + 1):
        on = d < len(days) and coh[d]
        if on and start is None:
            start = d
        if not on and start is not None:
            runs.append((start, d))
            start = None
    for a, b in runs:
        n = b - a
        kept = per[a]
        mid = days[a] + (days[b - 1] - days[a]) / 2
        text = (f"{n} days: kept" if kept
                else f"{n} day{'s' if n > 1 else ''}: dropped")
        ax.text(mid, tmax[a:b].max() + 0.8, text, ha="center", va="bottom",
                fontsize=8.5, color=HOT if kept else INK)
    ax.set_ylim(base, max(tmax.max(), bar.max()) + 4)
    ax.set_ylabel(r"daily high ($^\circ$C)", fontsize=9)
    ax.set_xticks(pd.to_datetime(["2026-06-08", "2026-06-15", "2026-06-22",
                                  "2026-06-29"]),
                  ["8 Jun", "15 Jun", "22 Jun", "29 Jun"])
    ax.tick_params(labelsize=8.5, length=2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor("none")
    ax.set_title(f"{STEP4_CITY[0]}, June 2026", fontsize=10, color=INK, pad=5,
                 loc="left")
    ax.legend(handles=[Patch(fc=HOT, label=r"hot $\ge$ 3 days in a row"),
                       Patch(fc="#8E877C", label="hot, too short"),
                       Patch(fc="#DDD7CD", label="not hot"),
                       Line2D([], [], color=TUNE, lw=1.6, label="bar")],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4,
              frameon=False, fontsize=8.5, handlelength=1.2)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.9, bottom=0.2)
    save(fig, "detector_step_persist")


def label_figure(r):
    events = r["events"].set_index("event_id")
    top = int(r["events"].iloc[0]["event_id"])
    fig, axes = plt.subplots(2, 3, figsize=(5.2, 3.7))
    for ax, day in zip(axes.flat, LABEL_DAYS):
        k = int(np.flatnonzero(r["time"] == pd.Timestamp(day))[0])
        lab = r["labels"][k]
        state = np.where(r["land"], 1, 0)
        state = np.where(lab > 0, 2, state)                     # too small
        state = np.where(np.isin(lab, events.index), 3, state)  # catalogued
        state = np.where(lab == top, 4, state)
        draw_map(ax, state, [SEA, LAND, "#BDB6AB", TUNE, HOT], r)
        ax.set_title(pd.Timestamp(day).strftime("%-d %b"), fontsize=9,
                     color=INK, pad=2)
    fig.legend(handles=[Patch(fc=HOT, label=r"event \#1"),
                        Patch(fc=TUNE, label="other events"),
                        Patch(fc="#BDB6AB", label="too small to count")],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               handlelength=1.2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.1,
                        wspace=0.04, hspace=0.14)
    save(fig, "detector_step_label")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    r = run_detector()
    pipeline_figure(r)
    threshold_figure()
    area_figure(r)
    persist_figure(r)
    label_figure(r)


if __name__ == "__main__":
    main()
