"""The catalogue's maps and the plot style they share.

Each map function returns (fig, ax); pass save=<path> to also write the file."""

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MPL_CONFIG = {
    'savefig.dpi': 300,  # Publication quality
    'figure.dpi': 150,   # Screen display
    'font.family': 'serif',  # Font family
    'font.size': 11,  # Global font size
    'font.serif': ['Computer Modern'],
    'text.usetex': True,  # Enable LaTeX rendering
    'axes.labelsize': 10,  # Font size for axis labels
    'axes.titlesize': 11,  # Font size for subplot titles
    'xtick.labelsize': 9,  # Legend text size
    'ytick.labelsize': 9,  # Legend text size
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.linewidth': 0.8,
    'savefig.bbox': 'tight',
    'xtick.direction': 'inout',
    'ytick.direction': 'inout',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
}


def mpl_apply(config: dict | None = None) -> None:
    """
    Apply Matplotlib configuration globally.

    Parameters
    ----------
    config : dict, optional
        Dictionary of rcParams to apply. Defaults to MPL_CONFIG.
    """
    cfg = dict(config if config is not None else MPL_CONFIG)
    if cfg.get("text.usetex") and shutil.which("latex") is None:
        # No TeX toolchain on this machine (e.g. Colab): render with mathtext
        # instead so $...$ labels still work without a latex binary.
        cfg["text.usetex"] = False
        cfg["mathtext.fontset"] = "cm"
        cfg["font.serif"] = ["DejaVu Serif"]
    mpl.rcParams.update(cfg)


mpl_apply()


def _finish(fig, save):
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
        print(f"Plot -> {save}")
    return fig


def plot_koppen_map(events: dict | None = None, extent=None, save=None,
                    title="Koppen-Geiger climate classes, 1991-2020 (Beck et al. 2023)"):
    """Global (or zoomed) map of the 30 Koppen-Geiger sub-zones in the official
    Beck et al. colors, with optional event markers.

    events: {label: (lat, lon)} — white-filled ringed markers with labels
            (lon in signed degrees East).
    extent: (lon_min, lon_max, lat_min, lat_max) to zoom a region.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    from hwcat.zones import KOPPEN_COLORS, koppen_grid

    classes, _, _, codes = koppen_grid()
    cmap = ListedColormap(["#ffffff"] + [KOPPEN_COLORS[c] for c in codes[1:]])
    norm = BoundaryNorm(np.arange(-0.5, 31.5), cmap.N)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.imshow(classes, cmap=cmap, norm=norm, interpolation="nearest",
              extent=[-180, 180, -90, 90], origin="upper")
    if events:
        for name, (la, lo) in events.items():
            lo = ((lo + 180.0) % 360.0) - 180.0
            ax.scatter([lo], [la], s=55, facecolor="white", edgecolor="black",
                       linewidth=1.2, zorder=5)
            ax.annotate(name, (lo, la), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color="black",
                        bbox=dict(facecolor="white", alpha=0.75,
                                  edgecolor="none", pad=1.5))
    if extent:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set(xlabel="longitude", ylabel="latitude", title=title)
    handles = [Patch(facecolor=KOPPEN_COLORS[c], label=c) for c in codes[1:]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=10, fontsize=7, frameon=False, handlelength=1.2,
              columnspacing=1.0)
    return _finish(fig, save), ax


def plot_europe_heatwaves(hw_days, lat, lon, events=None, top_n: int = 8,
                          year: int = 2026, save=None):
    """Map of heat-wave days per grid cell, with the largest events marked.

    hw_days: (lat, lon) count of days each cell spent in a detected event.
    events:  the event_table DataFrame; the top_n by severity are labelled at
             their centroids."""
    from matplotlib.colors import BoundaryNorm

    counts = np.where(hw_days > 0, hw_days, np.nan)
    vmax = max(6, int(np.nanmax(counts)) if np.isfinite(counts).any() else 6)
    levels = np.arange(0, vmax + 2)

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(counts, origin="upper", cmap="inferno_r",
                   norm=BoundaryNorm(levels, 256),
                   extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                   interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, label="heat-wave days", shrink=0.85, pad=0.02)

    if events is not None and len(events):
        for _, e in events.head(top_n).iterrows():
            ax.scatter([e.centroid_lon], [e.centroid_lat], s=48,
                       facecolor="white", edgecolor="black", linewidth=1.1,
                       zorder=5)
            ax.annotate(f"{int(e['rank'])}. {e['peak']}",
                        (e.centroid_lon, e.centroid_lat), xytext=(6, 5),
                        textcoords="offset points", fontsize=7.5,
                        bbox=dict(facecolor="white", alpha=0.8,
                                  edgecolor="none", pad=1.4), zorder=6)
    ax.set(xlabel="longitude", ylabel="latitude",
           title=f"European heat waves {year}: days above the calendar-day "
                 f"90th percentile\n(CTX90pct, 1991-2020 baseline; "
                 f"numbers mark the {top_n} most severe events)")
    return _finish(fig, save), ax
