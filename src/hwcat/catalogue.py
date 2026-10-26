"""One heat-wave catalogue: one year, one setting.

    from hwcat.catalogue import run
    from hwcat.workdir import Workdir
    run(2003, Workdir(Path("hwcat-data")), leave_one_out=True)

Applies Perkins & Alexander CTX90pct against a 1991-2020 calendar-day
baseline, labels the connected space-time blobs, and writes the ranked
catalogue, optionally its footprints and a map, into the working folder.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hwcat import detect as D
from hwcat import zones
from hwcat.workdir import Workdir


def footprint_koppen(det, event_id: int, zones) -> str:
    """Most common land Koppen class among the cells an event ever covered."""
    from collections import Counter
    hit = det.labels == event_id
    _, ii, jj = np.nonzero(hit)
    cells = set(zip(ii.tolist(), jj.tolist()))
    counts = Counter(zones.koppen_class(det.lat[i], det.lon[j])
                     for i, j in cells)
    counts.pop("ocean", None)
    return counts.most_common(1)[0][0] if counts else "ocean"


def baseline_years_for(year: int, first: int, last: int,
                       leave_one_out: bool) -> list[int]:
    """Baseline years for one target year; without it under leave-one-out."""
    years = list(range(first, last + 1))
    if leave_one_out and year in years:
        years.remove(year)
    return years


def threshold_name(region: str, first: int, last: int, pct: float, span: str,
                   window: int, baseline_years: list[int],
                   remove_cycle: bool = False) -> str:
    """Cache filename for a threshold, keyed by everything that shapes it.

    The default (15-day window, full baseline, raw pooling) keeps its
    original name so existing caches and notebooks still find it."""
    name = f"heatwave_threshold_{region}_{first}_{last}_p{round(pct * 100, 1):g}"
    if window != 15:
        name += f"_w{window}"
    if remove_cycle:
        name += "_sc"
    dropped = sorted(set(range(first, last + 1)) - set(baseline_years))
    if dropped:
        name += "_loo" + "_".join(str(y) for y in dropped)
    return f"{name}_{span}.nc"


def default_tag(window: int, first: int, last: int,
                baseline_years: list[int], remove_cycle: bool = True) -> str:
    """Output suffix for a run with no --tag, so a non-default window, raw
    pooling or a leave-one-out baseline never overwrites the reference
    catalogue. The seasonal cycle is removed by default (Brunner and Voigt
    2024), so it is raw pooling that gets marked."""
    tag = f"_w{window}" if window != 15 else ""
    if not remove_cycle:
        tag += "_raw"
    if len(baseline_years) < last - first + 1:
        tag += "_loo"
    return tag


def replace_atomically(write, path: Path) -> None:
    """Write through a temporary file, so a run killed mid-write never leaves
    a truncated file that a later run would reuse."""
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write(tmp)
    os.replace(tmp, path)


def save_labels_npz(det, events, path: Path) -> None:
    """Every cell-day of every catalogued event, as sparse coordinates.

    The full (time, lat, lon) label grid is mostly zeros; storing only the
    event cell-days keeps a season to a few hundred kB."""
    keep = np.isin(det.labels, events.event_id.to_numpy())
    t, i, j = np.nonzero(keep)
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_atomically(
        lambda p: np.savez_compressed(
            p, t=t.astype(np.int16), i=i.astype(np.int16),
            j=j.astype(np.int16), event_id=det.labels[t, i, j].astype(np.int32),
            shape=np.array(det.labels.shape)),
        path)


def download_hint(workdir: Workdir, years: str) -> str:
    """The hwcat download command that fetches what is missing, keeping a
    non-European region's name and asking for its box."""
    hint = f"hwcat download --years {years} --workdir {workdir.root}"
    if workdir.region != "europe":
        hint += f" --region-name {workdir.region} --region N W S E"
    return hint


REF_THRESHOLD = (0.90, 15, True)   # percentile, window, cycle removed


def domain_mask(lat: np.ndarray, lon: np.ndarray, land: np.ndarray,
                region: str) -> np.ndarray:
    """Land cells in the region's own countries (Europe: Natural Earth's
    European countries plus Cyprus), so a "European" catalogue holds no
    North African or Anatolian events. Regions without a country grid keep
    all land in the box."""
    from hwcat import viewer
    if region not in viewer.HOME:
        return land
    grid, _, home = viewer.country_grid(lat, lon, land, region)
    return land & np.asarray(home)[grid]


def run(year: int, workdir: Workdir, percentile: float = 0.90,
        min_days: int = 3, min_area: float = 50_000.0, window: int = 15,
        leave_one_out: bool = False, save_labels: bool = False,
        plot: bool = True, season: tuple[str, str] = ("05-01", "09-30"),
        baseline: tuple[int, int] = (1991, 2020), tag: str | None = None,
        rebuild_threshold: bool = False, remove_cycle: bool = True,
        connectivity: int = 10, whole_box: bool = False,
        rescore: bool = False) -> Path:
    """Catalogue one year under one setting; return the event CSV path.

    whole_box keeps every land cell in the box instead of the region's own
    countries. rescore adds two severities computed the same way for every
    setting, so rankings can be compared without the metric itself moving:
    severity_ref_C_km2_day, the exceedance above the REF_THRESHOLD bar, and
    severity_norm_km2_day, the exceedance above this run's own bar divided
    by the local interquartile range of daily maxima (from the REF_THRESHOLD
    window), as in Russo's HWMId."""
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be odd and positive, got {window}: a "
                         f"window centred on the target day has as many days "
                         f"on each side")
    first, last = baseline
    baseline_years = baseline_years_for(year, first, last, leave_one_out)
    target_path = workdir.era5_file(year)
    if not target_path.exists():
        raise FileNotFoundError(
            f"missing {target_path}\nrun: {download_hint(workdir, str(year))}")

    print(f"Target year {year}: {target_path.name}")
    target = D.open_daily_tmax([target_path])

    # The month span is part of the cache key: the window makes the
    # thresholds near either end depend on which months the baseline holds,
    # so a May-September cache must never be reused for April-October data.
    months = target["time"].dt.month
    span = f"m{int(months.min()):02d}-{int(months.max()):02d}"
    workdir.thresholds.mkdir(parents=True, exist_ok=True)
    baseline_da = None

    def threshold_for(pct, win, cycle):
        nonlocal baseline_da
        path = workdir.thresholds / threshold_name(
            workdir.region, first, last, pct, span, win, baseline_years, cycle)
        if path.exists() and not rebuild_threshold:
            return xr.open_dataarray(path).load()
        if baseline_da is None:
            paths = [workdir.era5_file(y) for y in baseline_years]
            missing = [p.name for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"missing {len(missing)} baseline year(s): "
                    f"{', '.join(missing[:5])}\n"
                    f"run: {download_hint(workdir, f'{first}-{last}')}")
            baseline_da = D.open_daily_tmax(paths)
        out = D.calendar_day_percentile(baseline_da, q=pct, window=win // 2,
                                        remove_cycle=cycle)
        replace_atomically(out.to_netcdf, path)
        return out

    threshold = threshold_for(percentile, window, remove_cycle)

    lat, lon = target["latitude"].values, target["longitude"].values
    land = D.land_mask_for(lat, lon)
    domain = land if whole_box else domain_mask(lat, lon, land, workdir.region)
    det = D.detect(target, threshold, land_mask=domain, min_days=min_days,
                   min_area_km2=min_area, connectivity=connectivity,
                   cut=D.cut_cells(land, domain))
    extra = None
    if rescore:
        def on_days(da):
            return da.sel(dayofyear=target["time"].dt.dayofyear).values
        ref_pct, ref_win, ref_cycle = REF_THRESHOLD
        ref_bar = on_days(threshold_for(ref_pct, ref_win, ref_cycle))
        iqr = (on_days(threshold_for(0.75, ref_win, ref_cycle))
               - on_days(threshold_for(0.25, ref_win, ref_cycle)))
        extra = {"severity_ref_C_km2_day":
                     np.clip(target.values - ref_bar, 0.0, None),
                 "severity_norm_km2_day":
                     det.anomaly / np.maximum(iqr, 0.1)}
    del baseline_da
    events = D.event_table(det, target, min_area_km2=min_area,
                           extra_severity=extra)
    events = D.in_season(events, *season)

    tag = default_tag(window, first, last, baseline_years,
                      remove_cycle) if tag is None else tag
    out_csv = workdir.catalogue_file(year, tag)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if events.empty:
        # An empty catalogue is a result, not an error: a strict setting in a
        # cool summer, or a box with no land, can find nothing.
        replace_atomically(
            lambda p: pd.DataFrame(columns=["rank", "start", "peak", "end"])
            .to_csv(p, index=False), out_csv)
        if save_labels:
            save_labels_npz(det, events.assign(event_id=pd.Series(dtype=int)),
                            workdir.labels_file(year, tag))
        print(f"No events above {min_area:,.0f} km2 -> {out_csv} (empty)")
        return out_csv

    # Tag each event with the modal Koppen class over its land footprint. The
    # centroid alone is wrong for an event wrapped around a coast.
    events["koppen"] = [footprint_koppen(det, int(eid), zones)
                        for eid in events.event_id]
    events["zone"] = [zones.GROUP_NAMES.get(zones.koppen_group(k), "")
                      for k in events.koppen]
    replace_atomically(lambda p: events.to_csv(p, index=False), out_csv)
    if save_labels:
        save_labels_npz(det, events, workdir.labels_file(year, tag))
    print(f"{len(events)} events above {min_area:,.0f} km2 -> {out_csv}")

    if plot:
        from hwcat import plots
        workdir.maps.mkdir(parents=True, exist_ok=True)
        hw_days = (det.labels > 0).sum(axis=0)
        plots.plot_europe_heatwaves(
            hw_days, det.lat, det.lon, events=events, year=year,
            save=workdir.maps / f"heatwave_{workdir.region}_{year}{tag}.pdf")
    return out_csv
