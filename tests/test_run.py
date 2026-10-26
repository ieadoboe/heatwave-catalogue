"""catalogue.run end to end on a small synthetic ERA5 working folder:
thresholds, the Europe mask, the edge flag and the rescored severities.

    python tests/test_run.py
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hwcat import catalogue as C
from hwcat.workdir import Workdir

rng = np.random.default_rng(1)
# A box over southern France, the Pyrenees and northern Spain, plus a
# strip of North Africa (Algeria) at its southern end: 35.5-46N, 0-4E.
LAT = np.arange(46.0, 35.49, -0.25)
LON = np.arange(0.0, 4.01, 0.25)


def year_file(wd, year, extra=0.0):
    t = pd.date_range(f"{year}-04-01", f"{year}-10-31", freq="D")
    doy = t.dayofyear.to_numpy()[:, None, None]
    field = 20 + 8 * np.sin(np.pi * (doy - 91) / 214) \
        + rng.normal(0, 2, (t.size, LAT.size, LON.size)) + extra
    da = xr.DataArray(field.astype("float32"), name="t2m",
                      coords={"valid_time": t, "latitude": LAT, "longitude": LON},
                      dims=("valid_time", "latitude", "longitude"))
    wd.era5.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(wd.era5_file(year))


def main():
    with tempfile.TemporaryDirectory() as d:
        wd = Workdir(Path(d))
        for y in range(2000, 2006):
            year_file(wd, y)
        # 2010: a 10-day wave over the whole box in July, Spain to Algeria.
        heat = np.zeros((214, LAT.size, LON.size), dtype="float32")
        heat[100:110] = 12.0
        t = pd.date_range("2010-04-01", "2010-10-31", freq="D")
        doy = t.dayofyear.to_numpy()[:, None, None]
        field = 20 + 8 * np.sin(np.pi * (doy - 91) / 214) \
            + rng.normal(0, 2, (t.size, LAT.size, LON.size)) + heat
        wd.era5.mkdir(parents=True, exist_ok=True)
        xr.DataArray(field.astype("float32"), name="t2m",
                     coords={"valid_time": t, "latitude": LAT, "longitude": LON},
                     dims=("valid_time", "latitude", "longitude")
                     ).to_netcdf(wd.era5_file(2010))

        kw = dict(baseline=(2000, 2005), plot=False, min_area=20_000.0)
        europe = pd.read_csv(C.run(2010, wd, rescore=True, **kw))
        box = pd.read_csv(C.run(2010, wd, whole_box=True, tag="_box", **kw))
        raw = C.run(2010, wd, remove_cycle=False, **kw)
        names = sorted(p.name for p in wd.thresholds.iterdir())

    lead = europe.iloc[0]
    assert {"severity_ref_C_km2_day", "severity_norm_km2_day",
            "touches_edge"} <= set(europe.columns), europe.columns
    # The reference bar is this run's own bar, so both severities agree.
    assert abs(lead.severity_ref_C_km2_day / lead.severity_C_km2_day - 1) < 0.05
    assert lead.severity_norm_km2_day > 0
    # The sea splits the wave: the whole box also catalogues its Algerian
    # part as a second event, the European run does not.
    assert len(box) == 2 and len(europe) == 1, (len(box), len(europe))
    assert box.centroid_lat.min() < 37.5 < europe.centroid_lat.min()
    assert lead.touches_edge          # it fills the box to its edges
    assert str(lead.start) == "2010-07-10", lead.start
    # Raw pooling is marked in the default tag; cycle removal is not.
    assert raw.name.endswith("_2010_raw.csv"), raw.name
    assert any("_p25_sc" in n for n in names) and any("_p75_sc" in n for n in names)
    print("ok")


if __name__ == "__main__":
    main()
