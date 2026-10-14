"""One catalogue run: the working-folder layout, refusals, an all-ocean box,
and equivalence with the reference outputs frozen before the move.

    python tests/test_catalogue.py
"""
import filecmp
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hwcat import catalogue as C
from hwcat.workdir import Workdir

REF = Path("/Users/isaacadoboe/Developer/hwcat-reference")
ERA5 = Path("/Users/isaacadoboe/Developer/heatwave-initial-conditions/data/era5_europe")


def test_layout():
    w = Workdir(Path("/tmp/x"))
    assert w.era5_file(2003) == Path("/tmp/x/era5/europe_tmax_2003.nc")
    assert w.catalogue_file(2003, "_a50k_p90_w15") == \
        Path("/tmp/x/catalogues/heatwave_events_europe_2003_a50k_p90_w15.csv")
    assert w.labels_file(2003, "_a50k_p90_w15") == \
        Path("/tmp/x/catalogues/labels/heatwave_labels_europe_2003_a50k_p90_w15.npz")


def test_missing_era5_names_the_download_command():
    with tempfile.TemporaryDirectory() as d:
        try:
            C.run(1999, Workdir(Path(d)))
        except FileNotFoundError as e:
            assert "hwcat download" in str(e) and "1999" in str(e), e
        else:
            raise AssertionError("expected FileNotFoundError")


def test_even_window_refused():
    with tempfile.TemporaryDirectory() as d:
        try:
            C.run(2003, Workdir(Path(d)), window=30)
        except ValueError as e:
            assert "odd" in str(e), e
        else:
            raise AssertionError("expected ValueError")


def test_all_ocean_box_gives_empty_catalogue():
    """A box over the open Atlantic has no land cells: an empty catalogue."""
    with tempfile.TemporaryDirectory() as d:
        w = Workdir(Path(d), region="atlantic")
        w.era5.mkdir(parents=True)
        time = pd.date_range("2000-04-01", "2000-10-31", freq="D")
        lat, lon = np.arange(40, 35, -0.25), np.arange(-40, -35, 0.25)
        for y in range(1991, 2021):
            t = time.map(lambda x: x.replace(year=y))
            xr.DataArray(np.full((len(t), len(lat), len(lon)), 20.0, "float32"),
                         coords={"valid_time": t, "latitude": lat, "longitude": lon},
                         dims=("valid_time", "latitude", "longitude"),
                         name="t2m").to_netcdf(w.era5_file(y))
        out = C.run(2000, w, plot=False)
        assert len(pd.read_csv(out)) == 0, out


def test_matches_reference_2003():
    """Leave-one-out reference setting for 2003, byte for byte."""
    with tempfile.TemporaryDirectory() as d:
        w = Workdir(Path(d))
        w.era5.parent.mkdir(parents=True, exist_ok=True)
        w.era5.symlink_to(ERA5)
        out = C.run(2003, w, leave_one_out=True, save_labels=True, plot=False,
                    tag="_a50k_p90_w15")
        ref = REF / "catalogues" / out.name
        assert filecmp.cmp(out, ref, shallow=False), (out, ref)
        a = np.load(w.labels_file(2003, "_a50k_p90_w15"))
        b = np.load(REF / "catalogues" / "labels" / w.labels_file(2003, "_a50k_p90_w15").name)
        for k in ("t", "i", "j", "event_id", "shape"):
            assert np.array_equal(a[k], b[k]), k


if __name__ == "__main__":
    test_layout()
    test_missing_era5_names_the_download_command()
    test_even_window_refused()
    test_all_ocean_box_gives_empty_catalogue()
    test_matches_reference_2003()
    print("ok")
