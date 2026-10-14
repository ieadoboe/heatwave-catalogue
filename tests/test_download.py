"""The downloader with a fake Copernicus client: the region reaches the
request, files are split per year, and years on disk are skipped.

    python tests/test_download.py
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hwcat import download as DL
from hwcat.workdir import Workdir


class FakeClient:
    def __init__(self):
        self.requests = []

    def retrieve(self, dataset, request, target):
        self.requests.append(request)
        years = [int(y) for y in request["year"]]
        t = pd.DatetimeIndex([pd.Timestamp(y, int(m), 1)
                              for y in years for m in request["month"]])
        xr.Dataset({"t2m": (("valid_time", "latitude", "longitude"),
                            np.zeros((len(t), 2, 2), "float32"))},
                   coords={"valid_time": t, "latitude": [1.0, 0.0],
                           "longitude": [0.0, 1.0]}).to_netcdf(target)


def test_region_and_per_year_files():
    with tempfile.TemporaryDirectory() as d:
        w = Workdir(Path(d), region="usa")
        fake = FakeClient()
        DL.download([2001, 2002], w, [50, -130, 25, -60], client=fake)
        assert fake.requests[0]["area"] == [50, -130, 25, -60], fake.requests
        assert w.era5_file(2001).exists() and w.era5_file(2002).exists()
        DL.download([2001, 2002], w, [50, -130, 25, -60], client=fake)
        assert len(fake.requests) == 1, "years on disk must be skipped"


if __name__ == "__main__":
    test_region_and_per_year_files()
    print("ok")
