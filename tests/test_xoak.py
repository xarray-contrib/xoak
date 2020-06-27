from xoak import __version__

import numpy as np
import xarray as xr

import xoak


def test_version():
    assert __version__ == "0.1.0"


def test_index_creation():
    # create dataset
    lat = [-70, -70, 55, 55]
    lon = [-180, 30, 30, -180]
    field = [2, 3, 1, 5]
    ds = xr.Dataset(
        coords={"lat": (("node",), lat), "lon": (("node",), lon)},
        data_vars={"field": (("node",), field)},
    )

    # create index
    ds.xoak.set_index(["lat", "lon"], transform=np.deg2rad, metric="haversine")

