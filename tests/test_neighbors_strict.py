import numpy as np
import xarray as xr

import xoak


def test_correct_nn_cartesian():
    # create four-point grid
    y = xr.DataArray(np.array([[0, 0], [1, 1]]))
    x = xr.DataArray(np.array([[0, 1], [0, 1]]))
    field = xr.DataArray(np.array([[0, 1], [2, 3]]))
    dataset = xr.Dataset(coords={"y": y, "x": x}, data_vars={"field": field},)

    # create indexer
    # we want to sample the poilnts
    y_points = xr.DataArray(np.array([0.1, 0.1, 0.9, 0.9]))
    x_points = xr.DataArray(np.array([0.1, 0.9, 0.1, 0.9]))
    indexer = xr.Dataset(coords={"y_points": y_points, "x_points": x_points})

    # create index
    dataset.xoak.set_index(["y", "x"])

    # select data and assert that the returned values are [0, 1, 2, 3]
    ds_sel = dataset.xoak.sel(y=indexer.y_points, x=indexer.x_points)
    assert all(fi == truth for fi, truth in zip(ds_sel.field, [0, 1, 2, 3]))


def test_correct_nn_latlon():
    """Test that we're honoring the date line."""
    # create four-point grid
    lat = xr.DataArray(np.array([[-30, -30], [30, 30]]))
    lon = xr.DataArray(np.array([[-180, 0], [180, 0]]))
    field = xr.DataArray(np.array([[0, 1], [2, 3]]))
    dataset = xr.Dataset(coords={"lat": lat, "lon": lon}, data_vars={"field": field},)

    # create indexer
    # we want to sample the points at the date line
    latitude = xr.DataArray(np.array([-29, -31, 32, 31]))
    longitude = xr.DataArray(np.array([-181, 179, 181, 540]))
    indexer = xr.Dataset(coords={"latitude": latitude, "longitude": longitude})

    # create index
    dataset.xoak.set_index(["lat", "lon"], metric="haversine", transform=np.deg2rad)

    # select data and assert that the returned values are [0, 0, 2, 2]
    ds_sel = dataset.xoak.sel(lat=indexer.latitude, lon=indexer.longitude)
    assert all(fi == truth for fi, truth in zip(ds_sel.field, [0, 0, 2, 2]))
