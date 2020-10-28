import numpy as np
import dask
import xarray as xr

import xoak

import pytest


@pytest.fixture(params=[np, dask.array])
def array_lib(request):
    """Array lib that is used for creation of the data and the indexer."""
    return request.param


@pytest.fixture(params=[(100, 100), (10, 1), (10_000,), (36, 18, 12)])
def dataset(request, array_lib):
    """Dataset with coords lon and lat on a grid of different shapes."""
    shape = request.param

    lat = xr.DataArray(array_lib.random.uniform(-90, 90, size=shape))
    lon = xr.DataArray(array_lib.random.uniform(-180, 180, size=shape))
    field = lat + lon  # artificial data of the same shape as lon and lat

    ds = xr.Dataset(
        coords={"lat": lat, "lon": lon},
        data_vars={"field": field},
    )

    return ds


@pytest.fixture(params=[(10,), (33, 69), (1_000,), (1, 2, 3, 4)])
def indexer(request, array_lib):
    """Indexer dataset with coords longitude and latitude of parametrized shapes."""
    shape = request.param

    latitude = xr.DataArray(array_lib.random.uniform(-90, 90, size=shape))
    longitude = xr.DataArray(array_lib.random.uniform(-180, 180, size=shape))

    ds = xr.Dataset(coords={"longitude": longitude, "latitude": latitude})

    return ds


def test_indexer(dataset, indexer, array_lib):
    """Select the dataset with positions from the indexer."""
    # create index
    dataset.xoak.set_index(
        ["lat", "lon"], transform=array_lib.deg2rad, metric="haversine"
    )

    # select with indexer
    ds_sel = dataset.xoak.sel(lat=indexer.latitude, lon=indexer.longitude)

    # ensure same shape
    assert ds_sel.field.shape == indexer.longitude.shape
