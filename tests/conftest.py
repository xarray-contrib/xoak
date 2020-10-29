import dask
import numpy as np
import pytest
import xarray as xr


@pytest.fixture(params=[np, dask.array])
def array_lib(request):
    """Array lib that is used for creation of the data and the indexer."""
    return request.param


@pytest.fixture(params=[(10, 10), (10, 1), (1000,), (3, 10, 5)])
def geo_dataset(request, array_lib):
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
def geo_indexer(request, array_lib):
    """Indexer dataset with coords longitude and latitude of parametrized shapes."""
    shape = request.param

    latitude = xr.DataArray(array_lib.random.uniform(-90, 90, size=shape))
    longitude = xr.DataArray(array_lib.random.uniform(-180, 180, size=shape))

    ds = xr.Dataset(coords={"longitude": longitude, "latitude": latitude})

    return ds
