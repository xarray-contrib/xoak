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
    """Dataset with cartesian coords x, y, z on a grid of different shapes."""
    shape = request.param

    z = xr.DataArray(array_lib.random.uniform(-10, 10, size=shape))
    y = xr.DataArray(array_lib.random.uniform(0, 100, size=shape))
    x = xr.DataArray(array_lib.random.uniform(-50, 50, size=shape))
    field = z + y + x  # artificial data of the same shape as x and y, and z

    ds = xr.Dataset(coords={"z": z, "y": y, "x": x}, data_vars={"field": field},)

    return ds


@pytest.fixture(params=[(10,), (33, 69), (1_000,), (1, 2, 3, 4)])
def indexer(request, array_lib):
    """Indexer dataset with coords x, y, z of parametrized shapes."""
    shape = request.param

    z_points = xr.DataArray(array_lib.random.uniform(-10, 10, size=shape))
    y_points = xr.DataArray(array_lib.random.uniform(0, 100, size=shape))
    x_points = xr.DataArray(array_lib.random.uniform(-50, 50, size=shape))

    ds = xr.Dataset(
        coords={"x_points": x_points, "y_points": y_points, "z_points": z_points}
    )

    return ds


@pytest.mark.parametrize("metric", ["minkowski", None])
def test_indexer(dataset, indexer, array_lib, metric):
    """Select the dataset with positions from the indexer."""
    # create index
    if metric is not None:
        kwargs = {"metric": metric}
    else:
        kwargs = {}
    dataset.xoak.set_index(["z", "y", "x"], **kwargs)

    # select with indexer
    ds_sel = dataset.xoak.sel(
        z=indexer.z_points, y=indexer.y_points, x=indexer.x_points
    )

    # ensure same shape
    assert ds_sel.field.shape == indexer.x_points.shape
