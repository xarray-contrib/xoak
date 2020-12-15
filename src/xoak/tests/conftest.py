import dask
import numpy as np
import pytest
import xarray as xr
from sklearn.metrics import pairwise_distances_argmin_min

# use single-threaded dask scheduler for all tests, as multi-threads or
# multi-processes may not be supported by some index adapters.
# TODO: enable multi-threaded and/or multi-processes per index
dask.config.set(scheduler='single-threaded')


@pytest.fixture(params=[np, dask.array], scope='session')
def array_lib(request):
    """Array lib that is used for creation of the data and the indexer."""
    return request.param


@pytest.fixture(params=[(200,), (20, 10), (4, 10, 5)], scope='session')
def geo_dataset(request, array_lib):
    """Dataset with coords lon and lat on a grid of different shapes."""
    shape = request.param

    lat = xr.DataArray(array_lib.random.uniform(-80, 80, size=shape))
    lon = xr.DataArray(array_lib.random.uniform(-160, 160, size=shape))

    ds = xr.Dataset(coords={'lat': lat, 'lon': lon})

    return ds


@pytest.fixture(params=[(100,), (10, 10), (2, 10, 5)], scope='session')
def geo_indexer(request, array_lib):
    """Indexer dataset with coords longitude and latitude of parametrized shapes."""
    shape = request.param

    latitude = xr.DataArray(array_lib.random.uniform(-80, 80, size=shape))
    longitude = xr.DataArray(array_lib.random.uniform(-160, 160, size=shape))

    ds = xr.Dataset(coords={'longitude': longitude, 'latitude': latitude})

    return ds


@pytest.fixture(scope='session')
def geo_expected(geo_dataset, geo_indexer):
    """Find nearest neighbors using brute-force approach."""
    X = np.stack([np.ravel(c) for c in (geo_indexer.latitude, geo_indexer.longitude)]).T
    Y = np.stack([np.ravel(c) for c in (geo_dataset.lat, geo_dataset.lon)]).T

    positions, _ = pairwise_distances_argmin_min(np.deg2rad(X), np.deg2rad(Y), metric='haversine')

    dataset_shape = geo_dataset.lat.shape
    dataset_dims = geo_dataset.lat.dims
    indexer_shape = geo_indexer.latitude.shape
    indexer_dims = geo_indexer.latitude.dims

    u_positions = list(np.unravel_index(positions.ravel(), dataset_shape))

    pos_indexers = {
        dim: xr.Variable(indexer_dims, ind.reshape(indexer_shape))
        for dim, ind in zip(dataset_dims, u_positions)
    }

    return geo_dataset.isel(indexers=pos_indexers)
