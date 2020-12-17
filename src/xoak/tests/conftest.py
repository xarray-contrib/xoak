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
def dataset_array_lib(request):
    """Array lib that is used for creation of the data."""
    return request.param


@pytest.fixture(params=[np, dask.array], scope='session')
def indexer_array_lib(request):
    """Array lib that is used for creation of the indexer."""
    return request.param


@pytest.fixture(
    params=[(('d1',), (200,)), (('d1', 'd2'), (20, 10)), (('d1', 'd2', 'd3'), (4, 10, 5))],
    scope='session',
)
def dataset_dims_shape(request):
    return request.param


@pytest.fixture(
    params=[(('i1',), (100,)), (('i1', 'i2'), (10, 10)), (('i1', 'i2', 'i3'), (2, 10, 5))],
    scope='session',
)
def indexer_dims_shape(request):
    return request.param


def query_brute_force(dataset, dataset_dims_shape, indexer, indexer_dims_shape, metric='euclidean'):
    """Find nearest neighbors using brute-force approach."""

    # for lat/lon coordinate, assume they are ordered lat, lon!!
    X = np.stack([np.ravel(c) for c in indexer.coords.values()]).T
    Y = np.stack([np.ravel(c) for c in dataset.coords.values()]).T

    if metric == 'haversine':
        X = np.deg2rad(X)
        Y = np.deg2rad(Y)

    positions, _ = pairwise_distances_argmin_min(X, Y, metric=metric)

    dataset_dims, dataset_shape = dataset_dims_shape
    indexer_dims, indexer_shape = indexer_dims_shape

    u_positions = list(np.unravel_index(positions.ravel(), dataset_shape))

    pos_indexers = {
        dim: xr.Variable(indexer_dims, ind.reshape(indexer_shape))
        for dim, ind in zip(dataset_dims, u_positions)
    }

    return dataset.isel(indexers=pos_indexers)


@pytest.fixture(scope='session')
def geo_dataset(dataset_dims_shape, dataset_array_lib):
    """Dataset with coords lon and lat on a grid of different shapes."""
    dims, shape = dataset_dims_shape

    lat = xr.DataArray(dataset_array_lib.random.uniform(-80, 80, size=shape), dims=dims)
    lon = xr.DataArray(dataset_array_lib.random.uniform(-160, 160, size=shape), dims=dims)

    ds = xr.Dataset(coords={'lat': lat, 'lon': lon})

    return ds


@pytest.fixture(scope='session')
def geo_indexer(indexer_dims_shape, indexer_array_lib):
    """Indexer dataset with coords longitude and latitude of parametrized shapes."""
    dims, shape = indexer_dims_shape

    latitude = xr.DataArray(indexer_array_lib.random.uniform(-80, 80, size=shape), dims=dims)
    longitude = xr.DataArray(indexer_array_lib.random.uniform(-160, 160, size=shape), dims=dims)

    ds = xr.Dataset(coords={'latitude': latitude, 'longitude': longitude})

    return ds


@pytest.fixture(scope='session')
def geo_expected(geo_dataset, dataset_dims_shape, geo_indexer, indexer_dims_shape):
    return query_brute_force(
        geo_dataset, dataset_dims_shape, geo_indexer, indexer_dims_shape, metric='haversine'
    )


@pytest.fixture(scope='session')
def xyz_dataset(dataset_dims_shape, dataset_array_lib):
    """Dataset with coords x, y, z on a grid of different shapes."""
    dims, shape = dataset_dims_shape

    x = xr.DataArray(dataset_array_lib.random.uniform(0, 10, size=shape), dims=dims)
    y = xr.DataArray(dataset_array_lib.random.uniform(0, 10, size=shape), dims=dims)
    z = xr.DataArray(dataset_array_lib.random.uniform(0, 10, size=shape), dims=dims)

    ds = xr.Dataset(coords={'x': x, 'y': y, 'z': z})

    return ds


@pytest.fixture(scope='session')
def xyz_indexer(indexer_dims_shape, indexer_array_lib):
    """Indexer dataset with coords xx, yy, zz of parametrized shapes."""
    dims, shape = indexer_dims_shape

    xx = xr.DataArray(indexer_array_lib.random.uniform(0, 10, size=shape), dims=dims)
    yy = xr.DataArray(indexer_array_lib.random.uniform(0, 10, size=shape), dims=dims)
    zz = xr.DataArray(indexer_array_lib.random.uniform(0, 10, size=shape), dims=dims)

    ds = xr.Dataset(coords={'xx': xx, 'yy': yy, 'zz': zz})

    return ds


@pytest.fixture(scope='session')
def xyz_expected(xyz_dataset, dataset_dims_shape, xyz_indexer, indexer_dims_shape):
    return query_brute_force(xyz_dataset, dataset_dims_shape, xyz_indexer, indexer_dims_shape)
