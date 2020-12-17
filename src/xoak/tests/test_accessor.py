import pytest
import xarray as xr
from scipy.spatial import cKDTree

import xoak  # noqa: F401


def test_set_index_error():
    ds = xr.Dataset(
        coords={
            'x': (('a', 'b'), [[0, 1], [2, 3]]),
            'y': (('b', 'a'), [[0, 1], [2, 3]]),
            'z': ('a', [0, 1]),
        }
    )

    with pytest.raises(ValueError, match='.*must all have the same dimensions.*'):
        ds.xoak.set_index(['x', 'y'], 'scipy_kdtree')

    with pytest.raises(ValueError, match='.*must all have the same dimensions.*'):
        ds.xoak.set_index(['x', 'z'], 'scipy_kdtree')


def test_set_index_persist_false():
    ds = xr.Dataset(
        coords={
            'x': ('a', [0, 1, 2, 3]),
            'y': ('a', [0, 1, 2, 3]),
        }
    )
    ds = ds.chunk(2)

    ds.xoak.set_index(['x', 'y'], 'scipy_kdtree', persist=False)

    assert isinstance(ds.xoak._index, tuple)


def test_sel_error():
    ds = xr.Dataset(
        coords={
            'x': ('a', [0, 1, 2, 3]),
            'y': ('a', [0, 1, 2, 3]),
        }
    )
    indexer = xr.Dataset(
        coords={
            'x': ('p', [1.2, 2.9]),
            'y': ('p', [1.2, 2.9]),
        }
    )

    with pytest.raises(ValueError, match='.*not been built yet.*'):
        ds.xoak.sel(x=indexer.x, y=indexer.y)

    ds.xoak.set_index(['x', 'y'], 'scipy_kdtree')
    indexer = xr.Dataset(
        coords={
            'x': ('p', [1.2, 2.9]),
            'y': ('p2', [1.2, 2.9]),
        }
    )

    with pytest.raises(ValueError, match='.*must have the same dimensions.*'):
        ds.xoak.sel(x=indexer.x, y=indexer.y)


def test_index_property():
    ds = xr.Dataset(
        coords={
            'x': ('a', [0, 1, 2, 3]),
            'y': ('a', [0, 1, 2, 3]),
        }
    )

    assert ds.xoak.index is None

    ds.xoak.set_index(['x', 'y'], 'scipy_kdtree')
    assert isinstance(ds.xoak.index, cKDTree)

    ds_chunk = ds.chunk(2)
    ds_chunk.xoak.set_index(['x', 'y'], 'scipy_kdtree')
    assert isinstance(ds_chunk.xoak.index, list)
