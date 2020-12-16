import pytest
import xarray as xr

import xoak  # noqa: F401

pytest.importorskip('sklearn')


def test_sklearn_kdtree(xyz_dataset, xyz_indexer, xyz_expected):
    xyz_dataset.xoak.set_index(['x', 'y', 'z'], 'sklearn_kdtree')
    ds_sel = xyz_dataset.xoak.sel(x=xyz_indexer.xx, y=xyz_indexer.yy, z=xyz_indexer.zz)

    xr.testing.assert_equal(ds_sel.load(), xyz_expected.load())


def test_sklearn_kdtree_options():
    ds = xr.Dataset(coords={'x': ('points', [1, 2]), 'y': ('points', [1, 2])})

    ds.xoak.set_index(['x', 'y'], 'sklearn_kdtree', leaf_size=10)

    # sklearn tree classes init options are not exposed as class properties
    assert ds.xoak._index._index_adapter._index_options == {'leaf_size': 10}


def test_sklearn_balltree(xyz_dataset, xyz_indexer, xyz_expected):
    xyz_dataset.xoak.set_index(['x', 'y', 'z'], 'sklearn_balltree')
    ds_sel = xyz_dataset.xoak.sel(x=xyz_indexer.xx, y=xyz_indexer.yy, z=xyz_indexer.zz)

    xr.testing.assert_equal(ds_sel.load(), xyz_expected.load())


def test_sklearn_balltree_options():
    ds = xr.Dataset(coords={'x': ('points', [1, 2]), 'y': ('points', [1, 2])})

    ds.xoak.set_index(['x', 'y'], 'sklearn_balltree', leaf_size=10)

    # sklearn tree classes init options are not exposed as class properties
    assert ds.xoak._index._index_adapter._index_options == {'leaf_size': 10}


def test_sklearn_geo_balltree(geo_dataset, geo_indexer, geo_expected):
    geo_dataset.xoak.set_index(['lat', 'lon'], 'sklearn_geo_balltree')
    ds_sel = geo_dataset.xoak.sel(lat=geo_indexer.latitude, lon=geo_indexer.longitude)

    xr.testing.assert_equal(ds_sel.load(), geo_expected.load())


def test_sklearn_geo_balltree_options():
    ds = xr.Dataset(coords={'x': ('points', [1, 2]), 'y': ('points', [1, 2])})

    ds.xoak.set_index(['x', 'y'], 'sklearn_geo_balltree', leaf_size=10, metric='euclidean')

    # sklearn tree classes init options are not exposed as class properties
    # user-defined metric should be ignored
    assert ds.xoak._index._index_adapter._index_options == {'leaf_size': 10, 'metric': 'haversine'}
