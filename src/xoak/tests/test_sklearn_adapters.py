import numpy as np
import pytest
import xarray as xr

import xoak  # noqa: F401

pytest.importorskip("sklearn")


def test_sklearn_kdtree(xyz_dataset, xyz_indexer, xyz_expected):
    xyz_dataset.xoak.set_index(["x", "y", "z"], "sklearn_kdtree")
    ds_sel = xyz_dataset.xoak.sel(x=xyz_indexer.xx, y=xyz_indexer.yy, z=xyz_indexer.zz)

    xr.testing.assert_equal(ds_sel.load(), xyz_expected.load())


def test_ndpointindex_kdtree(
    xyz_dataset, xyz_indexer, xyz_expected, dataset_array_lib, indexer_array_lib
):
    # TODO: remove when refactoring fixtures without dask
    if dataset_array_lib is not np or indexer_array_lib is not np:
        pytest.skip()

    xyz_dataset = xyz_dataset.compute()
    xyz_indexer = xyz_indexer.compute()
    xyz_expected = xyz_expected.compute()

    # TODO: remove when https://github.com/pydata/xarray/issues/10940 is fixed
    if xyz_dataset.x.ndim != 3:
        pytest.skip()

    xyz_dataset = xyz_dataset.set_xindex(
        ["x", "y", "z"], xr.indexes.NDPointIndex, tree_adapter_cls=xoak.SklearnKDTreeAdapter
    )
    ds_sel = xyz_dataset.sel(x=xyz_indexer.xx, y=xyz_indexer.yy, z=xyz_indexer.zz, method="nearest")

    xr.testing.assert_equal(ds_sel, xyz_expected)

    # NDPointIndex.equals() should return True (via merge)
    xr.testing.assert_identical(xr.merge([xyz_dataset, xyz_dataset]), xyz_dataset)


def test_sklearn_kdtree_options():
    ds = xr.Dataset(coords={"x": ("points", [1, 2]), "y": ("points", [1, 2])})

    ds.xoak.set_index(["x", "y"], "sklearn_kdtree", leaf_size=10)

    # sklearn tree classes init options are not exposed as class properties
    assert ds.xoak._index._index_adapter._index_options == {"leaf_size": 10}


def test_sklearn_balltree(xyz_dataset, xyz_indexer, xyz_expected):
    xyz_dataset.xoak.set_index(["x", "y", "z"], "sklearn_balltree")
    ds_sel = xyz_dataset.xoak.sel(x=xyz_indexer.xx, y=xyz_indexer.yy, z=xyz_indexer.zz)

    xr.testing.assert_equal(ds_sel.load(), xyz_expected.load())


def test_ndpointindex_balltree(
    xyz_dataset, xyz_indexer, xyz_expected, dataset_array_lib, indexer_array_lib
):
    # TODO: remove when refactoring fixtures without dask
    if dataset_array_lib is not np or indexer_array_lib is not np:
        pytest.skip()

    xyz_dataset = xyz_dataset.compute()
    xyz_indexer = xyz_indexer.compute()
    xyz_expected = xyz_expected.compute()

    # TODO: remove when https://github.com/pydata/xarray/issues/10940 is fixed
    if xyz_dataset.x.ndim != 3:
        pytest.skip()

    xyz_dataset = xyz_dataset.set_xindex(
        ["x", "y", "z"], xr.indexes.NDPointIndex, tree_adapter_cls=xoak.SklearnBallTreeAdapter
    )
    ds_sel = xyz_dataset.sel(x=xyz_indexer.xx, y=xyz_indexer.yy, z=xyz_indexer.zz, method="nearest")

    xr.testing.assert_equal(ds_sel, xyz_expected)

    # NDPointIndex.equals() should return True (via merge)
    xr.testing.assert_identical(xr.merge([xyz_dataset, xyz_dataset]), xyz_dataset)


def test_sklearn_balltree_options():
    ds = xr.Dataset(coords={"x": ("points", [1, 2]), "y": ("points", [1, 2])})

    ds.xoak.set_index(["x", "y"], "sklearn_balltree", leaf_size=10)

    # sklearn tree classes init options are not exposed as class properties
    assert ds.xoak._index._index_adapter._index_options == {"leaf_size": 10}


def test_sklearn_geo_balltree(geo_dataset, geo_indexer, geo_expected):
    geo_dataset.xoak.set_index(["lat", "lon"], "sklearn_geo_balltree")
    ds_sel = geo_dataset.xoak.sel(lat=geo_indexer.latitude, lon=geo_indexer.longitude)

    xr.testing.assert_equal(ds_sel.load(), geo_expected.load())


def test_ndpointindex_geo_balltree(
    geo_dataset, geo_indexer, geo_expected, dataset_array_lib, indexer_array_lib
):
    # TODO: remove when refactoring fixtures without dask
    if dataset_array_lib is not np or indexer_array_lib is not np:
        pytest.skip()

    geo_dataset = geo_dataset.compute()
    geo_indexer = geo_indexer.compute()
    geo_expected = geo_expected.compute()

    # TODO: remove when https://github.com/pydata/xarray/issues/10940 is fixed
    if geo_dataset.lat.ndim != 2:
        pytest.skip()

    geo_dataset = geo_dataset.set_xindex(
        ["lat", "lon"], xr.indexes.NDPointIndex, tree_adapter_cls=xoak.SklearnGeoBallTreeAdapter
    )
    ds_sel = geo_dataset.sel(lat=geo_indexer.latitude, lon=geo_indexer.longitude, method="nearest")

    xr.testing.assert_equal(ds_sel, geo_expected)

    # NDPointIndex.equals() should return True (via merge)
    xr.testing.assert_identical(xr.merge([geo_dataset, geo_dataset]), geo_dataset)


def test_sklearn_geo_balltree_options():
    ds = xr.Dataset(coords={"x": ("points", [1, 2]), "y": ("points", [1, 2])})

    ds.xoak.set_index(["x", "y"], "sklearn_geo_balltree", leaf_size=10, metric="euclidean")

    # sklearn tree classes init options are not exposed as class properties
    # user-defined metric should be ignored
    assert ds.xoak._index._index_adapter._index_options == {
        "leaf_size": 10,
        "metric": "haversine",
    }
