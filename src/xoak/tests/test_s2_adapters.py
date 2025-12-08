import sys

import numpy as np
import pytest
import xarray as xr

import xoak  # noqa:F401

pytest.importorskip("pys2index")


def test_s2point(geo_dataset, geo_indexer, geo_expected):
    geo_dataset.xoak.set_index(["lat", "lon"], "s2point")
    ds_sel = geo_dataset.xoak.sel(lat=geo_indexer.latitude, lon=geo_indexer.longitude)

    xr.testing.assert_equal(ds_sel.load(), geo_expected.load())


def test_ndpointindex_s2point(
    geo_dataset, geo_indexer, geo_expected, dataset_array_lib, indexer_array_lib
):
    # TODO: remove when refactoring fixtures without dask
    if dataset_array_lib is not np or indexer_array_lib is not np:
        pytest.skip()

    geo_dataset = geo_dataset.compute()
    geo_indexer = geo_indexer.compute()

    # TODO: remove when https://github.com/pydata/xarray/issues/10940 is fixed
    if geo_dataset.lat.ndim != 2:
        pytest.skip()

    geo_dataset = geo_dataset.set_xindex(
        ["lat", "lon"], xr.indexes.NDPointIndex, tree_adapter_cls=xoak.S2PointTreeAdapter
    )
    ds_sel = geo_dataset.sel(lat=geo_indexer.latitude, lon=geo_indexer.longitude, method="nearest")

    xr.testing.assert_equal(ds_sel, geo_expected)

    # NDPointIndex.equals() should return True (via merge)
    xr.testing.assert_identical(xr.merge([geo_dataset, geo_dataset]), geo_dataset)


def test_s2point_sizeof():
    ds = xr.Dataset(coords={"lat": ("points", [0.0, 10.0]), "lon": ("points", [-5.0, 5.0])})
    points = np.array([[0.0, -5.0], [10.0, 5.0]])

    ds.xoak.set_index(["lat", "lon"], "s2point")

    assert sys.getsizeof(ds.xoak._index._index_adapter) > points.nbytes
