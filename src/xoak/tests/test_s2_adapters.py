import sys

import numpy as np
import pytest
import xarray as xr

import xoak  # noqa:F401

pytest.importorskip('pys2index')


def test_s2point(geo_dataset, geo_indexer, geo_expected):
    geo_dataset.xoak.set_index(['lat', 'lon'], 's2point')
    ds_sel = geo_dataset.xoak.sel(lat=geo_indexer.latitude, lon=geo_indexer.longitude)

    xr.testing.assert_equal(ds_sel.load(), geo_expected.load())


def test_s2point_sizeof():
    ds = xr.Dataset(coords={'lat': ('points', [0.0, 10.0]), 'lon': ('points', [-5.0, 5.0])})
    points = np.array([[0.0, -5.0], [10.0, 5.0]])

    ds.xoak.set_index(['lat', 'lon'], 's2point')

    assert sys.getsizeof(ds.xoak._index._index_adapter) > points.nbytes
