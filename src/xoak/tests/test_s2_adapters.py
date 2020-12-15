import pytest
import xarray as xr

import xoak  # noqa:F401

pytest.importorskip('pys2index')


def test_s2point(geo_dataset, geo_indexer, geo_expected):
    geo_dataset.xoak.set_index(['lat', 'lon'], 's2point')
    ds_sel = geo_dataset.xoak.sel(lat=geo_indexer.latitude, lon=geo_indexer.longitude)

    xr.testing.assert_equal(ds_sel.load(), geo_expected.load())
