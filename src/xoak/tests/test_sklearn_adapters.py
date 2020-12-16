import pytest
import xarray as xr

import xoak  # noqa:F401

pytest.importorskip('sklearn')


def test_balltree(xyz_dataset, xyz_indexer, xyz_expected):
    xyz_dataset.xoak.set_index(['x', 'y', 'z'], 'sklearn_balltree')
    ds_sel = xyz_dataset.xoak.sel(x=xyz_indexer.xx, y=xyz_indexer.yy, z=xyz_indexer.zz)

    xr.testing.assert_equal(ds_sel.load(), xyz_expected.load())


def test_geo_balltree(geo_dataset, geo_indexer, geo_expected):
    geo_dataset.xoak.set_index(['lat', 'lon'], 'sklearn_geo_balltree')
    ds_sel = geo_dataset.xoak.sel(lat=geo_indexer.latitude, lon=geo_indexer.longitude)

    xr.testing.assert_equal(ds_sel.load(), geo_expected.load())
