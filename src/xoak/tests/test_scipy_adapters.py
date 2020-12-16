import pytest
import xarray as xr

import xoak  # noqa: F401

pytest.importorskip('scipy')


def test_scipy_kdtree(xyz_dataset, xyz_indexer, xyz_expected):
    xyz_dataset.xoak.set_index(['x', 'y', 'z'], 'scipy_kdtree')
    ds_sel = xyz_dataset.xoak.sel(x=xyz_indexer.xx, y=xyz_indexer.yy, z=xyz_indexer.zz)

    xr.testing.assert_equal(ds_sel.load(), xyz_expected.load())


def test_scipy_kdtree_options():
    ds = xr.Dataset(coords={'x': ('points', [1, 2]), 'y': ('points', [1, 2])})

    ds.xoak.set_index(['x', 'y'], 'scipy_kdtree', leafsize=10)

    assert ds.xoak.index.leafsize == 10
