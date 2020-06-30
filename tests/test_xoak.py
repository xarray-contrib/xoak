import xarray as xr
import xoak

import pytest


def test_version():
    assert xoak.__version__ == "0.1.0"


def test_accessor_is_present():
    ds = xr.Dataset()
    assert hasattr(ds, "xoak")


def test_index_can_be_set_explicitly():
    """Check that index is not set before / is set after construction."""
    # no index should be set right after initialization
    dataset = xr.Dataset(coords={"x": (["x",], range(3))})
    assert dataset.xoak.index is None

    # index should be set after set_index was called
    dataset.xoak.set_index(
        ["x",]
    )
    assert dataset.xoak.index is not None


def test_selecting_before_constructing_index_raises():
    """Ensure correct error if sel is called before index contruction."""
    dataset = xr.Dataset(coords={"x": (["x",], range(3))})
    in_message = r".*Call `\.xoak\.set_index\(\)` first.*"
    with pytest.raises(ValueError, match=in_message):
        dataset.xoak.sel(x=[1.5, 2.5])


def test_ensure_different_indexer_dims_raises():
    """Ensure error if multiple indexers are of different size."""
    dataset = xr.Dataset(
        coords={"x": (["pt",], range(3)), "y": (["pt",], range(-3, 0))}
    )
    dataset.xoak.set_index(["x", "y"])
    with pytest.raises(ValueError):
        dataset.xoak.sel(x=[1, 2], y=[-1,])
