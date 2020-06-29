import xarray
import xoak


def test_version():
    assert xoak.__version__ == "0.1.0"


def test_accessor_is_present():
    ds = xarray.Dataset()
    assert hasattr(ds, "xoak")
