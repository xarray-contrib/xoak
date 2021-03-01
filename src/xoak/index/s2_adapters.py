from pys2index import S2PointIndex

from .base import IndexAdapter, register_default


@register_default('s2point')
class S2PointIndexAdapter(IndexAdapter):
    """Xoak index adapter for :class:`pys2index.S2PointIndex`.

    It can be used for efficient indexing of a set of latitude / longitude points.

    When building the index, the coordinates must be given in the latitude,
    longitude order.

    Latitude and longitude values must be in degrees for both index and query
    points.

    See https://github.com/benbovy/pys2index.

    """

    def build(self, points):
        self._points_nbytes = points.nbytes
        return S2PointIndex(points)

    def query(self, s2index, points):
        return s2index.query(points)

    def __sizeof__(self):
        # a very crude approx. of the index memory consumption, useful for Dask.
        # Unfortunately, we cannot get the actual size of the underlying index, as
        # the internal data structures used by the index are not exposed to Python.
        return self._points_nbytes
