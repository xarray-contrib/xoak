import numpy as np
from sklearn.neighbors import BallTree, KDTree

from .base import IndexAdapter, register_default


@register_default('sklearn_kdtree')
class SklearnKDTreeAdapter(IndexAdapter):
    """Xoak index adapter for :class:`sklearn.neighbors.KDTree`."""

    def __init__(self, **kwargs):
        self._index_options = kwargs

    def build(self, points):
        return KDTree(points, **self._index_options)

    def query(self, kdtree, points):
        return kdtree.query(points)


@register_default('sklearn_balltree')
class SklearnBallTreeAdapter(IndexAdapter):
    """Xoak index adapter for :class:`sklearn.neighbors.BallTree`."""

    def __init__(self, **kwargs):
        self._index_options = kwargs

    def build(self, points):
        return BallTree(points, **self._index_options)

    def query(self, btree, points):
        return btree.query(points)


@register_default('sklearn_geo_balltree')
class SklearnGeoBallTreeAdapter(IndexAdapter):
    """Xoak index adapter for :class:`sklearn.neighbors.BallTree`, using
    the 'haversine' metric.

    It can be used for indexing a set of latitude / longitude points.

    When building the index, the coordinates must be given in the latitude,
    longitude order.

    Latitude and longitude values must be given in degrees for both index and
    query points (those values are converted in radians by this adapter).

    """

    def __init__(self, **kwargs):
        kwargs.update({'metric': 'haversine'})
        self._index_options = kwargs

    def build(self, points):
        return BallTree(np.deg2rad(points), **self._index_options)

    def query(self, btree, points):
        return btree.query(np.deg2rad(points))
