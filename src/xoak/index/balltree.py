import numpy as np
from sklearn.neighbors import BallTree

from .base import IndexAdapter, register_default


@register_default('balltree')
class BallTreeAdapter(IndexAdapter):
    def __init__(self, **kwargs):
        self.index_options = kwargs

    def build(self, points):
        return BallTree(points, **self.index_options)

    def query(self, btree, points):
        return btree.query(points)


@register_default('geo_balltree')
class GeoBallTreeAdapter(IndexAdapter):
    def __init__(self, **kwargs):
        kwargs.update({'metric': 'haversine'})
        self._index_options = kwargs

    def build(self, points):
        return BallTree(np.deg2rad(points), **self._index_options)

    def query(self, btree, points):
        return btree.query(np.deg2rad(points))
