import numpy as np
from sklearn.neighbors import BallTree

from .base import IndexWrapper, register_index


@register_index('balltree')
class BallTreeWrapper(IndexWrapper):
    index_cls = BallTree


@register_index('geo_balltree')
class GeoBallTreeWrapper(BallTreeWrapper):

    def build_index(self, points, **kwargs):
        kwargs.update({'metric': 'haversine'})
        return super().build_index(np.deg2rad(points), **kwargs)

    def query_index(self, points):
        return super().query_index(np.deg2rad(points))
