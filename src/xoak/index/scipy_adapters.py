from scipy.spatial import cKDTree

from .base import IndexAdapter, register_default


@register_default('scipy_kdtree')
class ScipyKDTreeAdapter(IndexAdapter):
    """Xoak index adapter for :class:`scipy.spatial.cKDTree`."""

    def __init__(self, **kwargs):
        self.index_options = kwargs

    def build(self, points):
        return cKDTree(points, **self.index_options)

    def query(self, kdtree, points, query_kwargs=None):
        if query_kwargs is None:
            query_kwargs = {}
        return kdtree.query(points, **query_kwargs)
