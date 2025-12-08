from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from xarray.indexes.nd_point_index import TreeAdapter  # type: ignore
except ImportError:

    class TreeAdapter: ...


if TYPE_CHECKING:
    import pys2index
    import sklearn.neighbors


class S2PointTreeAdapter(TreeAdapter):
    """:py:class:`pys2index.S2PointIndex` adapter for :py:class:`~xarray.indexes.NDPointIndex`."""

    _s2point_index: pys2index.S2PointIndex

    def __init__(self, points: np.ndarray, options: Mapping[str, Any]):
        from pys2index import S2PointIndex

        self._s2point_index = S2PointIndex(points)

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._s2point_index.query(points)

    def equals(self, other: S2PointTreeAdapter) -> bool:
        return np.array_equal(
            self._s2point_index.get_cell_ids(), other._s2point_index.get_cell_ids()
        )


class SklearnKDTreeAdapter(TreeAdapter):
    """:py:class:`sklearn.neighbors.KDTree` adapter for :py:class:`~xarray.indexes.NDPointIndex`."""

    _kdtree: sklearn.neighbors.KDTree

    def __init__(self, points: np.ndarray, options: Mapping[str, Any]):
        from sklearn.neighbors import KDTree

        self._kdtree = KDTree(points, **options)

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._kdtree.query(points)

    def equals(self, other: SklearnKDTreeAdapter) -> bool:
        return np.array_equal(self._kdtree.data, other._kdtree.data)


class SklearnBallTreeAdapter(TreeAdapter):
    """:py:class:`sklearn.neighbors.BallTree` adapter for
    :py:class:`~xarray.indexes.NDPointIndex`.

    """

    _balltree: sklearn.neighbors.BallTree

    def __init__(self, points: np.ndarray, options: Mapping[str, Any]):
        from sklearn.neighbors import BallTree

        self._balltree = BallTree(points, **options)

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._balltree.query(points)

    def equals(self, other: SklearnBallTreeAdapter) -> bool:
        return np.array_equal(self._balltree.data, other._balltree.data)


class SklearnGeoBallTreeAdapter(TreeAdapter):
    """:py:class:`sklearn.neighbors.BallTree` adapter for
    :py:class:`~xarray.indexes.NDPointIndex`, using the 'haversine' metric.

    It can be used for indexing a set of latitude / longitude points.

    When building the index, the coordinates must be given in the latitude,
    longitude order.

    Latitude and longitude values must be given in degrees for both index and
    query points (those values are converted in radians by this adapter).

    """

    _balltree: sklearn.neighbors.BallTree

    def __init__(self, points: np.ndarray, options: Mapping[str, Any]):
        from sklearn.neighbors import BallTree

        opts = dict(options)
        opts.update({"metric": "haversine"})

        self._balltree = BallTree(np.deg2rad(points), **opts)

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._balltree.query(np.deg2rad(points))

    def equals(self, other: SklearnGeoBallTreeAdapter) -> bool:
        return np.array_equal(self._balltree.data, other._balltree.data)
