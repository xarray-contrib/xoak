from typing import Any, Hashable, Iterable, List, Mapping, Tuple, Type, Union

import numpy as np
import xarray as xr
from xarray.core.utils import either_dict_or_kwargs

from .index.base import Index, IndexAdapter, XoakIndexWrapper

try:
    from dask.delayed import Delayed
except ImportError:  # pragma: no cover
    Delayed = Type[None]


def coords_to_point_array(coords: List[Any]) -> np.ndarray:
    """Re-arrange data from a list of xarray coordinates into a 2-d array of shape
    (npoints, ncoords).

    """
    c_chunks = [c.chunks for c in coords]

    if any([chunks is None for chunks in c_chunks]):
        # plain numpy arrays (maybe triggers compute)
        X = np.stack([np.ravel(c) for c in coords]).T

    else:
        import dask.array as da

        # TODO: check chunks are equal for all coords?

        X = da.stack([da.ravel(c.data) for c in coords]).T
        X = X.rechunk((X.chunks[0], len(coords)))

    return X


IndexAttr = Union[XoakIndexWrapper, Iterable[XoakIndexWrapper], Iterable[Delayed]]
IndexType = Union[str, Type[IndexAdapter]]


@xr.register_dataarray_accessor('xoak')
@xr.register_dataset_accessor('xoak')
class XoakAccessor:
    """A xarray Dataset or DataArray extension for indexing irregular,
    n-dimensional data using a ball tree.

    """

    _index: IndexAttr
    _index_type: IndexType
    _index_coords: Tuple[str]
    _index_coords_dims: Tuple[Hashable, ...]
    _index_coords_shape: Tuple[int, ...]

    def __init__(self, xarray_obj: Union[xr.Dataset, xr.DataArray]):
        self._xarray_obj = xarray_obj

    def _build_index_forest_delayed(self, X, persist=False, **kwargs) -> IndexAttr:
        import dask

        indexes = []
        offset = 0

        for i, chunk in enumerate(X.to_delayed().ravel()):
            indexes.append(
                dask.delayed(XoakIndexWrapper)(self._index_type, chunk, offset, **kwargs)
            )
            offset += X.chunks[0][i]

        if persist:
            return dask.persist(*indexes)
        else:
            return tuple(indexes)

    def set_index(
        self, coords: Iterable[str], index_type: IndexType, persist: bool = True, **kwargs
    ):
        """Create an index tree from a subset of coordinates of the DataArray / Dataset.

        If the given coordinates are chunked (Dask arrays), this method will (lazily) create
        a forest of index trees (one tree per chunk of the flattened coordinate arrays).

        Parameters
        ----------
        coords : iterable
            Coordinate names. Each given coordinate must have the same dimension(s),
            in the same order.
        index_type : str or :class:`~xoak.IndexAdapter` subclass
            Either a registered index adapter (see :class:`~xoak.IndexRegistry`) or a custom
            :class:`~xoak.IndexAdapter` subclass.
        persist: bool
            If True (default), this method will precompute and persist in memory the forest
            of index trees, if any.
        **kwargs
            Keyword arguments that will be passed to the underlying index constructor.

        """
        self._index_type = index_type
        self._index_coords = tuple(coords)

        coord_objs = [self._xarray_obj.coords[cn] for cn in coords]

        if len(set([c.dims for c in coord_objs])) > 1:
            raise ValueError(
                'Coordinates {coords} must all have the same dimensions in the same order'
            )

        self._index_coords_dims = coord_objs[0].dims
        self._index_coords_shape = coord_objs[0].shape

        X = coords_to_point_array([self._xarray_obj[c] for c in coords])

        if isinstance(X, np.ndarray):
            self._index = XoakIndexWrapper(self._index_type, X, 0, **kwargs)
        else:
            self._index = self._build_index_forest_delayed(X, persist=persist, **kwargs)

    @property
    def index(self) -> Union[None, Index, Iterable[Index]]:
        """Returns the underlying index object(s), or ``None`` if no index has
        been set yet.

        May trigger computation of lazy indexes.

        """
        if not getattr(self, '_index', False):
            return None
        elif isinstance(self._index, XoakIndexWrapper):
            return self._index.index
        else:
            import dask

            index_wrappers = dask.compute(*self._index)
            return [wrp.index for wrp in index_wrappers]

    def _query(self, indexers):
        X = coords_to_point_array([indexers[c] for c in self._index_coords])

        if isinstance(X, np.ndarray) and isinstance(self._index, XoakIndexWrapper):
            # directly call index wrapper's query method
            res = self._index.query(X)
            results = res['indices'][:, 0]

        else:
            # Two-stage lazy query with dask
            import dask
            import dask.array as da

            # coerce query array as a dask array and index(es) as an iterable
            if isinstance(X, np.ndarray):
                X = da.from_array(X, chunks=X.shape)

            if isinstance(self._index, XoakIndexWrapper):
                indexes = [self._index]
            else:
                indexes = self._index

            # 1st "map" stage:
            # - execute `IndexWrapperCls.query` for each query array chunk and each index instance
            # - concatenate all distances/positions results in two dask arrays of shape (n_points, n_indexes)

            res_chunk = []

            for i, chunk in enumerate(X.to_delayed().ravel()):
                res_chunk_idx = []

                chunk_npoints = X.chunks[0][i]
                shape = (chunk_npoints, 1)

                for idx in indexes:
                    dlyd = dask.delayed(idx.query)(chunk)
                    res_chunk_idx.append(
                        da.from_delayed(dlyd, shape, dtype=XoakIndexWrapper._query_result_dtype)
                    )

                res_chunk.append(da.concatenate(res_chunk_idx, axis=1))

            map_results = da.concatenate(res_chunk, axis=0)
            distances = map_results['distances']
            indices = map_results['indices']

            # 2nd "reduce" stage:
            # - brute force lookup over the indexes dimension (columns)

            indices_col = da.argmin(distances, axis=1)

            results = da.blockwise(
                lambda arr, icol: np.take_along_axis(arr, icol[:, None], 1),
                'i',
                indices,
                'ik',
                indices_col,
                'i',
                dtype=np.intp,
                concatenate=True,
            )

        return results

    def _get_pos_indexers(self, indices, indexers):
        """Returns positional indexers based on the query results and the
        original (label-based) indexers.

        1. Unravel the (flattened) indices returned from the query
        2. Reshape the unraveled indices according to indexers shapes
        3. Wrap the indices in xarray.Variable objects.

        """
        pos_indexers = {}

        indexer_dims = [idx.dims for idx in indexers.values()]
        indexer_shapes = [idx.shape for idx in indexers.values()]

        if len(set(indexer_dims)) > 1:
            raise ValueError('All indexers must have the same dimensions.')

        u_indices = list(np.unravel_index(indices.ravel(), self._index_coords_shape))

        for dim, ind in zip(self._index_coords_dims, u_indices):
            pos_indexers[dim] = xr.Variable(
                indexer_dims[0],
                ind.reshape(indexer_shapes[0]),
            )

        return pos_indexers

    def sel(
        self, indexers: Mapping[Hashable, Any] = None, **indexers_kwargs: Any
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Selection based on a ball tree index.

        The index must have been already built using `xoak.set_index()`.

        It behaves mostly like :meth:`xarray.Dataset.sel` and
        :meth:`xarray.DataArray.sel` methods, with some limitations:

        - Orthogonal indexing is not supported
        - For vectorized (point-wise) indexing, you need to supply xarray
          objects
        - Use it for nearest neighbor lookup only (it implicitly
          assumes method="nearest")

        This triggers :func:`dask.compute` if the given indexers and/or the index
        coordinates are chunked.

        """
        if not getattr(self, '_index', False):
            raise ValueError(
                'The index(es) has/have not been built yet. Call `.xoak.set_index()` first'
            )

        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'xoak.sel')
        indices = self._query(indexers)

        if not isinstance(indices, np.ndarray):
            # TODO: remove (see todo below)
            indices = indices.compute()

        pos_indexers = self._get_pos_indexers(indices, indexers)

        # TODO: issue in xarray. 1-dimensional xarray.Variables are always considered
        # as OuterIndexer, while we want here VectorizedIndexer
        # This would also allow lazy selection
        result = self._xarray_obj.isel(indexers=pos_indexers)

        return result
