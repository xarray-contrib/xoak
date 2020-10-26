import numbers
from typing import Any, Iterable, Hashable, List, Mapping, Type, Union

import numpy as np
import xarray as xr
from xarray.core.utils import either_dict_or_kwargs

from .indexes.base import Index, IndexWrapper, normalize_index


def coords_to_point_array(coords: List[Any]):
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


@xr.register_dataarray_accessor("xoak")
@xr.register_dataset_accessor("xoak")
class XoakAccessor:
    """A xarray Dataset or DataArray extension for indexing irregular,
    n-dimensional data using a ball tree.
    
    """

    def __init__(self, xarray_obj: Union[xr.Dataset, xr.DataArray]):

        self._xarray_obj = xarray_obj

        self._index = None
        self._index_cls = None
        self._index_coords = None
        self._index_coords_dims = None
        self._index_coords_shape = None

    def _build_index_forest_delayed(self, X, **kwargs) -> List[Any]:
        import dask

        indexes = []
        offset = 0

        for i, chunk in enumerate(X.to_delayed().ravel()):
            idx = dask.delayed(self._index_cls)(chunk, offset, **kwargs)
            indexes.append(idx)

            offset += X.chunks[0][i]

        return indexes

    def set_index(
            self,
            coords: Iterable[str],
            index_type: Union[str, Type[IndexWrapper]],
            persist: bool = True,
            **kwargs
    ):
        """Create an index tree from a subset of coordinates of the DataArray / Dataset.

        If the given coordinates are chunked (Dask arrays), this method will (lazily) create
        a forest of index trees (one tree per chunk of the flattened coordinate arrays).
        
        Parameters
        ----------
        coords : iterable
            Coordinate names. Each given coordinate must have
            the same dimension(s), in the same order.
        index_type : str or :class:`xoak.IndexWrapper` subclass
            Either one of the registered index types or a custom index wrapper class.
        persist: bool
            If True, this method will precompute and persist in memory the forest of index
            trees, if any (default: True).
        **kwargs
            Keyword arguments that will be passed to the underlying index constructor.

        """
        self._index_cls = normalize_index(index_type)
        self._index_coords = tuple(coords)

        coord_objs = [self._xarray_obj.coords[cn] for cn in coords]

        if len(set([c.dims for c in coord_objs])) > 1:
            raise ValueError(
                "Coordinates {coords} must all have the same dimensions in the same order"
            )

        self._index_coords_dims = coord_objs[0].dims
        self._index_coords_shape = coord_objs[0].shape

        X = coords_to_point_array([self._xarray_obj[c] for c in coords])

        if isinstance(X, np.ndarray):
            self._index = self._index_cls(X, 0, **kwargs)

        else:
            import dask

            self._index = self._build_index_forest_delayed(X, **kwargs)

            if persist:
                dask.persist(*self._index)

    @property
    def index(self) -> Index:
        """Returns the underlying index object."""

        if isinstance(self._index, list):
            import dask
            return dask.compute(*self._index)
        else:
            return self._index._index

    def _query(self, indexers):
        """Query the index. """
        X = coords_to_point_array([indexers[c] for c in self._index_coords])

        if isinstance(X, np.ndarray) and not isinstance(self._index, list):
            result = self._index.query(X)
            return result["positions"][:, 0]
        else:
            # TODO: implement two-stage query with dask
            raise NotImplementedError

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
            raise ValueError("All indexers must have the same dimensions.")

        u_indices = np.unravel_index(indices.ravel(), self._index_coords_shape)

        for dim, ind in zip(*[self._index_coords_dims, u_indices]):
            pos_indexers[dim] = xr.Variable(
                indexer_dims[0], ind.reshape(indexer_shapes[0])
            )

        return pos_indexers

    def sel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        **indexers_kwargs: Any
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Selection based on a ball tree index.
        
        The index must have been already built using
        `xoak.set_index()`.
        
        It behaves mostly like :meth:`xarray.Dataset.sel` and
        :meth:`xarray.DataArray.sel` methods, with some limitations:
        
        - Orthogonal indexing is not supported
        - For vectorized (point-wise) indexing, you need to supply xarray
          objects
        - Use it for nearest neighbor lookup only (it implicitly
          assumes method="nearest")
        
        """
        if self._index is None:
            raise ValueError(
                "The index(es) has/have not been built yet. Call `.xoak.set_index()` first"
            )

        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "xoak.sel")
        indices = self._query(indexers)
        pos_indexers = self._get_pos_indexers(indices, indexers)

        result = self._xarray_obj.isel(indexers=pos_indexers)

        return result
