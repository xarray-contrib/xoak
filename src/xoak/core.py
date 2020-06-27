import numbers
from typing import Any, Callable, Iterable, Hashable, List, Mapping, Union

import numpy as np
import xarray as xr
from xarray.core.utils import either_dict_or_kwargs
from sklearn.neighbors import BallTree


@xr.register_dataarray_accessor("xoak")
@xr.register_dataset_accessor("xoak")
class XoakAccessor:
    """A xarray Dataset or DataArray extension for indexing irregular,
    n-dimensional data using a ball tree.
    
    """

    def __init__(self, xarray_obj: Union[xr.Dataset, xr.DataArray]):

        self._xarray_obj = xarray_obj

        self._index = None
        self._index_coords = None
        self._index_coords_dims = None
        self._index_coords_shape = None

        self._transform = None

    def _stack(self, coords: List[Any]):
        """Stack and maybe transform coordinate labels into a format
        compliant with sklearn's BallTree, i.e., a 2-d array (npoints, nfeatures).
    
        """
        X = np.stack([np.ravel(arr) for arr in coords]).T

        if self._transform is not None:
            return self._transform(X)
        else:
            return X

    def set_index(self, coords: Iterable[str], transform: Callable = None, **kwargs):
        """Create a ball tree index from a subset of coordinates of
        the DataArray / Dataset.
        
        Parameters
        ----------
        coords : iterable
            Coordinate names. Each given coordinate must have
            the same dimension(s), in the same order.
        transform : callable, optional
            Any function used to convert coordinate labels. This is useful,
            e.g., for converting degrees to radians when using the haversine metric.
            This transform will also be applied each time before indexing (query). 
        **kwargs
            Arguments passed to :class:`sklearn.neighbors.BallTree`
            constructor.

        """
        if transform is not None:
            self._transform = transform

        self._index_coords = tuple(coords)

        coord_objs = [self._xarray_obj.coords[cn] for cn in coords]

        if len(set([c.dims for c in coord_objs])) > 1:
            raise ValueError(
                "Coordinates {coords} must all have the same dimensions in the same order"
            )

        self._index_coords_dims = coord_objs[0].dims
        self._index_coords_shape = coord_objs[0].shape

        X = self._stack([self._xarray_obj[c] for c in coords])

        self._index = BallTree(X, **kwargs)

    @property
    def index(self) -> BallTree:
        """Returns the underlying ball tree index."""

        return self._index

    def _query(self, indexers, tolerance):
        """Query the ball tree and maybe reject selected points
        based on tolerance (distance threshold).
        
        """
        X = self._stack([indexers[c] for c in self._index_coords])

        if tolerance is None:
            return self._index.query(X, return_distance=False)
        else:
            dist, indices = self._index.query(X, return_distance=True)
            return indices[dist <= tolerance]

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
        tolerance: numbers.Number = None,
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
                "The ball tree index has not been built yet. "
                "Call `.xoak.set_index()` first"
            )

        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "xoak.sel")
        indices = self._query(indexers, tolerance)
        pos_indexers = self._get_pos_indexers(indices, indexers)

        result = self._xarray_obj.isel(indexers=pos_indexers)

        return result
