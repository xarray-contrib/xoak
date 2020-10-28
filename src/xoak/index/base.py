import abc
from typing import Any, List, Mapping, Tuple, Type, TypeVar, Union
import warnings

import numpy as np


Index = TypeVar("Index")


class IndexAdapter(abc.ABC):
    """Base class for reusing a custom index to select data in
    :class:`xarray.DataArray` or :class:`xarray.Dataset` objects with xoak.

    Subclasses must implement the ``build()`` and ``query()`` methods,
    which are called to build a new index and query this index, respectively.

    If any options are necessary, they should be implemented as arguments to the
    ``__init__()`` method.

    """

    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def build(self, points: np.ndarray) -> Index:
        """Build the index from a set of points/samples and their coordinate labels.

        Parameters
        ----------
        points : ndarray of shape (n_points, n_coordinates)
            Two-dimensional array of points/samples (rows) and their
            corresponding coordinate labels (columns) to index.

        Returns
        -------
        index: object
            A new index object.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def query(self, index: Index, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Query points/samples,

        Parameters
        ----------
        index: object
            The index object returned by ``build()``.
        points: ndarray of shape (n_points, n_coordinates)
            Two-dimensional array of points/samples (rows) and their
            corresponding coordinate labels (columns) to query.

        Returns
        -------
        distances : ndarray of shape (n_points)
            Distances to the nearest neighbors.
        indices : ndarray of shape (n_points)
            Indices of the nearest neighbors in the array of the indexed
            points.

        """
        raise NotImplementedError()


class IndexRegistrationWarning(Warning):
    """Warning for conflicts in index registration."""


class IndexRegistry(Mapping[str, IndexAdapter]):
    """Mapping of all indexes that can be used to select data
    with xoak.

    """

    def __init__(self):
        self._indexes = {}

    def register(self, name: str, cls: Type[IndexAdapter]):
        """Register custom index in xoak.

        Parameters
        ----------
        name : str
            Name to give to this index type.
        cls: :class:`IndexAdapter` subclass
            The index adapter class to register.

        """
        if not issubclass(cls, IndexAdapter):
            raise TypeError("can only register IndexAdapter subclasses.")

        if name in self._indexes:
            warnings.warn(
                f"overriding an already registered index with the name '{name}'.",
                IndexRegistrationWarning,
                stacklevel=2,
            )

        self._indexes[name] = cls

    def __getitem__(self, key):
        return self._indexes[key]

    def __iter__(self):
        return len(self._indexes)

    def __len__(self):
        return len(self._indexes)

    def __repr__(self):
        header = f"<IndexRegistry ({len(self._indexes)} indexes)>\n"
        return header + "\n".join([name for name in self._indexes])


indexes = IndexRegistry()


def register_index(name):
    """A convenient decorator that can be applied to any
    :class:`~xoak.IndexAdapter` class to register a custom index in xoak.

    """

    def decorator(cls):
        indexes.register(name, cls)
        return cls

    return decorator


def normalize_index(name_or_cls: Union[str, Any]) -> Type[IndexAdapter]:

    if isinstance(name_or_cls, str):
        cls = indexes[name_or_cls]
    else:
        cls = name_or_cls

    if not issubclass(cls, IndexAdapter):
        raise TypeError(f"'{name_or_cls}' is not a subclass of IndexAdapter")

    return cls


class XoakIndexWrapper:
    """Thin wrapper used internally to build and query (registered)
    indexes, with dask support.

    """

    _query_result_dtype: List[Tuple[str, Any]] = [
        ("distances", np.double),
        ("indices", np.intp),
    ]

    def __init__(
        self,
        index_adapter: Union[str, Type[IndexAdapter]],
        points: np.ndarray,
        offset: int,
        **kwargs,
    ):
        index_adapter_cls = normalize_index(index_adapter)

        self._index_adapter = index_adapter_cls(**kwargs)
        self._index = self._index_adapter.build(points)
        self._offset = offset

    @property
    def index(self):
        return self._index

    def query(self, points: np.ndarray) -> np.ndarray:
        distances, positions = self._index_adapter.query(self._index, points)

        result = np.empty(shape=points.shape[0], dtype=self._query_result_dtype)
        result["distances"] = distances.ravel().astype(np.double)
        result["indices"] = positions.ravel().astype(np.intp) + self._offset

        return result[:, None]
