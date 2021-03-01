import abc
import warnings
from contextlib import suppress
from typing import Any, Dict, List, Mapping, Tuple, Type, TypeVar, Union

import numpy as np

Index = TypeVar('Index')


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


class IndexRegistry(Mapping[str, Type[IndexAdapter]]):
    """A registry of all indexes adapters that can be used to select data
    with xoak.

    """

    _default_indexes: Dict[str, Type[IndexAdapter]] = {}

    def __init__(self, use_default=True):
        """Creates a new index registry.

        This registry provides a dict-like interface as well as attribute-style
        access to index adapters.

        Parameters
        ----------
        use_default : bool, optional
            If True (default), pre-populates the registry with xoak's built-in
            index adapters.

        """
        self._indexes = {}

        if use_default:
            self._indexes.update(self._default_indexes)

    def register(self, name: str):
        """Register custom index in xoak.

        Parameters
        ----------
        name : str
            Name to give to this index type.
        cls: :class:`IndexAdapter` subclass
            The index adapter class to register.

        """

        def wrap(cls: Type[IndexAdapter]):
            if not issubclass(cls, IndexAdapter):
                raise TypeError('can only register IndexAdapter subclasses.')

            if name in self._indexes:
                warnings.warn(
                    f"overriding an already registered index with the name '{name}'.",
                    IndexRegistrationWarning,
                    stacklevel=2,
                )

            self._indexes[name] = cls

            return cls

        return wrap

    def __getattr__(self, name):
        if name not in {'__dict__', '__setstate__'}:
            # this avoids an infinite loop when pickle looks for the
            # __setstate__ attribute before the xarray object is initialized
            with suppress(KeyError):
                return self._indexes[name]
        raise AttributeError(f'IndexRegistry object has no attribute {name!r}')

    def __setattr__(self, name, value):
        if name == '_indexes':
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f'cannot set attribute {name!r} on a IndexRegistry object. '
                'Use `.register()` to add a new index adapter to the registry.'
            )

    def __dir__(self):
        extra_attrs = [k for k in self._indexes]
        return sorted(set(dir(type(self)) + extra_attrs))

    def _ipython_key_completions_(self):
        return list(self._indexes)

    def __getitem__(self, key):
        return self._indexes[key]

    def __iter__(self):
        return iter(self._indexes)

    def __len__(self):
        return len(self._indexes)

    def __repr__(self):
        header = f'<IndexRegistry ({len(self._indexes)} indexes)>\n'
        return header + '\n'.join([name for name in self._indexes])


def register_default(name: str):
    """A convenient decorator to register xoak's builtin indexes."""

    doc_extra = f"""

    This index adapter is registered in xoak under the name ``{name}``.
    You can use it in :meth:`xarray.Dataset.xoak.set_index` by simply providing
    its name for the ``index_type`` argument.
    Alternatively, you can access it via the index registry, i.e.,

    >>> import xoak
    >>> ireg = xoak.IndexRegistry()
    >>> ireg.{name}

    """

    def decorator(cls: Type[IndexAdapter]):
        if cls.__doc__ is not None:
            cls.__doc__ += doc_extra
        else:
            cls.__doc__ = doc_extra

        IndexRegistry._default_indexes[name] = cls
        return cls

    return decorator


def normalize_index(name_or_cls: Union[str, Any]) -> Type[IndexAdapter]:

    if isinstance(name_or_cls, str):
        cls = IndexRegistry._default_indexes[name_or_cls]
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
        ('distances', np.double),
        ('indices', np.intp),
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
        result['distances'] = distances.ravel().astype(np.double)
        result['indices'] = positions.ravel().astype(np.intp) + self._offset

        return result[:, None]
