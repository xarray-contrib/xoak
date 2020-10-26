from typing import Any, Dict, Tuple, Type, TypeVar, Union
import warnings

import numpy as np


Index = TypeVar('Index')


class IndexRegistrationWarning(Warning):
    """Warning for conflicts in accessor registration."""


class IndexWrapper:
    index_cls: Index = None

    _query_result_dtype = [("distances", np.double), ("positions", np.intp)]

    def __init__(self, points: np.ndarray, offset: int, **kwargs):
        self._index = self.build_index(points, **kwargs)
        self._offset = offset

    def build_index(self, points: np.ndarray, **kwargs) -> Index:
        return self.index_cls(points, **kwargs)

    def query_index(self, points: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self._index.query(points, **kwargs)

    def query(self, points: np.ndarray, **kwargs) -> np.ndarray:
        distances, positions = self.query_index(points, **kwargs)

        result = np.empty(shape=points.shape[0], dtype=self._query_result_dtype)
        result["distances"] = distances.ravel().astype(np.double)
        result["positions"] = positions.ravel().astype(np.intp) + self._offset

        return result[:, None]


registered_indexes: Dict[str, Type[IndexWrapper]] = {}


def register_index(name):

    def decorator(cls):
        if not issubclass(cls, IndexWrapper):
            raise TypeError("can only register IndexWrapper subclasses.")

        if name in registered_indexes:
            warnings.warn(
                f"overriding an already registered index with the name {name}.",
                IndexRegistrationWarning,
                stacklevel=2,
            )

        registered_indexes[name] = cls

        return cls

    return decorator


def normalize_index(name_or_cls: Union[str, Any]) -> Type[IndexWrapper]:

    if isinstance(name_or_cls, str):
        return registered_indexes[name_or_cls]

    if not issubclass(name_or_cls, IndexWrapper):
        raise TypeError(f"{name_or_cls} is not a subclass of IndexWrapper")

    return name_or_cls
