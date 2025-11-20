from importlib.metadata import version

from .accessor import XoakAccessor
from .index import IndexAdapter, IndexRegistry

__all__ = [
    "XoakAccessor",
    "IndexAdapter",
    "IndexRegistry",
]

__version__ = version("xoak")
