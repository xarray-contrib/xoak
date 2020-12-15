import importlib

from .base import IndexAdapter, IndexRegistry  # noqa: F401

adapters = [
    'sklearn_adapters',
    's2_adapters',
]

for mod in adapters:
    try:
        # importing the module registers the adapters
        importlib.import_module('.' + mod, package='xoak.index')
    except ImportError:
        pass

del adapters
