from .base import IndexAdapter, indexes, register_index


try:
    from .balltree import BallTreeAdapter, GeoBallTreeAdapter
except ImportError:
    pass
