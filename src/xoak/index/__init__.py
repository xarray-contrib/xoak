from .base import IndexAdapter, IndexRegistry

try:
    from .balltree import BallTreeAdapter, GeoBallTreeAdapter
except ImportError:
    pass
