from .base import IndexWrapper, register_index


try:
    from .balltree import BallTreeWrapper, GeoBallTreeWrapper
except ImportError:
    pass
