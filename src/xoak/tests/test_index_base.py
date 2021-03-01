import numpy as np
import pytest

from xoak import IndexAdapter, IndexRegistry
from xoak.index.base import (
    IndexRegistrationWarning,
    XoakIndexWrapper,
    normalize_index,
    register_default,
)
from xoak.index.scipy_adapters import ScipyKDTreeAdapter


class DummyIndex:
    def __init__(self, points, option=1):
        self.points = points
        self.option = option

    def query(self, points):
        distances = np.zeros((points.shape[0]))
        indices = np.ones((points.shape[0]))

        return distances, indices


class DummyIndexAdapter(IndexAdapter):
    def __init__(self, **kwargs):
        self.index_kwargs = kwargs

    def build(self, points):
        return DummyIndex(points, **self.index_kwargs)

    def query(self, index, points):
        return index.query(points)


def test_index_adapter_base():
    class IndexAdapterSubclass(IndexAdapter):
        def build(self, points):
            return super().build(points)

        def query(self, index, points):
            return super().query(index, points)

    adapter = IndexAdapterSubclass()

    with pytest.raises(NotImplementedError):
        adapter.build(np.zeros((10, 2)))

    with pytest.raises(NotImplementedError):
        adapter.query(None, np.zeros((10, 2)))


def test_index_registery_constructor():
    registry = IndexRegistry()
    assert dict(registry) == registry._default_indexes

    registry = IndexRegistry(use_default=False)
    assert len(registry) == 0


def test_index_registery_register():
    registry = IndexRegistry(use_default=False)
    registry.register('dummy')(DummyIndexAdapter)

    with pytest.warns(IndexRegistrationWarning, match='overriding an already registered index.*'):
        registry.register('dummy')(DummyIndexAdapter)

    with pytest.raises(TypeError, match='can only register IndexAdapter subclasses.'):
        registry.register('invalid')(DummyIndex)


def test_index_registry_dict_interface():
    registry = IndexRegistry(use_default=False)
    registry.register('dummy')(DummyIndexAdapter)

    assert registry['dummy'] is DummyIndexAdapter
    assert list(registry) == ['dummy']
    assert len(registry) == 1
    assert repr(registry) == '<IndexRegistry (1 indexes)>\ndummy'


def test_index_registry_attr_access():
    registry = IndexRegistry(use_default=False)
    registry.register('dummy')(DummyIndexAdapter)

    assert registry.dummy is DummyIndexAdapter
    assert 'dummy' in dir(registry)

    with pytest.raises(AttributeError, match='.*has no attribute.*'):
        registry.invalid_attr

    with pytest.raises(AttributeError, match='.*cannot set attribute.*'):
        registry.custom = DummyIndexAdapter


def test_index_registry_ipython_completion():
    registry = IndexRegistry(use_default=False)
    registry.register('dummy')(DummyIndexAdapter)

    assert 'dummy' in registry._ipython_key_completions_()


def test_register_default():
    # check that docstrings are updated
    assert 'This index adapter is registered in xoak' in ScipyKDTreeAdapter.__doc__

    register_default('dummy')(DummyIndexAdapter)
    assert 'This index adapter is registered in xoak' in DummyIndexAdapter.__doc__
    del IndexRegistry._default_indexes['dummy']


def test_normalize_index():
    assert normalize_index(DummyIndexAdapter) is DummyIndexAdapter
    assert normalize_index('scipy_kdtree') is ScipyKDTreeAdapter

    with pytest.raises(TypeError, match='.*is not a subclass of IndexAdapter'):
        normalize_index(DummyIndex)


def test_xoak_index_wrapper():
    idx_points = np.zeros((10, 2))
    offset = 1

    wrapper = XoakIndexWrapper(DummyIndexAdapter, idx_points, offset, option=2)
    wrapper2 = XoakIndexWrapper(DummyIndexAdapter, idx_points, 0)

    assert isinstance(wrapper.index, DummyIndex)
    assert wrapper.index.option == 2
    assert isinstance(wrapper2.index, DummyIndex)
    assert wrapper2.index.option == 1

    results = wrapper.query(np.zeros((5, 2))).ravel()

    assert results['distances'].dtype == np.double
    assert results['indices'].dtype == np.intp
    np.testing.assert_equal(results['distances'], np.zeros(5))
    np.testing.assert_equal(results['indices'], np.ones(5) + offset)
