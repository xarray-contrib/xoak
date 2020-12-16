.. _api:

API Reference
=============

This page provides an auto-generated summary of Xoak's API.

.. currentmodule:: xarray

Dataset.xoak
------------

This accessor extends :py:class:`xarray.Dataset` with all the methods and
properties listed below. Proper use of this accessor should be like:

.. code-block:: python

   >>> import xarray as xr         # first import xarray
   >>> import xoak                 # import xoak (the 'xoak' accessor is registered)
   >>> ds = xr.Dataset()           # create or load an xarray Dataset
   >>> ds.xoak.<meth_or_prop>      # access to the methods and properties listed below

**Properties**

.. autosummary::
   :toctree: _api_generated/
   :template: autosummary/accessor_attribute.rst

   Dataset.xoak.index

**Methods**

.. autosummary::
   :toctree: _api_generated/
   :template: autosummary/accessor_method.rst

    Dataset.xoak.set_index
    Dataset.xoak.sel

DataArray.xoak
--------------

The accessor above is also registered for :py:class:`xarray.DataArray`.

**Properties**

.. autosummary::
   :toctree: _api_generated/
   :template: autosummary/accessor_attribute.rst

   DataArray.xoak.index

**Methods**

.. autosummary::
   :toctree: _api_generated/
   :template: autosummary/accessor_method.rst

    DataArray.xoak.set_index
    DataArray.xoak.sel

Indexes
-------

.. currentmodule:: xoak

.. autosummary::
   :toctree: _api_generated/

    IndexAdapter
    IndexRegistry

**Xoak's built-in index adapters**

.. currentmodule:: xoak.index.scipy_adapters

.. autosummary::
   :toctree: _api_generated/

    ScipyKDTreeAdapter

.. currentmodule:: xoak.index.sklearn_adapters

.. autosummary::
   :toctree: _api_generated/

    SklearnKDTreeAdapter
    SklearnBallTreeAdapter
    SklearnGeoBallTreeAdapter

.. currentmodule:: xoak.index.s2_adapters

.. autosummary::
   :toctree: _api_generated/

    S2PointIndexAdapter
