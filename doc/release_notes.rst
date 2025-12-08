.. _release_notes:

Release Notes
=============

v0.2.0 (Unreleased)
-------------------

Features
~~~~~~~~

- Xoak now relies on :py:class:`xarray.indexes.NDPointIndex` for point-wise
  indexing of irregular data, by providing custom ``TreeAdapter`` classes.
  The current functionality remains the same. See documentation examples
  for more details (:pull:`44`).

Deprecations
------------

- Xoak specific API :py:meth:`xarray.Dataset.xoak.set_index` and
  :py:meth:`xarray.Dataset.xoak.sel` has been deprecated in favor of Xarray's
  API :py:meth:`xarray.Dataset.set_xindex` and :py:meth:`xarray.Dataset.sel`.
  See documentation examples for more details (:pull:`44`).
- Xoak experimental support for chunked coordinates (Dask arrays) has been
  deprecated (:pull:`44`).


v0.1.2 (20 November 2025)
-------------------------

Maintenance
~~~~~~~~~~~

- Replace deprecated pkg_resources (:pull:`45`).
- CI, packaging and doc maintenance (complete revamp) (:pull:`47`).

v0.1.1 (4 August 2021)
----------------------

Bug fixes
~~~~~~~~~

- Fix selection using indexers with more than one chunk (:pull:`38`).

Maintenance
~~~~~~~~~~~

- Re-enable tests with pys2index on Linux (:pull:`33`).
- Transfer repository to ``xarray-contrib`` organization (:pull:`36`).

v0.1.0 (17 December 2020)
-------------------------

Initial release.
