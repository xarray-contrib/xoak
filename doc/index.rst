Xoak: Xarray extension for indexing irregular grids
===================================================

**Xoak** is an Xarray extension that allows point-wise selection of irregular,
n-dimensional data encoded in coordinates with an arbitrary number of
dimensions.

It provides a built-in index adapter for Scipy_'s ``cKDTree``, as well as adapters
for index structures implemented in these 3rd-party libraries (optional
dependencies):

- Scikit-Learn_: ``BallTree`` and ``KDTree``, which support various distance metrics.
- pys2index_: ``S2PointIndex`` for efficient indexing of lat/lon point data,
  based on `s2geometry`.

Xoak also provides a mechanism for easily adding and registering custom index adapters.

.. _Scipy: https://docs.scipy.org/doc/scipy/reference/
.. _Scikit-Learn: https://scikit-learn.org
.. _pys2index: https://github.com/benbovy/pys2index

Table of contents
-----------------

**Getting Started**

* :doc:`install`
* :doc:`examples/index`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   install
   examples/index

**Help & Reference**

* :doc:`api`
* :doc:`release_notes`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help & Reference

   api
   release_notes

**For Contributors**

* :doc:`contribute`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: For Contributors

   contribute

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
