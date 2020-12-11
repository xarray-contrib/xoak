# xoak Design notes

## 1. Scope

xoak aims at extending [xarray](https://github.com/pydata/xarray)'s capabilities
to support efficient indexing of n-dimensional, irregular data using spatial
partitioning tree structures.

## 2. Spatial data partitioning trees

Spatial data partitioning trees are commonly used as indexes for accelerating
the following operations:

- range search (e.g., find all points within given boundaries)
- nearest neighbor lookup (e.g., find the point(s) in a given set that is/are
  closest to a given point).

There are many kinds of tree structures, e.g.,

- k-d tree
- ball tree
- r-tree
- quad-tree
- octree
- vantage-point tree
- ...

Each structure has its own advantages and limitations regarding index
construction time, query time, memory footprint and types of search supported,
that mostly depend on the properties of the space being partitioned, e.g., low
vs high dimensions, euclidean vs non-euclidean, etc.

## 3. Reuse vs. implement tree structures

A few structures are available in Python:

- A kd-tree is implemented in the
  [scipy.spatial](https://docs.scipy.org/doc/scipy/reference/spatial.html)
  sub-package
- An alternative kd-tree implementation can be found in
  [pykdtree](https://github.com/storpipfugl/pykdtree).
- [libpysal](https://github.com/pysal/libpysal) adds support for arc distance on
  top of scipy's k-d tree
- [scikit-learn](https://scikit-learn.org) has k-d tree and ball tree implementations
- The [rtree](https://github.com/Toblerity/rtree) package wraps
  [libspatialindex](https://github.com/libspatialindex/libspatialindex)

xoak will reuse (some of) those existing implementations. In the mid/long term,
however, xoak might benefit of its own implementations. Motivations are:

- Data structures not available elsewhere (Python packages)
- More flexible implementations (e.g., allow user-defined distance functions)
- More control of underlying parallelisation, e.g., with Dask

Possible solutions for efficient implementation and their main advantage:

- Cython: already used by scikit-learn and scipy
- Numba: mature enough and flexible (would allow user-defined metrics
  implemented as jitted functions)
- C++ / Xtensor: reuse the powerful features of the C++ standard library /
  ecosystem.

## 4. API

### xarray Dataset / DataArray accessor(s)

The current, recommended way to extend Xarray is via
[accessors](http://xarray.pydata.org/en/stable/internals.html#extending-xarray).

#### One vs. multiple accessors

We have two options:

- *A.* one accessor per kind of index, e.g., ``Dataset.kdtree``, ``Dataset.balltree``, etc.
- *B.* one unique accessor ``Dataset.xoak`` for handling all kinds of indexes.

Choosing option A vs option B will depend on how much the indexes may have in
common (regarding both the internal logic and the exposed API).

Option A has the advantage of having the possibility to assign multiple kinds of
indexes to the same set of coordinates. Some indexes are better for range
queries, other indexes are better for nearest neighbor lookup.

Option B has the advantage of not polluting too much ``Dataset``'s and
``DataArray``'s namespace, and may result in less code repetition. Supporting
both optimal range and nearest neighbor queries could be done, e.g., using
a hybrid/compound index.

We will consider option B further below in this document.

#### Consistency with Xarray indexing API

The API exposed in the accessor(s) should be as close as possible to xarray's
API used for setting new indexes and using them for selection. There should be a
clear separation between index construction and data selection (queries).

#### Index construction

```python
ds.xoak.set_index(coord_list, index_type='kdtree', transform=None, **index_kwargs)
```

- ``coord_list`` is a list of dataset coordinates used to build the index. Those
  coordinates can be either 1-dimensional or n-dimensional. All coordinates
  given here must have the same dimensions.

- ``index_type`` (str) is the type of index used. This could also accept classes
  if we want to allow extending xoak.

- ``tranform`` (callable) is optional and would allow some transformation
  applied to the coordinate labels before building (and querying) the index. For
  example, that could be converting lat/lon coordinates from degrees to radians
  (as input to the haversine formula) or from spherical to cartesian XYZ
  coordinates (for using it with k-d trees, similarly to the trick used in
  ``libpysal``'s Arc_KDTree).

- ``**index_kwargs`` are keyword arguments passed to the underlying tree index
  Python classes (e.g., for setting the metric, the tree leaf size, etc.).

#### Access to the underlying index

```python
ds.xoak.index
```

#### Selecting data

```python
ds.xoak.sel(tolerance=None, **indexers)
```

This should closely match ``Dataset.sel()`` and ``DataArray.sel()``, and should
follow the [indexing behavior and
rules](http://xarray.pydata.org/en/stable/indexing.html) defined in xarray:

- Simple indexer types such as integers, slices or unlabeled arrays should be
  used for orthogonal indexing (range search).

- Advanced indexer types such as xarray DataArray or Variable objects should be
  used for point-wise indexing (nearest neighbor lookup).

Additionally:

- For point-wise indexing, indexers must be given for all coordinates in the
  index. Also, indexers must all have the same dimensions (although it may
  differ from the dimensions of the coordinates used to build the index).

- It is not possible to mix orthogonal and point-wise indexing for indexers
  given in ``ds.xoak.sel()``.

- ``method='nearest'`` is implicitly assumed here.

### Xarray custom indexes

One important point of Xarray's development roadmap is the addition of [flexible
indexes](http://xarray.pydata.org/en/stable/roadmap.html#flexible-indexes). Once
this feature is available in xarray, the accessors mentioned here above should
be depreciated in favor of custom index classes that can be used directly in
xarray. Having an API close to xarray's current index creation and data selection
methods will help towards a smooth transition.

### Utility functions

For convenience, xoak may provide some utility functions, e.g., to transform
lat/lon coordinates into XYZ cartesian coordinates.

### Low-level API

In case xoak will eventually have its own index implementations, it should also
provide a low-level API for reusing or inspecting the indexes outside of Xarray.
Alternatively, those implementations could be maintained in a 3rd-party package.

### Indexing operations not supported by xarray

Data partitioning trees often allow queries other than just range and/or nearest
neighbor searches, e.g., k-nearest neighbor search, radius-based search, point
density/count, compute distances, selection of objects, etc. How to expose those
queries in xoak remains an open question.

### Other operations supported by xarray

Data partitioning trees have also the potential to be reused in important xarray
features other than data selection such as, e.g., group-by operations. How to
handle those operations with xoak remains an open question too.

## 5. Internals

Instead of directly reusing tree-based indexes in 3rd party packages, we might
need to write some thin wrappers around them in order to:

- store coordinate dimensions
- properly handle support for orthogonal vs. point-wise indexing
- re-arrange the data (coordinate or indexer values) without copy
- index serialization (see below)
- etc.

## 6. Index serialization / reconstruction

Building a tree index can be time consuming. xoak should allow saving/loading
indexes. This could be done independently of the data in xarray objects by
pickling the index objects.

A more general solution has still to be found on the xarray side. This will
probably be addressed during the implementation of flexible indexes.

## 7. dask support -- distributed / out-of-core indexes

Most data partitioning trees are centralized structures, making it hard to use
in the context of distributed or out-of-core computation.

There are two possible cases:

- large coordinate data
- large indexer data

Possible issues are:

- keeping all *coordinate* data in memory at the same time
- tree construction time
- keeping the whole tree in memory
- keeping all *indexer* data in memory at the same time
- selection (query) time

Some potential sources of inspiration:

- This
  [gist](https://gist.github.com/brendancol/a3dd4a35ecd94660411112999923d561),
  where a tree index is built for each data chunk and where each tree is queried
  during selection. This could probably be optimized a bit.
- A couple of papers on distributed R-trees:
  [SD-Rtree](https://ieeexplore.ieee.org/document/4221678) and
  [DD-Rtree](https://ieeexplore.ieee.org/document/7840586).
