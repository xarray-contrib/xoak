# xoak

[![Tests](https://github.com/ESM-VFC/xoak/workflows/test/badge.svg)](https://github.com/ESM-VFC/xoak/actions?query=workflow%3Atest)
[![Coverage](https://codecov.io/gh/ESM-VFC/xoak/branch/master/graphs/badge.svg?branch=master)](https://codecov.io/github/ESM-VFC/xoak?branch=master)
[![Documentation Status](https://readthedocs.org/projects/xoak/badge/?version=latest)](https://xoak.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ESM-VFC/xoak/master?filepath=doc%2Fexamples)

Xoak is an Xarray extension that allows point-wise selection of irregular,
n-dimensional data encoded in coordinates with an arbitrary number of
dimensions.

It provides a built-in index adapter for
[Scipy](https://docs.scipy.org/doc/scipy/reference/)'s `cKDTree`, as well as
adapters for index structures implemented in these 3rd-party libraries (optional
dependencies):

- [Scikit-Learn](https://scikit-learn.org): `BallTree` and `KDTree`, which
  support various distance metrics.
- [pys2index](https://github.com/benbovy/pys2index): `S2PointIndex` for
  efficient indexing of lat/lon point data, based on `s2geometry`.

Xoak also provides a mechanism for easily adding and registering custom index adapters.

## Install

Xoak can be installed using conda (or mamba):

```bash
$ conda install xoak -c conda-forge
```

or pip:

```bash
$ python -m pip install xoak
```

Xoak's optional dependencies can be installed using conda:

```bash
$ conda install scikit-learn pys2index -c conda-forge
```

## Documentation

Documentation is hosted on ReadTheDocs: https://xoak.readthedocs.io/

## License

MIT License, see LICENSE file.
