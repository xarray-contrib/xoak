.. _install:

Install xoak
============

Required dependencies
---------------------

- Python 3.6 or later.
- `xarray <http://xarray.pydata.org>`__
- `dask <https://docs.dask.org>`__
- `scipy <https://docs.scipy.org/doc/scipy/reference/>`__

Optional dependencies
---------------------

Xoak provides built-in adapters for some indexes implemented in the 3rd-party
libraries listed below. Those dependencies are optional: adapters won't be
added in Xoak's index registry if they are not installed.

- `scikit-learn <https://scikit-learn.org>`__
- `pys2index <https://github.com/benbovy/pys2index>`__

Install using conda
-------------------

Xoak can be installed or updated using conda_ (or mamba_)::

  $ conda install xoak -c conda-forge

This installs Xoak and all its required dependencies. The optional dependencies
listed above could be installed separately using conda too::

  $ conda install scikit-learn pys2index -c conda-forge

The xoak conda package is maintained on the `conda-forge`_ channel.

.. _conda-forge: https://conda-forge.org/
.. _conda: https://conda.io/docs/
.. _mamba: https://github.com/mamba-org/mamba

Install using pip
-----------------

You can also install Xoak and its required dependencies using pip_::

  $ python -m pip install xoak

.. _pip: https://pip.pypa.io

Install from source
-------------------

See Section :ref:`contribute` for instructions on how to setup a development
environment for Xoak.

Import xoak
-----------

To make sure that Xoak is correctly installed, try to import it by running this
line::

    $ python -c "import xoak"
