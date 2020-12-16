.. _contribute:

Contributor Guide
=================

Xoak is an open-source project. Contributions are welcome, and they are
greatly appreciated!

You can contribute in many ways, e.g., by reporting bugs, submitting feedbacks,
contributing to the development of the code and/or the documentation, etc.

This page provides resources on how best to contribute.

Issues
------

The `Github issue tracker`_ is the right place for reporting bugs and for
discussing about development ideas. Feel free to open a new issue if you have
found a bug or if you have suggestions about new features or changes.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting,
  specifically the Python interpreter version, installed libraries, and Xoak
  version.
* Detailed steps to reproduce the bug.

If you can write a demonstration test that currently fails but should pass, that
is a very useful commit to make as well, even if you cannot fix the bug itself.

For now, as the project is still very young, it is also a good place for
asking usage questions.

.. _`Github Issue Tracker`: https://github.com/ESM-VFC/xoak/issues

Development environment
-----------------------

If you wish to contribute to the development of the code and/or the
documentation, here are a few steps for setting up a development environment.

Fork the repository and download the code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To further be able to submit modifications, it is preferable to start by
forking the Xoak repository on GitHub_ (you need to have an account).

Then clone your fork locally::

  $ git clone git@github.com:your_name_here/xoak.git

Alternatively, if you don't plan to submit any modification, you can clone the
original Xoak git repository::

   $ git clone git@github.com:ESM-VFC/xoak.git

.. _GitHub: https://github.com

Install
~~~~~~~

To install the dependencies, we recommend using the conda_ package manager with
the conda-forge_ channel. For development purpose, you might consider installing
the packages in a new conda environment::

  $ conda create -n xoak_dev python xarray dask scipy scikit-learn -c conda-forge
  $ conda activate xoak_dev

Then install Xoak locally (in development mode) using ``pip``::

  $ cd xoak
  $ python -m pip install -e .

.. _conda: http://conda.pydata.org/docs/
.. _conda-forge: https://conda-forge.github.io/

Pre-commit
~~~~~~~~~~

Xoak provides a configuration for `pre-commit <https://pre-commit.com>`_, which
can be used to ensure that code-style and code formatting is consistent.

First install ``pre-commit``::

  $ conda install pre-commit -c conda-forge

Then run the following command to activate it in the current repository::

  $ pre-commit install

From now on ``pre-commit`` will run whenever you commit with ``git``.

Run tests
~~~~~~~~~

To make sure everything behaves as expected, you may want to run Xoak's unit
tests locally using the `pytest`_ package. You can first install it with conda::

  $ conda install pytest pytest-cov -c conda-forge

Then you can run tests from the main xoak directory::

  $ pytest . --verbose

.. _pytest: https://docs.pytest.org/en/latest/

Contributing to code
--------------------

Below are some useful pieces of information in case you want to contribute
to the code.

Local development
~~~~~~~~~~~~~~~~~

Once you have set up the development environment, the next step is to create
a new git branch for local development::

  $ git checkout -b name-of-your-bugfix-or-feature

Now you can make your changes locally.

Submit changes
~~~~~~~~~~~~~~

Once you are done with the changes, you can commit your changes to git and
push your branch to your Xoak fork on GitHub::

  $ git add .
  $ git commit -m "Your detailed description of your changes."
  $ git push origin name-of-your-bugfix-or-feature

(note: this operation may be repeated several times).

When committing, ``pre-commit`` will re-format the files if necessary.

We you are ready, you can create a new pull request through the GitHub_ website
(note that it is still possible to submit changes after your created a pull
request).

Contributing to documentation
-----------------------------

Xoak uses Sphinx_ for documentation, hosted on http://readthedocs.org .
Documentation is maintained in the RestructuredText markup language (``.rst``
files) in the ``doc`` folder.

To build the documentation locally, first install some extra requirements::

   $ conda install sphinx sphinx_rtd_theme sphinx-autosummary-accessors -c conda-forge

Then build the documentation with ``make``::

   $ cd doc
   $ make html

The resulting HTML files end up in the ``build/html`` directory.

You can now make edits to rst files and run ``make html`` again to update
the affected pages.

.. _Sphinx: http://www.sphinx-doc.org/

Docstrings
~~~~~~~~~~

Everything (i.e., classes, methods, functions...) that is part of the public API
should follow the numpydoc_ standard when possible.

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
