========
Overview
========

AtomsMM is an OpenMM customization developed by the ATOMS group at UFRJ/Brazil.

* Free software: MIT license

Installation
============

::

    pip install atomsmm

Documentation
=============

https://atomsmm.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
