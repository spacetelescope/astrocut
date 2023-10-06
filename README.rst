Cutout tools for astronomical images
------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge
    
.. image:: https://badge.fury.io/py/astrocut.svg
    :target: https://badge.fury.io/py/astrocut
    :alt: PyPi Status
    
.. image:: https://readthedocs.org/projects/astrocut/badge/?version=latest
    :target: https://astrocut.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Astrocut provides tools for making cutouts from sets of astronomical images with shared footprints. It is under active development.

Three main areas of functionality are included:

- Solving the specific problem of creating image cutouts from sectors of Transiting Exoplanet Survey Satellite (TESS) full-frame images.
- General fits file cutouts incuding from single images and sets of images with the shared WCS/pixel scale.
- Cutout post-processing functionality, including centering cutouts along a path (for moving targets) and combining cutouts.

Documentation is at https://astrocut.readthedocs.io.

Project Status
--------------
.. image:: https://github.com/spacetelescope/astrocut/workflows/CI/badge.svg?branch=master
    :target: https://github.com/spacetelescope/astrocut/actions
    :alt: Github actions CI status
    
.. image:: https://codecov.io/gh/spacetelescope/astrocut/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/spacetelescope/astrocut
  :alt: Codecov coverage status


Developer Documentation
-----------------------

Installation
============

.. code-block:: bash

    $ git clone https://github.com/spacetelescope/astrocut.git
    $ cd astrocut
    $ pip install .
    
For active developement intall in develop mode

.. code-block:: bash

    $ pip install -e .
    
Testing
=======
Testing is now run with `tox <https://tox.readthedocs.io>`_ (``pip install tox``).
Tests can be found in ``astrocut/tests/``.

.. code-block:: bash

    $ tox -e test

Tests can also be run directly with pytest:

.. code-block:: bash

    $ pip install -e .[test]
    $ pytest
    
Documentation
=============
Documentation files are found in ``docs/``.

We now build the documentation with `tox <https://tox.readthedocs.io>`_ (``pip install tox``):

.. code-block:: bash

    $ tox -e build_docs

You can also build the documentation with Sphinx directly using:

.. code-block:: bash
                
    $ pip install -e .[docs]
    $ cd docs
    $ make html
    
The built docs will be in ``docs/_build/html/``, to view them go to ``file:///path/to/astrocut/repo/docs/_build/html/index.html`` in the browser of your choice.
    

Release Protocol
================

GitHub Action Releases
^^^^^^^^^^^^^^^^^^

The `pypi-package.yml <.github/workflows/pypi-package.yml>`_ GitHub workflow creates a PyPI release. The job in this workflow is triggered when a tag is pushed or a GH release (+tag) is created, and uses `OpenAstronomy`'s `GitHub action workflow <https://github.com/OpenAstronomy/github-actions-workflows>`_
for publishing pure Python packages (`see here <https://github-actions-workflows.openastronomy.org/en/stable/publish_pure_python.html>`_ for documentation).

Manual Releases
^^^^^^^^^^^^^^^

For making releases manually, follow the `Astropy template release instructions <https://docs.astropy.org/en/latest/development/astropy-package-template.html>`_. 

*Requirements:*

- build (``pip install build``)
- twine (``pip install twine``)

*Notes:* 

- Astrocut uses setuptools_scm to manage version numbers.
- Astrocut does have a pyproject.toml file
- If the given twine command doesn't work you likely need ``python -m twine upload dist/*``
- You shouldn't have trigger a readthedocs build manually, it should run on it's own in ~20 min.


Contributing
------------

We love contributions! Astrocut is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
Astrocut based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.


License
-------

This project is Copyright (c) MAST Archive Developers and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.


