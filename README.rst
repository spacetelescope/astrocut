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

Tools for making image cutouts from sets of TESS full frame images.

This package is under active development, and will ultimately grow to encompass a range of cutout activities relevant to images from many missions, however at this time it is focussed on the specific problem of creating Target Pixel File cutouts from sectors of TESS full frame images.

Documentation is at https://astrocut.readthedocs.io.

Project Status
--------------
.. image:: https://travis-ci.org/spacetelescope/astrocut.svg?branch=master
    :target: https://travis-ci.org/spacetelescope/astrocut
    :alt: Travis CI status
    
.. image:: https://codecov.io/gh/spacetelescope/astrocut/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/spacetelescope/astrocut
  :alt: Codecov coverage status





Developer Documentation
-----------------------

Installation
^^^^^^^^^^^^

.. code-block:: bash

    $ git clone https://github.com/spacetelescope/astrocut.git
    $ cd astrocut
    $ pip install .
    
For active developement intall in develop mode

.. code-block:: bash

    $ pip install -e .
    
Testing
^^^^^^^
Testing is now run with `tox <https://tox.readthedocs.io>`_ (``pip install tox``).
Tests can be found in ``astrocut/tests/``.

.. code-block:: bash

    $ tox -e test

Tests can also be run directly with pytest:

.. code-block:: bash

    $ pip install -e .[test]
    $ pytest
    
Documentation
^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^

Follow the `Astropy template release instructions <https://docs.astropy.org/en/latest/development/astropy-package-template.html>`_.

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


