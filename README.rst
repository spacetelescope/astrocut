Cutout tools for astronomical images
------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

Cutout tools for single and multiple astronomical image fitsfiles that can
produce both single cutout files and target pixel file cutout stacks.

Installation
------------
.. code-block:: bash

    $ git clone https://github.com/spacetelescope/astrocut.git
    $ cd astrocut
    $ python setup.py install

Example Usage
-------------
.. code-block:: python

    >>> from astrocut.make_cube import make_cube
    >>> from astrocut.cube_cut import cube_cut

    >>> # Making the data cube
    >>> filelist = ['<list of FFIs>']
    >>> make_cube(infiles, "newCube.fits")

    >>> # Making the cutout tpf
    >>> cube_cut(cubeFile, "259.7 36.7", 5, verbose=True)


License
-------

This project is Copyright (c) MAST Archive Developers and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.
