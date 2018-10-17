Cutout tools for astronomical images
------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

Tools for making image cutouts from sets of TESS full frame images.

This package is under active development, and will ultimately grow to encompass a range of cutout activities relevant to images from many missions, however at this time it is focussed on the specific problem of creating Target Pixel File cutouts from sectors of TESS full frame images.

Documentation is at https://astrocut.readthedocs.io.

Installation
------------
.. code-block:: bash

    $ git clone https://github.com/spacetelescope/astrocut.git
    $ cd astrocut
    $ python setup.py install

Example Usage
-------------
.. code-block:: python

    >>> from astrocut import CubeFactory
    >>> from astrocut import CutoutFactory

    >>> # Making the data cube
    >>> filelist = ['<list of FFIs>']
    >>> CubeFactory().make_cube(infiles, "img-cube.fits")

    >>> # Making the cutout tpf
    >>> CutoutFactory().cube_cut("img-cube.fits", "259.7 36.7", 5, verbose=True)


License
-------

This project is Copyright (c) MAST Archive Developers and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.

