
Astrocut
========

Tools for making image cutouts from sets of images with shared footprints.

This package is under active development, and will
ultimately grow to encompass a range of cutout activities relevant to
images from many missions. Currently there are two modes of interaction:

    - Solving the specific problem of creating image cutouts from sectors of TESS full
      frame images (FFIs) ( `~astrocut.CubeFactory` and `~astrocut.CutoutFactory`)
    - More generalized cutouts from sets of images with the same WCS/pixel scale
      (`~astrocut.fits_cut`)

Astrocut lives on GitHub at: `github.com/spacetelescope/astrocut <https://github.com/spacetelescope/astrocut>`_.


Documentation
-------------

.. toctree::
  :maxdepth: 2

  astrocut/install.rst

.. toctree::
  :maxdepth: 3

  astrocut/index.rst


.. toctree::
  :maxdepth: 1

  license
