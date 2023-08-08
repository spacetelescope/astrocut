# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *  # noqa
# ----------------------------------------------------------------------------

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys

__minimum_python_version__ = "3.8"


class UnsupportedPythonError(Exception):
    pass


if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise UnsupportedPythonError("astrocut does not support Python < {}".format(__minimum_python_version__))


if not _ASTROPY_SETUP_:  # noqa
    from .make_cube import CubeFactory, TicaCubeFactory  # noqa
    from .cube_cut import CutoutFactory  # noqa
    from .cutouts import fits_cut, img_cut, normalize_img  # noqa
    from .cutout_processing import (path_to_footprints, center_on_path,  # noqa
                                    CutoutsCombiner, build_default_combine_function)  # noqa
