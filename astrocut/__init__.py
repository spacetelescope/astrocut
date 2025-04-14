# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *  # noqa
# ----------------------------------------------------------------------------

"""
This module initializes the astrocut package and performs essential setup tasks, including:
- Verifying the version of Python.
- Setting up package-wide logging.
- Importing key modules.
"""

import sys

from .exceptions import UnsupportedPythonError
from .utils.logger import setup_logger

# Enforce Python version check during package import.
__minimum_python_version__ = "3.9"  # minimum supported Python version
if sys.version_info < tuple(map(int, __minimum_python_version__.split('.'))):
    raise UnsupportedPythonError(f"astrocut does not support Python < {__minimum_python_version__}")

# Initialize package-wide logger using astropy's logging system
log = setup_logger()

# Import key submodules and functions if not in setup mode
if not _ASTROPY_SETUP_:  # noqa
    from .cube_factory import CubeFactory  # noqa
    from .tica_cube_factory import TicaCubeFactory  # noqa
    from .cutout_factory import CutoutFactory, cube_cut  # noqa
    from .cutout_processing import (  # noqa
        path_to_footprints, center_on_path, CutoutsCombiner, build_default_combine_function  # noqa
    )  # noqa
    from .image_cutout import normalize_img # noqa
    from .fits_cutout import FITSCutout, fits_cut, img_cut  # noqa
    from .asdf_cutout import ASDFCutout, asdf_cut, get_center_pixel  # noqa
    from .tess_cube_cutout import TessCubeCutout  # noqa
    from .footprint_cutout import ra_dec_crossmatch # noqa
    from .tess_footprint_cutout import TessFootprintCutout, cube_cut_from_footprint  # noqa