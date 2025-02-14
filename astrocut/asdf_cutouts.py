# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements cutout functionality similar to fitscut, but for the ASDF file format."""
from pathlib import Path
from typing import List, Union

import astropy
import gwcs
import numpy as np
from astropy.utils.decorators import deprecated_renamed_argument
from s3path import S3Path

from .ASDFCutout import ASDFCutout


def get_center_pixel(gwcsobj: gwcs.wcs.WCS, ra: float, dec: float) -> tuple:
    """ 
    Get the closest pixel location on an input image for a given set of coordinates.

    Parameters
    ----------
    gwcsobj : gwcs.wcs.WCS
        The GWCS object.
    ra : float
        The right ascension of the input coordinates.
    dec : float
        The declination of the input coordinates.

    Returns
    -------
    pixel_position
        The pixel position of the input coordinates.
    wcs_updated : `~astropy.wcs.WCS`
        The approximated FITS WCS object.
    """
    return ASDFCutout.get_center_pixel(gwcsobj, ra, dec)


@deprecated_renamed_argument('output_file', None, '1.0.0', warning_type=DeprecationWarning,
                             message='`output_file` is non-operational and will be removed in a future version.')
def asdf_cut(input_files: List[Union[str, Path, S3Path]], 
             ra: float, 
             dec: float, 
             cutout_size: int = 25,
             output_file: Union[str, Path] = "example_roman_cutout.fits",
             write_file: bool = True, 
             fill_value: Union[int, float] = np.nan,
             output_dir: Union[str, Path] = '.',
             output_format: str = '.asdf', 
             key: str = None,
             secret: str = None, 
             token: str = None, 
             verbose: bool = False) -> astropy.nddata.Cutout2D:
    """
    Takes one of more ASDF input files (`input_files`) and generates a cutout of designated size `cutout_size`
    around the given coordinates (`coordinates`). The cutout is written to a file or returned as an object.

    Parameters
    ----------
    input_file : str | Path | S3Path
        The input ASDF file.
    ra : float
        The right ascension of the central cutout.
    dec : float
        The declination of the central cutout.
    cutout_size : int
        Optional, default 25. The image cutout pixel size.
        Note: Odd values for `cutout_size` generally result in a cutout that is more accurately 
        centered on the target coordinates compared to even values, due to the symmetry of the 
        pixel grid. 
    output_file : str | Path
        Optional, default "example_roman_cutout.fits". The name of the output cutout file.
        This parameter is deprecated and will be removed in a future version.
    write_file : bool
        Optional, default True. Flag to write the cutout to a file or not.
    fill_value: int | float
        Optional, default `np.nan`. The fill value for pixels outside the original image.
    output_dir : str | Path
        Optional, default ".". The directory to write the cutout file(s) to.
    output_format : str
        Optional, default ".asdf". The format of the output cutout file. If `write_file` is False,
        then cutouts will be returned as `asdf.AsdfFile` objects if `output_format` is ".asdf" or
        as `astropy.io.fits.HDUList` objects if `output_format` is ".fits".
    key : string
        Default None. Access key ID for S3 file system. Only applicable if `input_file` is a
        cloud resource.
    secret : string
        Default None. Secret access key for S3 file system. Only applicable if `input_file` is a
        cloud resource.
    token : string
        Default None. Security token for S3 file system. Only applicable if `input_file` is a
        cloud resource.
    verbose : bool
        Default False. If True, intermediate information is printed.

    Returns
    -------
    response : str | list
        A list of cutout file paths if `write_file` is True, otherwise a list of cutout objects.
    """
    return ASDFCutout(input_files=input_files,
                      coordinates=(ra, dec),
                      cutout_size=cutout_size,
                      fill_value=fill_value,
                      memory_only=not write_file,
                      output_dir=output_dir,
                      output_format=output_format,
                      key=key,
                      secret=secret,
                      token=token,
                      verbose=verbose).cutout()
