# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements cutout functionality similar to fitscut."""

from pathlib import Path
from typing import List, Literal, Optional, Union, Tuple

from astropy.coordinates import SkyCoord
from astropy.io.fits import HDUList
from astropy.units import Quantity
from astropy.utils.decorators import deprecated_renamed_argument
import numpy as np
from s3path import S3Path
from .FITSCutout import FITSCutout
from .ImageCutout import ImageCutout


@deprecated_renamed_argument('correct_wcs', None, '1.0.0', warning_type=DeprecationWarning,
                             message='`correct_wcs` is non-operational and will be removed in a future version.')
def fits_cut(input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
             cutout_size: Union[int, np.ndarray, Quantity, List[int], Tuple[int]] = 25,
             correct_wcs: bool = False, extension: Optional[Union[int, List[int], Literal['all']]] = None,
             single_outfile: bool = True, cutout_prefix: str = "cutout", output_dir: Union[str, Path] = '.', 
             memory_only: bool = False, verbose=False) -> Union[str, List[str], List[HDUList]]:
    """
    Takes one or more fits files with the same WCS/pointing, makes the same cutout in each file,
    and returns the result either in a single FITS file with one cutout per extension or in 
    individual fits files. The memory_only flag allows the cutouts to be returned as 
    `~astropy.io.fits.HDUList` objects rather than saving to disk.

    Note: No checking is done on either the WCS pointing or pixel scale. If images don't line up
    the cutouts will also not line up.

    Parameters
    ----------
    input_files : list
        List of fits image files to cutout from. The image is assumed to be in the first extension.
    coordinates : str or `~astropy.coordinates.SkyCoord` object
        The position around which to cutout. It may be specified as a string ("ra dec" in degrees) 
        or as the appropriate `~astropy.coordinates.SkyCoord` object.
    cutout_size : int, array-like, `~astropy.units.Quantity`
        The size of the cutout array. If ``cutout_size`` is a scalar number or a scalar 
        `~astropy.units.Quantity`, then a square cutout of ``cutout_size`` will be created.  
        If ``cutout_size`` has two elements, they should be in ``(ny, nx)`` order.  Scalar numbers 
        in ``cutout_size`` are assumed to be in units of pixels. `~astropy.units.Quantity` objects 
        must be in pixel or angular units.
    extension : int, list of ints, None, or 'all'
       Optional, default None. Default is to cutout the first extension that has image data.
       The user can also supply one or more extensions to cutout from (integers), or 'all'.
    single_outfile : bool 
        Default True. If true return all cutouts in a single fits file with one cutout per extension,
        if False return cutouts in individual fits files. If returing a single file the filename will 
        have the form: <cutout_prefix>_<ra>_<dec>_<size x>_<size y>.fits. If returning multiple files
        each will be named: <original filemame base>_<ra>_<dec>_<size x>_<size y>.fits.
    cutout_prefix : str 
        Default value "cutout". Only used if single_outfile is True. A prefix to prepend to the cutout 
        filename. 
    output_dir : str
        Default value '.'. The directory to save the cutout file(s) to.
    memory_only : bool
        Default value False. If set to true, instead of the cutout file(s) being written to disk
        the cutout(s) are returned as a list of `~astropy.io.fit.HDUList` objects. If set to
        True cutout_prefix and output_dir are ignored, however single_outfile can still be used to
        set the number of returned `~astropy.io.fits.HDUList` objects.
    verbose : bool
        Default False. If true intermediate information is printed.
    fsspec_kwargs : any
        Default value {"anon": True}. This parameter should be used to provide cloud credentials to
        access private data buckets (e.g. {"key": "YOUR-SECRET-KEY-ID", "secret": "YOUR-SECRET-KEY"}).

    Returns
    -------
    response : str or list
        If single_outfile is True returns the single output filepath. Otherwise returns a list of all 
        the output filepaths.
        If memory_only is True a list of `~astropy.io.fit.HDUList` objects is returned instead of
        file name(s).
    """
    return FITSCutout(input_files,
                      coordinates=coordinates,
                      cutout_size=cutout_size,
                      memory_only=memory_only,
                      output_dir=output_dir,
                      output_format='.fits',
                      extension=extension,
                      single_outfile=single_outfile,
                      cutout_prefix=cutout_prefix,
                      verbose=verbose).cutout()


def normalize_img(img_arr: np.ndarray, stretch: str = 'asinh', minmax_percent: Optional[List[int]] = None, 
                  minmax_value: Optional[List[int]] = None, invert: bool = False) -> np.ndarray:
    """
    Apply given stretch and scaling to an image array.

    Parameters
    ----------
    img_arr : array
        The input image array.
    stretch : str
        Optional, default 'asinh'. The stretch to apply to the image array.
        Valid values are: asinh, sinh, sqrt, log, linear
    minmax_percent : array
        Optional. Interval based on a keeping a specified fraction of pixels (can be asymmetric) 
        when scaling the image. The format is [lower percentile, upper percentile], where pixel
        values below the lower percentile and above the upper percentile are clipped.
        Only one of minmax_percent and minmax_value shoul be specified.
    minmax_value : array
        Optional. Interval based on user-specified pixel values when scaling the image.
        The format is [min value, max value], where pixel values below the min value and above
        the max value are clipped.
        Only one of minmax_percent and minmax_value should be specified.
    invert : bool
        Optional, default False.  If True the image is inverted (light pixels become dark and vice versa).

    Returns
    -------
    response : array
        The normalized image array, in the form in an integer arrays with values in the range 0-255.
    """
    return ImageCutout.normalize_img(img_arr=img_arr,
                                     stretch=stretch,
                                     minmax_percent=minmax_percent,
                                     minmax_value=minmax_value,
                                     invert=invert)


def img_cut(input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
            cutout_size: Union[int, np.ndarray, Quantity, List[int], Tuple[int]] = 25, stretch: str = 'asinh', 
            minmax_percent: Optional[List[int]] = None, minmax_value: Optional[List[int]] = None, 
            invert: bool = False, img_format: str = '.jpg', colorize: bool = False,
            cutout_prefix: str = "cutout", output_dir: Union[str, Path] = '.', 
            extension: Optional[Union[int, List[int], Literal['all']]] = None, fill_value: Union[int, float] = np.nan,
            limit_rounding_method: str = 'round', verbose=False) -> Union[str, List[str]]:
    """
    Takes one or more fits files with the same WCS/pointing, makes the same cutout in each file,
    and returns the result either as a single color image or in individual image files.

    Note: No checking is done on either the WCS pointing or pixel scale. If images don't line up
    the cutouts will also not line up.

    Parameters
    ----------
    input_files : list
        List of fits image files to cutout from. The image is assumed to be in the first extension.
    coordinates : str or `~astropy.coordinates.SkyCoord` object
        The position around which to cutout. It may be specified as a string ("ra dec" in degrees) 
        or as the appropriate `~astropy.coordinates.SkyCoord` object.
    cutout_size : int, array-like, `~astropy.units.Quantity`
        The size of the cutout array. If ``cutout_size`` is a scalar number or a scalar 
        `~astropy.units.Quantity`, then a square cutout of ``cutout_size`` will be created.  
        If ``cutout_size`` has two elements, they should be in ``(ny, nx)`` order.  Scalar numbers 
        in ``cutout_size`` are assumed to be in units of pixels. `~astropy.units.Quantity` objects 
        must be in pixel or angular units.
    fill_value : int | float
        Value to fill the cutout with if the cutout is outside the image.
    stretch : str
        Optional, default 'asinh'. The stretch to apply to the image array.
        Valid values are: asinh, sinh, sqrt, log, linear
    minmax_percent : array
        Optional, default [0.5,99.5]. Interval based on a keeping a specified fraction of pixels 
        (can be asymmetric) when scaling the image. The format is [lower percentile, upper percentile], 
        where pixel values below the lower percentile and above the upper percentile are clipped.
        Only one of minmax_percent and minmax_value should be specified.
    minmax_value : array
        Optional. Interval based on user-specified pixel values when scaling the image.
        The format is [min value, max value], where pixel values below the min value and above
        the max value are clipped.
        Only one of minmax_percent and minmax_value should be specified.
    invert : bool
        Optional, default False.  If True the image is inverted (light pixels become dark and vice versa).
    img_format : str
        Optional, default 'jpg'. The output image file type. Valid values are "jpg" and "png".
    colorize : bool
        Optional, default False.  If True a single color image is produced as output, and it is expected
        that three files are given as input.
    cutout_prefix : str 
        Default value "cutout". Only used when producing a color image. A prefix to prepend to the 
        cutout filename. 
    output_dir : str
        Defaul value '.'. The directory to save the cutout file(s) to.
    limit_rounding_method : str
        Method to use for rounding the cutout limits. Options are 'round', 'ceil', and 'floor'.
    extension : int, list of ints, None, or 'all'
        Optional, default None. Default is to cutout the first extension that has image data.
        The user can also supply one or more extensions to cutout from (integers), or "all".
    verbose : bool
        Default False. If true intermediate information is printed.

    Returns
    -------
    response : str or list
        If colorize is True returns the single output filepath. Otherwise returns a list of all 
        the output filepaths.
    """

    return FITSCutout(input_files=input_files,
                      coordinates=coordinates,
                      cutout_size=cutout_size,
                      fill_value=fill_value,
                      output_dir=output_dir,
                      limit_rounding_method=limit_rounding_method,
                      stretch=stretch,
                      minmax_percent=minmax_percent,
                      minmax_value=minmax_value,
                      invert=invert,
                      colorize=colorize,
                      output_format=img_format,
                      cutout_prefix=cutout_prefix,
                      extension=extension,
                      verbose=verbose).cutout()
