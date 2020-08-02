# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements cutout functionality similar to fitscut."""

import os
import warnings
import numpy as np

from time import time
from datetime import date
from itertools import product

from astropy import log
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.visualization import (SqrtStretch, LogStretch, AsinhStretch, SinhStretch, LinearStretch,
                                   MinMaxInterval, ManualInterval, AsymmetricPercentileInterval)

from PIL import Image

from . import __version__
from .exceptions import InputWarning, DataWarning, InvalidQueryError, InvalidInputError


#### FUNCTIONS FOR UTILS ####
def _get_cutout_limits(img_wcs, center_coord, cutout_size):
    """
    Takes the center coordinates, cutout size, and the wcs from
    which the cutout is being taken and returns the x and y pixel limits
    for the cutout.

    Note: This function does no bounds checking, so the returned limits are not 
          guaranteed to overlap the original image.

    Parameters
    ----------
    img_wcs : `~astropy.wcs.WCS`
        The WCS for the image that the cutout is being cut from.
    center_coord : `~astropy.coordinates.SkyCoord`
        The central coordinate for the cutout
    cutout_size : array
        [nx,ny] in with ints (pixels) or astropy quantities

    Returns
    -------
    response : `numpy.array`
        The cutout pixel limits in an array of the form [[xmin,xmax],[ymin,ymax]]
    """
        
    # Note: This is returning the center pixel in 1-up
    try:
        center_pixel = center_coord.to_pixel(img_wcs, 1)
    except wcs.NoConvergence:  # If wcs can't converge, center coordinate is far from the footprint
        raise InvalidQueryError("Cutout location is not in image footprint!")

    # For some reason you can sometimes get nans without a no convergance error
    if np.isnan(center_pixel).all():
        raise InvalidQueryError("Cutout location is not in image footprint!")
    
    lims = np.zeros((2, 2), dtype=int)

    for axis, size in enumerate(cutout_size):
        
        if not isinstance(size, u.Quantity):  # assume pixels
            dim = size / 2
        elif size.unit == u.pixel:  # also pixels
            dim = size.value / 2
        elif size.unit.physical_type == 'angle':
            pixel_scale = u.Quantity(wcs.utils.proj_plane_pixel_scales(img_wcs)[axis],
                                     img_wcs.wcs.cunit[axis])
            dim = (size / pixel_scale).decompose() / 2

        lims[axis, 0] = int(np.round(center_pixel[axis] - 1 - dim))
        lims[axis, 1] = int(np.round(center_pixel[axis] - 1 + dim))

        # The case where the requested area is so small it rounds to zero
        if lims[axis, 0] == lims[axis, 1]:
            lims[axis, 0] = int(np.floor(center_pixel[axis] - 1))
            lims[axis, 1] = lims[axis, 0] + 1

    return lims

        
def _get_cutout_wcs(img_wcs, cutout_lims):
    """
    Starting with the full FFI WCS and adjusting it for the cutout WCS.
    Adjusts CRPIX values and adds physical WCS keywords.

    Parameters
    ----------
    img_wcs : `~astropy.wcs.WCS`
        WCS for the image the cutout is being cut from.
    cutout_lims : `numpy.array`
        The cutout pixel limits in an array of the form [[ymin,ymax],[xmin,xmax]]

    Returns
    --------
    response :  `~astropy.wcs.WCS`
        The cutout WCS object including SIP distortions if present.
    """

    # relax = True is important when the WCS has sip distortions, otherwise it has no effect
    wcs_header = img_wcs.to_header(relax=True) 

    # Adjusting the CRPIX values
    wcs_header["CRPIX1"] -= cutout_lims[0, 0]
    wcs_header["CRPIX2"] -= cutout_lims[1, 0]

    # Adding the physical wcs keywords
    wcs_header.set("WCSNAMEP", "PHYSICAL", "name of world coordinate system alternate P")
    wcs_header.set("WCSAXESP", 2, "number of WCS physical axes")
    
    wcs_header.set("CTYPE1P", "RAWX", "physical WCS axis 1 type CCD col")
    wcs_header.set("CUNIT1P", "PIXEL", "physical WCS axis 1 unit")
    wcs_header.set("CRPIX1P", 1, "reference CCD column")
    wcs_header.set("CRVAL1P", cutout_lims[0, 0] + 1, "value at reference CCD column")
    wcs_header.set("CDELT1P", 1.0, "physical WCS axis 1 step")
                
    wcs_header.set("CTYPE2P", "RAWY", "physical WCS axis 2 type CCD col")
    wcs_header.set("CUNIT2P", "PIXEL", "physical WCS axis 2 unit")
    wcs_header.set("CRPIX2P", 1, "reference CCD row")
    wcs_header.set("CRVAL2P", cutout_lims[1, 0] + 1, "value at reference CCD row")
    wcs_header.set("CDELT2P", 1.0, "physical WCS axis 2 step")

    return wcs.WCS(wcs_header)
        

def remove_sip_coefficients(hdu_header):
    """
    Remove standard sip coefficient keywords for a fits header.

    Parameters
    ----------
    hdu_header : ~astropy.io.fits.Header
        The header from which SIP keywords will be removed.  This is done in place.
    """

    for lets in product(["A", "B"], ["", "P"]):
        lets = ''.join(lets)

        key = "{}_ORDER".format(lets)
        if key in hdu_header.keys():
            del hdu_header["{}_ORDER".format(lets)]

        key = "{}_DMAX".format(lets)
        if key in hdu_header.keys():
            del hdu_header["{}_DMAX".format(lets)]
        
        for i, j in product([0, 1, 2, 3], [0, 1, 2, 3]):
            key = "{}_{}_{}".format(lets, i, j)
            if key in hdu_header.keys():
                del hdu_header["{}_{}_{}".format(lets, i, j)]
#### FUNCTIONS FOR UTILS ####


def _hducut(img_hdu, center_coord, cutout_size, correct_wcs=False, verbose=False):
    """
    Takes an ImageHDU (image and associated metatdata in the fits format), as well as a center 
    coordinate and size and make a cutout of that image, which is returned as another ImageHDU,
    including updated  WCS information.


    Parameters
    ----------
    img_hdu : `~astropy.io.fits.hdu.image.ImageHDU`
        The image and assciated metadata that is being cut out.
    center_coord : `~astropy.coordinates.sky_coordinate.SkyCoord`
        The coordinate to cut out around.
    cutout_size : array
        The size of the cutout as [nx,ny], where nx/ny can be integers (assumed to be pixels)
        or `~astropy.Quantity` values, either pixels or angular quantities.
    correct_wcs : bool
        Default False. If true a new WCS will be created for the cutout that is tangent projected
        and does not include distortions.  
    verbose : bool
        Default False. If true intermediate information is printed.

    Returns
    -------
    response : `~astropy.io.fits.hdu.image.ImageHDU` 
        The cutout image and associated metadata.
    """
    
    hdu_header = fits.Header(img_hdu.header, copy=True)

    # We are going to reroute the logging to a string stream temporarily so we can
    # intercept any message from astropy, chiefly the "Inconsistent SIP distortion information"
    # INFO message which will indicate that we need to remove existing SIP keywords
    # from a WCS whose CTYPE does not include SIP. In this we are taking the CTYPE to be
    # correct and adjusting the header keywords to match.
    hdlrs = log.handlers
    log.handlers = []
    with log.log_to_list() as log_list:        
        img_wcs = wcs.WCS(hdu_header, relax=True)

    for hd in hdlrs:
        log.addHandler(hd)

    no_sip = False
    if (len(log_list) > 0):
        if ("Inconsistent SIP distortion information" in log_list[0].msg):
            # Delete standard sip keywords
            remove_sip_coefficients(hdu_header)
        
            # load wcs ignoring any nonstandard keywords
            img_wcs = wcs.WCS(hdu_header, relax=False)

            # As an extra precaution make sure the img wcs has no sip coeefficients
            img_wcs.sip = None
            no_sip = True
            
        else:  # Message(s) we didn't prepare for we want to go ahead and display
            for log_rec in log_list:
                log.log(log_rec.levelno, log_rec.msg, extra={"origin": log_rec.name})

    img_data = img_hdu.data

    if verbose:
        print("Original image shape: {}".format(img_data.shape))

    # Get cutout limits
    cutout_lims = _get_cutout_limits(img_wcs, center_coord, cutout_size)

    if verbose:
        print("xmin,xmax: {}".format(cutout_lims[0]))
        print("ymin,ymax: {}".format(cutout_lims[1]))

    # These limits are not guarenteed to be within the image footprint
    xmin, xmax = cutout_lims[0]
    ymin, ymax = cutout_lims[1]

    ymax_img, xmax_img = img_data.shape

    # Check the cutout is on the image
    if (xmax <= 0) or (xmin >= xmax_img) or (ymax <= 0) or (ymin >= ymax_img):
        raise InvalidQueryError("Cutout location is not in image footprint!")

    # Adjust limits and figuring out the` padding
    padding = np.zeros((2, 2), dtype=int)
    if xmin < 0:
        padding[1, 0] = -xmin
        xmin = 0
    if ymin < 0:
        padding[0, 0] = -ymin
        ymin = 0
    if xmax > xmax_img:
        padding[1, 1] = xmax - xmax_img
        xmax = xmax_img
    if ymax > ymax_img:
        padding[0, 1] = ymax - ymax_img
        ymax = ymax_img  
        
    img_cutout = img_hdu.data[ymin:ymax, xmin:xmax]

    # Adding padding to the cutout so that it's the expected size
    if padding.any():  # only do if we need to pad
        img_cutout = np.pad(img_cutout, padding, 'constant', constant_values=np.nan)

    if verbose:
        print("Image cutout shape: {}".format(img_cutout.shape))

    # Getting the cutout wcs
    cutout_wcs = _get_cutout_wcs(img_wcs, cutout_lims)

    # Updating the header with the new wcs info
    if no_sip:
        hdu_header.update(cutout_wcs.to_header(relax=False))
    else:
        hdu_header.update(cutout_wcs.to_header(relax=True))  # relax arg is for sip distortions if they exist

    # Naming the extension
    hdu_header["EXTNAME"] = "CUTOUT"

    # Moving the filename, if present, into the ORIG_FLE keyword
    hdu_header["ORIG_FLE"] = (hdu_header.get("FILENAME"), "Original image filename.")
    hdu_header.remove("FILENAME", ignore_missing=True)

    hdu = fits.ImageHDU(header=hdu_header, data=img_cutout)

    return hdu


def _save_single_fits(cutout_hdus, output_path, center_coord):
    """
    Save a list of cutout hdus to a single fits file.

    Parameters
    ----------
    cutout_hdus : list
        List of `~astropy.io.fits.hdu.image.ImageHDU` objects to be written to a single fits file.
    output_path : str
        The full path to the output fits file.
    center_coord : `~astropy.coordinates.sky_coordinate.SkyCoord`
        The center coordinate of the image cutouts.
    """

    # Setting up the Primary HDU
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header.extend([("ORIGIN", 'STScI/MAST', "institution responsible for creating this file"),
                               ("DATE", str(date.today()), "file creation date"),
                               ('PROCVER', __version__, 'software version'),
                               ('RA_OBJ', center_coord.ra.deg, '[deg] right ascension'),
                               ('DEC_OBJ', center_coord.dec.deg, '[deg] declination')])

    cutout_hdulist = fits.HDUList([primary_hdu] + cutout_hdus)

    # Writing out the hdu often causes a warning as the ORIG_FLE card description is truncated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        cutout_hdulist.writeto(output_path, overwrite=True, checksum=True)


def _save_multiple_fits(cutout_hdus, output_paths, center_coord):
    """
    Save a list of cutout hdus to individual fits files.

    Parameters
    ----------
    cutout_hdus : list
        List of `~astropy.io.fits.hdu.image.ImageHDU` objects to be written to a single fits file.
    output_paths : list
        The cutout filepaths associated with the cutout hdus.
    center_coord : `~astropy.coordinates.sky_coordinate.SkyCoord`
        The center coordinate of the image cutouts.
    """

    if len(output_paths) != len(cutout_hdus):
        raise InvalidInputError("The number of filenames must match the number of cutouts.")

    # Adding aditional keywords
    for i, cutout in enumerate(cutout_hdus):
        # Turning our hdu into a primary hdu
        hdu = fits.PrimaryHDU(header=cutout.header, data=cutout.data)
        hdu.header.extend([("ORIGIN", 'STScI/MAST', "institution responsible for creating this file"),
                           ("DATE", str(date.today()), "file creation date"),
                           ('PROCVER', __version__, 'software version'),
                           ('RA_OBJ', center_coord.ra.deg, '[deg] right ascension'),
                           ('DEC_OBJ', center_coord.dec.deg, '[deg] declination')])

        cutout_hdulist = fits.HDUList([hdu])
        cutout_hdulist.writeto(output_paths[i], overwrite=True, checksum=True)

    
def _parse_size_input(cutout_size):
    """
    Makes the given cutout size into a length 2 array.

    Parameters
    ----------
    cutout_size : int, array-like, `~astropy.units.Quantity`
        The size of the cutout array. If ``cutout_size`` is a scalar number or a scalar 
        `~astropy.units.Quantity`, then a square cutout of ``cutout_size`` will be created.  
        If ``cutout_size`` has two elements, they should be in ``(ny, nx)`` order.  Scalar numbers 
        in ``cutout_size`` are assumed to be in units of pixels. `~astropy.units.Quantity` objects 
        must be in pixel or angular units.

    Returns
    -------
    response : array
        Length two cutout size array, in the form [ny, nx].
    """

    # Making size into an array [ny, nx]
    if np.isscalar(cutout_size):
        cutout_size = np.repeat(cutout_size, 2)

    if isinstance(cutout_size, u.Quantity):
        cutout_size = np.atleast_1d(cutout_size)
        if len(cutout_size) == 1:
            cutout_size = np.repeat(cutout_size, 2)

    if len(cutout_size) > 2:
        warnings.warn("Too many dimensions in cutout size, only the first two will be used.",
                      InputWarning)
        cutout_size = cutout_size[:2]

    return cutout_size

       
def fits_cut(input_files, coordinates, cutout_size, correct_wcs=False, drop_after=None, 
             single_outfile=True, cutout_prefix="cutout", output_dir='.', verbose=False):
    """
    Takes one or more fits files with the same WCS/pointing, makes the same cutout in each file,
    and returns the result either in a single fitsfile with one cutout per extension or in 
    individual fits files.

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
    correct_wcs : bool
        Default False. If true a new WCS will be created for the cutout that is tangent projected
        and does not include distortions.
    single_outfile : bool 
        Default True. If true return all cutouts in a single fits file with one cutout per extension,
        if False return cutouts in individual fits files. If returing a single file the filename will 
        have the form: <cutout_prefix>_<ra>_<dec>_<size x>_<size y>.fits. If returning multiple files
        each will be named: <original filemame base>_<ra>_<dec>_<size x>_<size y>.fits.
    cutout_prefix : str 
        Default value "cutout". Only used if single_outfile is True. A prefix to prepend to the cutout 
        filename. 
    output_dir : str
        Defaul value '.'. The directory to save the cutout file(s) to.
    verbose : bool
        Default False. If true intermediate information is printed.

    Returns
    -------
    response : str or list
        If single_outfile is True returns the single output filepath. Otherwise returns a list of all 
        the output filepaths.
    """

    # Dealing with deprecation
    if drop_after is not None:
        warnings.warn("Argument 'drop_after' is deprecated and will be ignored",
                      AstropyDeprecationWarning)
    
    if verbose:
        start_time = time()
            
    # Making sure we have an array of images
    if type(input_files) == str:
        input_files = [input_files]

    if not isinstance(coordinates, SkyCoord):
        coordinates = SkyCoord(coordinates, unit='deg')

    # Turning the cutout size into a 2 member array
    cutout_size = _parse_size_input(cutout_size)

    # Making the cutouts
    cutout_hdu_dict = {}
    num_empty = 0
    for in_fle in input_files:
        if verbose:
            print("\n{}".format(in_fle))

        warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)
        with fits.open(in_fle, mode='denywrite', memmap=True) as hdulist:
            try:
                cutout = _hducut(hdulist[0], coordinates, cutout_size,
                                 correct_wcs=correct_wcs, verbose=verbose)
            except OSError as err:
                warnings.warn("Error {} encountered when performing cutout on {}, skipping...".format(err, in_fle),
                              DataWarning)
        
        # Check that there is data in the cutout image
        if (cutout.data == 0).all() or (np.isnan(cutout.data)).all():
            cutout.header["EMPTY"] = (True, "Indicates no data in cutout image.")
            num_empty += 1
            
        cutout_hdu_dict[in_fle] = cutout

    # If no cutouts contain data, raise exception
    if num_empty == len(input_files):
        raise InvalidQueryError("Cutout contains no data! (Check image footprint.)")

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Setting up the output file(s) and writing them
    if single_outfile:

        cutout_path = "{}_{:7f}_{:7f}_{}-x-{}_astrocut.fits".format(cutout_prefix,
                                                                    coordinates.ra.value,
                                                                    coordinates.dec.value,
                                                                    str(cutout_size[0]).replace(' ', ''), 
                                                                    str(cutout_size[1]).replace(' ', ''))
        cutout_path = os.path.join(output_dir, cutout_path)
        cutout_hdus = [cutout_hdu_dict[fle] for fle in input_files]
        _save_single_fits(cutout_hdus, cutout_path, coordinates)

    else:
        cutout_hdus = []
        cutout_path = []
        for fle in input_files:
            cutout = cutout_hdu_dict[fle]
            if cutout.header.get("EMPTY"):
                warnings.warn("Cutout of {} contains to data and will not be written.".format(fle),
                              DataWarning)
                continue

            cutout_hdus.append(cutout)

            filename = "{}_{:7f}_{:7f}_{}-x-{}_astrocut.fits".format(os.path.basename(fle).rstrip('.fits'),
                                                                     coordinates.ra.value,
                                                                     coordinates.dec.value,
                                                                     str(cutout_size[0]).replace(' ', ''), 
                                                                     str(cutout_size[1]).replace(' ', ''))
            cutout_path.append(os.path.join(output_dir, filename))
                
        _save_multiple_fits(cutout_hdus, cutout_path, coordinates)

        
    if verbose:
        print("Cutout fits file(s): {}".format(cutout_path))
        print("Total time: {:.2} sec".format(time()-start_time))

    return cutout_path


def normalize_img(img_arr, stretch='asinh', minmax_percent=None, minmax_value=None, invert=False):
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


    # Setting up the transform with the stretch
    if stretch == 'asinh':
        transform = AsinhStretch()
    elif stretch == 'sinh':
        transform = SinhStretch()
    elif stretch == 'sqrt':
        transform = SqrtStretch()
    elif stretch == 'log':
        transform = LogStretch()
    elif stretch == 'linear':
        transform = LinearStretch()
    else:
        raise InvalidInputError("Stretch {} is not supported!".format(stretch))

    # Adding the scaling to the transform
    if minmax_percent is not None:
        transform += AsymmetricPercentileInterval(*minmax_percent)
        
        if minmax_value is not None:
            warnings.warn("Both minmax_percent and minmax_value are set, minmax_value will be ignored.",
                          InputWarning)
    elif minmax_value is not None:
        transform += ManualInterval(*minmax_value)
    else:  # Default, scale the entire image range to [0,1]
        transform += MinMaxInterval()
   
    # Performing the transform and then putting it into the integer range 0-255
    norm_img = transform(img_arr)
    norm_img = np.multiply(255, norm_img, out=norm_img)
    norm_img = norm_img.astype(np.uint8)

    # Applying invert if requested
    if invert:
        norm_img = 255 - norm_img

    return norm_img


def img_cut(input_files, coordinates, cutout_size, stretch='asinh', minmax_percent=None,
            minmax_value=None, invert=False, img_format='jpg', colorize=False,
            cutout_prefix="cutout", output_dir='.', drop_after=None, verbose=False):
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
    verbose : bool
        Default False. If true intermediate information is printed.

    Returns
    -------
    response : str or list
        If colorize is True returns the single output filepath. Otherwise returns a list of all 
        the output filepaths.
    """

    # Dealing with deprecation
    if drop_after is not None:
        warnings.warn("Argument 'drop_after' is deprecated and will be ignored",
                      AstropyDeprecationWarning)
        
    if verbose:
        start_time = time()
            
    # Making sure we have an array of images
    if type(input_files) == str:
        input_files = [input_files]
    
    # Doing image checks for color images
    if colorize:
        if len(input_files) < 3:
            raise InvalidInputError("Color cutouts require 3 imput files (RGB).")
        if len(input_files) > 3:
            warnings.warn("Too many inputs for a color cutout, only the first three will be used.",
                          InputWarning)
            input_files = input_files[:3]
            
    if not isinstance(coordinates, SkyCoord):
        coordinates = SkyCoord(coordinates, unit='deg')

    # Turning the cutout size into a 2 member array
    cutout_size = _parse_size_input(cutout_size)

    # Applying the default scaling
    if (minmax_percent is None) and (minmax_value is None):
        minmax_percent = [0.5, 99.5]
        
    # Making the cutouts
    cutout_hdu_dict = {}
    for in_fle in input_files:
        if verbose:
            print("\n{}".format(in_fle))
        hdulist = fits.open(in_fle, mode='denywrite', memmap=True)
        cutout = _hducut(hdulist[0], coordinates, cutout_size,
                         correct_wcs=False, verbose=verbose)
        hdulist.close()

        # We just want the data array
        cutout = cutout.data
        
        # Applying the appropriate normalization parameters
        normalized_cutout = normalize_img(cutout, stretch, minmax_percent, minmax_value, invert)
        
        # Check that there is data in the cutout image
        if not (cutout == 0).all():
            cutout_hdu_dict[in_fle] = normalized_cutout

    # If no cutouts contain data, raise exception
    if not cutout_hdu_dict:
        raise InvalidQueryError("Cutout contains to data! (Check image footprint.)")

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Setting up the output file(s) and writing them
    if colorize:

        cutout_path = "{}_{:7f}_{:7f}_{}-x-{}_astrocut.{}".format(cutout_prefix,
                                                                  coordinates.ra.value,
                                                                  coordinates.dec.value,
                                                                  str(cutout_size[0]).replace(' ', ''), 
                                                                  str(cutout_size[1]).replace(' ', ''),
                                                                  img_format.lower()) 
        cutout_path = os.path.join(output_dir, cutout_path)

        # TODO: This is not elegant or efficient, make it better
        red = cutout_hdu_dict.get(input_files[0])
        green = cutout_hdu_dict.get(input_files[1])
        blue = cutout_hdu_dict.get(input_files[2])

        cshape = ()
        for cutout in [red, green, blue]:
            if cutout is not None:
                cshape = cutout.shape
                break

        if red is None:
            red = np.zeros(cshape)
        if green is None:
            green = np.zeros(cshape)
        if blue is None:
            blue = np.zeros(cshape)

        Image.fromarray(np.dstack([red, green, blue]).astype(np.uint8)).save(cutout_path)
          
    else:
 
        cutout_path = []
        for fle in input_files:
            cutout = cutout_hdu_dict.get(fle)
            if cutout is None:
                warnings.warn("Cutout of {} contains to data and will not be written.".format(fle),
                              DataWarning)
                continue

            file_path = "{}_{:7f}_{:7f}_{}-x-{}_astrocut.{}".format(os.path.basename(fle).rstrip('.fits'),
                                                                    coordinates.ra.value,
                                                                    coordinates.dec.value,
                                                                    str(cutout_size[0]).replace(' ', ''), 
                                                                    str(cutout_size[1]).replace(' ', ''),
                                                                    img_format.lower())
            file_path = os.path.join(output_dir, file_path)
            cutout_path.append(file_path)
            
            Image.fromarray(cutout).save(file_path)
        
    if verbose:
        print("Cutout fits file(s): {}".format(cutout_path))
        print("Total time: {:.2} sec".format(time()-start_time))

    return cutout_path
    
    
    
