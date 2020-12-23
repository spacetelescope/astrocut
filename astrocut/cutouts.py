# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements cutout functionality similar to fitscut."""

import os
import warnings
import numpy as np

from time import time
from datetime import date
from itertools import product

from astropy import log
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.visualization import (SqrtStretch, LogStretch, AsinhStretch, SinhStretch, LinearStretch,
                                   MinMaxInterval, ManualInterval, AsymmetricPercentileInterval)

from PIL import Image

from .utils.utils import parse_size_input, get_cutout_limits, get_cutout_wcs, remove_sip_coefficients, save_fits
from .exceptions import InputWarning, DataWarning, InvalidQueryError, InvalidInputError


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

            # Remove sip coefficients
            img_wcs.sip = None
            no_sip = True
            
        else:  # Message(s) we didn't prepare for we want to go ahead and display
            for log_rec in log_list:
                log.log(log_rec.levelno, log_rec.msg, extra={"origin": log_rec.name})

    img_data = img_hdu.data

    if verbose:
        print("Original image shape: {}".format(img_data.shape))

    # Get cutout limits
    cutout_lims = get_cutout_limits(img_wcs, center_coord, cutout_size)

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
    cutout_wcs = get_cutout_wcs(img_wcs, cutout_lims)

    # Updating the header with the new wcs info
    if no_sip:
        hdu_header.update(cutout_wcs.to_header(relax=False))
    else:
        hdu_header.update(cutout_wcs.to_header(relax=True))  # relax arg is for sip distortions if they exist

    # Naming the extension and preserving the original name
    hdu_header["O_EXT_NM"] = (hdu_header.get("EXTNAME"), "Original extension name.")
    hdu_header["EXTNAME"] = "CUTOUT"

    # Moving the filename, if present, into the ORIG_FLE keyword
    hdu_header["ORIG_FLE"] = (hdu_header.get("FILENAME"), "Original image filename.")
    hdu_header.remove("FILENAME", ignore_missing=True)

    hdu = fits.ImageHDU(header=hdu_header, data=img_cutout)

    return hdu


def _parse_extensions(infile_exts, infile_name, user_exts):
    """
    Given a list of image extensions available in the file with infile_name, cross-match with
    user input extensions to figure out which extensions to use for cutout.

    Parameters
    ----------
    infile_exts : array
    infile_name : str
    user_exts : int, list of ints, None, or 'all'
        Optional, default None. Default is to cutout the first extension that has image data.
        The user can also supply one or more extensions to cutout from (integers), or "all".

    Returns
    -------
    response : array
        List of extensions to be cutout.
    """
 
    if len(infile_exts) == 0:
        warnings.warn(f"No image extensions with data found in {infile_name}, skipping...",
                      DataWarning)
        return []
            
    if user_exts is None:
        cutout_exts = infile_exts[:1]  # Take the first image extension
    elif user_exts == 'all':
        cutout_exts = infile_exts  # Take all the extensions
    else:  # User input extentions
        cutout_exts = [x for x in infile_exts if x in user_exts]
        if len(cutout_exts) < len(user_exts):
            warnings.warn((f"Not all requested extensions in {infile_name} are image extensions or have data, "
                           f"extension(s) {','.join([x for x in user_exts if x not in cutout_exts])} will be skipped."),
                          DataWarning)

    return cutout_exts

                    
def fits_cut(input_files, coordinates, cutout_size, correct_wcs=False, extension=None, 
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
    extension : int, list of ints, None, or 'all'
        Optional, default None. Default is to cutout the first extension that has image data.
       The user can also supply one or more extensions to cutout from (integers), or "all".
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
    
    if verbose:
        start_time = time()
            
    # Making sure we have an array of images
    if type(input_files) == str:
        input_files = [input_files]

    # If a single extension is given, make it a list
    if isinstance(extension, int):
        extension = [extension]

    if not isinstance(coordinates, SkyCoord):
        coordinates = SkyCoord(coordinates, unit='deg')

    # Turning the cutout size into a 2 member array
    cutout_size = parse_size_input(cutout_size)

    # Making the cutouts
    cutout_hdu_dict = {}
    num_empty = 0
    num_cutouts = 0
    for in_fle in input_files:
        if verbose:
            print("\n{}".format(in_fle))

        warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)
        with fits.open(in_fle, mode='denywrite', memmap=True) as hdulist:

            # Sorting out which extension(s) to cutout
            all_inds = np.where([x.is_image and (x.data is not None) for x in hdulist])[0]
            cutout_inds = _parse_extensions(all_inds, in_fle, extension)

            num_cutouts += len(cutout_inds)
            for ind in cutout_inds:            
                try:
                    cutout = _hducut(hdulist[ind], coordinates, cutout_size,
                                     correct_wcs=correct_wcs, verbose=verbose)

                    # Check that there is data in the cutout image
                    if (cutout.data == 0).all() or (np.isnan(cutout.data)).all():
                        cutout.header["EMPTY"] = (True, "Indicates no data in cutout image.")
                        num_empty += 1

                    # Adding a few more keywords
                    cutout.header["ORIG_EXT"] = (ind, "Extension in original file.")
                    if not cutout.header.get("ORIG_FLE") and hdulist[0].header.get("FILENAME"):
                        cutout.header["ORIG_FLE"] = hdulist[0].header.get("FILENAME")
                    
                    cutout_hdu_dict[in_fle] = cutout_hdu_dict.get(in_fle, []) + [cutout]
                    
                except OSError as err:
                    warnings.warn((f"Error {err} encountered when performing cutout on {in_fle}, "
                                   f"extension {ind}, skipping..."),
                                  DataWarning)
                    num_empty += 1

    # If no cutouts contain data, raise exception
    if num_empty == num_cutouts:
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
        cutout_hdus = [x for fle in cutout_hdu_dict for x in cutout_hdu_dict[fle]]
        save_fits(cutout_hdus, cutout_path, coordinates)

        if verbose:
            print("Cutout fits file: {}".format(cutout_path))

        all_paths = cutout_path

    else:  # one output file per input file
        all_paths = []
        
        if verbose:
            print("Cutout fits files:")
            
        for fle in input_files:
            cutout_list = cutout_hdu_dict[fle]
            if np.array([x.header.get("EMPTY") for x in cutout_list]).all():
                warnings.warn(f"Cutout of {fle} contains no data and will not be written.",
                              DataWarning)
                continue

            filename = "{}_{:7f}_{:7f}_{}-x-{}_astrocut.fits".format(os.path.basename(fle).rstrip('.fits'),
                                                                     coordinates.ra.value,
                                                                     coordinates.dec.value,
                                                                     str(cutout_size[0]).replace(' ', ''), 
                                                                     str(cutout_size[1]).replace(' ', ''))
            cutout_path = os.path.join(output_dir, filename)
            save_fits(cutout_list, cutout_path, coordinates)

            all_paths.append(cutout_path)
            
            if verbose:
                print(cutout_path)
        
    if verbose:
        print("Total time: {:.2} sec".format(time()-start_time))

    return all_paths


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
            cutout_prefix="cutout", output_dir='.', extension=None, verbose=False):
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
        
    if verbose:
        start_time = time()
            
    # Making sure we have an array of images
    if type(input_files) == str:
        input_files = [input_files]
            
    if not isinstance(coordinates, SkyCoord):
        coordinates = SkyCoord(coordinates, unit='deg')

    # Turning the cutout size into a 2 member array
    cutout_size = parse_size_input(cutout_size)

    # Applying the default scaling
    if (minmax_percent is None) and (minmax_value is None):
        minmax_percent = [0.5, 99.5]
        
    # Making the cutouts
    cutout_hdu_dict = {}
    for in_fle in input_files:
        if verbose:
            print("\n{}".format(in_fle))


        warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)
        with fits.open(in_fle, mode='denywrite', memmap=True) as hdulist:

            # Sorting out which extension(s) to cutout
            all_inds = np.where([x.is_image and (x.data is not None) for x in hdulist])[0]
            cutout_inds = _parse_extensions(all_inds, in_fle, extension)

            for ind in cutout_inds:   
                try:
                    cutout = _hducut(hdulist[ind], coordinates, cutout_size, correct_wcs=False, verbose=verbose)

                    # We just want the data array
                    cutout = cutout.data
        
                    # Applying the appropriate normalization parameters
                    normalized_cutout = normalize_img(cutout, stretch, minmax_percent, minmax_value, invert)
        
                    # Check that there is data in the cutout image
                    if not (cutout == 0).all():
                        cutout_hdu_dict[in_fle] = cutout_hdu_dict.get(in_fle, []) + [normalized_cutout]
                    
                except OSError as err:
                    warnings.warn("Error {} encountered when performing cutout on {}, skipping...".format(err, in_fle),
                                  DataWarning)

    # If no cutouts contain data, raise exception
    if not cutout_hdu_dict:
        raise InvalidQueryError("Cutout contains to data! (Check image footprint.)")

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Setting up the output file(s) and writing them
    if colorize:
        cutouts = [x for fle in input_files for x in cutout_hdu_dict.get(fle, [])]
        
        # Doing checks correct number of cutouts
        if len(cutouts) < 3:
            raise InvalidInputError(("Color cutouts require 3 input images (RGB)."
                                     "If you supplied 3 images one of the cutouts may have been empty."))
        if len(cutouts) > 3:
            warnings.warn("Too many inputs for a color cutout, only the first three will be used.",
                          InputWarning)
            cutouts = cutouts[:3]

            
        cutout_path = "{}_{:7f}_{:7f}_{}-x-{}_astrocut.{}".format(cutout_prefix,
                                                                  coordinates.ra.value,
                                                                  coordinates.dec.value,
                                                                  str(cutout_size[0]).replace(' ', ''), 
                                                                  str(cutout_size[1]).replace(' ', ''),
                                                                  img_format.lower()) 
        cutout_path = os.path.join(output_dir, cutout_path)

        Image.fromarray(np.dstack([cutouts[0], cutouts[1], cutouts[2]]).astype(np.uint8)).save(cutout_path)
          
    else:
 
        cutout_path = []
        for fle in input_files:

            for i, cutout in enumerate(cutout_hdu_dict.get(fle)):


                file_path = "{}_{:7f}_{:7f}_{}-x-{}_astrocut_{}.{}".format(os.path.basename(fle).rstrip('.fits'),
                                                                           coordinates.ra.value,
                                                                           coordinates.dec.value,
                                                                           str(cutout_size[0]).replace(' ', ''), 
                                                                           str(cutout_size[1]).replace(' ', ''),
                                                                           i,
                                                                           img_format.lower())
                file_path = os.path.join(output_dir, file_path)
                cutout_path.append(file_path)
            
                Image.fromarray(cutout).save(file_path)
        
    if verbose:
        print("Cutout fits file(s): {}".format(cutout_path))
        print("Total time: {:.2} sec".format(time()-start_time))

    return cutout_path
