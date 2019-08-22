# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements cutout functionality similar to fitscut."""


import numpy as np
import astropy.units as u

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import wcs

from time import time
from datetime import date

import os
import warnings

from . import __version__
from .exceptions import InputWarning, TypeWarning, InvalidQueryError, InvalidInputError




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

    #print(f"Center pixel: {center_pixel}")
    
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
            lims[axis, 1] = lims[axis, 0] + 1 #int(np.ceil(center_pixel[axis] - 1))

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


def _hducut(img_hdu, center_coord, cutout_size, correct_wcs=False, drop_after=None, verbose=False):
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
    drop_after : str or None
        Default None. When creating the header for the cutout (and crucially, before 
        building the WCS object) drop all header keywords starting with the one given.  This is
        useful particularly for drizzle files that contain a multitude of extranious keywords
        and sometimes leftover WCS keywords that astropy will try to parse even thought they should be
        ignored.
    verbose : bool
        Default False. If true intermediate information is printed.

    Returns
    -------
    response : `~astropy.io.fits.hdu.image.ImageHDU` 
        The cutout image and associated metadata.
    """

    # Pulling out the header
    max_ind = len(img_hdu.header)
    if drop_after is not None:
        try:
            max_ind = img_hdu.header.index(drop_after)
        except ValueError:
            warnings.warn("Last desired keyword not found in image header, using the entire header.")
    
    hdu_header = fits.Header(img_hdu.header[:max_ind], copy=True)
    img_wcs = wcs.WCS(hdu_header)
    img_data = img_hdu.data

    if verbose:
        print(f"Original image shape: {img_data.shape}")

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

    if verbose:
        print(cutout_wcs.__repr__())

    # Updating the header with the new wcs info
    hdu_header.update(cutout_wcs.to_header(relax=True)) # relax arg is for sip distortions if they exist

    # Naming the extension
    hdu_header["EXTNAME"] = "CUTOUT"

    # Moving the filename, if present, into the ORIG_FLE keyword
    hdu_header["ORIG_FLE"] = (hdu_header.get("FILENAME"),"Original image filename.")
    hdu_header.remove("FILENAME", ignore_missing=True)

    hdu = fits.ImageHDU(header=hdu_header, data=img_cutout)

    return hdu

def _save_single(cutout_hdus, output_path, center_coord):
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
    primary_hdu.header.extend([("ORIGIN",'STScI/MAST',"institution responsible for creating this file"),
                               ("DATE", str(date.today()), "file creation date"),
                               ('PROCVER',__version__, 'software version'),
                               ('RA_OBJ', center_coord.ra.deg, '[deg] right ascension'),
                               ('DEC_OBJ', center_coord.dec.deg, '[deg] declination')])

    cutout_hdulist = fits.HDUList([primary_hdu] + cutout_hdus)
    cutout_hdulist.writeto(output_path, overwrite=True, checksum=True)


def _save_multiple(cutout_hdus, output_dir, filenames, center_coord):
    """
    Save a list of cutout hdus to individual fits files.

    Parameters
    ----------
    cutout_hdus : list
        List of `~astropy.io.fits.hdu.image.ImageHDU` objects to be written to a single fits file.
    output_dir : str
        The path to the directory where the fits files will be written.
    filenames : list
        The filename associated with the cutout hdus.
    center_coord : `~astropy.coordinates.sky_coordinate.SkyCoord`
        The center coordinate of the image cutouts.
    """

    if len(filenames) != len(cutout_hdus):
        raise InvalidInputError("The number of filenames must match the number of cutouts.")

    # Adding aditional keywords
    for i,cutout in enumerate(cutout_hdus):
        # Turning our hdu into a primary hdu
        hdu = fits.PrimaryHDU(header=cutout.header, data=cutout.data)
        hdu.header.extend([("ORIGIN",'STScI/MAST',"institution responsible for creating this file"),
                           ("DATE", str(date.today()), "file creation date"),
                           ('PROCVER',__version__, 'software version'),
                           ('RA_OBJ', center_coord.ra.deg, '[deg] right ascension'),
                           ('DEC_OBJ', center_coord.dec.deg, '[deg] declination')])

        cutout_hdulist = fits.HDUList([hdu])
        cutout_hdulist.writeto(os.path.join(output_dir,filenames[i]), overwrite=True, checksum=True)    

        
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
    drop_after : str or None
        Default None. When creating the header for the cutout (and crucially, before 
        building the WCS object) drop all header keywords starting with the one given.  This is
        useful particularly for drizzle files that contain a multitude of extranious keywords
        and sometimes leftover WCS keywords that astropy will try to parse even thought they should be
        ignored.
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
        If single_outfile is True returns the single output filename. Otherwise returns a list of all 
        the output files.
    """

    if verbose:
        start_time = time()
            
    # Making sure we have an array of images
    if type(input_files) == str:
        input_files == [input_files]

    if not isinstance(coordinates, SkyCoord):
        coordinates = SkyCoord(coordinates, unit='deg')

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

    # Making the cutouts
    cutout_hdu_dict = {}
    num_empty = 0
    for in_fle in input_files:
        if verbose:
            print(f"\n{in_fle}")
        hdulist = fits.open(in_fle)
        cutout = _hducut(hdulist[0], coordinates, cutout_size,
                         correct_wcs=correct_wcs, drop_after=drop_after, verbose=verbose)
        hdulist.close()
        
        # Check that there is data in the cutout image
        if (cutout.data == 0).all() or (np.isnan(cutout.data)).all():
            cutout.header["EMPTY"] = (True, "Indicates no data in cutout image.")
            num_empty += 1
            
        cutout_hdu_dict[in_fle] =  cutout

    # If no cutouts contain data, raise exception
    if num_empty == len(input_files):
        raise InvalidQueryError("Cutout contains to data! (Check image footprint.)")

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Setting up the output file(s) and writing them
    if single_outfile:

        cutout_file = "{}_{:7f}_{:7f}_{}x{}_astrocut.fits".format(cutout_prefix,
                                                                  coordinates.ra.value,
                                                                  coordinates.dec.value,
                                                                  cutout_size[0], # TODO: make cutout size
                                                                  cutout_size[1]) # look nicer
        
        cutout_hdus = [cutout_hdu_dict[fle] for fle in input_files]

        _save_single(cutout_hdus, os.path.join(output_dir, cutout_file), coordinates)

    else:
        cutout_hdus = []
        cutout_file = []
        for fle in input_files:
            cutout = cutout_hdu_dict[fle]
            if cutout.header.get("EMPTY"):
                warnings.warn("Cutout of {} contains to data and will not be written.".format(fle))
                continue

            cutout_hdus.append(cutout)

            cutout_file.append("{}_{:7f}_{:7f}_{}x{}_astrocut.fits".format(fle.rstrip('.fits'),
                                                                           coordinates.ra.value,
                                                                           coordinates.dec.value,
                                                                           cutout.header["NAXIS1"],
                                                                           cutout.header["NAXIS2"]))
                
        _save_multiple(cutout_hdus, output_dir, cutout_file, coordinates)

        
    if verbose:
        print("Cutout fits file(s): {}".format(cutout_file))
        print("Total time: {:.2} sec".format(time()-start_time))

    return cutout_file