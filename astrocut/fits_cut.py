# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements cutout functionality similar to fitscut."""


import numpy as np
import astropy.units as u

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import wcs

from time import time

import os
import warnings

from . import __version__
from .exceptions import InputWarning, TypeWarning, InvalidQueryError




#### FUNCTIONS FOR UTILS ####
def _get_cutout_limits(img_wcs, center_coord, cutout_size):
    """
    Takes the center coordinates, cutout size, and the wcs from
    which the cutout is being taken and returns the x and y pixel limits
    for the cutout.

    Parameters
    ----------
    img_wcs : `~astropy.wcs.WCS`
        The WCS for the image that the cutout is being cut from.
    center_coord : `~astropy.coordinates.SkyCoord`
        The central coordinate for the cutout
    cutout_size : array
        [ny,nx] in with ints (pixels) or astropy quantities

    Returns
    -------
    response : `numpy.array`
        The cutout pixel limits in an array of the form [[ymin,ymax],[xmin,xmax]]
    """
        
    # Note: This is returning the center pixel in 1-up
    try:
        center_pixel = center_coord.to_pixel(img_wcs, 1)
    except wcs.NoConvergence:  # If wcs can't converge, center coordinate is far from the footprint
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
            lims[axis, 1] = int(np.ceil(center_pixel[axis] - 1))

    # Checking at least some of the cutout is on the image
    # TODO: I am not convinced by this check
    if ((lims[0, 0] <= 0) and (lims[0, 1] <= 0)) or ((lims[1, 0] <= 0) and (lims[1, 1] <= 0)):
        raise InvalidQueryError("Cutout location is not in cube footprint!")

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

    Resturns
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
    Does the actual cutting out.
    """
    
    img_wcs = wcs.WCS(img_hdu.header)
    img_wcs.sip = None # Necessary for CANDELS bc extra sip coeffs hanging around
    
    img_data = img_hdu.data

    # Get cutout limits
    cutout_lims = _get_cutout_limits(img_wcs, center_coord, cutout_size)

    if verbose:
        print("xmin,xmax: {}".format(cutout_lims[1]))
        print("ymin,ymax: {}".format(cutout_lims[0]))

    # These limits are not guarenteed to be within the image footprint
    xmin, xmax = cutout_lims[1]
    ymin, ymax = cutout_lims[0]

    ymax_img, xmax_img = img_data.shape

    # Adjust limits and figuring out the padding
    padding = np.zeros((2, 2), dtype=int)
    if xmin < 0:
        padding[1, 0] = -xmin
        xmin = 0
    if ymin < 0:
        padding[2, 0] = -ymin
        ymin = 0
    if xmax > xmax_img:
        padding[1, 1] = xmax - xmax_img
        xmax = xmax_img
    if ymax > ymax_img:
        padding[2, 1] = ymax - ymax_img
        ymax = ymax_img  
        
    img_cutout = img_hdu.data[ymin:ymax, xmin:xmax]

    # Adding padding to the cutout so that it's the expected size
    if padding.any():  # only do if we need to pad
        img_cutout = np.pad(img_cutout, padding, 'constant', constant_values=np.nan)

    if verbose:
        print("Image cutout shape: {}".format(img_cutout.shape))

    # Getting the cutout wcs
    cutout_wcs = _get_cutout_wcs(img_wcs, cutout_lims)

    # Building the HDU
    hdu_header = fits.Header(img_hdu.header, copy=True)
    hdu_header.update(cutout_wcs.to_header(relax=True)) # relax arg is for sip distortions if they exist
    # TODO: change filename -> original_file or something
    # Add/change any other keywords as needed

    
    hdu = fits.ImageHDU(header=hdu_header, data=img_cutout)

    return hdu
    


def fits_cut(input_files, coordinates, cutout_size, correct_wcs=False, single_outfile=True,
             cutout_files=None, output_path='.', verbose=False):
    """
    Takes one or more fits files with the same WCS/pointing (in future will support resampling),
    makes the same cutout in each file and returns the result in a single fitsfile with one cutout
    per extension (in future will support outputting multiple files).
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

    cutout_hdus = []
    for in_fle in input_files:
        hdulist = fits.open(in_fle)
        cutout_hdus.append( _hducut(hdulist[0], coordinates, cutout_size,
                                    correct_wcs=correct_wcs, verbose=verbose))
        hdulist.close()

    # Setting up the Promary HDU
    primary_hdu = fits.PrimaryHDU()
    # TODO: add some info about origin etc

    # TODO: implement multiple files
    #if single_outfile:
    cutout_hdulist = fits.HDUList([primary_hdu] + cutout_hdus)

    # Getting the output filename
    if not cutout_files:
        cutout_files = "cutout.fits" # TODO: make this a better filename
        # TODO: also need to deal with getting a list
    cutout_path = os.path.join(output_path, cutout_files)

    if verbose:
        print("Cutout fits file(s): {}".format(cutout_path))

    # Make sure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
            
    cutout_hdulist.writeto(cutout_path, overwrite=True, checksum=True)

    if verbose:
        print("Total time: {:.2} sec".format(time()-start_time))

    return cutout_path
        

