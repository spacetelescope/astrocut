# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module includes a variety of functions that may be used by multiple modules."""

import warnings
import numpy as np

from datetime import date

from astropy import wcs
from astropy.io import fits
from astropy import units as u
from astropy.utils import deprecated

from .. import __version__
from ..exceptions import InvalidQueryError, InputWarning


def parse_size_input(cutout_size):
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


def get_cutout_limits(img_wcs, center_coord, cutout_size):
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


def get_cutout_wcs(img_wcs, cutout_lims):
    """
    Starting with the full image WCS and adjusting it for the cutout WCS.
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


def _build_astrocut_primaryhdu(**keywords):
    """
    TODO: Document
    """

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header.extend([("ORIGIN", 'STScI/MAST', "institution responsible for creating this file"),
                               ("DATE", str(date.today()), "file creation date"),
                               ('PROCVER', __version__, 'software version')])
    for kwd in keywords:
        primary_hdu.header[kwd] = keywords[kwd]

    return primary_hdu


@deprecated(since="v0.9", alternative="make_fits")
def save_fits(cutout_hdus, output_path, center_coord):
    return get_fits(cutout_hdus, center_coord=center_coord, output_path=output_path)


def get_fits(cutout_hdus, center_coord=None, output_path=None):
    """
    Make one or more cutout hdus to a single fits object, optionally save the file to disk.

    Parameters
    ----------
    cutout_hdus : list or `~astropy.io.fits.hdu.image.ImageHDU`
        The `~astropy.io.fits.hdu.image.ImageHDU` object(s) to be written to the fits file.
    output_path : str
        The full path to the output fits file.
    center_coord : `~astropy.coordinates.sky_coordinate.SkyCoord`
        The center coordinate of the image cutouts.  TODO: make more general?
    """

    if isinstance(cutout_hdus, fits.hdu.image.ImageHDU):
        cutout_hdus = [cutout_hdus]
    
    # Setting up the Primary HDU
    keywords = dict()
    if center_coord:
        keywords = {"RA_OBJ": (center_coord.ra.deg, '[deg] right ascension'),
                    "DEC_OBJ": (center_coord.dec.deg, '[deg] declination')}
    primary_hdu = _build_astrocut_primaryhdu(**keywords)

    cutout_hdulist = fits.HDUList([primary_hdu] + cutout_hdus)

    if output_path:
        # Writing out the hdu often causes a warning as the ORIG_FLE card description is truncated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            cutout_hdulist.writeto(output_path, overwrite=True, checksum=True)

    return cutout_hdulist


