# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements cutout functionality similar to fitscut, but for the ASDF file format."""
from typing import Union

import asdf
import astropy
import gwcs

from astropy.coordinates import SkyCoord


def get_center_pixel(gwcs: gwcs.wcs.WCS, ra: float, dec: float) -> tuple:
    """ Get the center pixel from a roman 2d science image

    For an input RA, Dec sky coordinate, get the closest pixel location
    on the input Roman image.

    Parameters
    ----------
    gwcs : gwcs.wcs.WCS
        the Roman GWCS object
    ra : float
        the input Right Ascension
    dec : float
        the input Declination

    Returns
    -------
    tuple
        the pixel position, FITS wcs object
    """

    # Convert the gwcs object to an astropy FITS WCS header
    header = gwcs.to_fits_sip()

    # Update WCS header with some keywords that it's missing.
    # Otherwise, it won't work with astropy.wcs tools (TODO: Figure out why. What are these keywords for?)
    for k in ['cpdis1', 'cpdis2', 'det2im1', 'det2im2', 'sip']:
        if k not in header:
            header[k] = 'na'

    # New WCS object with updated header
    wcs_updated = astropy.wcs.WCS(header)

    # Turn input RA, Dec into a SkyCoord object
    coordinates = SkyCoord(ra, dec, unit='deg')

    # Map the coordinates to a pixel's location on the Roman 2d array (row, col)
    row, col = astropy.wcs.utils.skycoord_to_pixel(coords=coordinates, wcs=wcs_updated)

    return (row, col), wcs_updated


def get_cutout(data: asdf.tags.core.ndarray.NDArrayType, coords: Union[tuple, SkyCoord],
               wcs: astropy.wcs.wcs.WCS = None, size: int = 20, outfile: str = "example_roman_cutout.fits"):
    """ Get a Roman image cutout

    Cut out a square section from the input image data array.  The ``coords`` can either be a tuple of x, y
    pixel coordinates or an astropy SkyCoord object, in which case, a wcs is required.  Writes out a
    new output file containing the image cutout of the specified ``size``.  Default is 20 pixels.

    Parameters
    ----------
    data : asdf.tags.core.ndarray.NDArrayType
        the input Roman image data array
    coords : Union[tuple, SkyCoord]
        the input pixel or sky coordinates
    wcs : astropy.wcs.wcs.WCS, Optional
        the astropy FITS wcs object
    size : int, optional
        the image cutout pizel size, by default 20
    outfile : str, optional
        the name of the output cutout file, by default "example_roman_cutout.fits"

    Raises
    ------
    ValueError:
        when a wcs is not present when coords is a SkyCoord object
    """

    # check for correct inputs
    if isinstance(coords, SkyCoord) and not wcs:
        raise ValueError('wcs must be input if coords is a SkyCoord.')

    # create the cutout
    cutout = astropy.nddata.Cutout2D(data, position=coords, wcs=wcs, size=(size, size))

    # write the cutout to the output file
    astropy.io.fits.writeto(outfile, data=cutout.data, header=cutout.wcs.to_header(), overwrite=True)


def asdf_cut(input_file: str, ra: float, dec: float, cutout_size: int = 20, output_file: str = "example_roman_cutout.fits"):
    """ Preliminary proof-of-concept functionality.

    Takes a single ASDF input file (``input_file``) and generates a cutout of designated size ``cutout_size``
    around the given coordinates (``coordinates``).

    Parameters
    ----------
    input_file : str
        the input ASDF file
    ra : float
        the Right Ascension of the central cutout
    dec : float
        the Declination of the central cutout
    cutout_size : int, optional
        the image cutout pixel size, by default 20
    output_file : str, optional
        the name of the output cutout file, by default "example_roman_cutout.fits"
    """

    # get the 2d image data
    with asdf.open(input_file) as f:
        data = f['roman']['data']
        gwcs = f['roman']['meta']['wcs']

        # get the center pixel
        pixel_coordinates, wcs = get_center_pixel(gwcs, ra, dec)

        # create the 2d image cutout
        get_cutout(data, pixel_coordinates, wcs, size=cutout_size, outfile=output_file)
