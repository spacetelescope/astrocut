# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements cutout functionality similar to fitscut, but for the ASDF file format."""
from typing import Union

import asdf
import astropy
import gwcs
import numpy as np

from astropy.coordinates import SkyCoord


def get_center_pixel(gwcsobj: gwcs.wcs.WCS, ra: float, dec: float) -> tuple:
    """ Get the center pixel from a roman 2d science image

    For an input RA, Dec sky coordinate, get the closest pixel location
    on the input Roman image.

    Parameters
    ----------
    gwcsobj : gwcs.wcs.WCS
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
    header = gwcsobj.to_fits_sip()

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
    row, col = gwcsobj.invert(coordinates)

    return (row, col), wcs_updated


def get_cutout(data: asdf.tags.core.ndarray.NDArrayType, coords: Union[tuple, SkyCoord],
               wcs: astropy.wcs.wcs.WCS = None, size: int = 20, outfile: str = "example_roman_cutout.fits",
               write_file: bool = True, fill_value: Union[int, float] = np.nan) -> astropy.nddata.Cutout2D:
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
    write_file : bool, by default True
        Flag to write the cutout to a file or not
    fill_value: int, by default np.nan
        The fill value for pixels outside the original image.

    Returns
    -------
    astropy.nddata.Cutout2D:
        an image cutout object

    Raises
    ------
    ValueError:
        when a wcs is not present when coords is a SkyCoord object
    RuntimeError:
        when the requested cutout does not overlap with the original image
    """

    # check for correct inputs
    if isinstance(coords, SkyCoord) and not wcs:
        raise ValueError('wcs must be input if coords is a SkyCoord.')

    # create the cutout
    try:
        cutout = astropy.nddata.Cutout2D(data, position=coords, wcs=wcs, size=(size, size), mode='partial',
                                         fill_value=fill_value)
    except astropy.nddata.utils.NoOverlapError as e:
        raise RuntimeError('Could not create 2d cutout.  The requested cutout does not overlap with the '
                           'original image.') from e

    # check if the data is a quantity and get the array data
    if isinstance(cutout.data, astropy.units.Quantity):
        data = cutout.data.value
    else:
        data = cutout.data

    # write the cutout to the output file
    if write_file:
        astropy.io.fits.writeto(outfile, data=data, header=cutout.wcs.to_header(), overwrite=True)

    return cutout


def asdf_cut(input_file: str, ra: float, dec: float, cutout_size: int = 20,
             output_file: str = "example_roman_cutout.fits",
             write_file: bool = True, fill_value: Union[int, float] = np.nan) -> astropy.nddata.Cutout2D:
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
    write_file : bool, by default True
        Flag to write the cutout to a file or not
    fill_value: int, by default np.nan
        The fill value for pixels outside the original image.

    Returns
    -------
    astropy.nddata.Cutout2D:
        an image cutout object
    """

    # get the 2d image data
    with asdf.open(input_file) as f:
        data = f['roman']['data']
        gwcsobj = f['roman']['meta']['wcs']

        # get the center pixel
        pixel_coordinates, wcs = get_center_pixel(gwcsobj, ra, dec)

        # create the 2d image cutout
        return get_cutout(data, pixel_coordinates, wcs, size=cutout_size, outfile=output_file,
                          write_file=write_file, fill_value=fill_value)
