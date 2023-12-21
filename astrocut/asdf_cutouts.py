# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements cutout functionality similar to fitscut, but for the ASDF file format."""

import asdf
import astropy

def get_center_pixel(file, ra, dec):

    # Get the roman 2D science image
    with asdf.open(file) as f:
        # Get the WCS
        wcs = f['roman']['meta']['wcs']

        # Get the WCS header
        header = wcs.to_fits_sip()
    
    # Update WCS header with some keywords that it's missing.
    # Otherwise, it won't work with astropy.wcs tools (TODO: Figure out why. What are these keywords for?)
    for k in ['cpdis1', 'cpdis2', 'det2im1', 'det2im2', 'sip']:
        header[k] = 'na'
    
    # New WCS object with updated header
    wcs_updated = astropy.wcs.WCS(header)

    # Turn input RA, Dec into a SkyCoord object
    coordinates = astropy.coordinates.SkyCoord(ra, dec, unit='deg')

    # Map the coordinates to a pixel's location on the Roman 2d array (row, col)
    row, col = astropy.wcs.utils.skycoord_to_pixel(coords=coordinates, wcs=wcs_updated)

    return (row, col), wcs_updated


def get_cutout(file, coords, wcs, size=20, outfile="example_roman_cutout.fits"):
    
    # Get the 2D science image
    with asdf.open(file) as f:
        data = f['roman']['data']

        coordinates = coords
        cutout = astropy.nddata.Cutout2D(data, position=coordinates, wcs=wcs, size=(size, size))

        astropy.io.fits.writeto(outfile, data=cutout.data.value, header=cutout.wcs.to_header(), overwrite=True)


def asdf_cut(input_file, ra, dec, *, cutout_size=20, output_file="example_roman_cutout.fits"):
    """ Preliminary proof-of-concept functionality.
    Takes a single ASDF input file (``input_file``) and generates a cutout of designated size ``cutout_size``
    around the given coordinates (``coordinates``).
    """

    pixel_coordinates, wcs = get_center_pixel(input_file, ra, dec)

    get_cutout(input_file, pixel_coordinates, wcs, size=cutout_size, outfile=output_file)
    