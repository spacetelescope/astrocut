import numpy as np

from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.data import get_pkg_data_filename

from ..utils import utils


# Example FFI WCS for testing
with open(get_pkg_data_filename('data/ex_ffi_wcs.txt'), "r") as FLE:
    WCS_STR = FLE.read()

    
def test_get_cutout_limits():

    test_img_wcs_kwds = fits.Header(cards=[('NAXIS', 2, 'number of array dimensions'),
                                           ('NAXIS1', 20, ''),
                                           ('NAXIS2', 30, ''),
                                           ('CTYPE1', 'RA---TAN', 'Right ascension, gnomonic projection'),
                                           ('CTYPE2', 'DEC--TAN', 'Declination, gnomonic projection'),
                                           ('CRVAL1', 100, '[deg] Coordinate value at reference point'),
                                           ('CRVAL2', 20, '[deg] Coordinate value at reference point'),
                                           ('CRPIX1', 10, 'Pixel coordinate of reference point'),
                                           ('CRPIX2', 15, 'Pixel coordinate of reference point'),
                                           ('CDELT1', 1.0, '[deg] Coordinate increment at reference point'),
                                           ('CDELT2', 1.0, '[deg] Coordinate increment at reference point'),
                                           ('WCSAXES', 2, 'Number of coordinate axes'),
                                           ('PC1_1', 1, 'Coordinate transformation matrix element'),
                                           ('PC2_2', 1, 'Coordinate transformation matrix element'),
                                           ('CUNIT1', 'deg', 'Units of coordinate increment and value'),
                                           ('CUNIT2', 'deg', 'Units of coordinate increment and value')])
    
    test_img_wcs = wcs.WCS(test_img_wcs_kwds)

    center_coord = SkyCoord("100 20", unit='deg')
    cutout_size = [10, 10]

    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    assert (lims[0, 1] - lims[0, 0]) == (lims[1, 1] - lims[1, 0])
    assert (lims == np.array([[4, 14], [9, 19]])).all()

    cutout_size = [10, 5]
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    assert (lims[0, 1] - lims[0, 0]) == 10
    assert (lims[1, 1] - lims[1, 0]) == 5

    cutout_size = [.1, .1]*u.deg
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    assert (lims[0, 1] - lims[0, 0]) == (lims[1, 1] - lims[1, 0])
    assert (lims[0, 1] - lims[0, 0]) == 1

    cutout_size = [4, 5]*u.deg
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    assert (lims[0, 1] - lims[0, 0]) == 4
    assert (lims[1, 1] - lims[1, 0]) == 5

    center_coord = SkyCoord("90 20", unit='deg')
    cutout_size = [4, 5]*u.deg
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    assert lims[0, 0] < 0

    center_coord = SkyCoord("100 5", unit='deg')
    cutout_size = [4, 5]*u.pixel
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    assert lims[1, 0] < 0


def test_get_cutout_wcs():
    test_img_wcs_kwds = fits.Header(cards=[('NAXIS', 2, 'number of array dimensions'),
                                           ('NAXIS1', 20, ''),
                                           ('NAXIS2', 30, ''),
                                           ('CTYPE1', 'RA---TAN', 'Right ascension, gnomonic projection'),
                                           ('CTYPE2', 'DEC--TAN', 'Declination, gnomonic projection'),
                                           ('CRVAL1', 100, '[deg] Coordinate value at reference point'),
                                           ('CRVAL2', 20, '[deg] Coordinate value at reference point'),
                                           ('CRPIX1', 10, 'Pixel coordinate of reference point'),
                                           ('CRPIX2', 15, 'Pixel coordinate of reference point'),
                                           ('CDELT1', 1.0, '[deg] Coordinate increment at reference point'),
                                           ('CDELT2', 1.0, '[deg] Coordinate increment at reference point'),
                                           ('WCSAXES', 2, 'Number of coordinate axes'),
                                           ('PC1_1', 1, 'Coordinate transformation matrix element'),
                                           ('PC2_2', 1, 'Coordinate transformation matrix element'),
                                           ('CUNIT1', 'deg', 'Units of coordinate increment and value'),
                                           ('CUNIT2', 'deg', 'Units of coordinate increment and value')])
    
    test_img_wcs = wcs.WCS(test_img_wcs_kwds)

    center_coord = SkyCoord("100 20", unit='deg')
    cutout_size = [4, 5]*u.deg
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    cutout_wcs = utils.get_cutout_wcs(test_img_wcs, lims)
    assert (cutout_wcs.wcs.crval == [100, 20]).all()
    assert (cutout_wcs.wcs.crpix == [3, 4]).all()

    center_coord = SkyCoord("100 5", unit='deg')
    cutout_size = [4, 5]*u.deg
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    cutout_wcs = utils.get_cutout_wcs(test_img_wcs, lims)
    assert (cutout_wcs.wcs.crval == [100, 20]).all()
    assert (cutout_wcs.wcs.crpix == [3, 19]).all()

    center_coord = SkyCoord("110 20", unit='deg')
    cutout_size = [10, 10]*u.deg
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    cutout_wcs = utils.get_cutout_wcs(test_img_wcs, lims)
    assert (cutout_wcs.wcs.crval == [100, 20]).all()
    assert (cutout_wcs.wcs.crpix == [-3, 6]).all()

