import numpy as np
import pytest
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

from astrocut.exceptions import InputWarning, InvalidQueryError

from ..utils import utils

# Example FFI WCS for testing
with open(get_pkg_data_filename("data/ex_ffi_wcs.txt"), "r") as FLE:
    WCS_STR = FLE.read()


@pytest.mark.parametrize(
    "input_value, expected",
    [
        (5, np.array((5, 5))),  # scalar
        (10 * u.pix, np.array((10, 10)) * u.pix),  # Astropy quantity
        ((5, 10), np.array((5, 10))),  # tuple
        ([10, 5], np.array((10, 5))),  # list
        (np.array((5, 10)), np.array((5, 10))),  # array
    ],
)
def test_parse_size_input(input_value, expected):
    """Test that different types of input are accurately parsed into cutout sizes."""
    cutout_size = utils.parse_size_input(input_value)
    assert np.array_equal(cutout_size, expected)


def test_parse_size_input_dimension_warning():
    """Test that a warning is output when input has too many dimensions"""
    warning = "Too many dimensions in cutout size, only the first two will be used."
    with pytest.warns(InputWarning, match=warning):
        cutout_size = utils.parse_size_input((5, 5, 10))
        assert np.array_equal(cutout_size, np.array((5, 5)))


def test_parse_size_input_invalid():
    """Test that an error is raised when one of the size dimensions is not positive"""
    err = "Cutout size dimensions must be greater than zero."
    with pytest.raises(InvalidQueryError, match=err):
        utils.parse_size_input(0)

    with pytest.raises(InvalidQueryError, match=err):
        utils.parse_size_input((0, 5))

    with pytest.raises(InvalidQueryError, match=err):
        utils.parse_size_input((0, 5))


def test_get_cutout_limits():
    test_img_wcs_kwds = fits.Header(
        cards=[
            ("NAXIS", 2, "number of array dimensions"),
            ("NAXIS1", 20, ""),
            ("NAXIS2", 30, ""),
            ("CTYPE1", "RA---TAN", "Right ascension, gnomonic projection"),
            ("CTYPE2", "DEC--TAN", "Declination, gnomonic projection"),
            ("CRVAL1", 100, "[deg] Coordinate value at reference point"),
            ("CRVAL2", 20, "[deg] Coordinate value at reference point"),
            ("CRPIX1", 10, "Pixel coordinate of reference point"),
            ("CRPIX2", 15, "Pixel coordinate of reference point"),
            ("CDELT1", 1.0, "[deg] Coordinate increment at reference point"),
            ("CDELT2", 1.0, "[deg] Coordinate increment at reference point"),
            ("WCSAXES", 2, "Number of coordinate axes"),
            ("PC1_1", 1, "Coordinate transformation matrix element"),
            ("PC2_2", 1, "Coordinate transformation matrix element"),
            ("CUNIT1", "deg", "Units of coordinate increment and value"),
            ("CUNIT2", "deg", "Units of coordinate increment and value"),
        ]
    )

    test_img_wcs = wcs.WCS(test_img_wcs_kwds)

    center_coord = SkyCoord("100 20", unit="deg")
    cutout_size = [10, 10]

    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    assert (lims[0, 1] - lims[0, 0]) == (lims[1, 1] - lims[1, 0])
    assert (lims == np.array([[4, 14], [9, 19]])).all()

    cutout_size = [10, 5]
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    assert (lims[0, 1] - lims[0, 0]) == 10
    assert (lims[1, 1] - lims[1, 0]) == 5

    cutout_size = [0.1, 0.1] * u.deg
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    assert (lims[0, 1] - lims[0, 0]) == (lims[1, 1] - lims[1, 0])
    assert (lims[0, 1] - lims[0, 0]) == 1

    cutout_size = [4, 5] * u.deg
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    assert (lims[0, 1] - lims[0, 0]) == 4
    assert (lims[1, 1] - lims[1, 0]) == 5

    center_coord = SkyCoord("90 20", unit="deg")
    cutout_size = [4, 5] * u.deg
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    assert lims[0, 0] < 0

    center_coord = SkyCoord("100 5", unit="deg")
    cutout_size = [4, 5] * u.pixel
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    assert lims[1, 0] < 0


def _make_table_data(ctype2_values):
    """Build a minimal cube image-header table (as found in HDU 2 of a cube file) with one
    row per entry in ``ctype2_values``, for use in testing ``parse_table_wcs``."""

    n_rows = len(ctype2_values)

    columns = [
        fits.Column(name='CTYPE1', format='24A', array=np.full(n_rows, 'RA---TAN-SIP')),
        fits.Column(name='CTYPE2', format='24A', array=np.array(ctype2_values)),
        fits.Column(name='CRVAL1', format='D', array=np.full(n_rows, 100.0)),
        fits.Column(name='CRVAL2', format='D', array=np.full(n_rows, 20.0)),
        fits.Column(name='CRPIX1', format='D', array=np.full(n_rows, 10.0)),
        fits.Column(name='CRPIX2', format='D', array=np.full(n_rows, 15.0)),
        fits.Column(name='CDELT1', format='D', array=np.full(n_rows, 1.0)),
        fits.Column(name='CDELT2', format='D', array=np.full(n_rows, 1.0)),
        fits.Column(name='CUNIT1', format='8A', array=np.full(n_rows, 'deg')),
        fits.Column(name='CUNIT2', format='8A', array=np.full(n_rows, 'deg')),
        fits.Column(name='FFI_FILE', format='40A', array=np.array([f'ffi_{i}.fits' for i in range(n_rows)])),
    ]

    return fits.BinTableHDU.from_columns(columns).data


def test_parse_table_wcs():
    """Test that parse_table_wcs finds the valid WCS row and builds a matching WCS object."""

    ctype2_values = ['INVALID', 'INVALID', 'DEC--TAN-SIP', 'INVALID', 'INVALID']
    table_data = _make_table_data(ctype2_values)

    table_wcs = utils.parse_table_wcs(table_data, 'CTYPE2', 'DEC--TAN-SIP')
    assert isinstance(table_wcs, wcs.WCS)
    assert (table_wcs.wcs.crval == [100.0, 20.0]).all()
    assert (table_wcs.wcs.crpix == [10.0, 15.0]).all()

    # return_header=True should also give back the header the WCS was built from, including
    # non-WCS columns like FFI_FILE
    table_wcs, wcs_header = utils.parse_table_wcs(table_data, 'CTYPE2', 'DEC--TAN-SIP', return_header=True)
    assert isinstance(table_wcs, wcs.WCS)
    assert wcs_header['FFI_FILE'] == 'ffi_2.fits'


def test_parse_table_wcs_walks_outward_from_middle():
    """Test that parse_table_wcs starts at the middle row and wraps around to find a match."""

    # Valid row is before the middle, so the search must wrap around after reaching the end
    ctype2_values = ['DEC--TAN-SIP', 'INVALID', 'INVALID', 'INVALID', 'INVALID']
    table_data = _make_table_data(ctype2_values)

    table_wcs, wcs_header = utils.parse_table_wcs(table_data, 'CTYPE2', 'DEC--TAN-SIP', return_header=True)
    assert wcs_header['FFI_FILE'] == 'ffi_0.fits'


def test_parse_table_wcs_no_valid_row():
    """Test that a NoWcsKeywordsFoundError is raised when no row has valid WCS keywords."""

    ctype2_values = ['INVALID'] * 6
    table_data = _make_table_data(ctype2_values)

    with pytest.raises(wcs.NoWcsKeywordsFoundError, match='No FFI rows contain valid WCS keywords.'):
        utils.parse_table_wcs(table_data, 'CTYPE2', 'DEC--TAN-SIP')


def test_get_cutout_wcs():
    test_img_wcs_kwds = fits.Header(
        cards=[
            ("NAXIS", 2, "number of array dimensions"),
            ("NAXIS1", 20, ""),
            ("NAXIS2", 30, ""),
            ("CTYPE1", "RA---TAN", "Right ascension, gnomonic projection"),
            ("CTYPE2", "DEC--TAN", "Declination, gnomonic projection"),
            ("CRVAL1", 100, "[deg] Coordinate value at reference point"),
            ("CRVAL2", 20, "[deg] Coordinate value at reference point"),
            ("CRPIX1", 10, "Pixel coordinate of reference point"),
            ("CRPIX2", 15, "Pixel coordinate of reference point"),
            ("CDELT1", 1.0, "[deg] Coordinate increment at reference point"),
            ("CDELT2", 1.0, "[deg] Coordinate increment at reference point"),
            ("WCSAXES", 2, "Number of coordinate axes"),
            ("PC1_1", 1, "Coordinate transformation matrix element"),
            ("PC2_2", 1, "Coordinate transformation matrix element"),
            ("CUNIT1", "deg", "Units of coordinate increment and value"),
            ("CUNIT2", "deg", "Units of coordinate increment and value"),
        ]
    )

    test_img_wcs = wcs.WCS(test_img_wcs_kwds)

    center_coord = SkyCoord("100 20", unit="deg")
    cutout_size = [4, 5] * u.deg
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    cutout_wcs = utils.get_cutout_wcs(test_img_wcs, lims)
    assert (cutout_wcs.wcs.crval == [100, 20]).all()
    assert (cutout_wcs.wcs.crpix == [3, 4]).all()

    center_coord = SkyCoord("100 5", unit="deg")
    cutout_size = [4, 5] * u.deg
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    cutout_wcs = utils.get_cutout_wcs(test_img_wcs, lims)
    assert (cutout_wcs.wcs.crval == [100, 20]).all()
    assert (cutout_wcs.wcs.crpix == [3, 19]).all()

    center_coord = SkyCoord("110 20", unit="deg")
    cutout_size = [10, 10] * u.deg
    lims = utils.get_cutout_limits(test_img_wcs, center_coord, cutout_size)
    cutout_wcs = utils.get_cutout_wcs(test_img_wcs, lims)
    assert (cutout_wcs.wcs.crval == [100, 20]).all()
    assert (cutout_wcs.wcs.crpix == [-3, 6]).all()
