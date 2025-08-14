import warnings
from os import path
from pathlib import Path
from typing import List

import astropy.units as u
import numpy as np
import pytest
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import FITSFixedWarning

from ..cutout_factory import CutoutFactory, cube_cut
from ..exceptions import InputWarning
from ..cube_factory import CubeFactory
from .utils_for_test import create_test_ffis


@pytest.fixture
def ffi_files(tmp_path):
    """Pytest fixture for creating test ffi files"""

    tmpdir = str(tmp_path)

    img_sz = 10
    num_im = 100

    return create_test_ffis(img_sz, num_im, dir_name=tmpdir)


@pytest.fixture
def cube_file(ffi_files: List[str], tmp_path):
    """Pytest fixture for creating a cube file"""

    tmpdir = str(tmp_path)

    # Making the test cube
    cube_maker = CubeFactory()
    cube_file = cube_maker.make_cube(ffi_files, path.join(tmpdir, "test_cube.fits"), verbose=False)

    return cube_file


def check_cutout(hdulist, center_pix, center_world, cutout_size, ecube, eps=1.e-7):
    """Check FITS cutout for correctness
    
    Checks RA_OBJ/DEC_OBJ in primary header, and TIME, FLUX, and
    FLUX_ERR in table.
    
    Inputs:
    cutfile  Name of FITS cutout file
    pixcrd   [2] pixel coordinates for cutout center [cy,cx]
    world    [2] RA, Dec in degrees for cutout center
    csize    Integer size of cutout (probably should be odd)
    ecube    Simulated data cube
    eps      Maximum allowed distance offset in degrees
    Returns True on success, False on failure
    """
    # Compute cutout bounds
    ix, iy = int(center_pix[1]), int(center_pix[0])
    x1, x2 = ix - cutout_size // 2, ix + cutout_size // 2
    y1, y2 = iy - cutout_size // 2, iy + cutout_size // 2

    # Check RA and Dec in primary header
    ra_obj = hdulist[0].header['RA_OBJ']
    dec_obj = hdulist[0].header['DEC_OBJ']
    expected_coords = SkyCoord(center_world[0], center_world[1], frame='icrs', unit='deg')
    actual_coords = SkyCoord(ra_obj, dec_obj, frame='icrs', unit='deg')
    
    dist = expected_coords.separation(actual_coords).degree
    assert dist <= eps, "{} separation in primary header {} too large".format(hdulist, dist)
        
    # Check timeseries data
    ntimes = ecube.shape[2]
    tab = hdulist[1].data
    assert len(tab) == ntimes, f"Expected {ntimes} entries, found {len(tab)}"
    assert (tab['TIME'] == (np.arange(ntimes)+0.5)).all(), "Some time values are incorrect"

    check_flux(tab['FLUX'], x1, x2, y1, y2, ecube[:, :, :, 0], 'FLUX', hdulist)
    check_flux(tab['FLUX_ERR'], x1, x2, y1, y2, ecube[:, :, :, 1], 'FLUX_ERR', hdulist)
    
    # Ensure correct data type in third HDU
    assert hdulist[2].data.dtype.type == np.int32


def check_flux(flux, x1, x2, y1, y2, ecube, label, hdulist):
    """ Checking to make sure the right corresponding pixels 
    are replaced by NaNs when cutout goes off the TESS camera.
    Test one of flux or error
    """
    cx, cy = ecube.shape[:2]

    # Check NaNs in regions outside valid cutout bounds
    if x1 < 0:
        assert np.isnan(flux[:, :-x1, :]).all(), "{} {} x1 NaN failure".format(hdulist, label)
    if y1 < 0:
        assert np.isnan(flux[:, :, :-y1]).all(), "{} {} y1 NaN failure".format(hdulist, label)
    if x2 >= cx:
        assert np.isnan(flux[:, -(x2-cx+1):, :]).all(), "{} {} x2 NaN failure".format(hdulist, label)
    if y2 >= cy:
        assert np.isnan(flux[:, :, -(y2-cy+1):]).all(), "{} {} y2 NaN failure".format(hdulist, label)
        
    # Compute valid indices within cutout bounds
    x1c, x2c = max(x1, 0), min(x2, cx - 1)
    y1c, y2c = max(y1, 0), min(y2, cy - 1)

    # Extract and compare valid data
    expected_flux = ecube[x1c:x2c, y1c:y2c, :]
    cutout_flux = np.moveaxis(flux[:, x1c-x1:x2c-x1, y1c-y1:y2c-y1], 0, -1)

    assert np.array_equal(expected_flux, cutout_flux)


@pytest.mark.parametrize("use_factory", [True, False])
def test_cube_cutout(cube_file, ffi_files, use_factory, tmp_path):
    """
    Testing the cube cutout functionality.
    """
    tmpdir = str(tmp_path)
    img_sz = 10
    num_im = 100

    # Read one of the input images to get the WCS
    n = 1
    img_header = fits.getheader(ffi_files[0], n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FITSFixedWarning)
        cube_wcs = wcs.WCS(img_header)

    # get pixel positions at edges and center of image
    # somewhat cryptic one-liner to get the grid of points
    pval = np.array([0, img_sz // 2, img_sz - 1], dtype=float)
    pixcrd = pval[np.transpose(np.reshape(np.mgrid[0:3, 0:3], (2, 9)))]
    
    # add one more giant cutout that goes off all 4 edges
    pixcrd = np.append(pixcrd, pixcrd[4].reshape(1, 2), axis=0)

    # getting the world coordinates
    world_coords = cube_wcs.all_pix2world(pixcrd, 0)

    # Getting the cutouts
    cutbasename = 'make_cube_cutout_{}.fits'
    cutlist = [path.join(tmpdir, cutbasename.format(i)) for i in range(len(world_coords))]
    csize = [img_sz//2]*len(world_coords)
    csize[-1] = img_sz+5
    for i, v in enumerate(world_coords):
        coord = SkyCoord(v[0], v[1], frame='icrs', unit='deg')
        if use_factory:
            cutout_maker = CutoutFactory()
            cutout_maker.cube_cut(cube_file, coord, csize[i], target_pixel_file=cutlist[i],
                                  output_path=tmpdir, verbose=False)
        else:
            cube_cut(cube_file, coord, csize[i], target_pixel_file=cutlist[i],
                     output_path=tmpdir, verbose=False)


    # expected values for cube
    ecube = np.zeros((img_sz, img_sz, num_im, 2))
    plane = np.arange(img_sz*img_sz, dtype=np.float32).reshape((img_sz, img_sz))
    for i in range(num_im):
        ecube[:, :, i, 0] = -plane
        ecube[:, :, i, 1] = plane
        plane += img_sz*img_sz

    # Doing the actual checking
    for i, cutfile in enumerate(cutlist):
        with fits.open(cutfile) as hdulist:
            check_cutout(hdulist, pixcrd[i], world_coords[i], csize[i], ecube)


def test_parse_table_info(cube_file, tmp_path):
    """Test _parse_table_info"""

    tmpdir = str(tmp_path)

    cutout_maker = CutoutFactory()
    coord = "256.88 6.38"
    cutout_size = [5, 3]
    out_file = cutout_maker.cube_cut(
        cube_file, coord, cutout_size, output_path=path.join(tmpdir, "out_dir"), verbose=False
    )

    assert "256.880000_6.380000_5x3_astrocut.fits" in out_file

    assert isinstance(cutout_maker.cube_wcs, wcs.WCS)
    ra, dec = cutout_maker.cube_wcs.wcs.crval
    assert round(ra, 4) == 250.3497
    assert round(dec, 4) == 2.2809

    # checking on the center coordinate too
    coord = SkyCoord(256.88, 6.38, frame="icrs", unit="deg")
    assert cutout_maker.center_coord.separation(coord) == 0


def test_header_keywords_quality(cube_file, tmp_path):
    """Test header keywords quality"""

    tmpdir = str(tmp_path)

    cutout_maker = CutoutFactory()
    coord = "256.88 6.38"
    cutout_size = [5, 3]
    out_file = cutout_maker.cube_cut(
        cube_file, coord, cutout_size, output_path=path.join(tmpdir, "out_dir"), verbose=False
    )

    with fits.open(out_file) as hdulist:

        # Primary header checks
        primary_header = hdulist[0].header
        assert primary_header['FFI_TYPE'] == 'SPOC'
        assert primary_header['CREATOR'] == 'astrocut'
        assert primary_header['BJDREFI'] == 2457000

        # Checking for header keyword propagation in EXT 1 and 2
        ext1_header = hdulist[1].header
        ext2_header = hdulist[2].header
        assert ext1_header['FFI_TYPE'] == ext2_header['FFI_TYPE'] == primary_header['FFI_TYPE']
        assert ext1_header['CREATOR'] == ext2_header['CREATOR'] == primary_header['CREATOR']
        assert ext1_header['BJDREFI'] == ext2_header['BJDREFI'] == primary_header['BJDREFI']


def test_header_keywords_diffs(cube_file, tmp_path):
    """Test known header keywords differences"""

    tmpdir = str(tmp_path)

    cutout_maker = CutoutFactory()
    coord = "256.88 6.38"
    cutout_size = [5, 3]
    out_file = cutout_maker.cube_cut(
        cube_file, coord, cutout_size, output_path=path.join(tmpdir, "out_dir"), verbose=False
    )

    with fits.open(out_file) as hdulist:

        # Get units from BinTableHDU
        cols = hdulist[1].columns.info('name, unit', output=False)
        cols_dict = dict(zip(*cols.values()))

        assert len(cols_dict) == 12
        assert hdulist[0].header['TIMEREF'] == 'SOLARSYSTEM', 'TIMEREF keyword does not match expected'
        assert hdulist[0].header['TASSIGN'] == 'SPACECRAFT', 'TASSIGN keyword does not match expected'
        assert cols_dict['FLUX'] == 'e-/s', f'Expected `FLUX` units of "e-/s", got units of "{cols_dict["FLUX"]}"'
        assert hdulist[1].data.field('CADENCENO').all() == 0.0


def test_get_cutout_limits(cube_file, tmp_path):
    """Test _get_cutout_limits"""

    tmpdir = str(tmp_path)

    cutout_maker = CutoutFactory()

    # Making the cutout
    coord = "256.88 6.38"
    cutout_size = [5, 3]
    out_file = cutout_maker.cube_cut(
        cube_file, coord, cutout_size, output_path=path.join(tmpdir, "out_dir"), verbose=False
    )

    xmin, xmax = cutout_maker.cutout_lims[0]
    ymin, ymax = cutout_maker.cutout_lims[1]

    assert (xmax-xmin) == cutout_size[0]
    assert (ymax-ymin) == cutout_size[1]

    cutout_size = [5*u.pixel, 7*u.pixel]
    out_file = cutout_maker.cube_cut(cube_file, coord, cutout_size, verbose=False, output_path=tmpdir)
    assert "256.880000_6.380000_5x7_astrocut.fits" in out_file

    xmin, xmax = cutout_maker.cutout_lims[0]
    ymin, ymax = cutout_maker.cutout_lims[1]

    assert (xmax-xmin) == cutout_size[0].value
    assert (ymax-ymin) == cutout_size[1].value

    cutout_size = [3*u.arcmin, 5*u.arcmin]
    out_file = cutout_maker.cube_cut(cube_file, coord, cutout_size, verbose=False, output_path=tmpdir)
    assert "256.880000_6.380000_8x15_astrocut.fits" in out_file

    xmin, xmax = cutout_maker.cutout_lims[0]
    ymin, ymax = cutout_maker.cutout_lims[1]

    assert (xmax - xmin) == 8
    assert (ymax - ymin) == 15


def test_small_cutout(cube_file, tmp_path):

    tmpdir = str(tmp_path)

    cutout_maker = CutoutFactory()
    cutout_size = [1 * u.arcsec, 5 * u.arcsec]
    coord = "256.88 6.38"

    out_file = cutout_maker.cube_cut(cube_file, coord, cutout_size, verbose=False, output_path=tmpdir)
    assert "256.880000_6.380000_1x1_astrocut.fits" in out_file

    xmin, xmax = cutout_maker.cutout_lims[0]
    ymin, ymax = cutout_maker.cutout_lims[1]

    assert (xmax - xmin) == 1
    assert (ymax - ymin) == 1


def test_get_full_cutout_wcs(cube_file, tmp_path):
    """Test _get_full_cutout_wcs"""

    tmpdir = str(tmp_path)

    cutout_maker = CutoutFactory()
    cutout_size = [5, 3]
    coord = "256.88 6.38"

    cutout_maker.cube_cut(cube_file, coord, cutout_size, verbose=False, output_path=tmpdir)


def test_fit_cutout_wcs(cube_file, tmp_path):
    """Test _fit_cutout_wcs"""

    tmpdir = str(tmp_path)

    cutout_maker = CutoutFactory()
    cutout_size = [5, 3]
    coord = "256.88 6.38"

    cutout_maker.cube_cut(cube_file, coord, cutout_size, verbose=False, output_path=tmpdir)
    cry, crx = cutout_maker.cutout_wcs.wcs.crpix
    assert round(cry) == 3
    assert round(crx) == 2


def test_target_pixel_file(cube_file, tmp_path):
    """Test target pixel file"""
    
    tmpdir = str(tmp_path)

    cutout_maker = CutoutFactory()
    cutout_size = [5, 3]
    coord = "256.88 6.38"

    out_file = cutout_maker.cube_cut(cube_file, coord, cutout_size, verbose=False, output_path=tmpdir)

    # Testing the cutout content is in test_cube_cutout
    # this tests that the format of the tpf is what it should be
    tpf = fits.open(out_file)

    assert tpf[0].header["ORIGIN"] == 'STScI/MAST'

    tpf_table = tpf[1].data
    # SPOC cutouts have 1 extra columns in EXT 1
    assert len(tpf_table.columns) == 12
    assert "TIME" in tpf_table.columns.names
    assert "FLUX" in tpf_table.columns.names
    assert "FLUX_ERR" in tpf_table.columns.names
    assert "FFI_FILE" in tpf_table.columns.names

    # Check img cutout shape and data type
    cutout_img = tpf_table[0]['FLUX']
    assert cutout_img.shape == (3, 5)
    assert cutout_img.dtype.name == 'float32'

    # Check error cutout shape, contents, and data type
    cutout_err = tpf_table[0]['FLUX_ERR']
    assert cutout_err.shape == (3, 5)
    assert cutout_err.dtype.name == 'float32'

    # Check aperture shape and data type
    aperture = tpf[2].data
    assert aperture.shape == (3, 5)
    assert aperture.dtype.name == 'int32'

    tpf.close()


def test_ffi_cube_header(cube_file, ffi_files):
    """Test FFI headers versus cube headers"""

    ffi_header_keys = list(fits.getheader(ffi_files[0], 0).keys())
    cube_header_keys = list(fits.getheader(cube_file, 0).keys())

    # Lists of expected keys
    # CHECKSUM in FFI header, not in cube header
    # SECTOR, DATE, ORIGIN in cube header, not in FFI header
    effi_keys = ['CHECKSUM']
    ecube_keys = ['SECTOR', 'DATE', 'ORIGIN']

    for k in effi_keys:
        assert k in ffi_header_keys
        assert k not in cube_header_keys

    for k in ecube_keys:
        assert k not in ffi_header_keys
        assert k in cube_header_keys


def test_inputs(cube_file, tmp_path, caplog):
    """
    Testing with different user input types/combos. And verbose.
    """

    tmpdir = str(tmp_path)

    cutout_maker = CutoutFactory()

    # Setting up
    coord = "256.88 6.38"

    cutout_size = [5, 3]*u.pixel
    cutout_file = cutout_maker.cube_cut(cube_file, coord, cutout_size, output_path=tmpdir, verbose=True)
    captured = caplog.text
    assert "Image cutout cube shape: (100, 3, 5)" in captured
    assert "Using WCS from row 50 out of 100" in captured
    assert "Cutout center coordinate: 256.88, 6.38" in captured
    assert "5x3" in cutout_file

    cutout_size = [5, 3]*u.arcmin
    cutout_file = cutout_maker.cube_cut(cube_file, coord, cutout_size, output_path=tmpdir, verbose=False)
    assert "14x9" in cutout_file

    cutout_size = [5, 3, 9]*u.pixel
    with pytest.warns(InputWarning):
        cutout_file = cutout_maker.cube_cut(cube_file, coord, cutout_size, output_path=tmpdir, verbose=False)
    assert "5x3" in cutout_file
    assert "x9" not in cutout_file


def test_s3_cube_cut(tmp_path: Path):
    """Does using an S3-hosted TESS cube yield correct results?

    This test implements a spot check which verifies whether a cutout
    for Proxima Cen (Sector 38) obtained from an S3-hosted cube
    file yields results that are identical to those returned
    by the Tesscut service.

    To speed up the test and avoid adding astroquery as a dependency,
    the test uses hard-coded reference values which were obtained
    as follows:

    >>> from astroquery.mast import Tesscut  # doctest: +SKIP
    >>> crd = SkyCoord(217.42893801, -62.67949189, unit="deg")  # doctest: +SKIP
    >>> cut = Tesscut.get_cutouts(crd, size=3, sector=38)  # doctest: +SKIP
    >>> cut[0][1].data.shape  # doctest: +SKIP
    (3705,)
    >>> cut[0][1].data['TIME'][0]  # doctest: +SKIP
    2333.8614060219998
    >>> cut[0][1].data['FLUX'][100][0, 0]  # doctest: +SKIP
    2329.8127
    >>> cut[0][1].data['FLUX_ERR'][200][1, 2]  # doctest: +SKIP
    1.1239403
    >>> cut[0][0].header['CAMERA']  # doctest: +SKIP
    2
    """
    # Test case: Proxima Cen in Sector 38 (Camera 2, CCD 2)
    coord = SkyCoord(217.42893801, -62.67949189, unit="deg", frame="icrs")
    cube_file = "s3://stpubdata/tess/public/mast/tess-s0038-2-2-cube.fits"
    cutout_file = CutoutFactory().cube_cut(cube_file, coord, 3, output_path=str(tmp_path))
    hdulist = fits.open(cutout_file)
    assert hdulist[1].data.shape == (3705,)
    assert np.isclose(hdulist[1].data["TIME"][0], 2333.8614060219998)
    assert np.isclose(hdulist[1].data["FLUX"][100][0, 0], 2329.8127)
    assert np.isclose(hdulist[1].data["FLUX_ERR"][200][1, 2], 1.1239403)
    assert hdulist[0].header["CAMERA"] == 2
    hdulist.close()


def test_multithreading(tmp_path):
    tmpdir = str(tmp_path)

    cutout_size = 10
    coord = SkyCoord(217.42893801, -62.67949189, unit="deg", frame="icrs")
    cube_file = "s3://stpubdata/tess/public/mast/tess-s0038-2-2-cube.fits"

    cut_factory = CutoutFactory()

    # 1 thread condition (no threading) used as verification for the different thread conditions
    cutout_no_threads = cut_factory.cube_cut(
        cube_file, coordinates=coord, output_path=tmpdir, verbose=False, cutout_size=cutout_size, threads=1
    )
    data_no_threads = fits.getdata(cutout_no_threads)

    # when threads="auto", number of threads is system dependent: cpu_count + 4, limited to max of 32
    # https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor
    for threads in (4, "auto"):
        cutout_threads = cut_factory.cube_cut(
            cube_file, coordinates=coord, output_path=tmpdir, verbose=False, cutout_size=cutout_size, threads=threads
        )
        data_threads = fits.getdata(cutout_threads)

        for ext_name in ("FLUX", "FLUX_ERR"):
            assert np.array_equal(data_no_threads[ext_name], data_threads[ext_name])
