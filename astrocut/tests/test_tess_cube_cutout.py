
import shutil
import numpy as np
import pytest
import warnings
from pathlib import Path
from typing import List, Literal

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

from .utils_for_test import create_test_ffis
from ..exceptions import DataWarning, InvalidInputError, InvalidQueryError
from ..cube_factory import CubeFactory
from ..tica_cube_factory import TicaCubeFactory
from ..cube_cutout import CubeCutout
from ..tess_cube_cutout import TessCubeCutout


@pytest.fixture
def img_size():
    """Fixture to return the image size"""
    return 10


@pytest.fixture
def num_images():
    """Fixture to return the number of images"""
    return 10


@pytest.fixture
def cutout_size():
    """Fixture to return the cutout size"""
    return 6


@pytest.fixture
def coordinates(cube_wcs, img_size):
    """Fixture to return the coordinates at the center of the images"""
    return cube_wcs.pixel_to_world(img_size // 2, img_size // 2)


@pytest.fixture
def cutout_lims(img_size, cutout_size):
    """Fixture to return the expected cutout limits"""
    return np.array([[img_size // 2 - cutout_size // 2, img_size // 2 + cutout_size // 2],
                     [img_size // 2 - cutout_size // 2, img_size // 2 + cutout_size // 2]])


@pytest.fixture
def ffi_files(tmpdir, img_size, num_images, ffi_type: Literal["SPOC", "TICA"]):
    """Fixture for creating test ffi files"""
    return create_test_ffis(img_size, num_images, dir_name=tmpdir, product=ffi_type)


@pytest.fixture
def cube_file(ffi_files: List[str], tmpdir, ffi_type: Literal['SPOC', "TICA"]):
    """Fixture for creating a cube file"""
    # Making the test cube
    if ffi_type == "SPOC":
        cube_maker = CubeFactory()
    else:
        cube_maker = TicaCubeFactory()

    cube_file = cube_maker.make_cube(ffi_files, Path(tmpdir, "test_cube.fits"), verbose=False)

    return cube_file


@pytest.fixture
def cube_wcs(ffi_files: List[str], tmpdir, ffi_type: Literal["SPOC", "TICA"]):
    """Fixture to return the WCS of the cube"""
    # Read one of the input images to get the WCS
    n = 1 if ffi_type == 'SPOC' else 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FITSFixedWarning)
        return WCS(fits.getheader(ffi_files[0], n))


@pytest.mark.parametrize("ffi_type", ["SPOC", "TICA"])
def test_tess_cube_cutout(cube_file, num_images, ffi_type, cutout_size, coordinates, cutout_lims):
    # Make Cutout
    cutouts = TessCubeCutout(cube_file, coordinates, cutout_size, product=ffi_type).cutouts
    cutout = cutouts[0]

    # Should return a list of CubeCutoutInstance objects
    assert isinstance(cutouts, list)
    assert isinstance(cutout, CubeCutout.CubeCutoutInstance)

    # Check cutout attributes
    assert cutout.shape == (num_images, cutout_size, cutout_size)
    assert np.all(cutout.cutout_lims == cutout_lims)
    assert np.all(cutout.aperture == 1)  # Cutout is entirely within the image

    # Compare data with input cube file
    with fits.open(cube_file) as hdul:
        data = np.transpose(hdul[1].data, (3, 2, 0, 1))
        assert np.all(data[0, :, cutout_lims[0, 0]:cutout_lims[0, 1], cutout_lims[1, 0]:cutout_lims[1, 1]] == 
                      cutout.data)
        if ffi_type == 'SPOC':
            assert np.all(data[1, :, cutout_lims[0, 0]:cutout_lims[0, 1], cutout_lims[1, 0]:cutout_lims[1, 1]] == 
                          cutout.uncertainty)

    # Check the cutout WCS
    cutout_wcs = cutout.wcs
    assert isinstance(cutout_wcs, WCS)
    assert np.all(np.round(cutout_wcs.world_to_pixel(coordinates)) == (cutout_size // 2, cutout_size // 2))
    assert cutout_wcs.pixel_shape == (cutout_size, cutout_size)
    assert cutout.wcs_fit['WCS_MSEP'][0] == 0
    assert cutout.wcs_fit['WCS_SIG'][0] == 0


@pytest.mark.parametrize("ffi_type", ["SPOC", "TICA"])
def test_tess_cube_cutout_tpf(cube_file, num_images, ffi_type, cutout_size, coordinates, cutout_lims):
    # Make Cutout
    tpfs = TessCubeCutout(cube_file, coordinates, cutout_size, product=ffi_type).tpf_cutouts
    tpf = tpfs[0]

    # Should return a list of HDUList objects
    assert isinstance(tpfs, list)
    assert isinstance(tpf, fits.HDUList)

    # Check primary header values
    primary_header = tpf[0].header
    assert primary_header['RA_OBJ'] == coordinates.ra.deg
    assert primary_header['DEC_OBJ'] == coordinates.dec.deg
    assert primary_header['FFI_TYPE'] == ffi_type
    assert primary_header['CREATOR'] == 'astrocut'
    assert primary_header['ORIGIN'] == 'STScI/MAST'
    assert primary_header['TELAPSE'] == primary_header['TSTOP'] - primary_header['TSTART']

    # Get units from BinTableHDU
    cols = tpf[1].columns.info('name, unit', output=False)
    cols_dict = dict(zip(*cols.values()))

    # Check differences in primary header values and units for SPOC and TICA
    if ffi_type == 'SPOC':
        assert primary_header['TIMEREF'] == 'SOLARSYSTEM'
        assert primary_header['TASSIGN'] == 'SPACECRAFT'
        assert cols_dict['FLUX'] == 'e-/s'
        assert tpf[1].data.field('CADENCENO').all() == 0.0

    if ffi_type == 'TICA':
        assert primary_header['TIMEREF'] is None
        assert primary_header['TASSIGN'] is None
        assert cols_dict['FLUX'] == 'e-'
        assert tpf[1].data.field('CADENCENO').all() != 0.0

        # Verifying DATE-OBS calculation in TICA
        date_obs = primary_header['DATE-OBS']
        tstart = Time(date_obs).jd - primary_header['BJDREFI']
        assert primary_header['TSTART'] == tstart

        # Verifying DATE-END calculation in TICA
        date_end = primary_header['DATE-END']
        tstop = Time(date_end).jd - primary_header['BJDREFI']
        assert primary_header['TSTOP'] == tstop

    # Check for header keyword propagation in EXT 1 and 2
    ext1_header = tpf[1].header
    ext2_header = tpf[2].header
    assert ext1_header['FFI_TYPE'] == ext2_header['FFI_TYPE'] == primary_header['FFI_TYPE']
    assert ext1_header['CREATOR'] == ext2_header['CREATOR'] == primary_header['CREATOR']
    assert ext1_header['BJDREFI'] == ext2_header['BJDREFI'] == primary_header['BJDREFI']

    # Check timeseries data
    table = tpf[1].data
    assert tpf[1].data.shape[0] == num_images
    assert np.all(table['TIME'] == (np.arange(num_images) + 0.5))

    # Check data table columns
    num_cols = 12 if ffi_type == 'SPOC' else 11
    assert len(table.columns) == num_cols
    assert 'TIME' in table.columns.names
    assert 'FLUX' in table.columns.names
    assert 'FLUX_ERR' in table.columns.names
    assert 'FFI_FILE' in table.columns.names

    # Check flux data
    flux = table['FLUX']
    assert flux.shape == (num_images, cutout_size, cutout_size)
    assert flux.dtype.type == np.float32

    # Check flux error data
    err = table['FLUX_ERR']
    assert err.shape == (num_images, cutout_size, cutout_size)
    if ffi_type == 'TICA':
        assert np.mean(err) == 0
    assert err.dtype.type == np.float32

    # Compare data with input cube file
    with fits.open(cube_file) as hdul:
        data = np.transpose(hdul[1].data, (3, 2, 0, 1))
        assert np.all(data[0, :, cutout_lims[0, 0]:cutout_lims[0, 1], cutout_lims[1, 0]:cutout_lims[1, 1]] == flux)
        if ffi_type == 'SPOC':
            assert np.all(data[1, :, cutout_lims[0, 0]:cutout_lims[0, 1], cutout_lims[1, 0]:cutout_lims[1, 1]] == err)

    # Check aperture HDU
    aper = tpf[2].data
    assert np.all(aper == 1)
    assert aper.shape == (cutout_size, cutout_size)
    assert aper.dtype.type == np.int32

    tpf.close()


@pytest.mark.parametrize("ffi_type", ["SPOC", "TICA"])
def test_tess_cutout_partial(cube_file, cutout_size, ffi_type, cube_wcs, img_size, coordinates, num_images):
    # Off the top
    coord = cube_wcs.pixel_to_world(0, img_size // 2)
    cutout = TessCubeCutout(cube_file, coord, cutout_size, product=ffi_type).cutouts[0]
    offset = cutout_size // 2
    assert np.all(np.isnan(cutout.data[:, :, :offset]))
    assert np.all(cutout.aperture[:, :offset] == 0)
    if ffi_type == 'SPOC':
        assert np.all(np.isnan(cutout.uncertainty[:, :, :offset]))

    # Off the bottom
    coord = cube_wcs.pixel_to_world(img_size, img_size // 2)
    cutout = TessCubeCutout(cube_file, coord, cutout_size, product=ffi_type).cutouts[0]
    assert np.all(np.isnan(cutout.data[:, :, offset:]))
    assert np.all(cutout.aperture[:, offset:] == 0)
    if ffi_type == 'SPOC':
        assert np.all(np.isnan(cutout.uncertainty[:, :, offset:]))

    # Off the left, integer fill value
    coord = cube_wcs.pixel_to_world(img_size // 2, 0)
    cutout = TessCubeCutout(cube_file, coord, cutout_size, product=ffi_type, fill_value=0).cutouts[0]
    assert np.all(cutout.data[:, :offset, :] == 0)
    assert np.all(cutout.aperture[:offset, :] == 0)
    if ffi_type == 'SPOC':
        assert np.all(cutout.uncertainty[:, :offset, :] == 0)

    # Off the right, float fill value
    coord = cube_wcs.pixel_to_world(img_size // 2, img_size)
    cutout = TessCubeCutout(cube_file, coord, cutout_size, product=ffi_type, fill_value=1.5).cutouts[0]
    assert np.all(cutout.data[:, offset:, :] == 1.5)
    assert np.all(cutout.aperture[offset:, :] == 0)
    if ffi_type == 'SPOC':
        assert np.all(cutout.uncertainty[:, offset:, :] == 1.5)

    # Large cutout that goes off all sides. Use center coordinate
    cutout_size = [120, 140]
    diff_width = (cutout_size[0] - img_size) // 2
    diff_height = (cutout_size[1] - img_size) // 2
    cutout = TessCubeCutout(cube_file, coordinates, cutout_size, product=ffi_type).cutouts[0]
    cutout_mask = np.zeros((num_images, cutout_size[1], cutout_size[0]), dtype=bool)
    cutout_mask[:, diff_height:diff_height + img_size, diff_width:diff_width + img_size] = True
    aper_mask = np.zeros((cutout_size[1], cutout_size[0]), dtype=bool)
    aper_mask[diff_height:diff_height + img_size, diff_width:diff_width + img_size] = True
    assert np.all(np.isnan(cutout.data[~cutout_mask]))
    assert np.all(cutout.aperture[~aper_mask] == 0)
    if ffi_type == 'SPOC':
        assert np.all(np.isnan(cutout.uncertainty[~cutout_mask]))


@pytest.mark.parametrize("ffi_type", ["SPOC"])
def test_tess_cube_cutout_batching(cube_file, tmpdir, cutout_size, coordinates, num_images):
    # Make copies of cube file
    copy1 = Path(tmpdir, "test_cube_1.fits")
    copy2 = Path(tmpdir, "test_cube_2.fits")
    shutil.copy(cube_file, copy1)
    shutil.copy(cube_file, copy2)
    input_files = [cube_file, copy1, copy2]

    # Make multiple cutouts at once
    cutout_size = (5, 3)
    cutouts = TessCubeCutout(input_files, coordinates, cutout_size).cutouts

    # Should return a list of CubeCutoutInstance objects
    assert isinstance(cutouts, list)
    assert len(cutouts) == 3
    assert isinstance(cutouts[0], CubeCutout.CubeCutoutInstance)

    # Check shape of data in cutout
    expected_shape = (num_images, cutout_size[1], cutout_size[0])
    assert cutouts[0].shape == cutouts[1].shape == cutouts[2].shape == expected_shape
    assert np.all(cutouts[0].data == cutouts[1].data)
    assert np.all(cutouts[0].data == cutouts[2].data)
    assert (cutouts[0].uncertainty.shape == cutouts[1].uncertainty.shape == cutouts[2].uncertainty.shape ==
            expected_shape)
    assert np.all(cutouts[0].uncertainty == cutouts[1].uncertainty)
    assert np.all(cutouts[0].uncertainty == cutouts[2].uncertainty)
    assert (cutouts[0].aperture.shape == cutouts[1].aperture.shape == cutouts[2].aperture.shape ==
            (cutout_size[1], cutout_size[0]))
    assert np.all(cutouts[0].aperture == cutouts[1].aperture)
    assert np.all(cutouts[0].aperture == cutouts[2].aperture)


@pytest.mark.parametrize("ffi_type", ["SPOC", "TICA"])
def test_tess_cube_cutout_write_to_tpf(cube_file, tmpdir, cutout_size, coordinates, ffi_type):
    # Make cutout
    cutout = TessCubeCutout(cube_file, coordinates, cutout_size, product=ffi_type)

    # Write to TPF with output_file specified
    cutout_paths = cutout.write_as_tpf(tmpdir, 'cutout.fits')
    cutout_path = cutout_paths[0]

    # Should return a list of filepaths
    assert isinstance(cutout_paths, list)
    assert isinstance(cutout_path, str)

    # Check pathname
    assert Path(cutout_path).exists()
    assert 'cutout.fits' in cutout_path
    assert str(tmpdir) in cutout_path

    # Check that file can be opened with fits
    with fits.open(cutout_path) as hdul:
        assert isinstance(hdul, fits.HDUList)

    # Write to TPF without output_file specified
    cutout_path = cutout.write_as_tpf(tmpdir)[0]

    # Check pathname
    assert Path(cutout_path).exists()
    assert str(tmpdir) in cutout_path
    assert 'test' in cutout_path
    assert f'{coordinates.ra.value:7f}' in cutout_path
    assert f'{coordinates.dec.value:7f}' in cutout_path
    assert f'{cutout_size}x{cutout_size}' in cutout_path
    assert 'astrocut' in cutout_path


def test_tess_cube_cutout_s3():
    # Test case: Proxima Cen in Sector 38 (Camera 2, CCD 2)
    # Make a cutout from cloud cube file
    coord = SkyCoord(217.42893801, -62.67949189, unit="deg", frame="icrs")
    cube_file = "s3://stpubdata/tess/public/mast/tess-s0038-2-2-cube.fits"
    tpf = TessCubeCutout(cube_file, coord, 3).tpf_cutouts[0]

    # Check the cutout for proper shape and some values
    assert tpf[1].data.shape == (3705,)
    table = tpf[1].data
    assert np.isclose(table["TIME"][0], 2333.8614060219998)
    assert np.isclose(table["FLUX"][100][0, 0], 2329.8127)
    assert np.isclose(table["FLUX_ERR"][200][1, 2], 1.1239403)
    assert tpf[0].header["CAMERA"] == 2
    tpf.close()


def test_tess_cube_cutout_threads():
    # Using a cloud file to test multithreading
    cutout_size = 5
    coord = SkyCoord(217.42893801, -62.67949189, unit="deg", frame="icrs")
    cube_file = "s3://stpubdata/tess/public/mast/tess-s0038-2-2-cube.fits"

    # 1 thread condition (no threading) used as verification for the different thread conditions
    cutout_no_threads = TessCubeCutout(cube_file, coord, cutout_size, threads=1).tpf_cutouts[0]
    data_no_threads = cutout_no_threads[1].data

    # when threads="auto", number of threads is system dependent: cpu_count + 4, limited to max of 32
    # https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor
    for threads in (4, "auto"):
        cutout_threads = TessCubeCutout(cube_file, coord, cutout_size, threads=threads).tpf_cutouts[0]
        data_threads = cutout_threads[1].data

        for ext_name in ("FLUX", "FLUX_ERR"):
            assert np.array_equal(data_no_threads[ext_name], data_threads[ext_name])


@pytest.mark.parametrize("ffi_type", ["SPOC"])
def test_tess_cube_cutout_not_in_footprint(cube_file):
    # Make a cutout with a coordinate outside the image footprint
    warnings.simplefilter('error')
    coord = SkyCoord(10, 10, unit="deg", frame="icrs")
    with pytest.warns(DataWarning, match='Cutout footprint does not overlap'):
        with pytest.raises(InvalidQueryError, match='Cube cutout contains no data!'):
            TessCubeCutout(cube_file, coord, 3)


@pytest.mark.parametrize("ffi_type", ["SPOC"])
def test_tess_cube_cutout_invalid_product(cube_file):
    # Error if an invalid product name is input
    with pytest.raises(InvalidInputError, match='Product for TESS cube cutouts must be'):
        TessCubeCutout(cube_file, SkyCoord(0, 0, unit='deg'), 3, product='INVALID')
