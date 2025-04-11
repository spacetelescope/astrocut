from pathlib import Path
import numpy as np
import pytest

import asdf
from astropy import coordinates as coord
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.nddata import Cutout2D
from astropy.io import fits
from gwcs import wcs, coordinate_frames
from PIL import Image

from astrocut.asdf_cutout import ASDFCutout, asdf_cut
from astrocut.exceptions import DataWarning, InvalidInputError, InvalidQueryError


def make_wcs(xsize, ysize, ra=30., dec=45.):
    """ Create a fake gwcs object """
    # todo - refine this to better reflect roman wcs

    # create transformations
    # - shift coords so array center is at 0, 0 ; reference pixel
    # - scale pixels to correct angular scale
    # - project coords onto sky with TAN projection
    # - transform center pixel to the input celestial coordinate
    pixelshift = models.Shift(-xsize) & models.Shift(-ysize)
    pixelscale = models.Scale(0.1 / 3600.) & models.Scale(0.1 / 3600.)  # 0.1 arcsec/pixel
    tangent_projection = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(ra, dec, 180.)

    # net transforms pixels to sky
    det2sky = pixelshift | pixelscale | tangent_projection | celestial_rotation

    # define the wcs object
    detector_frame = coordinate_frames.Frame2D(name='detector', axes_names=('x', 'y'), unit=(u.pix, u.pix))
    sky_frame = coordinate_frames.CelestialFrame(reference_frame=coord.ICRS(), name='world', unit=(u.deg, u.deg))
    return wcs.WCS([(detector_frame, det2sky), (sky_frame, None)])


@pytest.fixture()
def makefake():
    """ Fixture factory to make a fake gwcs and dataset """

    def _make_fake(nx, ny, ra, dec, zero=False, asint=False):
        # create the wcs
        wcsobj = make_wcs(nx/2, ny/2, ra=ra, dec=dec)
        wcsobj.bounding_box = ((0, nx), (0, ny))

        # create the data
        if zero:
            data = np.zeros([nx, ny])
        else:
            size = nx * ny
            data = np.arange(size).reshape(nx, ny)

        # make a quantity
        data *= (u.electron / u.second)

        # make integer array
        if asint:
            data = data.astype(int)

        return data, wcsobj

    yield _make_fake


@pytest.fixture()
def fakedata(makefake):
    """ Fixture to create fake data and wcs """
    # set up initial parameters
    nx = 1000
    ny = 1000
    ra = 30.
    dec = 45.

    yield makefake(nx, ny, ra, dec)


@pytest.fixture()
def test_images(tmp_path, fakedata):
    """ Fixture to create a fake dataset of 3 images """
    # get the fake data
    data, wcsobj = fakedata

    # create meta
    meta = {'wcs': wcsobj}

    # create and write the asdf file
    tree = {'roman': {'data': data, 'meta': meta}}
    af = asdf.AsdfFile(tree)

    path = tmp_path / 'roman'
    path.mkdir(exist_ok=True)

    files = []
    for i in range(3):
        filename = path / f'test_roman_{i}.asdf'
        af.write_to(filename)
        files.append(filename)

    return files


@pytest.fixture
def center_coord():
    """ Fixture to return a center coordinate """
    return SkyCoord('29.99901792 44.99930555', unit='deg')


@pytest.fixture
def cutout_size():
    """ Fixture to return a cutout size """
    return 10


def test_asdf_cutout(test_images, center_coord, cutout_size):
    cutout = ASDFCutout(test_images, center_coord, cutout_size)
    cutouts = cutout.cutouts
    # Should output a list of strings for multiple input files
    assert isinstance(cutouts, list)
    assert isinstance(cutouts[0], Cutout2D)
    assert len(cutouts) == 3
    assert isinstance(cutout.asdf_cutouts, list)
    assert isinstance(cutout.asdf_cutouts[0], asdf.AsdfFile)
    assert isinstance(cutout.fits_cutouts, list)
    assert isinstance(cutout.fits_cutouts[0], fits.HDUList)

    # Open output files
    for i, cutout in enumerate(cutouts):
        # Check shape of data
        cutout_data = cutout.data
        cutout_wcs = cutout.wcs
        assert cutout_data.shape == (10, 10)

        # Check that data is equal between cutout and original image
        with asdf.open(test_images[i]) as input_af:
            assert np.all(cutout_data == input_af['roman']['data'].value[470:480, 471:481])

        # Check WCS and that center coordinate matches input
        s_coord = cutout_wcs.pixel_to_world(cutout_size / 2, cutout_size / 2)
        assert cutout_wcs.pixel_shape == (10, 10)
        assert np.isclose(s_coord.ra.deg, center_coord.ra.deg)
        assert np.isclose(s_coord.dec.deg, center_coord.dec.deg)


def test_asdf_cutout_write_to_file(test_images, center_coord, cutout_size, tmpdir):
    # Write cutouts to ASDF files on disk
    cutout = ASDFCutout(test_images, center_coord, cutout_size)
    asdf_files = cutout.write_as_asdf(output_dir=tmpdir)
    assert len(asdf_files) == 3
    for i, asdf_file in enumerate(asdf_files):
        with asdf.open(asdf_file) as af:
            assert 'roman' in af.tree
            assert 'data' in af.tree['roman']
            assert 'meta' in af.tree['roman']
            assert np.all(af.tree['roman']['data'] == cutout.cutouts[i].data)
            assert af.tree['roman']['meta']['wcs'].pixel_shape == (10, 10)
            assert Path(asdf_file).stat().st_size < Path(test_images[i]).stat().st_size

    # Write cutouts to FITS files on disk
    cutout = ASDFCutout(test_images, center_coord, cutout_size)
    fits_files = cutout.write_as_fits(output_dir=tmpdir)
    assert len(fits_files) == 3
    for i, fits_file in enumerate(fits_files):
        with fits.open(fits_file) as hdul:
            assert np.all(hdul[0].data == cutout.cutouts[i].data)
            assert hdul[0].header['NAXIS1'] == 10
            assert hdul[0].header['NAXIS2'] == 10
            assert Path(fits_file).stat().st_size < Path(test_images[i]).stat().st_size


def test_asdf_cutout_partial(test_images, center_coord, cutout_size):
    # Off the top
    center_coord = SkyCoord('29.99901792 44.9861', unit='deg')
    cutout = ASDFCutout(test_images[0], center_coord, cutout_size).cutouts[0]
    assert cutout.data.shape == (10, 10)
    assert np.isnan(cutout.data[:cutout_size//2, :]).all()

    # Off the bottom
    center_coord = SkyCoord('29.99901792 45.01387', unit='deg')
    cutout = ASDFCutout(test_images[0], center_coord, cutout_size).cutouts[0]
    assert np.isnan(cutout.data[cutout_size//2:, :]).all()

    # Off the left, with integer fill value
    center_coord = SkyCoord('29.98035835 44.99930555', unit='deg')
    cutout = ASDFCutout(test_images[0], center_coord, cutout_size, fill_value=1).cutouts[0]
    assert np.all(cutout.data[:, :cutout_size//2] == 1)

    # Off the right, with float fill value
    center_coord = SkyCoord('30.01961 44.99930555', unit='deg')
    cutout = ASDFCutout(test_images[0], center_coord, cutout_size, fill_value=1.5).cutouts[0]
    assert np.all(cutout.data[:, cutout_size//2:] == 1.5)

    # Error if unexpected fill value
    with pytest.raises(InvalidInputError, match='Fill value must be an integer or a float.'):
        ASDFCutout(test_images[0], center_coord, cutout_size, fill_value='invalid')


def test_asdf_cutout_poles(cutout_size, makefake, tmp_path):
    """ Test we can make cutouts around poles """
    # Make fake zero data around the pole
    ra, dec = 315.0, 89.995
    data, gwcs = makefake(1000, 1000, ra, dec, zero=True)

    # Add some values (5x5 array)
    data.value[245:250, 245:250] = 1

    # Check central pixel is correct
    ss = gwcs(500, 500)
    assert ss == (ra, dec)

    # Set input cutout coord
    center_coord = SkyCoord(284.702, 89.986, unit='deg')

    # create and write the asdf file
    meta = {'wcs': gwcs}
    tree = {'roman': {'data': data, 'meta': meta}}
    af = asdf.AsdfFile(tree)
    path = tmp_path / 'roman'
    path.mkdir(exist_ok=True)
    filename = path / 'test_roman_poles.asdf'
    af.write_to(filename)

    # Get cutout
    cutout = ASDFCutout(filename, center_coord, cutout_size).cutouts[0]

    # Check cutout contains all data
    assert len(np.where(cutout.data == 1)[0]) == 25


def test_asdf_cutout_not_in_footprint(test_images, center_coord, cutout_size):
    # Throw error if cutout location is not in image footprint
    with pytest.warns(DataWarning, match='Cutout footprint does not overlap'):
        with pytest.raises(InvalidQueryError, match='Cutout contains no data!'):
            ASDFCutout(test_images[0], SkyCoord('0 0', unit='deg'), cutout_size)

    # Alter one of the test images to only contain zeros in cutout footprint
    with asdf.open(test_images[0], mode='rw') as af:
        af['roman']['data'][470:480, 471:481] = 0
        af.update()

    # Should warn about first image containing no data, but not fail
    with pytest.warns(DataWarning, match='contains no data, skipping...'):
        cutouts = ASDFCutout(test_images, center_coord, cutout_size).cutouts
    assert len(cutouts) == 2


def test_asdf_cutout_no_gwcs(test_images, center_coord, cutout_size):
    # Remove WCS from test image
    with asdf.open(test_images[0], mode='rw') as af:
        del af['roman']['meta']['wcs']
        af.update()

    # Should warn about missing WCS for first image, but not fail
    with pytest.warns(DataWarning, match='does not contain a GWCS object'):
        cutouts = ASDFCutout(test_images, center_coord, cutout_size).cutouts
    assert len(cutouts) == 2


def test_asdf_cutout_invalid_params(test_images, center_coord, cutout_size, tmpdir):
    # Invalid units for cutout size
    cutout_size = 1 * u.m  # meters are not valid
    with pytest.raises(InvalidInputError, match='Cutout size unit meter is not supported.'):
        ASDFCutout(test_images, center_coord, cutout_size)


def test_asdf_cutout_img_output(test_images, center_coord, cutout_size, tmpdir):
    # Basic JPG image
    jpg_files = ASDFCutout(test_images, center_coord, cutout_size).write_as_img(output_dir=tmpdir, 
                                                                                output_format='jpg')
    assert len(jpg_files) == len(test_images)
    with open(jpg_files[0], 'rb') as IMGFLE:
        assert IMGFLE.read(3) == b'\xFF\xD8\xFF'  # JPG

    # PNG (single input file, not as list)
    png_files = ASDFCutout(test_images[0], center_coord, cutout_size).write_as_img(output_dir=tmpdir, 
                                                                                   output_format='png')
    with open(png_files[0], 'rb') as IMGFLE:
        assert IMGFLE.read(8) == b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'  # PNG
    assert len(png_files) == 1

    # Save to memory only
    img_cutouts = ASDFCutout(test_images[0], center_coord, cutout_size).get_image_cutouts()
    assert len(img_cutouts) == 1
    assert isinstance(img_cutouts[0], Image.Image)
    assert np.array(img_cutouts[0]).shape == (10, 10)

    # Color image
    color_jpg = ASDFCutout(test_images, center_coord, cutout_size).write_as_img(output_dir=tmpdir, colorize=True)
    img = Image.open(color_jpg)
    assert img.mode == 'RGB'


def test_get_center_pixel(fakedata):
    """ Test get_center_pixel function """
    # get the fake data
    __, gwcs = fakedata

    pixel_coordinates, wcs = ASDFCutout.get_center_pixel(gwcs, 30, 45)
    assert np.allclose(pixel_coordinates, (np.array(500), np.array(500)))
    assert np.allclose(wcs.celestial.wcs.crval, np.array([30, 45]))


def test_asdf_cut(test_images, center_coord, cutout_size, tmpdir):
    """ Test convenience function to create ASDF cutouts """
    def check_paths(cutout_paths, ext):
        assert isinstance(cutout_paths, list)
        assert isinstance(cutout_paths[0], str)
        assert len(cutout_paths) == 3
        for i, path in enumerate(cutout_paths):
            assert isinstance(path, str)
            assert path.endswith(ext)
            assert Path(path).exists()
            assert str(tmpdir) in path
            assert Path(test_images[i]).stem in path
            assert center_coord.ra.to_string(unit='deg', decimal=True) in path
            assert center_coord.dec.to_string(unit='deg', decimal=True) in path
            assert '10-x-10' in path
    
    # Write files to disk as ASDF files
    asdf_paths = asdf_cut(test_images, center_coord.ra.deg, center_coord.dec.deg, cutout_size, output_dir=tmpdir)
    check_paths(asdf_paths, '.asdf')

    # Write files to disk as FITS files
    fits_paths = asdf_cut(test_images, center_coord.ra.deg, center_coord.dec.deg, cutout_size, output_dir=tmpdir, 
                          output_format='fits')
    check_paths = (fits_paths, '.fits')

    # Write cutouts to memory as Cutout2D objects
    cutouts = asdf_cut(test_images, center_coord.ra.deg, center_coord.dec.deg, cutout_size, write_file=False)
    assert isinstance(cutouts, list)
    assert isinstance(cutouts[0], Cutout2D)
    assert len(cutouts) == 3

    # Error if output format is not supported
    with pytest.raises(InvalidInputError, match='Output format .invalid is not recognized.'):
        asdf_cut(test_images, center_coord.ra.deg, center_coord.dec.deg, cutout_size, output_format='invalid')
