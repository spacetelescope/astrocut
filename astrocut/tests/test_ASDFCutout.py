from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import asdf
import astropy.wcs as fits_wcs
from astropy import coordinates as coord
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.nddata import Cutout2D
from astropy.io import fits
from gwcs import wcs, coordinate_frames
from s3path import S3Path
from PIL import Image

from astrocut.ASDFCutout import ASDFCutout
from astrocut.asdf_cutouts import asdf_cut
from astrocut.exceptions import DataWarning, InputWarning, InvalidInputError, InvalidQueryError


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


@pytest.mark.parametrize('output_format', ['asdf', 'fits'])
def test_asdf_cutout(test_images, center_coord, cutout_size, tmpdir, output_format):
    cutout = ASDFCutout(test_images[0], center_coord, cutout_size, output_format=output_format, 
                        output_dir=tmpdir).cutout()
    # Should output a list of memory objects
    assert isinstance(cutout, list)
    assert isinstance(cutout[0], fits.HDUList if output_format == 'fits' else asdf.AsdfFile)

    cutouts = ASDFCutout(test_images, center_coord, cutout_size, output_format=output_format, 
                         output_dir=tmpdir).cutout()
    # Should output a list of strings for multiple input files
    assert isinstance(cutouts, list)
    assert isinstance(cutouts[0], fits.HDUList if output_format == 'fits' else asdf.AsdfFile)
    assert len(cutouts) == 3

    # Open output files
    for i, cutout in enumerate(cutouts):
        # Get cutout data and WCS based on output format
        if output_format == 'asdf':
            cutout_data = np.copy(cutout['roman']['data'])
            cutout_wcs = cutout['roman']['meta']['wcs']
        else:
            cutout_data = cutout[0].data
            cutout_wcs = fits_wcs.WCS(cutout[0].header)

        # Check shape of data
        assert cutout_data.shape == (10, 10)

        # Check that data is equal between cutout and original image
        with asdf.open(test_images[i]) as input_af:
            assert np.all(cutout_data == input_af['roman']['data'].value[470:480, 471:481])

        # Check WCS and that center coordinate matches input
        s_coord = cutout_wcs.pixel_to_world(cutout_size / 2, cutout_size / 2)
        assert cutout_wcs.pixel_shape == (10, 10)
        assert np.isclose(s_coord.ra.deg, center_coord.ra.deg)
        assert np.isclose(s_coord.dec.deg, center_coord.dec.deg)


@pytest.mark.parametrize('output_format', ['asdf', 'fits'])
def test_asdf_cutout_memory_only(test_images, center_coord, cutout_size, output_format):
    cutouts = ASDFCutout(test_images, center_coord, cutout_size, output_format=output_format, memory_only=True).cutout()
    # Should output a list of memory objects
    assert isinstance(cutouts, list)
    expected_type = asdf.AsdfFile if output_format == 'asdf' else fits.HDUList
    assert isinstance(cutouts[0], expected_type)
    assert len(cutouts) == 3

    for i, cutout in enumerate(cutouts):
        # Get cutout data and WCS based on output format
        if output_format == 'asdf':
            cutout_data = cutout['roman']['data']
            cutout_wcs = cutout['roman']['meta']['wcs']
        else:
            cutout_data = cutout[0].data
            cutout_wcs = fits_wcs.WCS(cutout[0].header)

        # Check shape of data
        assert cutout_data.shape == (10, 10)

        # Check that data is equal between cutout and original image
        with asdf.open(test_images[i]) as input_af:
            assert np.all(cutout_data == input_af['roman']['data'].value[470:480, 471:481])

        # Check WCS and that center coordinate matches input
        s_coord = cutout_wcs.pixel_to_world(cutout_size / 2, cutout_size / 2)
        assert cutout_wcs.pixel_shape == (10, 10)
        assert np.isclose(s_coord.ra.deg, center_coord.ra.deg)
        assert np.isclose(s_coord.dec.deg, center_coord.dec.deg)


def test_asdf_cutout_cutout2D(test_images, center_coord, cutout_size):
    cutouts = ASDFCutout(test_images, center_coord, cutout_size, memory_only=True, return_cutout2D=True).cutout()
    # Should output a list of Cutout2D objects
    assert isinstance(cutouts, list)
    assert isinstance(cutouts[0], Cutout2D)
    assert len(cutouts) == 3

    for i, cutout in enumerate(cutouts):
        # Check shape of data
        assert cutout.data.shape == (10, 10)

        # Check that data is equal between cutout and original image
        with asdf.open(test_images[i]) as input_af:
            assert np.all(cutout.data == input_af['roman']['data'].value[470:480, 471:481])

        # Check WCS and that center coordinate matches input
        s_coord = cutout.wcs.pixel_to_world(cutout_size / 2, cutout_size / 2)
        assert cutout.wcs.pixel_shape == (10, 10)
        assert np.isclose(s_coord.ra.deg, center_coord.ra.deg)
        assert np.isclose(s_coord.dec.deg, center_coord.dec.deg)

    # Can also access `cutouts` attribute directly
    asdf_cut = ASDFCutout(test_images, center_coord, cutout_size, memory_only=True)
    cutouts = asdf_cut.cutout()
    assert isinstance(cutouts[0], asdf.AsdfFile)
    cutouts2D = asdf_cut.cutouts
    assert isinstance(cutouts2D[0], Cutout2D)
    assert len(cutouts2D) == 3


def test_asdf_cutout_partial(test_images, center_coord, cutout_size):
    # Off the top
    center_coord = SkyCoord('29.99901792 44.9861', unit='deg')
    cutout = ASDFCutout(test_images[0], center_coord, cutout_size, memory_only=True).cutout()
    assert cutout[0]['roman']['data'].shape == (10, 10)
    assert np.isnan(cutout[0]['roman']['data'][:cutout_size//2, :]).all()

    # Off the bottom
    center_coord = SkyCoord('29.99901792 45.01387', unit='deg')
    cutout = ASDFCutout(test_images[0], center_coord, cutout_size, memory_only=True).cutout()
    assert np.isnan(cutout[0]['roman']['data'][cutout_size//2:, :]).all()

    # Off the left, with integer fill value
    center_coord = SkyCoord('29.98035835 44.99930555', unit='deg')
    cutout = ASDFCutout(test_images[0], center_coord, cutout_size, fill_value=1, memory_only=True).cutout()
    assert np.all(cutout[0]['roman']['data'][:, :cutout_size//2] == 1)

    # Off the right, with float fill value
    center_coord = SkyCoord('30.01961 44.99930555', unit='deg')
    cutout = ASDFCutout(test_images[0], center_coord, cutout_size, fill_value=1.5, memory_only=True).cutout()
    assert np.all(cutout[0]['roman']['data'][:, cutout_size//2:] == 1.5)

    # Error if unexpected fill value
    with pytest.raises(InvalidInputError, match='Fill value must be an integer or a float.'):
        ASDFCutout(test_images[0], center_coord, cutout_size, fill_value='invalid', memory_only=True).cutout()


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
    cutout = ASDFCutout(filename, center_coord, cutout_size, memory_only=True).cutout()[0]

    # Check cutout contains all data
    assert len(np.where(cutout['roman']['data'] == 1)[0]) == 25


def test_asdf_cutout_not_in_footprint(test_images, center_coord, cutout_size):
    # Throw error if cutout location is not in image footprint
    with pytest.warns(DataWarning, match='Cutout footprint does not overlap'):
        with pytest.raises(InvalidQueryError, match='Cutout contains no data!'):
            ASDFCutout(test_images[0], SkyCoord('0 0', unit='deg'), cutout_size).cutout()

    # Alter one of the test images to only contain zeros in cutout footprint
    with asdf.open(test_images[0], mode='rw') as af:
        af['roman']['data'][470:480, 471:481] = 0
        af.update()

    # Should warn about first image containing no data, but not fail
    with pytest.warns(DataWarning, match='contains no data, skipping...'):
        cutouts = ASDFCutout(test_images, center_coord, cutout_size, memory_only=True).cutout()
    assert len(cutouts) == 2


def test_asdf_cutout_no_gwcs(test_images, center_coord, cutout_size):
    # Remove WCS from test image
    with asdf.open(test_images[0], mode='rw') as af:
        del af['roman']['meta']['wcs']
        af.update()

    # Should warn about missing WCS for first image, but not fail
    with pytest.warns(DataWarning, match='does not contain a GWCS object'):
        cutouts = ASDFCutout(test_images, center_coord, cutout_size, memory_only=True).cutout()
    assert len(cutouts) == 2


def test_asdf_cutout_invalid_params(test_images, center_coord, cutout_size, tmpdir):
    # Warning when image options are given
    with pytest.warns(InputWarning, match='are not supported for FITS or ASDF output and will be ignored.'):
        ASDFCutout(test_images, center_coord, cutout_size, stretch='asinh', output_dir=tmpdir).cutout()

    # Invalid units for cutout size
    cutout_size = 1 * u.m  # meters are not valid
    with pytest.raises(InvalidInputError, match='Cutout size unit meter is not supported.'):
        ASDFCutout(test_images, center_coord, cutout_size, output_dir=tmpdir).cutout()


def test_asdf_cutout_img_output(test_images, center_coord, cutout_size, tmpdir):
    # Basic JPG image
    jpg_files = ASDFCutout(test_images, center_coord, cutout_size, output_dir=tmpdir, output_format='jpg', 
                           return_paths=True).cutout()
    assert len(jpg_files) == len(test_images)
    with open(jpg_files[0], 'rb') as IMGFLE:
        assert IMGFLE.read(3) == b'\xFF\xD8\xFF'  # JPG

    # PNG (single input file, not as list)
    png_files = ASDFCutout(test_images[0], center_coord, cutout_size, output_dir=tmpdir, output_format='png', 
                           return_paths=True).cutout()
    with open(png_files[0], 'rb') as IMGFLE:
        assert IMGFLE.read(8) == b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'  # PNG
    assert len(png_files) == 1

    # Save to memory only
    img_cutouts = ASDFCutout(test_images[0], center_coord, cutout_size, output_dir=tmpdir, output_format='png', 
                             memory_only=True).cutout()
    assert len(img_cutouts) == 1
    assert isinstance(img_cutouts[0], Image.Image)
    assert np.array(img_cutouts[0]).shape == (10, 10)

    # Color image
    color_jpg = ASDFCutout(test_images, center_coord, cutout_size, output_format='jpg', colorize=True, 
                           output_dir=tmpdir, return_paths=True).cutout()
    img = Image.open(color_jpg)
    assert img.mode == 'RGB'

        
@patch('requests.head')
@patch('s3fs.S3FileSystem')
def test_asdf_get_cloud_http(mock_s3fs, mock_requests, center_coord, cutout_size):
    """ Test we can get HTTP URI of cloud resource """
    # Mock HTTP response
    mock_resp = MagicMock()
    mock_resp.status_code = 200  # public bucket
    mock_requests.return_value = mock_resp

    # Mock s3 file system operations
    HTTP_URI = 'http_test'
    mock_fs = mock_s3fs.return_value
    mock_file = MagicMock()
    mock_file.url.return_value = HTTP_URI
    mock_fs.open.return_value.__enter__.return_value = mock_file

    # Test function with string input
    s3_uri = 's3://test_bucket/test_file.asdf'
    cutout = ASDFCutout(s3_uri, center_coord, cutout_size)
    http_uri = cutout._get_cloud_http(s3_uri)
    assert http_uri == HTTP_URI
    mock_s3fs.assert_called_with(anon=True, key=None, secret=None, token=None)
    mock_fs.open.assert_called_once_with(s3_uri, 'rb')
    mock_file.url.assert_called_once()

    # Test function with S3Path input
    s3_uri_path = S3Path('/test_bucket/test_file_2.asdf')
    http_uri_path = cutout._get_cloud_http(s3_uri_path)
    assert http_uri_path == HTTP_URI
    mock_fs.open.assert_called_with(s3_uri_path, 'rb')

    # Test function with private bucket
    mock_resp.status_code = 403
    cutout = ASDFCutout(s3_uri, center_coord, cutout_size, key='access')
    http_uri = cutout._get_cloud_http(s3_uri)
    mock_s3fs.assert_called_with(anon=False, key='access', secret=None, token=None)


def test_get_center_pixel(fakedata):
    """ Test get_center_pixel function """
    # get the fake data
    __, gwcs = fakedata

    pixel_coordinates, wcs = ASDFCutout.get_center_pixel(gwcs, 30, 45)
    assert np.allclose(pixel_coordinates, (np.array(500), np.array(500)))
    assert np.allclose(wcs.celestial.wcs.crval, np.array([30, 45]))


def test_asdf_cut(test_images, center_coord, cutout_size, tmpdir):
    """ Test convenience function to create ASDF cutouts """
    # Write files to disk
    cutouts = asdf_cut(test_images, center_coord.ra.deg, center_coord.dec.deg, cutout_size, output_dir=tmpdir, 
                       return_paths=True, return_cutout2D=False)
    assert isinstance(cutouts, list)
    assert isinstance(cutouts[0], str)
    assert len(cutouts) == 3
    for i, path in enumerate(cutouts):
        assert isinstance(path, str)
        assert path.endswith('.asdf')
        assert Path(path).exists()
        assert str(tmpdir) in path
        assert Path(test_images[i]).stem in path
        assert center_coord.ra.to_string(unit='deg', decimal=True) in path
        assert center_coord.dec.to_string(unit='deg', decimal=True) in path
        assert '10-x-10' in path

    # Write cutouts to memory as Cutout2D objects
    cutouts = asdf_cut(test_images, center_coord.ra.deg, center_coord.dec.deg, cutout_size, write_file=False)
    assert isinstance(cutouts, list)
    assert isinstance(cutouts[0], Cutout2D)
    assert len(cutouts) == 3

    # Write cutouts to memory as AsdfFile objects
    cutouts = asdf_cut(test_images, center_coord.ra.deg, center_coord.dec.deg, cutout_size, write_file=False, 
                       return_cutout2D=False)
    assert isinstance(cutouts[0], asdf.AsdfFile)

    # Write cutouts to memory as HDUList objects
    cutouts = asdf_cut(test_images, center_coord.ra.deg, center_coord.dec.deg, cutout_size, output_format='fits', 
                       write_file=False, return_cutout2D=False)
    assert isinstance(cutouts[0], fits.HDUList)
