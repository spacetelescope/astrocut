
import pathlib
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import asdf
from astropy.modeling import models
from astropy import coordinates as coord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from gwcs import wcs
from gwcs import coordinate_frames as cf
from s3path import S3Path
from astrocut.asdf_cutouts import get_center_pixel, asdf_cut, _get_cutout, _slice_gwcs, _get_cloud_http


def make_wcs(xsize, ysize, ra=30., dec=45.):
    """ create a fake gwcs object """
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
    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='world', unit=(u.deg, u.deg))
    return wcs.WCS([(detector_frame, det2sky), (sky_frame, None)])


@pytest.fixture()
def makefake():
    """ fixture factory to make a fake gwcs and dataset """

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
    """ fixture to create fake data and wcs """
    # set up initial parameters
    nx = 1000
    ny = 1000
    ra = 30.
    dec = 45.

    yield makefake(nx, ny, ra, dec)


@pytest.fixture()
def make_file(tmp_path, fakedata):
    """ fixture to create a fake dataset """
    # get the fake data
    data, wcsobj = fakedata

    # create meta
    meta = {'wcs': wcsobj}

    # create and write the asdf file
    tree = {'roman': {'data': data, 'meta': meta}}
    af = asdf.AsdfFile(tree)

    path = tmp_path / "roman"
    path.mkdir(exist_ok=True)
    filename = path / "test_roman.asdf"
    af.write_to(filename)

    yield filename


@pytest.fixture()
def output(tmp_path):
    """ fixture to create the output path """
    def _output_file(ext='fits'):
        # create output fits path
        out = tmp_path / "roman"
        out.mkdir(exist_ok=True, parents=True)
        output_file = out / f"test_output_cutout.{ext}" if ext else "test_output_cutout"
        return output_file
    yield _output_file


def test_get_center_pixel(fakedata):
    """ test we can get the correct center pixel """
    # get the fake data
    __, gwcs = fakedata

    pixel_coordinates, wcs = get_center_pixel(gwcs, 30., 45.)
    assert np.allclose(pixel_coordinates, (np.array(500.), np.array(500.)))
    assert np.allclose(wcs.celestial.wcs.crval, np.array([30., 45.]))


@pytest.mark.parametrize('quantity', [True, False], ids=['quantity', 'array'])
def test_get_cutout(output, fakedata, quantity):
    """ test we can create a cutout """
    output_file = output('fits')

    # get the input wcs
    data, gwcs = fakedata
    skycoord = gwcs(25, 25, with_units=True)
    wcs = WCS(gwcs.to_fits_sip())

    # convert quanity data back to array
    if not quantity:
        data = data.value

    # create cutout
    cutout = _get_cutout(data, skycoord, wcs, size=10, outfile=output_file)

    assert_same_coord(5, 10, cutout, wcs)

    # test output
    with fits.open(output_file) as hdulist:
        data = hdulist[0].data
        assert data.shape == (10, 10)
        assert data[5, 5] == 25025


def test_asdf_cutout(make_file, output):
    """ test we can make a cutout """
    output_file = output('fits')
    # make cutout
    ra, dec = (29.99901792, 44.99930555)
    asdf_cut(make_file, ra, dec, cutout_size=10, output_file=output_file)

    # test output
    with fits.open(output_file) as hdulist:
        data = hdulist[0].data
        assert data.shape == (10, 10)
        assert data[5, 5] == 475476


@pytest.mark.parametrize('suffix', ['fits', 'asdf', None])
def test_write_file(make_file, suffix, output):
    """ test we can write an different file types """
    output_file = output(suffix)

    # make cutout
    ra, dec = (29.99901792, 44.99930555)
    asdf_cut(make_file, ra, dec, cutout_size=10, output_file=output_file)

    # if no suffix provided, check that the default output is fits
    if not suffix:
        output_file += ".fits"

    assert pathlib.Path(output_file).exists()


def test_fail_write_asdf(fakedata, output):
    """ test we fail to write an asdf if no gwcs given """
    with pytest.raises(ValueError, match='The original gwcs object is needed when writing to asdf file.'):
        output_file = output('asdf')
        data, gwcs = fakedata
        skycoord = gwcs(25, 25, with_units=True)
        wcs = WCS(gwcs.to_fits_sip())
        _get_cutout(data, skycoord, wcs, size=10, outfile=output_file)


def test_cutout_nofile(make_file, output):
    """ test we can make a cutout with no file output """
    output_file = output()
    # make cutout
    ra, dec = (29.99901792, 44.99930555)
    cutout = asdf_cut(make_file, ra, dec, cutout_size=10, output_file=output_file, write_file=False)

    assert not pathlib.Path(output_file).exists()
    assert cutout.shape == (10, 10)


def test_cutout_poles(makefake):
    """ test we can make cutouts around poles """
    # make fake zero data around the pole
    ra, dec = 315.0, 89.995
    data, gwcs = makefake(1000, 1000, ra, dec, zero=True)

    # add some values (5x5 array)
    data.value[245:250, 245:250] = 1

    # check central pixel is correct
    ss = gwcs(500, 500)
    assert ss == (ra, dec)

    # set input cutout coord
    cc = coord.SkyCoord(284.702, 89.986, unit=u.degree)
    wcs = WCS(gwcs.to_fits_sip())

    # get cutout
    cutout = _get_cutout(data, cc, wcs, size=50, write_file=False)
    assert_same_coord(5, 10, cutout, wcs)

    # check cutout contains all data
    assert len(np.where(cutout.data.value == 1)[0]) == 25


def test_fail_cutout_outside(fakedata):
    """ test we fail when cutout completely outside range """
    data, gwcs = fakedata
    wcs = WCS(gwcs.to_fits_sip())
    cc = coord.SkyCoord(200.0, 50.0, unit=u.degree)

    with pytest.raises(RuntimeError, match='Could not create 2d cutout.  The requested '
                       'cutout does not overlap with the original image'):
        _get_cutout(data, cc, wcs, size=50, write_file=False)


def assert_same_coord(x, y, cutout, wcs):
    """ assert we get the same sky coordinate from cutout and original wcs """
    cutout_coord = pixel_to_skycoord(x, y, cutout.wcs)
    ox, oy = cutout.to_original_position((x, y))
    orig_coord = pixel_to_skycoord(ox, oy, wcs)
    assert cutout_coord == orig_coord


@pytest.mark.parametrize('asint, fill', [(False, None), (True, -9999)], ids=['fillfloat', 'fillint'])
def test_partial_cutout(makefake, asint, fill):
    """ test we get a partial cutout with nans or fill value """
    ra, dec = 30.0, 45.0
    data, gwcs = makefake(100, 100, ra, dec, asint=asint)

    wcs = WCS(gwcs.to_fits_sip())
    cc = coord.SkyCoord(29.999, 44.998, unit=u.degree)
    cutout = _get_cutout(data, cc, wcs, size=50, write_file=False, fill_value=fill)
    assert cutout.shape == (50, 50)
    if asint:
        assert -9999 in cutout.data
    else:
        assert np.isnan(cutout.data).any()


def test_bad_fill(makefake):
    """ test error is raised on bad fill value """
    ra, dec = 30.0, 45.0
    data, gwcs = makefake(100, 100, ra, dec, asint=True)
    wcs = WCS(gwcs.to_fits_sip())
    cc = coord.SkyCoord(29.999, 44.998, unit=u.degree)
    with pytest.raises(ValueError, match='fill_value is inconsistent with the data type of the input array'):
        _get_cutout(data, cc, wcs, size=50, write_file=False)


def test_cutout_raedge(makefake):
    """ test we can make cutouts around ra=0 """
    # make fake zero data around the ra edge
    ra, dec = 0.0, 10.0
    data, gg = makefake(2000, 2000, ra, dec, zero=True)

    # check central pixel is correct
    ss = gg(1001, 1001)
    assert pytest.approx(ss, abs=1e-3) == (ra, dec)

    # set input cutout coord
    cc = coord.SkyCoord(0.001, 9.999, unit=u.degree)
    wcs = WCS(gg.to_fits_sip())

    # get cutout
    cutout = _get_cutout(data, cc, wcs, size=100, write_file=False)
    assert_same_coord(5, 10, cutout, wcs)

    # assert the RA cutout bounds are > 359 and < 0
    bounds = gg(*cutout.bbox_original, with_units=True)
    assert bounds[0].ra.value > 359
    assert bounds[1].ra.value < 0.1


def test_slice_gwcs(fakedata):
    """ test we can slice a gwcs object """
    data, gwcsobj = fakedata
    skycoord = gwcsobj(250, 250)
    wcs = WCS(gwcsobj.to_fits_sip())

    cutout = _get_cutout(data, skycoord, wcs, size=50, write_file=False)

    sliced = _slice_gwcs(gwcsobj, cutout.slices_original)

    # check coords between slice and original gwcs
    assert cutout.center_cutout == (24.5, 24.5)
    assert sliced.array_shape == (50, 50)
    assert sliced(*cutout.input_position_cutout) == gwcsobj(*cutout.input_position_original)
    assert gwcsobj(*cutout.center_original) == sliced(*cutout.center_cutout)

    # assert same sky footprint between slice and original
    # gwcs footprint/bounding_box expects ((x0, x1), (y0, y1)) but cutout.bbox is in ((y0, y1), (x0, x1))
    assert (gwcsobj.footprint(bounding_box=tuple(reversed(cutout.bbox_original))) == sliced.footprint()).all()


@patch('requests.head')
@patch('s3fs.S3FileSystem')
def test_get_cloud_http(mock_s3fs, mock_requests):
    """ test we can get HTTP URI of cloud resource """
    # mock HTTP response
    mock_resp = MagicMock()
    mock_resp.status_code = 200  # public bucket
    mock_requests.return_value = mock_resp

    # mock s3 file system operations
    HTTP_URI = "http_test"
    mock_fs = mock_s3fs.return_value
    mock_file = MagicMock()
    mock_file.url.return_value = HTTP_URI
    mock_fs.open.return_value.__enter__.return_value = mock_file

    # test function with string input
    s3_uri = "s3://test_bucket/test_file.asdf"
    http_uri = _get_cloud_http(s3_uri)
    assert http_uri == HTTP_URI
    mock_s3fs.assert_called_with(anon=True)
    mock_fs.open.assert_called_once_with(s3_uri, 'rb')
    mock_file.url.assert_called_once()

    # test function with S3Path input
    s3_uri_path = S3Path("/test_bucket/test_file_2.asdf")
    http_uri_path = _get_cloud_http(s3_uri_path)
    assert http_uri_path == HTTP_URI
    mock_fs.open.assert_called_with(s3_uri_path, 'rb')

    # test function with private bucket
    mock_resp.status_code = 403
    http_uri = _get_cloud_http(s3_uri)
    mock_s3fs.assert_called_with(anon=False)
