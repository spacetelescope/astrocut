
import pathlib
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
from astrocut.asdf_cutouts import get_center_pixel, get_cutout, asdf_cut


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
    celestial_rotation = models.RotateNative2Celestial(ra, dec, 180.)  #

    # net transforms pixels to sky
    det2sky = pixelshift | pixelscale | tangent_projection | celestial_rotation

    # define the wcs object
    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='world', unit=(u.deg, u.deg))
    return wcs.WCS([(detector_frame, det2sky), (sky_frame, None)])


@pytest.fixture()
def makefake():
    """ fixture factory to make a fake gwcs and dataset """

    def _make_fake(nx, ny, ra, dec, zero=False):
        # create the wcs
        wcsobj = make_wcs(nx/2, ny/2, ra=ra, dec=dec)
        wcsobj.bounding_box = ((0, nx), (0, ny))

        # create the data
        if zero:
            data = np.zeros([nx, ny])
        else:
            size = nx *  ny
            data = np.arange(size).reshape(nx, ny)
        data *= (u.electron / u.second)

        return data, wcsobj

    yield _make_fake


@pytest.fixture()
def fakedata(makefake):
    """ fixture to create fake data and wcs """
    # set up initial parameters
    nx = 100
    ny = 100
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


def test_get_center_pixel(fakedata):
    """ test we can get the correct center pixel """
    # get the fake data
    __, gwcs = fakedata

    pixel_coordinates, wcs = get_center_pixel(gwcs, 30., 45.)
    assert np.allclose(pixel_coordinates, (np.array(50.), np.array(50.)))
    assert np.allclose(wcs.celestial.wcs.crval, np.array([30., 45.]))


@pytest.fixture()
def output_file(tmp_path):
    """ fixture to create the output path """
    # create output fits path
    out = tmp_path / "roman"
    out.mkdir(exist_ok=True, parents=True)
    output_file = out / "test_output_cutout.fits"
    yield output_file


@pytest.mark.parametrize('quantity', [True, False], ids=['quantity', 'array'])
def test_get_cutout(output_file, fakedata, quantity):
    """ test we can create a cutout """

    # get the input wcs
    data, gwcs = fakedata
    skycoord = gwcs(25, 25, with_units=True)
    wcs = WCS(gwcs.to_fits_sip())

    # convert quanity data back to array
    if not quantity:
        data = data.value

    # create cutout
    cutout = get_cutout(data, skycoord, wcs, size=10, outfile=output_file)

    assert_same_coord(5, 10, cutout, wcs)

    # test output
    with fits.open(output_file) as hdulist:
        data = hdulist[0].data
        assert data.shape == (10, 10)
        assert data[5, 5] == 2525


def test_asdf_cutout(make_file, output_file):
    """ test we can make a cutout """
    # make cutout
    ra, dec = (29.99901792, 44.99930555)
    asdf_cut(make_file, ra, dec, cutout_size=10, output_file=output_file)

    # test output
    with fits.open(output_file) as hdulist:
        data = hdulist[0].data
        assert data.shape == (10, 10)
        assert data[5, 5] == 2526


def test_cutout_nofile(make_file, output_file):
    """ test we can make a cutout with no file output """
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
    cutout = get_cutout(data, cc, wcs, size=50, write_file=False)
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
        get_cutout(data, cc, wcs, size=50, write_file=False)


def assert_same_coord(x, y, cutout, wcs):
    """ assert we get the same sky coordinate from cutout and original wcs """
    cutout_coord = pixel_to_skycoord(x, y, cutout.wcs)
    ox, oy = cutout.to_original_position((x, y))
    orig_coord = pixel_to_skycoord(ox, oy, wcs)
    assert cutout_coord == orig_coord





