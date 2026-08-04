import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import asdf
import numpy as np
import pytest
from astropy import coordinates as coord
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import models
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from gwcs import coordinate_frames, wcs
from PIL import Image

from astrocut.asdf_cutout import ASDFCutout, asdf_cut, get_center_pixel
from astrocut.exceptions import DataWarning, InputWarning, InvalidInputError, InvalidQueryError, ModuleWarning

try:
    from stdatamodels import asdf_in_fits

    HAS_ASDF_IN_FITS = True
except ImportError:
    HAS_ASDF_IN_FITS = False


def make_wcs(xsize, ysize, ra=30.0, dec=45.0):
    """Create a fake gwcs object"""
    # todo - refine this to better reflect roman wcs

    # create transformations
    # - shift coords so array center is at 0, 0 ; reference pixel
    # - scale pixels to correct angular scale
    # - project coords onto sky with TAN projection
    # - transform center pixel to the input celestial coordinate
    pixelshift = models.Shift(-xsize) & models.Shift(-ysize)
    pixelscale = models.Scale(0.1 / 3600.0) & models.Scale(0.1 / 3600.0)  # 0.1 arcsec/pixel
    tangent_projection = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(ra, dec, 180.0)

    # net transforms pixels to sky
    det2sky = pixelshift | pixelscale | tangent_projection | celestial_rotation

    # define the wcs object
    detector_frame = coordinate_frames.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
    sky_frame = coordinate_frames.CelestialFrame(reference_frame=coord.ICRS(), name="world", unit=(u.deg, u.deg))
    return wcs.WCS([(detector_frame, det2sky), (sky_frame, None)])


def make_fake(nx, ny, ra, dec, zero=False, asint=False):
    """Fixture factory to make a fake gwcs and dataset"""
    wcsobj = make_wcs(nx / 2, ny / 2, ra=ra, dec=dec)
    wcsobj.bounding_box = ((0, nx), (0, ny))

    # create the data
    if zero:
        data = np.zeros([nx, ny])
    else:
        size = nx * ny
        data = np.arange(size).reshape(nx, ny)

    # make a quantity
    data *= u.electron / u.second

    # make integer array
    if asint:
        data = data.astype(int)

    return data, wcsobj


@pytest.fixture()
def fake_data():
    """Fixture to create fake data and wcs"""
    # set up initial parameters
    nx = 1000
    ny = 1000
    ra = 30.0
    dec = 45.0

    yield make_fake(nx, ny, ra, dec)


@pytest.fixture()
def images(tmp_path):
    """Fixture to create a fake dataset of 3 images"""
    path = tmp_path / "roman"
    path.mkdir(exist_ok=True)

    files = []
    for i in range(3):
        # get the fake data
        data, wcsobj = make_fake(1000, 1000, 30.0 + i * 0.001, 45.0 + i * 0.001)

        # create meta
        meta = {
            "wcs": wcsobj,
            "product_type": "l2",
            "origin": "STSCI/SOC",
            "file_date": Time("2023-10-01T00:00:00", format="isot"),
        }

        # create and write the asdf file
        tree = {
            "roman": {
                "meta": meta,
                "data": data,
                "dq": data.astype(int),  # DQ is typically an integer array
                "err": data,
                "context": np.expand_dims(data, axis=0),
                "invalid_dims": np.ndarray(shape=(10)),
            }
        }
        af = asdf.AsdfFile(tree)

        filename = path / f"test_roman_{i}.asdf"
        af.write_to(filename)
        files.append(filename)

    return files


@pytest.fixture
def center_coord():
    """Fixture to return a center coordinate"""
    return SkyCoord("29.99901792 44.99930555", unit="deg")


@pytest.fixture
def multi_coord():
    """Fixture to return a list of coordinates"""
    return [
        SkyCoord("29.99901792 44.99930555", unit="deg"),
        SkyCoord("30.00098208 44.99930555", unit="deg"),
        "29.98201792 45.00069445",
    ]


@pytest.fixture
def cutout_size():
    """Fixture to return a cutout size"""
    return 10


def test_asdf_cutout(images, center_coord, cutout_size):
    cutout = ASDFCutout(images, center_coord, cutout_size)
    cutouts = cutout.cutouts
    # Should output a list of strings for multiple input files
    assert isinstance(cutouts, Table)
    assert isinstance(cutouts["cutout"][0], Cutout2D)
    assert len(cutouts) == 3
    assert isinstance(cutout.asdf_cutouts, Table)
    assert isinstance(cutout.asdf_cutouts["cutout"][0], asdf.AsdfFile)
    assert isinstance(cutout.fits_cutouts, Table)
    assert isinstance(cutout.fits_cutouts["cutout"][0], fits.HDUList)

    # Open output files
    for i, cutout_row in enumerate(cutouts):
        # Check shape of data
        cutout_data = cutout_row["cutout"].data
        cutout_wcs = cutout_row["cutout"].wcs
        bbox = cutout_row["cutout"].bbox_original
        assert cutout_data.shape == (10, 10)

        # Check that data is equal between cutout and original image
        with asdf.open(images[i]) as input_af:
            assert np.all(
                cutout_data == input_af["roman"]["data"].value[bbox[0][0] : bbox[0][1] + 1, bbox[1][0] : bbox[1][1] + 1]
            )

        # Check WCS and that center coordinate matches input
        s_coord = cutout_wcs.pixel_to_world(cutout_size / 2, cutout_size / 2)
        assert cutout_wcs.pixel_shape == (10, 10)
        assert np.isclose(s_coord.ra.deg, center_coord.ra.deg)
        assert np.isclose(s_coord.dec.deg, center_coord.dec.deg)


@pytest.mark.parametrize("lite", [False, True])
def test_asdf_cutout_get_asdf_cutouts(images, multi_coord, cutout_size, lite):
    with pytest.warns(DataWarning, match="does not overlap the image"):
        cutout = ASDFCutout(images, multi_coord, cutout_size, lite=lite)

    # With input files and coordinates specified
    # Choose 2 files and 2 coordinates
    asdf_cutouts = cutout.get_asdf_cutouts(input_files=images[1:], coordinates=multi_coord[1:])
    assert isinstance(asdf_cutouts, Table)
    assert len(asdf_cutouts) == 3  # one coordinate will not have a cutout for one of the images
    for af in asdf_cutouts["cutout"]:
        assert isinstance(af, asdf.AsdfFile)
        assert "roman" in af
        assert "data" in af["roman"]
        assert af["roman"]["data"].shape == (cutout_size, cutout_size)

    # Error if a file is not found
    with pytest.raises(InvalidInputError, match="is not in the cutout results."):
        cutout.get_asdf_cutouts(input_files=["nonexistent_file.asdf"], coordinates=multi_coord[1:])

    # Error if a coordinate is not found
    with pytest.raises(InvalidInputError, match="is not in the cutout results."):
        cutout.get_asdf_cutouts(input_files=images[1:], coordinates=[SkyCoord("0 0", unit="deg")])

    # Error if invalid coordinate string is provided
    with pytest.raises(InvalidInputError, match="Invalid coordinate string"):
        cutout.get_asdf_cutouts(input_files=images[1:], coordinates=["invalid_coord"])

    # Error if invalid coordinate type is provided
    with pytest.raises(InvalidInputError, match="is not a valid SkyCoord or string"):
        cutout.get_asdf_cutouts(input_files=images[1:], coordinates=[12345])


def test_asdf_cutout_iter_asdf_cutouts(images, multi_coord, cutout_size):
    with pytest.warns(DataWarning, match="does not overlap the image"):
        cutout = ASDFCutout(images, multi_coord, cutout_size)

    asdf_cutouts = list(cutout.iter_asdf_cutouts(input_files=images[1:], coordinates=multi_coord[1:]))
    assert len(asdf_cutouts) == 3
    for file, coordinate, af in asdf_cutouts:
        assert file in {image.as_posix() for image in images[1:]}
        assert isinstance(coordinate, SkyCoord)
        assert isinstance(af, asdf.AsdfFile)
        assert "roman" in af
        assert af["roman"]["data"].shape == (cutout_size, cutout_size)


@pytest.mark.parametrize("lite", [False, True])
def test_asdf_cutout_get_fits_cutouts(images, multi_coord, cutout_size, lite):
    with pytest.warns(DataWarning, match="does not overlap the image"):
        cutout = ASDFCutout(images, multi_coord, cutout_size, lite=lite)

    # With input files and coordinates specified
    # Choose 2 files and 2 coordinates
    fits_cutouts = cutout.get_fits_cutouts(input_files=images[1:], coordinates=multi_coord[1:])
    assert isinstance(fits_cutouts, Table)
    assert len(fits_cutouts) == 3  # one coordinate will not have a cutout for one of the images
    for hdul in fits_cutouts["cutout"]:
        assert isinstance(hdul, fits.HDUList)
        assert len(hdul) == 2 if not HAS_ASDF_IN_FITS else 3  # primary + cutout HDU + optional ASDF extension
        assert hdul[0].name == "PRIMARY"
        assert hdul[1].name == "CUTOUT"
        assert hdul[1].data.shape == (cutout_size, cutout_size)


def test_asdf_cutout_iter_fits_cutouts(images, multi_coord, cutout_size):
    with pytest.warns(DataWarning, match="does not overlap the image"):
        cutout = ASDFCutout(images, multi_coord, cutout_size)

    fits_cutouts = list(cutout.iter_fits_cutouts(input_files=images[1:], coordinates=multi_coord[1:]))
    assert len(fits_cutouts) == 3
    for file, coordinate, hdul in fits_cutouts:
        assert file in {image.as_posix() for image in images[1:]}
        assert isinstance(coordinate, SkyCoord)
        assert isinstance(hdul, fits.HDUList)
        assert len(hdul) == 2 if not HAS_ASDF_IN_FITS else 3
        assert hdul[0].name == "PRIMARY"
        assert hdul[1].name == "CUTOUT"
        assert hdul[1].data.shape == (cutout_size, cutout_size)


@pytest.mark.parametrize(
    ("method_name", "kwargs", "blocked_getter"),
    [
        ("write_as_asdf", {"output_dir": "."}, "get_asdf_cutouts"),
        ("write_as_fits", {"output_dir": "."}, "get_fits_cutouts"),
        ("write_as_zip", {"output_dir": ".", "output_format": ".asdf"}, "get_asdf_cutouts"),
        ("write_as_zip", {"output_dir": ".", "output_format": ".fits"}, "get_fits_cutouts"),
    ],
)
def test_asdf_cutout_write_streams_from_iterators(
    images,
    center_coord,
    cutout_size,
    tmpdir,
    method_name,
    kwargs,
    blocked_getter,
):
    cutout = ASDFCutout(images, center_coord, cutout_size)
    call_kwargs = dict(kwargs)
    call_kwargs["output_dir"] = tmpdir

    with patch.object(
        ASDFCutout,
        blocked_getter,
        side_effect=AssertionError(f"{blocked_getter} should not be used"),
    ):
        output = getattr(cutout, method_name)(**call_kwargs)

    if isinstance(output, list):
        assert len(output) == len(images)
        for file_path in output:
            assert Path(file_path).exists()
    else:
        assert Path(output).exists()


def test_asdf_cutout_write_to_file(images, center_coord, cutout_size, tmpdir):
    def check_asdf_metadata(af, original_file, cutout_data, meta_only=False):
        """Check that ASDF file contains correct metadata"""
        assert "roman" in af
        assert "meta" in af["roman"]
        meta = af["roman"]["meta"]
        assert meta["wcs"].pixel_shape == (10, 10)
        assert meta["product_type"] == "l2"
        assert meta["file_date"] == Time("2023-10-01T00:00:00", format="isot")
        assert meta["origin"] == "STSCI/SOC"
        assert meta["orig_file"] == original_file.as_posix()
        assert meta["coordinate"] == center_coord.to_string(precision=8)

        if not meta_only:
            # Check cutout data and metadata
            for key in ["data", "dq", "err", "context"]:
                assert key in af["roman"]
                assert np.all(af["roman"][key] == cutout_data)

    # Write cutouts to ASDF files on disk
    cutout = ASDFCutout(images, center_coord, cutout_size, lite=False)
    asdf_files = cutout.write_as_asdf(output_dir=tmpdir)
    assert len(asdf_files) == 3
    for i, asdf_file in enumerate(asdf_files):
        with asdf.open(asdf_file) as af:
            check_asdf_metadata(af, images[i], cutout.cutouts["cutout"][i].data)
            # Check file size is smaller than original
            assert Path(asdf_file).stat().st_size < Path(images[i]).stat().st_size

    # Write cutouts to FITS files on disk
    cutout = ASDFCutout(images, center_coord, cutout_size, lite=False)
    fits_files = cutout.write_as_fits(output_dir=tmpdir)
    assert len(fits_files) == 3
    for i, fits_file in enumerate(fits_files):
        with fits.open(fits_file) as hdul:
            assert hdul[0].name == "PRIMARY"
            assert hdul[1].name == "CUTOUT"
            assert np.all(hdul[1].data == cutout.cutouts["cutout"][i].data)
            assert hdul[1].header["NAXIS1"] == 10
            assert hdul[1].header["NAXIS2"] == 10
            assert hdul[1].header["ORIG_FLE"] == images[i].as_posix()
            assert Path(fits_file).stat().st_size < Path(images[i]).stat().st_size

        if HAS_ASDF_IN_FITS:
            with asdf_in_fits.open(fits_file) as af:
                check_asdf_metadata(af, images[i], cutout.cutouts["cutout"][i].data, meta_only=True)


@pytest.mark.parametrize("output_format", [".asdf", ".fits"])
def test_asdf_cutout_write_to_zip(tmpdir, images, multi_coord, cutout_size, output_format):
    # Zip ASDF representations
    cutout = ASDFCutout(images, multi_coord[:2], cutout_size)
    zip_path = cutout.write_as_zip(output_dir=tmpdir, output_format=output_format)
    assert Path(zip_path).stem == f"astrocut_{output_format[1:]}_cutouts"
    assert Path(zip_path).exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert len(names) == 6  # 3 images * 2 coordinates = 6 cutouts
        for name in names:
            assert name.endswith(f"_astrocut{output_format}")

        # Open one file and check contents
        data = zf.read(names[0])
        if output_format == ".asdf":
            with asdf.open(io.BytesIO(data)) as af:
                assert "roman" in af
                assert "data" in af["roman"]
                assert af["roman"]["data"].shape == (cutout_size, cutout_size)
        else:
            with fits.open(io.BytesIO(data)) as hdul:
                assert isinstance(hdul, fits.HDUList)
                assert len(hdul) == 3 if HAS_ASDF_IN_FITS else 2
                assert hdul[1].data.shape == (cutout_size, cutout_size)


def test_asdf_cutout_write_to_zip_invalid_format(tmpdir, images, center_coord, cutout_size):
    # Invalid output format for zip
    cutout = ASDFCutout(images, center_coord, cutout_size)
    with pytest.raises(InvalidInputError, match="File format must be either '.asdf' or '.fits'"):
        cutout.write_as_zip(output_dir=tmpdir, output_format=".invalid")


def test_asdf_cutout_lite(images, center_coord, cutout_size):
    def check_lite_metadata(af, meta_only=False):
        """Check that ASDF file contains only lite metadata"""
        assert "roman" in af
        assert "meta" in af["roman"]
        assert "wcs" in af["roman"]["meta"]
        assert "orig_file" in af["roman"]["meta"]
        assert "coordinate" in af["roman"]["meta"]
        assert len(af["roman"]) == (1 if meta_only else 2)
        assert len(af["roman"]["meta"]) == 3  # only wcs, original filename, and coordinate
        if not meta_only:
            assert "data" in af["roman"]

    # Write cutouts to ASDF objects in lite mode
    cutout = ASDFCutout(images, center_coord, cutout_size, lite=True)
    for af in cutout.asdf_cutouts["cutout"]:
        check_lite_metadata(af)

    # Write cutouts to HDUList objects in lite mode
    cutout = ASDFCutout(images, center_coord, cutout_size, lite=True)
    for hdul in cutout.fits_cutouts["cutout"]:
        assert len(hdul) == 3 if HAS_ASDF_IN_FITS else 2  # primary HDU + cutout HDU + embedded ASDF extension
        assert hdul[0].name == "PRIMARY"
        assert hdul[1].name == "CUTOUT"

        # Check ASDF extension contents (stdatamodels optional)
        if HAS_ASDF_IN_FITS:
            assert hdul[2].name == "ASDF"
            with asdf_in_fits.open(hdul) as af:
                check_lite_metadata(af, meta_only=True)


def test_asdf_cutout_partial(images, center_coord, cutout_size):
    # Off the top
    center_coord = SkyCoord("29.99901792 44.9861", unit="deg")
    asdf_cutout = ASDFCutout(images[0], center_coord, cutout_size, lite=False)
    cutout = asdf_cutout.cutouts["cutout"][0]
    cutout_asdf = list(asdf_cutout.asdf_cutouts["cutout"])[0]
    assert cutout.data.shape == (10, 10)
    assert np.isnan(cutout.data[: cutout_size // 2, :]).all()
    assert np.isnan(cutout_asdf["roman"]["err"][: cutout_size // 2, :]).all()
    assert np.isnan(cutout_asdf["roman"]["context"][:, : cutout_size // 2, :]).all()
    # Default to 0 for integer arrays
    assert np.all(cutout_asdf["roman"]["dq"][: cutout_size // 2, :] == 0)

    # Off the bottom
    center_coord = SkyCoord("29.99901792 45.01387", unit="deg")
    cutout = ASDFCutout(images[0], center_coord, cutout_size).cutouts["cutout"][0]
    assert np.isnan(cutout.data[cutout_size // 2 :, :]).all()

    # Off the left, with integer fill value
    center_coord = SkyCoord("29.98035835 44.99930555", unit="deg")
    cutout = ASDFCutout(images[0], center_coord, cutout_size, fill_value=1).cutouts["cutout"][0]
    assert np.all(cutout.data[:, : cutout_size // 2] == 1)

    # Off the right, with float fill value
    center_coord = SkyCoord("30.01961 44.99930555", unit="deg")
    asdf_cutout = ASDFCutout(images[0], center_coord, cutout_size, fill_value=1.5, lite=False)
    cutout = asdf_cutout.cutouts["cutout"][0]
    cutout_asdf = list(asdf_cutout.asdf_cutouts["cutout"])[0]
    assert np.all(cutout.data[:, cutout_size // 2 :] == 1.5)
    # Convert to integer fill value for DQ array
    assert np.all(cutout_asdf["roman"]["dq"][:, cutout_size // 2 :] == 1)

    # Error if unexpected fill value
    with pytest.raises(InvalidInputError, match="Fill value must be an integer or a float."):
        ASDFCutout(images[0], center_coord, cutout_size, fill_value="invalid")


def test_asdf_cutout_poles(cutout_size, tmp_path):
    """Test we can make cutouts around poles"""
    # Make fake zero data around the pole
    ra, dec = 315.0, 89.995
    data, gwcs = make_fake(1000, 1000, ra, dec, zero=True)

    # Add some values (5x5 array)
    data.value[245:250, 245:250] = 1

    # Check central pixel is correct
    ss = gwcs(500, 500)
    assert ss == (ra, dec)

    # Set input cutout coord
    center_coord = SkyCoord(284.702, 89.986, unit="deg")

    # create and write the asdf file
    meta = {"wcs": gwcs}
    tree = {"roman": {"data": data, "meta": meta}}
    af = asdf.AsdfFile(tree)
    path = tmp_path / "roman"
    path.mkdir(exist_ok=True)
    filename = path / "test_roman_poles.asdf"
    af.write_to(filename)

    # Get cutout
    cutout = ASDFCutout(filename, center_coord, cutout_size).cutouts["cutout"][0]

    # Check cutout contains all data
    assert len(np.where(cutout.data == 1)[0]) == 25


def test_asdf_cutout_not_in_footprint(images, center_coord, cutout_size):
    # Throw error if cutout location is not in image footprint
    with pytest.warns(DataWarning, match="does not overlap"):
        with pytest.raises(InvalidQueryError, match="Cutout contains no data!"):
            ASDFCutout(images[0], SkyCoord("0 0", unit="deg"), cutout_size)

    # Alter one of the test images to only contain zeros in cutout footprint
    with asdf.open(images[0], mode="rw") as af:
        af["roman"]["data"][470:480, 471:481] = 0
        af.update()

    # Should warn about first image containing no data, but not fail
    with pytest.warns(DataWarning, match="contains no data, skipping..."):
        cutouts = ASDFCutout(images, center_coord, cutout_size).cutouts
    assert len(cutouts) == 2


def test_asdf_cutout_no_gwcs(images, center_coord, cutout_size):
    # Remove WCS from test image
    with asdf.open(images[0], mode="rw") as af:
        del af["roman"]["meta"]["wcs"]
        af.update()

    # Should warn about missing WCS for first image, but not fail
    with pytest.warns(DataWarning, match="does not contain a GWCS object"):
        cutouts = ASDFCutout(images, center_coord, cutout_size).cutouts
    assert len(cutouts) == 2


def test_asdf_cutout_invalid_params(images, center_coord, cutout_size, tmpdir):
    # Invalid units for cutout size
    cutout_size = 1 * u.m  # meters are not valid
    with pytest.raises(InvalidInputError, match="Cutout size unit meter is not supported."):
        ASDFCutout(images, center_coord, cutout_size)

    # No coordinates provided
    with pytest.raises(InvalidInputError, match="At least one coordinate must be provided."):
        ASDFCutout(images, [], cutout_size)


def test_asdf_cutout_img_output(images, center_coord, cutout_size, tmpdir):
    # Basic JPG image
    jpg_files = ASDFCutout(images, center_coord, cutout_size).write_as_img(output_dir=tmpdir, output_format="jpg")
    assert len(jpg_files) == len(images)
    with open(jpg_files[0], "rb") as IMGFLE:
        assert IMGFLE.read(3) == b"\xff\xd8\xff"  # JPG

    # PNG (single input file, not as list)
    png_files = ASDFCutout(images[0], center_coord, cutout_size).write_as_img(output_dir=tmpdir, output_format="png")
    with open(png_files[0], "rb") as IMGFLE:
        assert IMGFLE.read(8) == b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"  # PNG
    assert len(png_files) == 1

    # Save to memory only
    img_cutouts = ASDFCutout(images[0], center_coord, cutout_size).get_image_cutouts()
    assert len(img_cutouts) == 1
    assert isinstance(img_cutouts["cutout"][0], Image.Image)
    assert np.array(img_cutouts["cutout"][0]).shape == (10, 10)


def test_asdf_cutout_img_output_colorize(images, multi_coord, cutout_size, tmpdir):
    # Make a copy of one of the input images
    with asdf.open(images[0], mode="rw") as af:
        af["roman"]["data"] = af["roman"]["data"][::-1, ::-1]  # flip data to make it different
        af.update()
        af.write_to(images[0].with_name("test_roman_extra.asdf"))
    images_extra = images + [images[0].with_name("test_roman_extra.asdf")]

    # Color image
    cutout = ASDFCutout(images_extra, multi_coord[:2], cutout_size)
    color_jpgs = cutout.write_as_img(output_dir=tmpdir, colorize=True, input_files=images_extra[:3])
    assert len(color_jpgs) == 2
    img = Image.open(color_jpgs[0])
    assert img.mode == "RGB"

    # Warn if not enough input files for colorization
    with pytest.warns(InputWarning, match="Color cutouts require 3 input images"):
        color_imgs = cutout.get_image_cutouts(colorize=True, input_files=images_extra[:2])
    assert len(color_imgs) == 0

    # Warn if too many input files for colorization
    with pytest.warns(InputWarning, match="More than 3 cutouts found for coordinate"):
        color_imgs = cutout.get_image_cutouts(colorize=True)
    assert len(color_imgs) == 2


def test_asdf_cutout_iter_image_cutouts(images, center_coord, cutout_size):
    cutout = ASDFCutout(images, center_coord, cutout_size)

    image_cutouts = list(cutout.iter_image_cutouts(input_files=images[1:]))
    assert len(image_cutouts) == 2
    for file, coordinate, image in image_cutouts:
        assert file in {image_path.as_posix() for image_path in images[1:]}
        assert isinstance(coordinate, SkyCoord)
        assert isinstance(image, Image.Image)
        assert np.array(image).shape == (cutout_size, cutout_size)


def test_asdf_cutout_iter_image_cutouts_colorize(images, center_coord, cutout_size):
    cutout = ASDFCutout(images, center_coord, cutout_size)

    image_cutouts = list(cutout.iter_image_cutouts(colorize=True))
    assert len(image_cutouts) == 1
    files, coordinate, image = image_cutouts[0]
    assert all(image_path.as_posix() in files for image_path in images)
    assert isinstance(coordinate, SkyCoord)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"


@pytest.mark.parametrize("colorize", [False, True])
def test_asdf_cutout_write_img_streams_from_iterator(images, center_coord, cutout_size, tmpdir, colorize):
    cutout = ASDFCutout(images, center_coord, cutout_size)

    with patch.object(
        ASDFCutout,
        "get_image_cutouts",
        side_effect=AssertionError("get_image_cutouts should not be used"),
    ):
        output = cutout.write_as_img(output_dir=tmpdir, colorize=colorize)

    expected_count = 1 if colorize else len(images)
    assert len(output) == expected_count
    for file_path in output:
        assert Path(file_path).exists()


def test_asdf_cutout_cube_angular_size(images, center_coord):
    """Test that cube-like arrays use the computed cutout shape for angular sizes."""
    cutout = ASDFCutout(images[0], center_coord, 2 * u.arcsec, lite=False)

    assert cutout.cutouts["cutout"][0].data.shape == (20, 20)
    assert cutout.asdf_cutouts["cutout"][0]["roman"]["context"].shape == (1, 20, 20)


def test_asdf_cutout_gwcs(images, center_coord):
    """Test creating a rectangular cutout to make sure cutout gwcs is correct"""
    cutout = ASDFCutout(images[0], center_coord, cutout_size=[20, 40])
    asdf_cutouts = cutout.asdf_cutouts["cutout"]
    gwcs = asdf_cutouts[0]["roman"]["meta"]["wcs"]
    assert isinstance(gwcs, wcs.WCS)
    assert gwcs.pixel_shape == (20, 40)
    assert gwcs.array_shape == (40, 20)
    assert gwcs.bounding_box.intervals[0].lower == 0
    assert gwcs.bounding_box.intervals[0].upper == 19
    assert gwcs.bounding_box.intervals[1].lower == 0
    assert gwcs.bounding_box.intervals[1].upper == 39


@pytest.mark.parametrize(
    ("is_installed", "warn_msg"),
    [(True, "not available in the correct version"), (False, "package cannot be imported")],
)
def test_asdf_cutout_stdatamodels(images, center_coord, cutout_size, is_installed, warn_msg):
    """Test that warning is emitted about ASDF-in-FITS embedding for stdatamodels issues"""
    mock_stdatamodels = None
    if is_installed:
        mock_stdatamodels = MagicMock()
        mock_stdatamodels.__version__ = "1.0.0"
        mock_stdatamodels.asdf_in_fits = MagicMock()
    patch_dict = {"stdatamodels": mock_stdatamodels}

    with patch.dict("sys.modules", patch_dict):
        with patch("sys.version_info", (3, 11, 0)):
            with pytest.warns(ModuleWarning, match=warn_msg):
                cutout = ASDFCutout(images, center_coord, cutout_size)
                fits_cutouts = cutout.fits_cutouts["cutout"]
            assert len(fits_cutouts[0]) == 2  # primary + cutout HDU only


def test_asdf_cutout_python_version(images, center_coord, cutout_size):
    """Test that warning is emitted about ASDF-in-FITS embedding for Python <3.11"""
    with patch("sys.version_info", (3, 10, 0)):
        with pytest.warns(ModuleWarning, match="requires Python 3.11 or higher"):
            cutout = ASDFCutout(images, center_coord, cutout_size)
            fits_cutouts = cutout.fits_cutouts["cutout"]
        assert cutout._py311_or_higher is False
        assert cutout._asdf_in_fits is None
        assert len(fits_cutouts[0]) == 2  # primary + cutout HDU only


def test_asdf_cutout_convert_gwcs_to_fits_wcs(fake_data):
    """Test that we can convert a gwcs to a FITS WCS for cutout output"""
    # Get the fake data
    __, gwcs = fake_data

    cutout = ASDFCutout.__new__(ASDFCutout)  # create instance without calling __init__
    cutout._gwcs_to_fits_cache = {}  # initialize cache
    fits_wcs = cutout._convert_gwcs_to_fits_wcs(gwcs)
    assert isinstance(fits_wcs, WCS)
    assert fits_wcs.pixel_shape == (1001, 1001)
    assert fits_wcs.wcs.crval[0] == 30.0
    assert fits_wcs.wcs.crval[1] == 45.0


def test_get_center_pixel(fake_data):
    """Test get_center_pixel function"""
    # Get the fake data
    __, gwcs = fake_data

    # Using center coordinates
    pixel_coordinates = get_center_pixel(gwcs, 30, 45)
    assert np.allclose(pixel_coordinates, (np.array(500), np.array(500)))

    # Using upper left corner
    # Running this without parametrization to make sure that gwcs is not corrupted
    coord = gwcs.pixel_to_world(0, 0)
    pixel_coordinates = get_center_pixel(gwcs, coord.ra.deg, coord.dec.deg)
    assert np.allclose(pixel_coordinates, (np.array(0), np.array(0)))


def test_asdf_cut(images, center_coord, cutout_size, tmpdir):
    """Test convenience function to create ASDF cutouts"""

    def check_paths(cutout_paths, ext):
        assert isinstance(cutout_paths, list)
        assert isinstance(cutout_paths[0], str)
        assert len(cutout_paths) == 3
        for i, path in enumerate(cutout_paths):
            assert isinstance(path, str)
            assert path.endswith(ext)
            assert Path(path).exists()
            assert str(tmpdir) in path
            assert Path(images[i]).stem in path
            assert center_coord.ra.to_string(unit="deg", decimal=True) in path
            assert center_coord.dec.to_string(unit="deg", decimal=True) in path
            assert "10-x-10" in path

    # Write files to disk as ASDF files
    asdf_paths = asdf_cut(images, center_coord.ra.deg, center_coord.dec.deg, cutout_size, output_dir=tmpdir)
    check_paths(asdf_paths, ".asdf")

    # Write files to disk as FITS files
    fits_paths = asdf_cut(
        images, center_coord.ra.deg, center_coord.dec.deg, cutout_size, output_dir=tmpdir, output_format="fits"
    )
    check_paths(fits_paths, ".fits")

    # Write cutouts to memory as Cutout2D objects
    cutouts = asdf_cut(images, center_coord.ra.deg, center_coord.dec.deg, cutout_size, write_file=False)
    assert isinstance(cutouts, Table)
    assert isinstance(cutouts["cutout"][0], Cutout2D)
    assert len(cutouts) == 3

    # Error if output format is not supported
    with pytest.raises(InvalidInputError, match="Output format .invalid is not recognized."):
        asdf_cut(images, center_coord.ra.deg, center_coord.dec.deg, cutout_size, output_format="invalid")
