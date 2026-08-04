import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from PIL import Image

from astrocut.image_cutout import ImageCutout, normalize_img

from ..exceptions import InputWarning, InvalidInputError


class _DummyImageCutout(ImageCutout):
    """Minimal concrete ImageCutout for testing helper methods."""

    def __init__(self):
        # Avoid parent initialization and set only what tests need.
        self._coordinates = SkyCoord("1 2", unit="deg")
        self.cutouts_by_file = {}

    def _cutout_file(self, file):
        return None

    def cutout(self):
        return None


def _make_fake_cutout(shape=(4, 6)):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [30.0, 45.0]
    wcs.wcs.cdelt = [-0.0002777778, 0.0002777778]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    class _FakeCutout:
        pass

    cutout = _FakeCutout()
    cutout.shape = shape
    cutout.wcs = wcs
    cutout.data = np.arange(shape[0] * shape[1], dtype=float).reshape(shape)
    return cutout


def test_normalize_img():
    # basic linear stretch
    img_arr = np.array([[1, 0], [0.25, 0.75]])
    assert ((img_arr * 255).astype(int) == ImageCutout.normalize_img(img_arr, stretch="linear")).all()

    # invert
    assert (
        255 - (img_arr * 255).astype(int) == ImageCutout.normalize_img(img_arr, stretch="linear", invert=True)
    ).all()

    # linear stretch where input image must be scaled
    img_arr = np.array([[10, 5], [2.5, 7.5]])
    norm_img = ((img_arr - img_arr.min()) / (img_arr.max() - img_arr.min()) * 255).astype(int)
    assert (norm_img == ImageCutout.normalize_img(img_arr, stretch="linear")).all()

    # min_max val
    minval, maxval = 0, 1
    img_arr = np.array([[1, 0], [-1, 2]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch="linear", minmax_value=[minval, maxval])
    img_arr[img_arr < minval] = minval
    img_arr[img_arr > maxval] = maxval
    assert ((img_arr * 255).astype(int) == norm_img).all()

    minval, maxval = 0, 1
    img_arr = np.array([[1, 0], [0.1, 0.2]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch="linear", minmax_value=[minval, maxval])
    img_arr[img_arr < minval] = minval
    img_arr[img_arr > maxval] = maxval
    ((img_arr * 255).astype(int) == norm_img).all()

    # min_max percent
    img_arr = np.array([[1, 0], [0.1, 0.9], [0.25, 0.75]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch="linear", minmax_percent=[25, 75])
    assert (norm_img == [[255, 0], [0, 255], [39, 215]]).all()

    # asinh
    img_arr = np.array([[1, 0], [0.25, 0.75]])
    norm_img = ImageCutout.normalize_img(img_arr)
    assert ((np.arcsinh(img_arr * 10) / np.arcsinh(10) * 255).astype(int) == norm_img).all()

    # sinh
    img_arr = np.array([[1, 0], [0.25, 0.75]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch="sinh")
    assert ((np.sinh(img_arr * 3) / np.sinh(3) * 255).astype(int) == norm_img).all()

    # sqrt
    img_arr = np.array([[1, 0], [0.25, 0.75]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch="sqrt")
    assert ((np.sqrt(img_arr) * 255).astype(int) == norm_img).all()

    # log
    img_arr = np.array([[1, 0], [0.25, 0.75]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch="log")
    assert ((np.log(img_arr * 1000 + 1) / np.log(1000) * 255).astype(int) == norm_img).all()


def test_normalize_img_errors():
    # Bad stretch
    with pytest.raises(InvalidInputError):
        img_arr = np.array([[1, 0], [0.25, 0.75]])
        ImageCutout.normalize_img(img_arr, stretch="lin")

    # Giving both minmax percent and cut
    img_arr = np.array([[1, 0], [0.25, 0.75]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch="asinh", minmax_percent=[0.7, 99.3])
    with pytest.warns(
        InputWarning, match="Both minmax_percent and minmax_value are set, minmax_value will be ignored."
    ):
        test_img = ImageCutout.normalize_img(
            img_arr, stretch="asinh", minmax_value=[5, 2000], minmax_percent=[0.7, 99.3]
        )
    assert (test_img == norm_img).all()

    # Raise error if image array is empty
    img_arr = np.array([])
    with pytest.raises(InvalidInputError):
        ImageCutout.normalize_img(img_arr)


def test_prepare_render_options_invalid_stretch():
    cutout = _DummyImageCutout()
    with pytest.raises(InvalidInputError, match="is not recognized"):
        cutout._prepare_image_render_options("bad-stretch", None, None)


def test_coerce_coordinate_list_variants():
    assert ImageCutout._coerce_coordinate_list(None) == []
    assert ImageCutout._coerce_coordinate_list(["1 2"]) == ["1 2"]
    assert ImageCutout._coerce_coordinate_list(("1 2", "3 4")) == ["1 2", "3 4"]
    assert ImageCutout._coerce_coordinate_list("1 2") == ["1 2"]


def test_build_image_cutout_table_populates_cutout_column():
    coord = SkyCoord("1 2", unit="deg")
    img = Image.fromarray(np.zeros((3, 3), dtype=np.uint8))
    table = ImageCutout._build_image_cutout_table([("file_1", coord, img), ("file_2", coord, img)])
    assert len(table) == 2
    assert table["file"][0] == "file_1"
    assert table["cutout"][1] is img


def test_build_cutout_metadata_with_string_coordinate():
    cutout = _DummyImageCutout()
    fake_cutout = _make_fake_cutout()
    meta = cutout._build_cutout_metadata(["input_a.fits", "input_b.fits"], fake_cutout, "30.0 45.0")
    assert "input_files" in meta
    assert meta["center_ra_deg"] == 30.0
    assert meta["center_dec_deg"] == 45.0


def test_iter_selected_cutouts_branches_and_errors():
    cutout = _DummyImageCutout()
    fake_cutout = _make_fake_cutout()
    cutout.cutouts_by_file = {"file_a": [fake_cutout], "file_b": [fake_cutout]}

    with pytest.raises(InvalidInputError, match="Selecting image cutouts by coordinates is not supported"):
        list(cutout._iter_selected_cutouts(coordinates=[SkyCoord("1 2", unit="deg")]))

    with pytest.raises(InvalidInputError, match="is not in the cutout results"):
        list(cutout._iter_selected_cutouts(input_files=["missing_file"]))

    rows = list(cutout._iter_selected_cutouts(input_files="file_a"))
    assert len(rows) == 1
    assert rows[0][0] == "file_a"
    assert rows[0][2] is fake_cutout

    # Default path selects all files from cutouts_by_file.
    all_rows = list(cutout._iter_selected_cutouts())
    assert len(all_rows) == 2


def test_coord_object_and_colorize_helper_warnings():
    cutout = _DummyImageCutout()

    coord = cutout._coerce_coord_object("30.0 45.0")
    assert isinstance(coord, SkyCoord)

    with pytest.warns(InputWarning, match="Too many inputs for a color cutout"):
        cutout._warn_too_many_color_cutouts(coord)

    with pytest.raises(InvalidInputError, match="Color cutouts require 3 input images"):
        cutout._handle_insufficient_color_cutouts(coord)


def test_get_img_save_kwargs_without_metadata_and_save_oserror(tmp_path):
    cutout = _DummyImageCutout()
    img = Image.fromarray(np.zeros((4, 4), dtype=np.uint8))

    assert cutout._get_img_save_kwargs(img, "no_meta.png") == {}

    def _raise_oserror(*args, **kwargs):
        raise OSError("disk full")

    img.save = _raise_oserror
    with pytest.warns(Warning, match="Cutout could not be saved"):
        success = cutout._save_img_to_file(img, (tmp_path / "x.png").as_posix())
    assert success is False


def test_module_normalize_img_wrapper():
    img_arr = np.array([[1.0, 0.0], [0.25, 0.75]])
    assert (normalize_img(img_arr, stretch="linear") == ImageCutout.normalize_img(img_arr, stretch="linear")).all()
