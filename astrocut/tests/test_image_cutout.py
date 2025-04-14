import pytest

import numpy as np

from astrocut.image_cutout import ImageCutout

from ..exceptions import InputWarning, InvalidInputError


def test_normalize_img():
    # basic linear stretch
    img_arr = np.array([[1, 0], [.25, .75]])
    assert ((img_arr*255).astype(int) == ImageCutout.normalize_img(img_arr, stretch='linear')).all()

    # invert
    assert (255-(img_arr*255).astype(int) == ImageCutout.normalize_img(img_arr, stretch='linear', invert=True)).all()

    # linear stretch where input image must be scaled 
    img_arr = np.array([[10, 5], [2.5, 7.5]])
    norm_img = ((img_arr - img_arr.min())/(img_arr.max()-img_arr.min())*255).astype(int)
    assert (norm_img == ImageCutout.normalize_img(img_arr, stretch='linear')).all()

    # min_max val
    minval, maxval = 0, 1
    img_arr = np.array([[1, 0], [-1, 2]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch='linear', minmax_value=[minval, maxval])
    img_arr[img_arr < minval] = minval
    img_arr[img_arr > maxval] = maxval
    assert ((img_arr*255).astype(int) == norm_img).all()

    minval, maxval = 0, 1
    img_arr = np.array([[1, 0], [.1, .2]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch='linear', minmax_value=[minval, maxval])
    img_arr[img_arr < minval] = minval
    img_arr[img_arr > maxval] = maxval
    ((img_arr*255).astype(int) == norm_img).all()

    # min_max percent
    img_arr = np.array([[1, 0], [0.1, 0.9], [.25, .75]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch='linear', minmax_percent=[25, 75])
    assert (norm_img == [[255, 0], [0, 255], [39, 215]]).all()

    # asinh
    img_arr = np.array([[1, 0], [.25, .75]])
    norm_img = ImageCutout.normalize_img(img_arr)
    assert ((np.arcsinh(img_arr*10)/np.arcsinh(10)*255).astype(int) == norm_img).all()

    # sinh
    img_arr = np.array([[1, 0], [.25, .75]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch='sinh')
    assert ((np.sinh(img_arr*3)/np.sinh(3)*255).astype(int) == norm_img).all()

    # sqrt
    img_arr = np.array([[1, 0], [.25, .75]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch='sqrt')
    assert ((np.sqrt(img_arr)*255).astype(int) == norm_img).all()

    # log
    img_arr = np.array([[1, 0], [.25, .75]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch='log')
    assert ((np.log(img_arr*1000+1)/np.log(1000)*255).astype(int) == norm_img).all()


def test_normalize_img_errors():
    # Bad stretch
    with pytest.raises(InvalidInputError):
        img_arr = np.array([[1, 0], [.25, .75]])
        ImageCutout.normalize_img(img_arr, stretch='lin')

    # Giving both minmax percent and cut
    img_arr = np.array([[1, 0], [.25, .75]])
    norm_img = ImageCutout.normalize_img(img_arr, stretch='asinh', minmax_percent=[0.7, 99.3])
    with pytest.warns(InputWarning, 
                      match='Both minmax_percent and minmax_value are set, minmax_value will be ignored.'):
        test_img = ImageCutout.normalize_img(img_arr, stretch='asinh', minmax_value=[5, 2000], 
                                             minmax_percent=[0.7, 99.3])
    assert (test_img == norm_img).all()

    # Raise error if image array is empty
    img_arr = np.array([])
    with pytest.raises(InvalidInputError):
        ImageCutout.normalize_img(img_arr)
