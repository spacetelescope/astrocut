import pytest

import numpy as np
from os import path
from re import findall

from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u

from PIL import Image

from .utils_for_test import create_test_imgs
from .. import cutouts
from ..exceptions import InputWarning, InvalidInputError, InvalidQueryError 


def test_fits_cut(tmpdir, capsys):

    test_images = create_test_imgs(50, 6, dir_name=tmpdir)

    # Single file
    center_coord = SkyCoord("150.1163213 2.200973097", unit='deg')
    cutout_size = 10
    cutout_file = cutouts.fits_cut(test_images, center_coord, cutout_size, single_outfile=True, output_dir=tmpdir)
    assert isinstance(cutout_file, str)
    
    cutout_hdulist = fits.open(cutout_file)
    assert len(cutout_hdulist) == len(test_images) + 1  # num imgs + primary header

    cut1 = cutout_hdulist[1].data
    assert cut1.shape == (cutout_size, cutout_size)
    assert cutout_hdulist[1].data.shape == cutout_hdulist[2].data.shape
    assert cutout_hdulist[2].data.shape == cutout_hdulist[3].data.shape
    assert cutout_hdulist[3].data.shape == cutout_hdulist[4].data.shape
    assert cutout_hdulist[4].data.shape == cutout_hdulist[5].data.shape
    assert cutout_hdulist[5].data.shape == cutout_hdulist[6].data.shape
    
    cut_wcs = wcs.WCS(cutout_hdulist[1].header)
    sra, sdec = cut_wcs.all_pix2world(cutout_size/2, cutout_size/2, 0)
    assert round(float(sra), 4) == round(center_coord.ra.deg, 4)
    assert round(float(sdec), 4) == round(center_coord.dec.deg, 4)

    cutout_hdulist.close()

    # Multiple files
    cutout_files = cutouts.fits_cut(test_images, center_coord, cutout_size, single_outfile=False, output_dir=tmpdir)

    assert isinstance(cutout_files, list)
    assert len(cutout_files) == len(test_images)

    cutout_hdulist = fits.open(cutout_files[0])
    assert len(cutout_hdulist) == 2
    
    cut1 = cutout_hdulist[1].data
    assert cut1.shape == (cutout_size, cutout_size)
    
    cut_wcs = wcs.WCS(cutout_hdulist[1].header)
    sra, sdec = cut_wcs.all_pix2world(cutout_size/2, cutout_size/2, 0)
    assert round(float(sra), 4) == round(center_coord.ra.deg, 4)
    assert round(float(sdec), 4) == round(center_coord.dec.deg, 4)

    cutout_hdulist.close()

    # Output directory that has to be made
    cutout_files = cutouts.fits_cut(test_images, center_coord, cutout_size,
                                    output_dir=path.join(tmpdir, "cutout_files"), single_outfile=False)

    assert isinstance(cutout_files, list)
    assert len(cutout_files) == len(test_images)
    assert "cutout_files" in cutout_files[0]

    # Memory only flag
    cutout_list = cutouts.fits_cut(test_images, center_coord, cutout_size, single_outfile=True, memory_only=True)
    assert isinstance(cutout_list, list)
    assert len(cutout_list) == 1
    assert isinstance(cutout_list[0], fits.HDUList)
    
    # Do an off the edge test
    center_coord = SkyCoord("150.1163213 2.2005731", unit='deg')
    cutout_file = cutouts.fits_cut(test_images, center_coord, cutout_size, single_outfile=True, output_dir=tmpdir)
    assert isinstance(cutout_file, str)
    
    cutout_hdulist = fits.open(cutout_file)
    assert len(cutout_hdulist) == len(test_images) + 1  # num imgs + primary header

    cut1 = cutout_hdulist[1].data
    assert cut1.shape == (cutout_size, cutout_size)
    assert np.isnan(cut1[:cutout_size//2, :]).all()

    cutout_hdulist.close()

    # Test when the requested cutout is not on the image
    center_coord = SkyCoord("140.1163213 2.2005731", unit='deg')
    with pytest.raises(Exception, match='Cutout location is not in image footprint!') as e:
        cutout_file = cutouts.fits_cut(test_images, center_coord, cutout_size, single_outfile=True)
        assert e.type is InvalidQueryError

    center_coord = SkyCoord("15.1163213 2.2005731", unit='deg')
    with pytest.raises(Exception, match='Cutout location is not in image footprint!') as e:
        cutout_file = cutouts.fits_cut(test_images, center_coord, cutout_size, single_outfile=True)
        assert e.type is InvalidQueryError
    

    # Test when cutout is in some images not others

    # Putting zeros into 2 images
    for img in test_images[:2]:
        hdu = fits.open(img, mode="update")
        hdu[0].data[:20, :] = 0
        hdu.flush()
        hdu.close()
        
        
    center_coord = SkyCoord("150.1163213 2.2007", unit='deg')
    cutout_file = cutouts.fits_cut(test_images, center_coord, cutout_size, single_outfile=True, output_dir=tmpdir)
    
    cutout_hdulist = fits.open(cutout_file)
    assert len(cutout_hdulist) == len(test_images) + 1  # num imgs + primary header
    assert (cutout_hdulist[1].data == 0).all()
    assert (cutout_hdulist[2].data == 0).all()
    assert ~(cutout_hdulist[3].data == 0).any()
    assert ~(cutout_hdulist[4].data == 0).any()
    assert ~(cutout_hdulist[5].data == 0).any()
    assert ~(cutout_hdulist[6].data == 0).any()

    
    cutout_files = cutouts.fits_cut(test_images, center_coord, cutout_size, single_outfile=False, output_dir=tmpdir)
    assert isinstance(cutout_files, list)
    assert len(cutout_files) == len(test_images) - 2

    # Test when cutout is in no images
    for img in test_images[2:]:
        hdu = fits.open(img, mode="update")
        hdu[0].data[:20, :] = 0
        hdu.flush()
        hdu.close()

    with pytest.raises(Exception) as e:
        cutout_file = cutouts.fits_cut(test_images, center_coord, cutout_size, single_outfile=True, output_dir=tmpdir)
        assert e.type is InvalidQueryError
        assert "Cutout contains no data! (Check image footprint.)" in str(e.value)

    # test single image and also conflicting sip keywords
    test_image = create_test_imgs(50, 1, dir_name=tmpdir, basename="img_badsip_{:04d}.fits", bad_sip_keywords=True)[0]

    center_coord = SkyCoord("150.1163213 2.2007", unit='deg')
    cutout_size = [10, 15]
    cutout_file = cutouts.fits_cut(test_image, center_coord, cutout_size, output_dir=tmpdir)
    assert isinstance(cutout_file, str)
    assert "10-x-15" in cutout_file
    cutout_hdulist = fits.open(cutout_file)
    assert cutout_hdulist[1].data.shape == (15, 10)

    center_coord = SkyCoord("150.1159 2.2006", unit='deg')
    cutout_size = [10, 15]*u.pixel
    cutout_file = cutouts.fits_cut(test_image, center_coord, cutout_size, output_dir=tmpdir)
    assert isinstance(cutout_file, str)
    assert "10.0pix-x-15.0pix" in cutout_file
    cutout_hdulist = fits.open(cutout_file)
    assert cutout_hdulist[1].data.shape == (15, 10)

    cutout_size = [1, 2]*u.arcsec
    cutout_file = cutouts.fits_cut(test_image, center_coord, cutout_size, output_dir=tmpdir, verbose=True)
    assert isinstance(cutout_file, str)
    assert "1.0arcsec-x-2.0arcsec" in cutout_file
    cutout_hdulist = fits.open(cutout_file)
    assert cutout_hdulist[1].data.shape == (33, 17)
    captured = capsys.readouterr()
    assert "Original image shape: (50, 50)" in captured.out
    assert "Image cutout shape: (33, 17)" in captured.out
    assert "Total time:" in captured.out

    center_coord = "150.1159 2.2006"
    cutout_size = [10, 15, 20]
    with pytest.warns(InputWarning):
        cutout_file = cutouts.fits_cut(test_image, center_coord, cutout_size, output_dir=tmpdir)
    assert isinstance(cutout_file, str)
    assert "10-x-15" in cutout_file
    assert "x-20" not in cutout_file


def test_normalize_img():

    # basic linear stretch
    img_arr = np.array([[1, 0], [.25, .75]])
    assert ((img_arr*255).astype(int) == cutouts.normalize_img(img_arr, stretch='linear')).all()

    # invert
    assert (255-(img_arr*255).astype(int) == cutouts.normalize_img(img_arr, stretch='linear', invert=True)).all()

    # linear stretch where input image must be scaled 
    img_arr = np.array([[10, 5], [2.5, 7.5]])
    norm_img = ((img_arr - img_arr.min())/(img_arr.max()-img_arr.min())*255).astype(int)
    assert (norm_img == cutouts.normalize_img(img_arr, stretch='linear')).all()

    # min_max val
    minval, maxval = 0, 1
    img_arr = np.array([[1, 0], [-1, 2]])
    norm_img = cutouts.normalize_img(img_arr, stretch='linear', minmax_value=[minval, maxval])
    img_arr[img_arr < minval] = minval
    img_arr[img_arr > maxval] = maxval
    assert ((img_arr*255).astype(int) == norm_img).all()

    minval, maxval = 0, 1
    img_arr = np.array([[1, 0], [.1, .2]])
    norm_img = cutouts.normalize_img(img_arr, stretch='linear', minmax_value=[minval, maxval])
    img_arr[img_arr < minval] = minval
    img_arr[img_arr > maxval] = maxval
    ((img_arr*255).astype(int) == norm_img).all()

    # min_max percent
    img_arr = np.array([[1, 0], [0.1, 0.9], [.25, .75]])
    norm_img = cutouts.normalize_img(img_arr, stretch='linear', minmax_percent=[25, 75])
    assert (norm_img == [[255, 0], [0, 255], [39, 215]]).all()

    # asinh
    img_arr = np.array([[1, 0], [.25, .75]])
    norm_img = cutouts.normalize_img(img_arr)
    assert ((np.arcsinh(img_arr*10)/np.arcsinh(10)*255).astype(int) == norm_img).all()

    # sinh
    img_arr = np.array([[1, 0], [.25, .75]])
    norm_img = cutouts.normalize_img(img_arr, stretch='sinh')
    assert ((np.sinh(img_arr*3)/np.sinh(3)*255).astype(int) == norm_img).all()

    # sqrt
    img_arr = np.array([[1, 0], [.25, .75]])
    norm_img = cutouts.normalize_img(img_arr, stretch='sqrt')
    assert ((np.sqrt(img_arr)*255).astype(int) == norm_img).all()

    # log
    img_arr = np.array([[1, 0], [.25, .75]])
    norm_img = cutouts.normalize_img(img_arr, stretch='log')
    assert ((np.log(img_arr*1000+1)/np.log(1000)*255).astype(int) == norm_img).all()

    # Bad stretch
    with pytest.raises(InvalidInputError):
        img_arr = np.array([[1, 0], [.25, .75]])
        cutouts.normalize_img(img_arr, stretch='lin')

    # Giving both minmax percent and cut
    img_arr = np.array([[1, 0], [.25, .75]])
    norm_img = cutouts.normalize_img(img_arr, stretch='asinh', minmax_percent=[0.7, 99.3])
    with pytest.warns(InputWarning):
        test_img = cutouts.normalize_img(img_arr, stretch='asinh', minmax_value=[5, 2000], minmax_percent=[0.7, 99.3])
    assert (test_img == norm_img).all()


def test_img_cut(tmpdir, capsys):

    test_images = create_test_imgs(50, 6, dir_name=tmpdir)
    center_coord = SkyCoord("150.1163213 2.200973097", unit='deg')
    cutout_size = 10

    # Basic jpg image
    jpg_files = cutouts.img_cut(test_images, center_coord, cutout_size, output_dir=tmpdir)
    
    assert len(jpg_files) == len(test_images)
    with open(jpg_files[0], 'rb') as IMGFLE:
        assert IMGFLE.read(3) == b'\xFF\xD8\xFF'  # JPG

    # Png (single input file, not as list)
    img_files = cutouts.img_cut(test_images[0], center_coord, cutout_size, img_format='png', output_dir=tmpdir)
    with open(img_files[0], 'rb') as IMGFLE:
        assert IMGFLE.read(8) == b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'  # PNG
    assert len(img_files) == 1

    # Color image
    color_jpg = cutouts.img_cut(test_images[:3], center_coord, cutout_size, colorize=True, output_dir=tmpdir)
    img = Image.open(color_jpg)
    assert img.mode == 'RGB'

    # Too few input images
    with pytest.raises(InvalidInputError):
        cutouts.img_cut(test_images[0], center_coord, cutout_size, colorize=True, output_dir=tmpdir)

    # Too many input images
    with pytest.warns(InputWarning):
        color_jpg = cutouts.img_cut(test_images, center_coord, cutout_size, colorize=True, output_dir=tmpdir)
    img = Image.open(color_jpg)
    assert img.mode == 'RGB'

    # string coordinates and verbose
    center_coord = "150.1163213 2.200973097"
    jpg_files = cutouts.img_cut(test_images, center_coord, cutout_size,
                                output_dir=path.join(tmpdir, "image_path"), verbose=True)
    captured = capsys.readouterr()
    assert len(findall("Original image shape", captured.out)) == 6
    assert "Cutout fits file(s)" in captured.out
    assert "Total time" in captured.out

    # test color image where one of the images is all zeros
    hdu = fits.open(test_images[0], mode='update')
    hdu[0].data[:, :] = 0
    hdu.flush()
    hdu.close()

    with pytest.raises(InvalidInputError):
        cutouts.img_cut(test_images[:3], center_coord, cutout_size,
                        colorize=True, img_format='png', output_dir=tmpdir)

    
    
