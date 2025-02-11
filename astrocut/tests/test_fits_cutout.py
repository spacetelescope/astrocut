from pathlib import Path
import pytest

import numpy as np
from os import path
from re import findall

from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from PIL import Image

from astrocut.fits_cutout import FITSCutout

from .utils_for_test import create_test_imgs
from ..exceptions import DataWarning, InputWarning, InvalidInputError, InvalidQueryError


# Fixture to create test images for both SPOC and TICA
@pytest.fixture(params=['SPOC', 'TICA'])
def test_images(request, tmpdir):
    return create_test_imgs(request.param, 50, 6, dir_name=tmpdir)


# Fixture to create a test image with bad SIP keywords
@pytest.fixture(params=['SPOC', 'TICA'])
def test_image_bad_sip(request, tmpdir):
    return create_test_imgs(request.param, 50, 1, dir_name=tmpdir,
                            basename="img_badsip_{:04d}.fits", bad_sip_keywords=True)[0]
    

# Fixture to return a center coordinate
@pytest.fixture
def center_coord():
    return SkyCoord('150.1163213 2.200973097', unit='deg')


# Fixture to return a cutout size
@pytest.fixture
def cutout_size():
    return 10


def test_fits_cutout_single_outfile(test_images, center_coord, cutout_size, tmpdir):
    # Create cutout with single output file
    cutouts = FITSCutout(test_images, center_coord, cutout_size, single_outfile=True).fits_cutouts

    # Should output a list of objects
    assert isinstance(cutouts, list)
    assert isinstance(cutouts[0], fits.HDUList)
    assert len(cutouts) == 1

    # Check shape of data
    cutout = cutouts[0]
    assert len(cutout) == len(test_images) + 1  # num imgs + primary header
    assert cutout[1].data.shape == (cutout_size, cutout_size)
    assert cutout[1].data.shape == cutout[2].data.shape
    assert cutout[2].data.shape == cutout[3].data.shape
    assert cutout[3].data.shape == cutout[4].data.shape
    assert cutout[4].data.shape == cutout[5].data.shape
    assert cutout[5].data.shape == cutout[6].data.shape

    # Check that data is equal between cutout and original image
    for i, img in enumerate(test_images):
        with fits.open(img) as test_hdu:
            assert np.all(cutout[i + 1].data == test_hdu[0].data[19:29, 19:29])
    
    # Check WCS and position of center
    cut_wcs = wcs.WCS(cutout[1].header)
    sra, sdec = cut_wcs.all_pix2world(cutout_size/2, cutout_size/2, 0)
    assert round(float(sra), 4) == round(center_coord.ra.deg, 4)
    assert round(float(sdec), 4) == round(center_coord.dec.deg, 4)


def test_fits_cutout_multiple_files(tmpdir, test_images, center_coord, cutout_size):
    # Output is multiple files
    cutouts = FITSCutout(test_images, center_coord, cutout_size, single_outfile=False).fits_cutouts

    # Should output a list of objects
    assert isinstance(cutouts, list)
    assert isinstance(cutouts[0], fits.HDUList)
    assert len(cutouts) == len(test_images)
    
    for i, cutout in enumerate(cutouts):
        cut1 = cutout[1].data

        # Check shape of data
        assert len(cutout) == 2  # primary header + 1 image
        assert cut1.shape == (cutout_size, cutout_size)
        
        # Check that data is equal between cutout and original image
        with fits.open(test_images[i]) as test_hdu:
            assert np.all(cut1 == test_hdu[0].data[19:29, 19:29])
    
        # Check WCS and position of center
        cut_wcs = wcs.WCS(cutout[1].header)
        sra, sdec = cut_wcs.all_pix2world(cutout_size/2, cutout_size/2, 0)
        assert round(float(sra), 4) == round(center_coord.ra.deg, 4)
        assert round(float(sdec), 4) == round(center_coord.dec.deg, 4)

    # Test case where output directory does not exist
    new_dir = path.join(tmpdir, 'cutout_files')  # non-existing directory to write files to
    cutouts = FITSCutout(test_images[0], center_coord, cutout_size, single_outfile=False)
    paths = cutouts.write_as_fits(output_dir=new_dir)

    assert isinstance(paths, list)
    assert isinstance(paths[0], str)
    assert new_dir in paths[0]
    assert path.exists(new_dir)  # new directory should now exist


def test_fits_cutout_memory_only(test_images, center_coord, cutout_size):
    # Memory only, single file
    nonexisting_dir = 'nonexisting'  # non-existing directory to check that no files are written
    cutout_list = FITSCutout(test_images, center_coord, cutout_size, single_outfile=True).fits_cutouts
    assert isinstance(cutout_list, list)
    assert len(cutout_list) == 1
    assert isinstance(cutout_list[0], fits.HDUList)
    assert not path.exists(nonexisting_dir)  # no files should be written

    # Memory only, multiple files
    cutout_list = FITSCutout(test_images, center_coord, cutout_size, single_outfile=False).fits_cutouts
    assert isinstance(cutout_list, list)
    assert len(cutout_list) == len(test_images)
    assert isinstance(cutout_list[0], fits.HDUList)
    assert not path.exists(nonexisting_dir)  # no files should be written


def test_fits_cutout_return_paths(test_images, center_coord, cutout_size, tmpdir):
    # Return filepath for single output file
    cutout_file = FITSCutout(test_images, center_coord, cutout_size).write_as_fits(output_dir=tmpdir, 
                                                                                   cutout_prefix='prefix')[0]
    assert isinstance(cutout_file, str)
    assert path.exists(cutout_file)
    assert str(tmpdir) in cutout_file
    assert 'prefix' in cutout_file
    assert center_coord.ra.to_string(unit='deg', decimal=True) in cutout_file
    assert center_coord.dec.to_string(unit='deg', decimal=True) in cutout_file
    assert '10-x-10' in cutout_file

    # Return list of filepaths for multiple output files
    cutout_files = FITSCutout(test_images, center_coord, cutout_size, 
                              single_outfile=False).write_as_fits(output_dir=tmpdir)
    assert isinstance(cutout_files, list)
    assert len(cutout_files) == len(test_images)
    for i, cutout_file in enumerate(cutout_files):
        assert path.exists(cutout_file)
        assert str(tmpdir) in cutout_file
        assert Path(test_images[i]).stem in cutout_file


def test_fits_cutout_off_edge(test_images, cutout_size):
    #  Off the top
    center_coord = SkyCoord("150.1163213 2.2005731", unit='deg')
    cutout = FITSCutout(test_images, center_coord, cutout_size, single_outfile=True).fits_cutouts[0]
    assert isinstance(cutout, fits.HDUList)
    
    assert len(cutout) == len(test_images) + 1  # num imgs + primary header

    cut1 = cutout[1].data
    assert cut1.shape == (cutout_size, cutout_size)
    assert np.isnan(cut1[:cutout_size//2, :]).all()

    # Off the bottom
    center_coord = SkyCoord("150.1163213 2.2014", unit='deg')
    cutout = FITSCutout(test_images[0], center_coord, cutout_size).fits_cutouts[0]
    assert np.isnan(cutout[1].data[cutout_size//2:, :]).all()

    # Off the left, with integer fill value
    center_coord = SkyCoord('150.11672 2.200973097', unit='deg')
    cutout = FITSCutout(test_images[0], center_coord, cutout_size, fill_value=1).fits_cutouts[0]
    assert np.all(cutout[1].data[:, :cutout_size//2] == 1)

    # Off the right, with float fill value
    center_coord = SkyCoord('150.11588 2.200973097', unit='deg')
    cutout = FITSCutout(test_images[0], center_coord, cutout_size, fill_value=1.5).fits_cutouts[0]
    assert np.all(cutout[1].data[:, cutout_size//2:] == 1.5)

    # Error if unexpected fill value
    with pytest.raises(InvalidInputError, match='Fill value must be an integer or a float.'):
        FITSCutout(test_images[0], center_coord, cutout_size, fill_value='invalid')


def test_fits_cutout_cloud():
    # Test single cloud image
    test_s3_uri = "s3://stpubdata/hst/public/j8pu/j8pu0y010/j8pu0y010_drc.fits"
    center_coord = SkyCoord("150.4275416667 2.42155", unit='deg')
    cutout_size = [10, 15]
    cutout = FITSCutout(test_s3_uri, center_coord, cutout_size).fits_cutouts[0]
    assert cutout[1].data.shape == (15, 10)


def test_fits_cutout_rounding(test_images, cutout_size):
    # Rounding normally
    center_coord = SkyCoord("150.1163117 2.200973097", unit='deg')
    cutout = FITSCutout(test_images[0], center_coord, cutout_size).fits_cutouts[0]
    with fits.open(test_images[0]) as test_hdu:
        assert np.all(cutout[1].data == test_hdu[0].data[19:29, 20:30])

        # Rounding to ceiling
        cutout = FITSCutout(test_images[0], center_coord, cutout_size, limit_rounding_method='ceil').fits_cutouts[0]
        assert np.all(cutout[1].data == test_hdu[0].data[20:30, 20:30])

        # Rounding to floor
        cutout = FITSCutout(test_images[0], center_coord, cutout_size, limit_rounding_method='floor').fits_cutouts[0]
        assert np.all(cutout[1].data == test_hdu[0].data[19:29, 19:29])

        # Case that the cutout rounds to zero
        cutout_size = 0.57557495
        cutout = FITSCutout(test_images[0], center_coord, cutout_size, limit_rounding_method='round').fits_cutouts[0]
        assert np.all(cutout[1].data == test_hdu[0].data[24:25, 24:25])


def test_fits_cutout_extension(test_images, center_coord, cutout_size):
    # Add additional extensions to one of the input files
    with fits.open(test_images[0], mode='update') as hdul:
        # Create a copy of first extension
        source_hdu = hdul[0]
        new_hdu = source_hdu.copy()
        hdul.append(new_hdu)
        hdul.append(new_hdu)  # add two copies
        hdul.flush()  # save changes

    # Cutout all extensions
    cutout_list = FITSCutout(test_images[0], center_coord, cutout_size, extension='all').fits_cutouts
    assert len(cutout_list[0]) == 4  # primary header + 3 images

    # Specify a single extension
    cutout_list = FITSCutout(test_images[0], center_coord, cutout_size, extension=2).fits_cutouts
    assert len(cutout_list[0]) == 2  # primary header + 1 image

    # Specify a list of extensions
    cutout_list = FITSCutout(test_images[0], center_coord, cutout_size, extension=[0, 1]).fits_cutouts
    assert len(cutout_list[0]) == 3  # primary header + 2 images

    # Warning if a non-existing extension is specified
    with pytest.warns(DataWarning, match=r'extension\(s\) 3 will be skipped.'):
        cutout_list = FITSCutout(test_images[0], center_coord, cutout_size, extension=[1, 3]).fits_cutouts
        assert len(cutout_list[0]) == 2  # primary header + 1 image

    # Remove image data from one of the input files
    with fits.open(test_images[1], mode='update') as hdul:
        primary = hdul[0]
        primary.data = None
        table_data = Table({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        table_hdu = fits.BinTableHDU(data=table_data, name='TABLE')
        hdul.append(table_hdu)
        hdul.flush()

    with pytest.warns(DataWarning, match='No image extensions with data found.'):
        with pytest.raises(InvalidQueryError, match='Cutout contains no data!'):
            FITSCutout(test_images[1], center_coord, cutout_size)


def test_fits_cutout_not_in_footprint(test_images, cutout_size):
    # Test when the requested cutout is not on the image
    center_coord = SkyCoord("140.1163213 2.2005731", unit='deg')
    with pytest.warns(DataWarning, match='does not overlap'):
        with pytest.warns(DataWarning, match='contains no data, skipping...'):
            with pytest.raises(InvalidQueryError, match='Cutout contains no data!'):
                FITSCutout(test_images, center_coord, cutout_size, single_outfile=True)

    center_coord = SkyCoord("15.1163213 2.2005731", unit='deg')
    with pytest.warns(DataWarning, match='does not overlap'):
        with pytest.warns(DataWarning, match='contains no data, skipping...'):
            with pytest.raises(InvalidQueryError, match='Cutout contains no data!'):
                FITSCutout(test_images, center_coord, cutout_size, single_outfile=True)


def test_fits_cutout_no_data(tmpdir, test_images, cutout_size):
    # Test behavior when some input files contain zeros in cutout footprint
    # Putting zeros into 2 images
    for img in test_images[:2]:
        with fits.open(img, mode="update") as hdu:
            hdu[0].data[:20, :] = 0
            hdu.flush()
        
    # Single outfile should include empty files as extensions
    center_coord = SkyCoord("150.1163213 2.2007", unit='deg')
    with pytest.warns(DataWarning, match='contains no data, skipping...'):
        cutout = FITSCutout(test_images, center_coord, cutout_size, single_outfile=True).fits_cutouts[0]
    assert len(cutout) == len(test_images) - 1  # 6 images - 2 empty + 1 primary header
    assert ~(cutout[1].data == 0).any()
    assert ~(cutout[2].data == 0).any()
    assert ~(cutout[3].data == 0).any()
    assert ~(cutout[4].data == 0).any()
    
    # Empty files should not be written to their own file
    with pytest.warns(DataWarning, match='contains no data, skipping...'):
        cutout_files = FITSCutout(test_images, center_coord, cutout_size, 
                                  single_outfile=False).write_as_fits(output_dir=tmpdir)
    assert isinstance(cutout_files, list)
    assert len(cutout_files) == len(test_images) - 2

    # Test when all input files contain only zeros in cutout footprint
    # Putting zeros into the rest of the images
    for img in test_images[2:]:
        with fits.open(img, mode="update") as hdu:
            hdu[0].data[:20, :] = 0
            hdu.flush()

    with pytest.warns(DataWarning, match='contains no data, skipping...'):
        with pytest.raises(InvalidQueryError, match='Cutout contains no data!'):
            FITSCutout(test_images, center_coord, cutout_size, single_outfile=True)


def test_fits_cutout_bad_sip(tmpdir, caplog, test_image_bad_sip):
    # Test single image and also conflicting sip keywords
    center_coord = SkyCoord("150.1163213 2.2007", unit='deg')
    cutout_size = [10, 15]
    cutout_file = FITSCutout(test_image_bad_sip, center_coord, cutout_size).write_as_fits(output_dir=tmpdir)[0]
    assert isinstance(cutout_file, str)
    assert "10-x-15" in cutout_file
    with fits.open(cutout_file) as cutout_hdulist:
        assert cutout_hdulist[1].data.shape == (15, 10)

    center_coord = SkyCoord("150.1159 2.2006", unit='deg')
    cutout_size = [10, 15]*u.pixel
    cutout_file = FITSCutout(test_image_bad_sip, center_coord, cutout_size).write_as_fits(output_dir=tmpdir)[0]
    assert isinstance(cutout_file, str)
    assert "10.0pix-x-15.0pix" in cutout_file
    with fits.open(cutout_file) as cutout_hdulist:
        assert cutout_hdulist[1].data.shape == (15, 10)

    cutout_size = [1, 2]*u.arcsec
    cutout_file = FITSCutout(test_image_bad_sip, center_coord, cutout_size, 
                             verbose=True).write_as_fits(output_dir=tmpdir)[0]
    assert isinstance(cutout_file, str)
    assert "1.0arcsec-x-2.0arcsec" in cutout_file
    with fits.open(cutout_file) as cutout_hdulist:
        assert cutout_hdulist[1].data.shape == (33, 17)
    captured = caplog.text
    assert "Original image shape: (50, 50)" in captured
    assert "Image cutout shape: (33, 17)" in captured
    assert "Total time:" in captured

    center_coord = "150.1159 2.2006"
    cutout_size = [10, 15, 20]
    with pytest.warns(InputWarning, match='Too many dimensions in cutout size, only the first two will be used.'):
        cutout_file = FITSCutout(test_image_bad_sip, center_coord, cutout_size).write_as_fits(output_dir=tmpdir)[0]
    assert isinstance(cutout_file, str)
    assert "10-x-15" in cutout_file
    assert "x-20" not in cutout_file


def test_fits_cutout_invalid_params(tmpdir, test_images, center_coord, cutout_size):
    # Invalid limit rounding method
    with pytest.raises(InvalidInputError, match='Limit rounding method invalid is not recognized.'):
        FITSCutout(test_images, center_coord, cutout_size, limit_rounding_method='invalid')

    # Invalid units for cutout size
    cutout_size = 1 * u.m  # meters are not valid
    with pytest.raises(InvalidInputError, match='Cutout size unit meter is not supported.'):
        FITSCutout(test_images, center_coord, cutout_size)


def test_fits_cutout_img_output(tmpdir, test_images, caplog, center_coord, cutout_size):
    # Basic jpg image
    jpg_files = FITSCutout(test_images, center_coord, cutout_size).write_as_img(output_format='jpg', output_dir=tmpdir)
    assert len(jpg_files) == len(test_images)
    with open(jpg_files[0], 'rb') as IMGFLE:
        assert IMGFLE.read(3) == b'\xFF\xD8\xFF'  # JPG

    # Png (single input file, not as list)
    img_files = FITSCutout(test_images[0], center_coord, cutout_size).write_as_img(output_format='png', 
                                                                                   output_dir=tmpdir)
    with open(img_files[0], 'rb') as IMGFLE:
        assert IMGFLE.read(8) == b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'  # PNG
    assert len(img_files) == 1

    # String coordinates and verbose
    center_coord = "150.1163213 2.200973097"
    jpg_files = FITSCutout(test_images, center_coord, cutout_size, verbose=True).write_as_img(output_format='jpg', 
                                                                                              output_dir=tmpdir)
    captured = caplog.text
    assert len(findall('Original image shape', captured)) == 6
    assert 'Total time' in captured


def test_fits_cutout_img_color(tmpdir, test_images, center_coord, cutout_size):
    # Color image
    color_jpg = FITSCutout(test_images[:3], center_coord, cutout_size).write_as_img(output_format='jpg', colorize=True,
                                                                                    output_dir=tmpdir)
    img = Image.open(color_jpg)
    assert img.mode == 'RGB'


def test_fits_cutout_img_memory_only(test_images, center_coord, cutout_size):
    # Save black and white image to memory
    imgs = FITSCutout(test_images[0], center_coord, cutout_size).image_cutouts
    assert isinstance(imgs, list)
    assert len(imgs) == 1
    assert isinstance(imgs[0], Image.Image)

    # Save color image to memory
    color_imgs = FITSCutout(test_images[:3], center_coord, cutout_size).get_image_cutouts(colorize=True)
    assert isinstance(color_imgs, list)
    assert len(color_imgs) == 1
    assert isinstance(color_imgs[0], Image.Image)
    assert color_imgs[0].mode == 'RGB'


def test_fits_cutout_img_errors(tmpdir, test_images, center_coord, cutout_size):
    # Error when too few input images
    with pytest.raises(InvalidInputError):
        FITSCutout(test_images[0], center_coord, cutout_size).get_image_cutouts(colorize=True)

    # Warning when too many input images
    with pytest.warns(InputWarning, match='Too many inputs for a color cutout, only the first three will be used.'):
        color_jpg = FITSCutout(test_images, center_coord, cutout_size).write_as_img(colorize=True, output_dir=tmpdir)
    img = Image.open(color_jpg)
    assert img.mode == 'RGB'

    # Warning when saving image to unsupported image formats
    with pytest.warns(DataWarning, match='Cutout could not be saved in .blp format'):
        FITSCutout(test_images[0], center_coord, cutout_size).write_as_img(output_format='blp', output_dir=tmpdir)

    with pytest.warns(DataWarning, match='Cutout could not be saved in .mpg format'):
        FITSCutout(test_images[:3], center_coord, cutout_size).write_as_img(output_format='mpg', output_dir=tmpdir, 
                                                                            colorize=True)
        
    # Invalid stretch error
    with pytest.raises(InvalidInputError, match='Stretch invalid is not recognized.'):
        FITSCutout(test_images[0], center_coord, cutout_size).write_as_img(stretch='invalid', output_format='png',
                                                                           output_dir=tmpdir)

    # Invalid output format
    with pytest.raises(InvalidInputError, match='Output format .invalid is not supported'):
        FITSCutout(test_images[0], center_coord, cutout_size).write_as_img(output_format='invalid', output_dir=tmpdir)

    # Change first input file to be all zeros
    with fits.open(test_images[0], mode='update') as hdu:
        hdu[0].data[:, :] = 0
        hdu.flush()

    # Warning when outputting non-color images
    with pytest.warns(DataWarning, match='contains no data, skipping...'):
        with pytest.raises(InvalidQueryError, match='Cutout contains no data'):
            FITSCutout(test_images[0], center_coord, cutout_size).write_as_img(output_format='png', output_dir=tmpdir)

    # Error when outputting color image
    with pytest.warns(DataWarning, match='contains no data, skipping...'):
        with pytest.raises(InvalidInputError):
            FITSCutout(test_images[:3], center_coord, cutout_size).write_as_img(colorize=True, output_format='png', 
                                                                                output_dir=tmpdir)
