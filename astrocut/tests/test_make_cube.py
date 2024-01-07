import numpy as np
import os
import pytest

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations
from re import findall
from os import path

from .utils_for_test import create_test_ffis
from ..make_cube import CubeFactory, TicaCubeFactory


def test_make_cube(tmpdir):
    """
    Testing the make cube functionality by making a bunch of test FFIs, 
    making the cube, and checking the results.
    """

    cube_maker = CubeFactory()
    
    img_sz = 10
    num_im = 100
    
    ffi_files = create_test_ffis(img_sz, num_im, dir_name=tmpdir)
    cube_file = cube_maker.make_cube(ffi_files, path.join(tmpdir, "out_dir", "test_cube.fits"), verbose=False)
    hdu = fits.open(cube_file)
    cube = hdu[1].data
    
    # expected values for cube
    ecube = np.zeros((img_sz, img_sz, num_im, 2))
    plane = np.arange(img_sz*img_sz, dtype=np.float32).reshape((img_sz, img_sz))
    assert cube.shape == ecube.shape, "Mismatch between cube shape and expected shape"

    for i in range(num_im):
        ecube[:, :, i, 0] = -plane
        ecube[:, :, i, 1] = plane
        plane += img_sz*img_sz
    assert np.all(cube == ecube), "Cube values do not match expected values"
    
    tab = Table(hdu[2].data)
    assert np.all(tab['TSTART'] == np.arange(num_im)), "TSTART mismatch in table"
    assert np.all(tab['TSTOP'] == np.arange(num_im)+1), "TSTOP mismatch in table"

    filenames = np.array([path.split(x)[1] for x in ffi_files])
    assert np.all(tab['FFI_FILE'] == np.array(filenames)), "FFI_FILE mismatch in table"

    hdu.close()
    

def test_make_and_update_cube(tmpdir):
    """
    Testing the make cube and update cube functionality for TICACubeFactory by making a bunch of test FFIs, 
    making the cube with first half of the FFIs, updating the same cube with the second half,
    and checking the results.
    """

    cube_maker = TicaCubeFactory()

    img_sz = 10
    num_im = 100
    
    ffi_files = create_test_ffis(img_sz, num_im, product='TICA', dir_name=tmpdir)

    cube_file = path.join(os.getcwd(), "out_dir", "test_update_cube.fits")

    # Testing make_cube
    
    cube_maker.make_cube(ffi_files[0:num_im // 2], cube_file, verbose=False)

    hdu = fits.open(cube_file)
    cube = hdu[1].data

    # expected values for cube before update_cube
    ecube = np.zeros((img_sz, img_sz, num_im // 2, 1))
    plane = np.arange(img_sz*img_sz, dtype=np.float32).reshape((img_sz, img_sz))

    assert cube.shape == ecube.shape, "Mismatch between cube shape and expected shape"

    for i in range(num_im // 2):
        ecube[:, :, i, 0] = -plane
        # we don't need to test error array because TICA doesnt come with error arrays
        # so index 1 will always be blank
        # ecube[:, :, i, 1] = plane
        plane += img_sz*img_sz

    assert np.all(cube == ecube), "Cube values do not match expected values"

    hdu.close()

    # Testing update_cube

    cube_file = cube_maker._update_cube(ffi_files[num_im // 2:], cube_file, verbose=False)

    hdu = fits.open(cube_file)
    cube = hdu[1].data
    
    # expected values for cube after update_cube
    ecube = np.zeros((img_sz, img_sz, num_im, 1))
    plane = np.arange(img_sz*img_sz, dtype=np.float32).reshape((img_sz, img_sz))

    assert cube.shape == ecube.shape, "Mismatch between cube shape and expected shape"

    for i in range(num_im):
        ecube[:, :, i, 0] = -plane
        # we don't need to test error array because TICA doesnt come with error arrays
        # so index 1 will always be blank
        # ecube[:, :, i, 1] = plane
        plane += img_sz*img_sz

    assert np.all(cube == ecube), "Cube values do not match expected values"

    tab = Table(hdu[2].data)
    assert np.all(tab['STARTTJD'] == np.arange(num_im)), "STARTTJD mismatch in table"
    assert np.all(tab['ENDTJD'] == np.arange(num_im)+1), "ENDTJD mismatch in table"

    filenames = np.array([path.split(x)[1] for x in ffi_files])
    assert np.all(tab['FFI_FILE'] == np.array(filenames)), "FFI_FILE mismatch in table"

    hdu.close()

    os.remove(cube_file)
    

def test_iteration(tmpdir, capsys):
    """
    Testing cubes made with different numbers of iterations against each other.
    """

    cube_maker = CubeFactory()

    img_sz = 1000
    num_im = 10

    ffi_files = create_test_ffis(img_sz, num_im, dir_name=tmpdir)

    # Should produce cube in single iteration
    cube_file_1 = cube_maker.make_cube(ffi_files, path.join(tmpdir, "iterated_cube_1.fits"),
                                       max_memory=0.5, verbose=True)
    captured = capsys.readouterr()
    assert len(findall("Completed block", captured.out)) == 1, "Incorrect number of iterations"
    assert len(findall("Completed file", captured.out)) == num_im, "Incorrect number of complete files"

    cube_file_2 = cube_maker.make_cube(ffi_files, path.join(tmpdir, "iterated_cube_2.fits"),
                                       max_memory=0.05, verbose=True)
    captured = capsys.readouterr()
    assert len(findall("Completed block", captured.out)) == 2, "Incorrect number of iterations"
    assert len(findall("Completed file", captured.out)) == num_im*2, "Incorrect number of complete files"
    
    hdu_1 = fits.open(cube_file_1)
    cube_1 = hdu_1[1].data

    hdu_2 = fits.open(cube_file_2)
    cube_2 = hdu_2[1].data

    assert cube_1.shape == cube_2.shape, "Mismatch between cube shape for 1 vs 2 iterations"
    assert np.all(cube_1 == cube_2), "Cubes made in 1 vs 2 iterations do not match"

    # expected values for cube
    ecube = np.zeros((img_sz, img_sz, num_im, 2))
    plane = np.arange(img_sz*img_sz, dtype=np.float32).reshape((img_sz, img_sz))
    assert cube_1.shape == ecube.shape, "Mismatch between cube shape and expected shape"

    for i in range(num_im):
        ecube[:, :, i, 0] = -plane
        ecube[:, :, i, 1] = plane
        plane += img_sz*img_sz

    assert np.all(cube_1 == ecube), "Cube values do not match expected values"


@pytest.mark.parametrize("ffi_type", ["TICA", "SPOC"])
def test_invalid_inputs(tmpdir, ffi_type):

    coordinates = SkyCoord(289.0979, -29.3370, unit="deg")

    # Assigning some variables
    target_name = "TICA FFI" if ffi_type == "TICA" else "TESS FFI"
    value_error = "One or more incorrect file types were input. Please input TICA FFI files when using\
                   ``TicaCubeFactory``, and SPOC FFI files when using ``CubeFactory``."

    # Getting TESS sector 27 observations for the given coordinate
    observations = Observations.query_criteria(coordinates=coordinates,
                                               target_name=target_name,
                                               dataproduct_type="image",
                                               sequence_number=27)
    
    # Getting a list of products. Keeping it small so we don't have to download so many.
    products = Observations.get_product_list(observations[0])[:2]

    manifest = Observations.download_products(products, download_dir=str(tmpdir))

    if ffi_type == "TICA":
        cube_maker = CubeFactory()
    elif ffi_type == "SPOC":
        cube_maker = TicaCubeFactory()

    with pytest.raises(ValueError) as error_msg:
        cube_maker.make_cube(manifest["Local Path"])
    assert value_error in str(error_msg.value)

    

    

    
