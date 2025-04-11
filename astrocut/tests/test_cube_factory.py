from pathlib import Path
import numpy as np
import pytest

from astropy.io import fits
from astropy.table import Table
from re import findall

from astrocut.exceptions import DataWarning, InvalidInputError

from .utils_for_test import create_test_ffis
from ..cube_factory import CubeFactory
from ..tica_cube_factory import TicaCubeFactory


@pytest.fixture
def img_size():
    """ Fixture for the size of the test images to be created. """
    return 10


@pytest.fixture
def num_images():
    """ Fixture for the number of test images to be created. """
    return 100


def test_make_cube(tmpdir, img_size, num_images, tmp_path):
    """
    Testing the make cube functionality by making a bunch of test FFIs, 
    making the cube, and checking the results.
    """
    cube_maker = CubeFactory()
    tmp_path = Path(tmpdir)
    
    # Create test FFIs
    ffi_files = create_test_ffis(img_size, num_images, dir_name=tmpdir)
    cube_path = tmp_path / 'out_dir' / 'test_cube.fits'

    # Generate cube
    cube_file = cube_maker.make_cube(ffi_files, cube_path, verbose=False)

    # Open FITS file and extract cube data
    with fits.open(cube_file) as hdu:
        cube = hdu[1].data
        tab = Table(hdu[2].data)
        filenames = np.array([Path(x).name for x in ffi_files])
    
        # Expected cube shape and values
        ecube = np.zeros((img_size, img_size, num_images, 2))
        plane = np.arange(img_size*img_size, dtype=np.float32).reshape((img_size, img_size))
        assert cube.shape == ecube.shape, 'Mismatch between cube shape and expected shape'

        for i in range(num_images):
            ecube[:, :, i, 0] = -plane
            ecube[:, :, i, 1] = plane
            plane += img_size * img_size
        assert np.all(cube == ecube), 'Cube values do not match expected values'
        
        assert np.all(tab['TSTART'] == np.arange(num_images)), 'TSTART mismatch in table'
        assert np.all(tab['TSTOP'] == np.arange(num_images)+1), 'TSTOP mismatch in table'
        assert np.all(tab['FFI_FILE'] == np.array(filenames)), 'FFI_FILE mismatch in table'
    

def test_make_and_update_cube(tmpdir, img_size, num_images):
    """
    Testing the make cube and update cube functionality for TICACubeFactory by making a bunch of test FFIs, 
    making the cube with first half of the FFIs, updating the same cube with the second half,
    and checking the results.
    """
    cube_maker = TicaCubeFactory()
    tmp_path = Path(tmpdir)
    
    # Create test FFIs
    ffi_files = create_test_ffis(img_size, num_images, product='TICA', dir_name=tmpdir)
    cube_path = tmp_path / 'out_dir' / 'test_update_cube.fits'

    # Generate cube
    cube_file = cube_maker.make_cube(ffi_files[:num_images // 2], cube_path, verbose=False)

    with fits.open(cube_file) as hdu:
        cube = hdu[1].data

        # Expected values for cube before update_cube
        ecube = np.zeros((img_size, img_size, num_images // 2, 1))
        plane = np.arange(img_size*img_size, dtype=np.float32).reshape((img_size, img_size))

        assert cube.shape == ecube.shape, 'Mismatch between cube shape and expected shape'

        for i in range(num_images // 2):
            ecube[:, :, i, 0] = -plane
            # we don't need to test error array because TICA doesnt come with error arrays
            # so index 1 will always be blank
            # ecube[:, :, i, 1] = plane
            plane += img_size * img_size

        assert np.all(cube == ecube), 'Cube values do not match expected values'

    # Update cube
    cube_file = cube_maker.update_cube(ffi_files[num_images // 2:], cube_file, verbose=False)

    with fits.open(cube_file) as hdu:
        cube = hdu[1].data
        tab = Table(hdu[2].data)
        filenames = np.array([Path(x).name for x in ffi_files])
    
        # Expected values for cube after update_cube
        ecube = np.zeros((img_size, img_size, num_images, 1))
        plane = np.arange(img_size*img_size, dtype=np.float32).reshape((img_size, img_size))

        assert cube.shape == ecube.shape, 'Mismatch between cube shape and expected shape'

        for i in range(num_images):
            ecube[:, :, i, 0] = -plane
            # we don't need to test error array because TICA doesnt come with error arrays
            # so index 1 will always be blank
            # ecube[:, :, i, 1] = plane
            plane += img_size * img_size

        assert np.all(cube == ecube), 'Cube values do not match expected values'
        assert np.all(tab['STARTTJD'] == np.arange(num_images)), 'STARTTJD mismatch in table'
        assert np.all(tab['ENDTJD'] == np.arange(num_images)+1), 'ENDTJD mismatch in table'
        assert np.all(tab['FFI_FILE'] == np.array(filenames)), 'FFI_FILE mismatch in table'

    # Fail if trying to update a cube with no new files
    with pytest.raises(InvalidInputError, match='No new images were found'):
        with pytest.warns(DataWarning, match='Removed duplicate file'):
            cube_maker.update_cube(ffi_files[:1], cube_file)
    

def test_iteration(tmpdir, caplog):
    """
    Testing cubes made with different numbers of iterations against each other.
    """
    cube_maker = CubeFactory()
    tmp_path = Path(tmpdir)
    img_size = 1000
    num_images = 10

    # Create test FFIs
    ffi_files = create_test_ffis(img_size, num_images, dir_name=tmpdir)

    # Single iteration (higher memory usage)
    cube_file_1 = cube_maker.make_cube(ffi_files, tmp_path / 'iterated_cube_1.fits',
                                       max_memory=0.5, verbose=True)
    assert len(findall('Completed block', caplog.text)) == 1, 'Incorrect number of iterations'
    assert len(findall('Completed file', caplog.text)) == num_images, 'Incorrect number of complete files'
    caplog.clear()

    # Multiple iterations (lower memory usage)
    cube_file_2 = cube_maker.make_cube(ffi_files, tmp_path / 'iterated_cube_2.fits',
                                       max_memory=0.05, verbose=True)
    print(len(findall('Completed block', caplog.text)))
    assert len(findall('Completed block', caplog.text)) == 2, 'Incorrect number of iterations'
    assert len(findall('Completed file', caplog.text)) == num_images * 2, 'Incorrect number of complete files'

    # Open FITS files and compare cubes
    with fits.open(cube_file_1) as hdu_1, fits.open(cube_file_2) as hdu_2:
        cube_1 = hdu_1[1].data
        cube_2 = hdu_2[1].data

        assert cube_1.shape == cube_2.shape, 'Mismatch between cube shape for 1 vs 2 iterations'
        assert np.all(cube_1 == cube_2), 'Cubes made in 1 vs 2 iterations do not match'

        # Expected values for cube
        ecube = np.zeros((img_size, img_size, num_images, 2))
        plane = np.arange(img_size * img_size, dtype=np.float32).reshape((img_size, img_size))
        assert cube_1.shape == ecube.shape, 'Mismatch between cube shape and expected shape'

        for i in range(num_images):
            ecube[:, :, i, 0] = -plane
            ecube[:, :, i, 1] = plane
            plane += img_size * img_size

        assert np.all(cube_1 == ecube), 'Cube values do not match expected values'


@pytest.mark.parametrize('ffi_type', ['TICA', 'SPOC'])
def test_invalid_inputs(tmpdir, ffi_type, img_size, num_images):
    """
    Test that an error is raised when users attempt to make cubes with an invalid file type.
    """
    # Assigning some variables
    product = 'TICA' if ffi_type == 'TICA' else 'SPOC'
    value_error = ('One or more incorrect file types were input. Please input TICA FFI files when using '
                   '``TicaCubeFactory``, and SPOC FFI files when using ``CubeFactory``.')
    
    # Create test FFI files
    ffi_files = create_test_ffis(img_size=img_size, 
                                 num_images=num_images,
                                 dir_name=tmpdir, 
                                 product=product)
    
    # Create opposite cube factory of input
    cube_maker = CubeFactory() if ffi_type == 'TICA' else TicaCubeFactory()

    # Should raise a Value Error due to incorrect file type
    with pytest.raises(ValueError, match=value_error):
        cube_maker.make_cube(ffi_files)

    # Fail if trying to update a cube file that doesn't exist
    new_ffi_files = create_test_ffis(img_size=10, 
                                     num_images=10,
                                     dir_name=tmpdir, 
                                     product=product)
    with pytest.raises(InvalidInputError, match='Cube file was not found'):
        cube_maker.update_cube(new_ffi_files, 'non_existent_file.fits')
