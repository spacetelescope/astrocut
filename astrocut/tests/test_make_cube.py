import numpy as np

from astropy.io import fits
from astropy.table import Table
from re import findall
from os import path

from .utils_for_test import create_test_ffis
from ..make_cube import CubeFactory


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

    assert np.alltrue(cube == ecube), "Cube values do not match expected values"
        
    tab = Table(hdu[2].data)
    assert np.alltrue(tab['TSTART'] == np.arange(num_im)), "TSTART mismatch in table"
    assert np.alltrue(tab['TSTOP'] == np.arange(num_im)+1), "TSTOP mismatch in table"

    filenames = np.array([path.split(x)[1] for x in ffi_files])
    assert np.alltrue(tab['FFI_FILE'] == np.array(filenames)), "FFI_FILE mismatch in table"

    hdu.close()
    

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
    assert np.alltrue(cube_1 == cube_2), "Cubes made in 1 vs 2 iterations do not match"

    # expected values for cube
    ecube = np.zeros((img_sz, img_sz, num_im, 2))
    plane = np.arange(img_sz*img_sz, dtype=np.float32).reshape((img_sz, img_sz))
    assert cube_1.shape == ecube.shape, "Mismatch between cube shape and expected shape"

    for i in range(num_im):
        ecube[:, :, i, 0] = -plane
        ecube[:, :, i, 1] = plane
        plane += img_sz*img_sz

    assert np.alltrue(cube_1 == ecube), "Cube values do not match expected values"

    
    

    

    
