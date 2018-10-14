import numpy as np

from astropy.io import fits
from astropy.table import Table

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
    
    ffi_files = create_test_ffis(img_sz, num_im)
    cube_file = cube_maker.make_cube(ffi_files, "make_cube-test-cube", verbose=False)

    hdu = fits.open(cube_file)
    cube = hdu[1].data
    
    # expected values for cube
    ecube = np.zeros((img_sz,img_sz,num_im,2))
    plane = np.arange(img_sz*img_sz,dtype=np.float32).reshape((img_sz,img_sz))
    assert cube.shape == ecube.shape, "Mismatch between cube shape and expected shape"

    for i in range(num_im):
        ecube[:,:,i,0] = -plane
        ecube[:,:,i,1] = plane
        plane += img_sz*img_sz

    assert np.alltrue(cube == ecube), "Cube values do not match expected values"
        
    tab = Table(hdu[2].data)
    assert np.alltrue(tab['TSTART'] == np.arange(num_im)), "TSTART mismatch in table"
    assert np.alltrue(tab['TSTOP'] == np.arange(num_im)+1), "TSTOP mismatch in table"
    assert np.alltrue(tab['FFI_FILE'] == np.array(ffi_files)), "FFI_FILE mismatch in table"

    hdu.close()
    
