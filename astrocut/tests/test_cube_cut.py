import pytest
import numpy as np
from os import path

from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
import astropy.units as u

from .utils_for_test import create_test_ffis
from ..make_cube import CubeFactory
from ..cube_cut import CutoutFactory
from ..exceptions import InvalidQueryError, InputWarning


def checkcutout(cutfile, pixcrd, world, csize, ecube, eps=1.e-7):
    """Check FITS cutout for correctness
    
    Checks RA_OBJ/DEC_OBJ in primary header, and TIME, FLUX, and
    FLUX_ERR in table.
    
    Inputs:
    cutfile  Name of FITS cutout file
    pixcrd   [2] pixel coordinates for cutout center [cy,cx]
    world    [2] RA, Dec in degrees for cutout center
    csize    Integer size of cutout (probably should be odd)
    ecube    Simulated data cube
    eps      Maximum allowed distance offset in degrees
    Returns True on success, False on failure
    """
    
    ix = int(pixcrd[1])
    iy = int(pixcrd[0])
    x1 = ix - csize//2
    x2 = x1 + csize - 1
    y1 = iy - csize//2
    y2 = y1 + csize - 1
    hdulist = fits.open(cutfile)
    ra_obj = hdulist[0].header['RA_OBJ']
    dec_obj = hdulist[0].header['DEC_OBJ']
    pinput = SkyCoord(world[0], world[1], frame='icrs', unit='deg')
    poutput = SkyCoord(ra_obj, dec_obj, frame='icrs', unit='deg')
    
    dist = pinput.separation(poutput).degree
    assert dist <= eps, "{} separation in primary header {} too large".format(cutfile, dist)
        
    ntimes = ecube.shape[2]
    tab = hdulist[1].data
    assert len(tab) == ntimes, "{} expected {} entries, found {}".format(cutfile, ntimes, len(tab))
    assert (tab['TIME'] == (np.arange(ntimes)+0.5)).all(), "{} some time values are incorrect".format(cutfile)

    check1(tab['FLUX'], x1, x2, y1, y2, ecube[:, :, :, 0], 'FLUX', cutfile)
    check1(tab['FLUX_ERR'], x1, x2, y1, y2, ecube[:, :, :, 1], 'FLUX_ERR', cutfile)
    
    # Regression test for PR #6
    assert hdulist[2].data.dtype.type == np.int32

    return 


def check1(flux, x1, x2, y1, y2, ecube, label, cutfile):
    """Test one of flux or error"""
    cx = ecube.shape[0]
    cy = ecube.shape[1]
    if x1 < 0:
        assert np.isnan(flux[:, :-x1, :]).all(), "{} {} x1 NaN failure".format(cutfile, label)
    
    if y1 < 0:
        assert np.isnan(flux[:, :, :-y1]).all(), "{} {} y1 NaN failure".format(cutfile, label)
        
    if x2 >= cx:
        assert np.isnan(flux[:, -(x2-cx+1):, :]).all(), "{} {} x2 NaN failure".format(cutfile, label)
        
    if y2 >= cy:
        assert np.isnan(flux[:, :, -(y2-cy+1):]).all(), "{} {} y2 NaN failure".format(cutfile, label)
        
    x1c = max(x1, 0)
    y1c = max(y1, 0)
    x2c = min(x2, cx-1)
    y2c = min(y2, cy-1)
    scube = ecube[x1c:x2c, y1c:y2c, :]
    sflux = np.moveaxis(flux[:, x1c-x1:x2c-x1, y1c-y1:y2c-y1], 0, -1)
    assert (scube == sflux).all(), "{} {} comparison failure".format(cutfile, label)

    return


def test_cube_cutout(tmpdir):
    """
    Testing the cube cutout functionality.
    """

    # Making the test cube
    cube_maker = CubeFactory()
    
    img_sz = 10
    num_im = 100
    
    ffi_files = create_test_ffis(img_sz, num_im, dir_name=tmpdir)
    cube_file = cube_maker.make_cube(ffi_files, path.join(tmpdir, "test_cube.fits"), verbose=False)

    # Read one of the input images to get the WCS
    img_header = fits.getheader(ffi_files[0], 1)
    cube_wcs = wcs.WCS(img_header)

    # get pixel positions at edges and center of image
    # somewhat cryptic one-liner to get the grid of points
    pval = np.array([0, img_sz//2, img_sz-1], dtype=np.float)
    pixcrd = pval[np.transpose(np.reshape(np.mgrid[0:3, 0:3], (2, 9)))]
    
    # add one more giant cutout that goes off all 4 edges
    pixcrd = np.append(pixcrd, pixcrd[4].reshape(1, 2), axis=0)

    # getting the world coordinates
    world_coords = cube_wcs.all_pix2world(pixcrd, 0)

    # Getting the cutouts
    cutbasename = 'make_cube_cutout_{}.fits'
    cutlist = [path.join(tmpdir, cutbasename.format(i)) for i in range(len(world_coords))]
    csize = [img_sz//2]*len(world_coords)
    csize[-1] = img_sz+5
    for i, v in enumerate(world_coords):
        coord = SkyCoord(v[0], v[1], frame='icrs', unit='deg')
        CutoutFactory().cube_cut(cube_file, coord, csize[i], target_pixel_file=cutlist[i],
                                 output_path=tmpdir, verbose=False)

    # expected values for cube
    ecube = np.zeros((img_sz, img_sz, num_im, 2))
    plane = np.arange(img_sz*img_sz, dtype=np.float32).reshape((img_sz, img_sz))
    for i in range(num_im):
        ecube[:, :, i, 0] = -plane
        ecube[:, :, i, 1] = plane
        plane += img_sz*img_sz

    # Doing the actual checking
    for i, cutfile in enumerate(cutlist):
        checkcutout(cutfile, pixcrd[i], world_coords[i], csize[i], ecube)
           
    
        
def test_cutout_extras(tmpdir):

    # Making the test cube
    cube_maker = CubeFactory()
    
    img_sz = 10
    num_im = 100
    
    ffi_files = create_test_ffis(img_sz, num_im)
    cube_file = cube_maker.make_cube(ffi_files, path.join(tmpdir, "test_cube.fits"), verbose=False)

    # Making the cutout
    myfactory = CutoutFactory()
    coord = "256.88 6.38"

    ###########################
    # Test  _parse_table_info #
    ###########################
    cutout_size = [5, 3]
    out_file = myfactory.cube_cut(cube_file, coord, cutout_size,
                                  output_path=path.join(tmpdir, "out_dir"), verbose=False)
    assert "256.880000_6.380000_5x3_astrocut.fits" in out_file

    assert isinstance(myfactory.cube_wcs, wcs.WCS)
    ra, dec = myfactory.cube_wcs.wcs.crval
    assert round(ra, 4) == 250.3497
    assert round(dec, 4) == 2.2809

    # checking on the center coordinate too
    coord = SkyCoord(256.88, 6.38, frame='icrs', unit='deg')
    assert myfactory.center_coord.separation(coord) == 0

    ############################
    # Test  _get_cutout_limits #
    ############################
    xmin, xmax = myfactory.cutout_lims[0]
    ymin, ymax = myfactory.cutout_lims[1]
    
    assert (xmax-xmin) == cutout_size[0]
    assert (ymax-ymin) == cutout_size[1]

    cutout_size = [5*u.pixel, 7*u.pixel]
    out_file = myfactory.cube_cut(cube_file, coord, cutout_size, verbose=False)
    assert "256.880000_6.380000_5x7_astrocut.fits" in out_file
    
    xmin, xmax = myfactory.cutout_lims[0]
    ymin, ymax = myfactory.cutout_lims[1]
    
    assert (xmax-xmin) == cutout_size[0].value
    assert (ymax-ymin) == cutout_size[1].value

    cutout_size = [3*u.arcmin, 5*u.arcmin]
    out_file = myfactory.cube_cut(cube_file, coord, cutout_size, verbose=False)
    assert "256.880000_6.380000_8x15_astrocut.fits" in out_file
    
    xmin, xmax = myfactory.cutout_lims[0]
    ymin, ymax = myfactory.cutout_lims[1]
    
    assert (xmax-xmin) == 8
    assert (ymax-ymin) == 15

    cutout_size = [1*u.arcsec, 5*u.arcsec]
    out_file = myfactory.cube_cut(cube_file, coord, cutout_size, verbose=False)
    assert "256.880000_6.380000_1x1_astrocut.fits" in out_file
    
    xmin, xmax = myfactory.cutout_lims[0]
    ymin, ymax = myfactory.cutout_lims[1]
    
    assert (xmax-xmin) == 1
    assert (ymax-ymin) == 1

    #############################
    # Test _get_full_cutout_wcs #
    #############################
    cutout_size = [5, 3]
    out_file = myfactory.cube_cut(cube_file, coord, cutout_size, verbose=False)

    cutout_wcs_full = myfactory._get_full_cutout_wcs(fits.getheader(cube_file, 2))
    assert (cutout_wcs_full.wcs.crpix == [1045 - myfactory.cutout_lims[0, 0],
                                          1001 - myfactory.cutout_lims[1, 0]]).all()    

    ########################
    # Test _fit_cutout_wcs #
    ########################
    max_dist, sigma = myfactory._fit_cutout_wcs(cutout_wcs_full, (3, 5))
    assert max_dist.deg < 1e-05
    assert sigma < 1e-05

    cry, crx = myfactory.cutout_wcs.wcs.crpix
    assert round(cry) == 3
    assert round(crx) == 2
    

    ##########################
    # Test target pixel file #
    ##########################

    # Testing the cutout content is in test_cube_cutout
    # this tests that the format of the tpf is what it should be
    tpf = fits.open(out_file)

    assert tpf[0].header["ORIGIN"] == 'STScI/MAST'

    tpf_table = tpf[1].data
    assert len(tpf_table.columns) == 12
    assert "TIME" in tpf_table.columns.names
    assert "FLUX" in tpf_table.columns.names
    assert "FLUX_ERR" in tpf_table.columns.names
    assert "FFI_FILE" in tpf_table.columns.names

    cutout_img = tpf_table[0]['FLUX']
    assert cutout_img.shape == (3, 5)
    assert cutout_img.dtype.name == 'float32'

    aperture = tpf[2].data
    assert aperture.shape == (3, 5)
    assert aperture.dtype.name == 'int32'

    tpf.close()


def test_exceptions(tmpdir):
    """
    Testing various error conditions.
    """
    
    # Making the test cube
    cube_maker = CubeFactory()
    
    img_sz = 10
    num_im = 100
    
    ffi_files = create_test_ffis(img_sz, num_im)
    cube_file = cube_maker.make_cube(ffi_files, path.join(tmpdir, "test_cube.fits"), verbose=False)

    # Setting up
    myfactory = CutoutFactory()
   
    hdu = fits.open(cube_file)
    cube_table = hdu[2].data
     
    # Testing when none of the FFIs have good wcs info
    cube_table["WCSAXES"] = 0
    with pytest.raises(Exception, match='No FFI rows contain valid WCS keywords.') as e:
        myfactory._parse_table_info(cube_table)
        assert e.type is wcs.NoWcsKeywordsFoundError
    cube_table["WCSAXES"] = 2

    # Testing when nans are present 
    myfactory._parse_table_info(cube_table)
    wcs_orig = myfactory.cube_wcs
    cube_table["BARYCORR"] = np.nan
    myfactory._parse_table_info(cube_table)
    assert wcs_orig.to_header_string() == myfactory.cube_wcs.to_header_string()

    hdu.close()

    # Testing various off the cube inputs
    myfactory.center_coord = SkyCoord("50.91092264 6.40588255", unit='deg')
    with pytest.raises(Exception, match='Cutout location is not in cube footprint!') as e:
        myfactory._get_cutout_limits(np.array([5, 5]))
        assert e.type is InvalidQueryError
         
    myfactory.center_coord = SkyCoord("257.91092264 6.40588255", unit='deg')
    with pytest.raises(Exception, match='Cutout location is not in cube footprint!') as e:
        myfactory._get_cutout_limits(np.array([5, 5]))
        assert e.type is InvalidQueryError


    # Testing the WCS fitting function
    distmax, sigma = myfactory._fit_cutout_wcs(myfactory.cube_wcs, (100, 100))
    assert distmax.deg < 0.003
    assert sigma < 0.03

    distmax, sigma = myfactory._fit_cutout_wcs(myfactory.cube_wcs, (1, 100))
    assert distmax.deg < 0.003
    assert sigma < 0.03

    distmax, sigma = myfactory._fit_cutout_wcs(myfactory.cube_wcs, (100, 2))
    assert distmax.deg < 0.03
    assert sigma < 0.03

    myfactory.center_coord = SkyCoord("256.38994124 4.88986771", unit='deg')
    myfactory._get_cutout_limits(np.array([5, 500]))

    hdu = fits.open(cube_file)
    cutout_wcs = myfactory._get_full_cutout_wcs(hdu[2].header)
    hdu.close()

    distmax, sigma = myfactory._fit_cutout_wcs(cutout_wcs, (200, 200))
    assert distmax.deg < 0.004
    assert sigma < 0.2

    distmax, sigma = myfactory._fit_cutout_wcs(cutout_wcs, (100, 5))
    assert distmax.deg < 0.003
    assert sigma < 0.003

    distmax, sigma = myfactory._fit_cutout_wcs(cutout_wcs, (3, 100))
    assert distmax.deg < 0.003
    assert sigma < 0.003
    

def test_inputs(tmpdir, capsys):
    """
    Testing with different user input types/combos. And verbose.
    """

    # Making the test cube
    cube_maker = CubeFactory()
    
    img_sz = 10
    num_im = 100
    
    ffi_files = create_test_ffis(img_sz, num_im)
    cube_file = cube_maker.make_cube(ffi_files, path.join(tmpdir, "test_cube.fits"), verbose=False)

    # Setting up
    myfactory = CutoutFactory()
    coord = "256.88 6.38"

    cutout_size = [5, 3]*u.pixel
    cutout_file = myfactory.cube_cut(cube_file, coord, cutout_size, output_path=tmpdir, verbose=True)
    captured = capsys.readouterr()
    assert "Image cutout cube shape: (100, 3, 5)" in captured.out
    assert "Using WCS from row 50 out of 100" in captured.out
    assert "Cutout center coordinate: 256.88,6.38" in captured.out
    assert "5x3" in cutout_file

    cutout_size = [5, 3]*u.arcmin
    cutout_file = myfactory.cube_cut(cube_file, coord, cutout_size, output_path=tmpdir, verbose=False)
    assert "14x9" in cutout_file

    
    cutout_size = [5, 3, 9]*u.pixel
    with pytest.warns(InputWarning):
        cutout_file = myfactory.cube_cut(cube_file, coord, cutout_size, output_path=tmpdir, verbose=False)
    assert "5x3" in cutout_file
    assert "x9" not in cutout_file
