import os
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy import wcs
from astropy.coordinates import SkyCoord

from .utils_for_test import create_test_ffis
from ..make_cube import CubeFactory
from ..cube_cut import CutoutFactory

def checkcutout(cutfile,pixcrd,world,csize,ecube,eps=1.e-7):
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
    pinput = SkyCoord(world[0],world[1],frame='icrs',unit='deg')
    poutput = SkyCoord(ra_obj,dec_obj,frame='icrs',unit='deg')
    
    dist = pinput.separation(poutput).degree
    assert dist <= eps, "{} separation in primary header {} too large".format(cutfile,dist)
        
    cx = ecube.shape[0]
    cy = ecube.shape[1]
    ntimes = ecube.shape[2]
    tab = hdulist[1].data
    assert len(tab) == ntimes, "{} expected {} entries, found {}".format(cutfile,ntimes,len(tab))
    assert (tab['TIME']==(np.arange(ntimes)+0.5)).all(), "{} some time values are incorrect".format(cutfile)

    check1(tab['FLUX'],x1,x2,y1,y2,ecube[:,:,:,0],'FLUX',cutfile)
    check1(tab['FLUX_ERR'],x1,x2,y1,y2,ecube[:,:,:,1],'FLUX_ERR',cutfile)
    
    return 

def check1(flux,x1,x2,y1,y2,ecube,label,cutfile):
    """Test one of flux or error"""
    cx = ecube.shape[0]
    cy = ecube.shape[1]
    if x1 < 0:
        assert np.isnan(flux[:,:-x1,:]).all(), "{} {} x1 NaN failure".format(cutfile,label)
    
    if y1 < 0:
        assert np.isnan(flux[:,:,:-y1]).all(), "{} {} y1 NaN failure".format(cutfile,label)
        
    if x2>=cx:
        assert np.isnan(flux[:,-(x2-cx+1):,:]).all(), "{} {} x2 NaN failure".format(cutfile,label)
        
    if y2>=cy:
        assert np.isnan(flux[:,:,-(y2-cy+1):]).all(), "{} {} y2 NaN failure".format(cutfile,label)
        
    x1c = max(x1,0)
    y1c = max(y1,0)
    x2c = min(x2,cx-1)
    y2c = min(y2,cy-1)
    scube = ecube[x1c:x2c,y1c:y2c,:]
    sflux = np.moveaxis(flux[:,x1c-x1:x2c-x1,y1c-y1:y2c-y1],0,-1)
    assert (scube==sflux).all(), "{} {} comparison failure".format(cutfile,label)

    return



def test_cube_cutout(tmpdir):
    """
    Testing the cube cutout functionality.
    """

    # Making the test cube
    cube_maker = CubeFactory()
    
    img_sz = 10
    num_im = 100
    
    ffi_files = create_test_ffis(img_sz, num_im)
    cube_file = cube_maker.make_cube(ffi_files, "make_cube-test-cube", verbose=False)

    # Read one of the input images to get the WCS
    img_header = fits.getheader(ffi_files[0], 1)
    cube_wcs = wcs.WCS(img_header)

    # get pixel positions at edges and center of image
    # somewhat cryptic one-liner to get the grid of points
    pval = np.array([0,img_sz//2,img_sz-1],dtype=np.float)
    pixcrd = pval[np.transpose(np.reshape(np.mgrid[0:3,0:3],(2,9)))]
    
    # add one more giant cutout that goes off all 4 edges
    pixcrd = np.append(pixcrd,pixcrd[4].reshape(1,2),axis=0)

    # getting the world coordinates
    world_coords = cube_wcs.all_pix2world(pixcrd,0)

    # Getting the cutouts
    cutbasename = 'make_cube-cutout{}.fits'
    cutlist = [cutbasename.format(i) for i in range(len(world_coords))]
    csize = [img_sz//2]*len(world_coords)
    csize[-1] = img_sz+5
    for i, v in enumerate(world_coords):
        coord = SkyCoord(v[0],v[1],frame='icrs',unit='deg')
        cut = CutoutFactory().cube_cut(cube_file,coord,csize[i],target_pixel_file=cutlist[i],verbose=False)

    # expected values for cube
    ecube = np.zeros((img_sz,img_sz,num_im,2))
    plane = np.arange(img_sz*img_sz,dtype=np.float32).reshape((img_sz,img_sz))
    for i in range(num_im):
        ecube[:,:,i,0] = -plane
        ecube[:,:,i,1] = plane
        plane += img_sz*img_sz

    # Doing the actual checking
    for i, cutfile in enumerate(cutlist):
        checkcutout(cutfile,pixcrd[i],world_coords[i],csize[i],ecube)
           
    
