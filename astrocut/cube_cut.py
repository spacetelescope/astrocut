import numpy as np
import astropy.units as u

from astropy.io import fits
from astropy.table import Table,Column
from astropy.coordinates import SkyCoord
from astropy import wcs

from time import time
from copy import deepcopy

import os


def getCubeWcs(tableHeader,tableRow):
    """
    Takes the header and one entry from the cube table of data and returns 
    a WCS object that encalpsulates the given WCS information.
    """
    
    wcsHeader = fits.header.Header()

    for headerKey, headerVal in tableHeader.items():
        if not 'TTYPE' in headerKey:
            continue
        colNum = int(headerKey[5:])-1
        if 'NAXIS' in headerVal:
            wcsHeader[headerVal] =  int(tableRow[colNum])
        else:
            wcsHeader[headerVal] =  tableRow[colNum]

    return wcs.WCS(wcsHeader)


def get_cutout_limits(center_coord, cutout_size, cube_wcs):
    """
    Takes the center coordinates, cutout size, and the wcs from
    which the cutout is being taken and returns the x and y pixel limits
    for the cutout.

    cutout_size : array
         [nx,ny] in with ints (pixels) or astropy quantities
    """

    center_pixel = center_coord.to_pixel(cube_wcs)

    lims = np.zeros((2,2),dtype=int)

    for axis, size in enumerate(cutout_size):
        
        if not isinstance(size, u.Quantity): # assume pixels
            dim = size/2
        elif size.unit == u.pixel: # also pixels
            dim = size.value/2
        elif size.unit.physical_type == 'angle':
            pixel_scale = u.Quantity(proj_plane_pixel_scales(wcs)[axis], wcs.wcs.cunit[axis])
            dim = (size / pixel_scale).decompose()/2

        lims[axis,0] = int(np.round(center_pixel[axis] - dim))
        lims[axis,1] = int(np.round(center_pixel[axis] + dim))

        # Adjust bounds if necessary
        lims[axis,0] = lims[axis,0] if lims[axis,0] >=0 else 0
        lims[axis,1] = lims[axis,1]  if lims[axis,1] < cube_wcs._naxis[axis] else cube_wcs._naxis[axis]-1
    
    return lims


def get_cutout_wcs(cutout_lims, cube_wcs):
    """
    Get cutout wcs object, with updated NAXIS keywords and crpix. 
    """

    cutout_wcs = deepcopy(cube_wcs)
    
    cutout_wcs.wcs.crpix -= cutout_lims[:,0]
    
    cutout_wcs._naxis = [cutout_lims[0,1]-cutout_lims[0,0],
                         cutout_lims[1,1]-cutout_lims[1,0]]

    return cutout_wcs


def getCutout(cutout_lims,transposedCube, verbose=True):
    """
    Making a cutout from an image/uncertainty cube that has been transposed 
    to have time on the longest axis.

    Returns the untransposed image cutout and uncertainty cutout.
    """

    xmin,xmax = cutout_lims[0]
    ymin,ymax = cutout_lims[1]

    cutout = transposedCube[xmin:xmax,ymin:ymax,:,:]
    
    imgCutout = cutout[:,:,:,0].transpose((2,0,1))
    uncertCutout = cutout[:,:,:,1].transpose((2,0,1))

    if verbose:
        print("Image cutout cube shape: {}".format(imgCutout.shape))
        print("Uncertainty cutout cube shape: {}".format(uncertCutout.shape))
    
    return imgCutout, uncertCutout


def update_primary_header(primary_header, coordinates):

    # Adding cutout specific headers
    primary_header['RA_OBJ'] = (coordinates.ra.deg,'[deg] right ascension')
    primary_header['DEC_OBJ'] = (coordinates.dec.deg,'[deg] declination')
    
    # These are all the things in the TESS pipeline tpfs about the object that we can't fill
    primary_header['OBJECT'] = ("",'string version of target id ')
    primary_header['TCID'] = (0,'unique tess target identifier')
    primary_header['PXTABLE'] = (0,'pixel table id') 
    primary_header['PMRA'] = (0.0,'[mas/yr] RA proper motion') 
    primary_header['PMDEC'] = (0.0,'[mas/yr] Dec proper motion') 
    primary_header['PMTOTAL'] = (0.0,'[mas/yr] total proper motion') 
    primary_header['TESSMAG'] = (0.0,'[mag] TESS magnitude') 
    primary_header['TEFF'] = (0.0,'[K] Effective temperature') 
    primary_header['LOGG'] = (0.0,'[cm/s2] log10 surface gravity') 
    primary_header['MH'] =(0.0,'[log10([M/H])] metallicity') 
    primary_header['RADIUS'] = (0.0,'[solar radii] stellar radius')
    primary_header['TICVER'] = (0,'TICVER') 


def add_column_wcs(header, colnums, wcs_info):
    """
    Take WCS info in wcs_info and add it to the header as 
    """
    
    wcs_keywords = {'CTYPE1':'1CTYP{}',
                    'CTYPE2':'2CTYP{}',
                    'CRPIX1':'1CRPX{}',
                    'CRPIX2':'2CRPX{}',
                    'CRVAL1':'1CRVL{}',
                    'CRVAL2':'2CRVL{}',
                    'CUNIT1':'1CUNI{}',
                    'CUNIT2':'2CUNI{}',
                    'CDELT1':'1CDLT{}',
                    'CDELT2':'2CDLT{}',
                    'PC1_1':'11PC{}',
                    'PC1_2':'12PC{}',
                    'PC2_1':'21PC{}',
                    'PC2_2':'22PC{}'}


    for col in colnums:
        for kw in wcs_keywords:
            if kw not in wcs_info.keys():
                continue ## TODO: Something better than this?
            header[wcs_keywords[kw].format(col)] = wcs_info[kw]

            

def buildTpf(cubeFits, imgCube, uncertCube, cutoutWcs, coordinates, verbose=True):
    """
    Building the target pixel file.
    """

    # The primary hdu is just the main header, which is the same
    # as the one on the cube file
    primaryHdu = cubeFits[0]
    update_primary_header(primaryHdu.header, coordinates)

    cols = list()

    # Adding the Time relates columns
    cols.append(fits.Column(name='TIME', format='D', unit='BJD - 2457000, days', disp='D14.7',
                            array=(cubeFits[2].columns['TSTART'].array + cubeFits[2].columns['TSTOP'].array)/2))

    cols.append(fits.Column(name='TIMECORR', format='E', unit='d', disp='E14.7',
                            array=cubeFits[2].columns['BARYCORR'].array))

    cols.append(fits.Column(name='CADENCENO', format='J', disp='I10', array=np.array(range(len(imgCube)))))
    
    # Adding the cutouts
    tform = str(imgCube[0].size) + "E"
    dims = str(imgCube[0].shape)
    emptyArr = np.zeros(imgCube.shape)

    if verbose:
        print("TFORM: {}".format(tform))
        print("DIMS: {}".format(dims))
        print("Array shape: {}".format(emptyArr.shape))
        

    
    
    cols.append(fits.Column(name='RAW_CNTS', format=tform.replace('E','J'), unit='count', dim=dims, disp='I8',
                            array=emptyArr)) 
    cols.append(fits.Column(name='FLUX', format=tform, dim=dims, unit='e-/s', disp='E14.7', array=imgCube))
    cols.append(fits.Column(name='FLUX_ERR', format=tform, dim=dims, unit='e-/s', disp='E14.7', array=uncertCube)) 
   
    # Adding the background info (zeros b.c we don't have this info)
    cols.append(fits.Column(name='FLUX_BKG', format=tform, dim=dims, unit='e-/s', disp='E14.7',array=emptyArr))
    cols.append(fits.Column(name='FLUX_BKG_ERR', format=tform, dim=dims, unit='e-/s', disp='E14.7',array=emptyArr))

    # Adding the quality flags
    cols.append(fits.Column(name='QUALITY', format='j', disp='B16.16', array=cubeFits[2].columns['DQUALITY'].array))

    # Adding the position correction info (zeros b.c we don't have this info)
    cols.append(fits.Column(name='POS_CORR1', format='E', unit='pixel', disp='E14.7',array=emptyArr[:,0,0]))
    cols.append(fits.Column(name='POS_CORR2', format='E', unit='pixel', disp='E14.7',array=emptyArr[:,0,0]))

    # Adding the FFI_FILE column (not in the pipeline tpfs)
    cols.append(fits.Column(name='FFI_FILE', format='38A', unit='pixel',array=cubeFits[2].columns['FFI_FILE'].array))
        
    # making the table HDU
    tableHdu = fits.BinTableHDU.from_columns(cols)

    #print('raw counts', tableHdu.data['RAW_CNTS'].shape)

    tableHdu.header['EXTNAME'] = 'PIXELS'
    
    # Adding the wcs keywords to the columns and removing from the header
    wcs_header = cutoutWcs.to_header()
    add_column_wcs(tableHdu.header, [4,5,6,7,8], wcs_header) # TODO: can I not hard code the array?
    for kword in wcs_header:
        #del tableHdu.header[kword]
        tableHdu.header.remove(kword, ignore_missing=True)
    
    cutoutHduList = fits.HDUList([primaryHdu,tableHdu])

    return cutoutHduList



def cube_cut(cube_file, coordinates, cutout_size, target_pixel_file=None, verbose=None):
    """
    Takes a cube file (as created by ~astrocut.make_cube), 
    and makes a cutout stak of the given size around the given coordinates.

    Parameters
    ----------
    cube_file : str
        The cube file containing all the images to be cutout.  
        Must be in the format returned by ~astrocut.make_cube.
    coordinates : str or `astropy.coordinates` object
        The position around which to cutout. It may be specified as a
        string or as the appropriate `astropy.coordinates` object.
    cutout_size : int, array-like, `~astropy.units.Quantity`
        TODO: Is there a default size that makes sense?
        The size of the cutout array. If ``size``
        is a scalar number or a scalar `~astropy.units.Quantity`,
        then a square cutout of ``size`` will be created.  If
        ``size`` has two elements, they should be in ``(ny, nx)``
        order.  Scalar numbers in ``size`` are assumed to be in
        units of pixels. `~astropy.units.Quantity` objects must be in pixel or
        angular units.
    target_pixel_file : str
        Optional. The name for the output target pixel file. 
        If no name is supplied, the file will be named: 
        <cube_file>_<ra>_<dec>_<cutout_size>_cutout.fits
    verbose : bool
        Optional. If true intermediate information is printed. 

    Returns
    -------
    response: string or None
        If successfull, returns the path to the target pixel file, 
        if unsuccessfull returns None.
    """

    if verbose:
        startTime = time()

    cube = fits.open(cube_file) # TODO: add checking

    # Get the WCS and figure out which pixels are in the cutout
    wcsInd = int(len(cube[2].data)/2) # using the middle file for wcs info
    cubeWcs = getCubeWcs(cube[2].header, cube[2].data[wcsInd])

    if not isinstance(coordinates, SkyCoord):
        coordinates = SkyCoord.from_name(coordinates) # TODO: more checking here

    if verbose:
        print(coordinates)

    # making size into an array [nx, ny]
    cutout_size = np.atleast_1d(cutout_size)
    if len(cutout_size) == 1:
        cutout_size = np.repeat(cutout_size, 2)

    if len(cutout_size) > 2:
        print("To many dimensions in cutout size, only the first two will be used") # TODO: Make this into a warning
        
    # Get cutout limits
    cutout_lims = get_cutout_limits(coordinates, cutout_size, cubeWcs)

    if verbose:
        print("xmin,xmax:",cutout_lims[0])
        print("ymin,ymax:",cutout_lims[1])

    # Make the cutout
    imgCutout, uncertCutout = getCutout(cutout_lims,cube[1].data)

    # Get cutout wcs info
    cutout_wcs = get_cutout_wcs(cutout_lims, cubeWcs)
    
    if verbose:
        print(cubeWcs)
        print(cutout_wcs)
    
    # Build the TPF
    tpfObject = buildTpf(cube, imgCutout, uncertCutout, cutout_wcs, coordinates)

    print("raw counts:",tpfObject[1].data['RAW_CNTS'].shape)

    if verbose:
        writeTime = time()

    if not target_pixel_file:
        # TODO: also strip off excess path from cube file
        _, flename = os.path.split(cube_file)
        target_pixel_file = "{}_{}_{}_{}x{}_cutout.fits".format(flename.rstrip('.fits'),
                                                                coordinates.ra.value, coordinates.dec.value,
                                                                cutout_size[0],cutout_size[1])
    
    if verbose:
        print(target_pixel_file)
        
    # Write the TPF
    tpfObject.writeto(target_pixel_file, overwrite=True)

    # Close the cube file
    cube.close()

    if verbose:
        print("Write time: {:.2} sec".format(time()-writeTime))
        print("Total time: {:.2} sec".format(time()-startTime))

