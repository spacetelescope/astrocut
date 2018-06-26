import numpy as np

from astropy.io import fits
from astropy.table import Table,Column
from astropy.coordinates import SkyCoord
from astropy import wcs

from time import time

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


def getCutoutLims(centerCoord, dims, cubeWcs):
    """
    Takes the centra coordinates, cutout dimensions, and the wcs from
    which the cutout is being taken and returns the x and y pixel limits
    for the cutout.
    """

    centerPixel = centerCoord.to_pixel(cubeWcs) # TODO: add checks!!
      
    xlim = (int(round(centerPixel[0] - dims[0])),
            int(round(centerPixel[0] + dims[0])))
    ylim = (int(round(centerPixel[1] - dims[1])),
            int(round(centerPixel[1] + dims[1])))
    # TODO: Make sure entire cutout is on image or adjust bounds

    return xlim,ylim


def getCutout(xlim,ylim,transposedCube):
    """
    Making a cutout from an image cube that has been transposed 
    to have time on the longest axis.

    Returns the untransposed cutout.
    """

    cutout = transposedCube[xlim[0]:xlim[1]+1,ylim[0]:ylim[1]+1,:]
    return cutout.transpose((2,0,1))


def buildTpf(cubeFits, cutoutCube, cutoutWcs):
    """
    Building the target pixel file.
    """

    # The primary hdu is just the main header, which is the same
    # as the one on the cube file
    primaryHdu = cubeFits[0]

    # We need to build a fits column of the cutouts
    tform = str(cutoutCube[0].size) + "J"
    dims = str(cutoutCube[0].shape)
    cubeCol = fits.Column(name='CUTOUT', format=tform, dim=dims,
                          array=cutoutCube) # Not sure what to call this b/c not sure whats in the image, must check

    # TODO: REPLACE THESE WITH THE CORRECT COLUMNS
    cols = [cubeCol]
    for colname in ['TSTART','TSTOP','DATE-OBS','DATE-END']:
        #print(type(cubeFits[2].data[colname]))
        cols.append(cubeFits[2].columns[colname])

    # making the table HDU
    tableHdu = fits.BinTableHDU.from_columns(cols)

    # TODO: Sort by time here?

    # adding the wcs keywords 
    wcsHeader = cutoutWcs.to_header()
    for entry in wcsHeader:
        tableHdu.header[entry] = wcsHeader[entry]

    # TODO: I think there is at lease one more hdu in a tpf
    
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
    size : int or array 
        TODO: Is there a default size that makes sense?
        The size in pixels of the cutout, if one int is given the cutout will be
        square, otherwise it will be a rectangle with dimentions size[0]xsize[1]
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
        coordinates = SkyCoord.from_name(coordinates) # TODO: more cheking here

    if verbose:
        print(coordinates)

    if isinstance(cutout_size, int): # TODO: more checking
        cutout_size = [cutout_size,cutout_size]
    elif len(cutout_size) < 2:
        cutout_size = [cutout_size[0],cutout_size[0]]
        
    xlim,ylim = getCutoutLims(coordinates, cutout_size, cubeWcs) 

    # Make the cutout
    cutoutCube = getCutout(xlim,ylim,cube[1].data)

    # Build the TPF
    tpfObject = buildTpf(cube, cutoutCube, cubeWcs)

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

