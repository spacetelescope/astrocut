import numpy as np

from astropy.io import fits
from astropy.table import Table,Column

from time import time

# TODO: Find a better solution to these unweildy lists
specImgKwds = ['TSTART','TSTOP',
               'DATE-OBS','DATE-END',
               'BARYCORR',
               "NAXIS", "NAXIS1", "NAXIS2",
               'CTYPE1','CTYPE2',
               'CRPIX1','CRPIX2',
               'CRVAL1','CRVAL2',
               'CD1_1','CD1_2','CD2_1','CD2_2',
               'A_2_0','A_0_2','A_1_1',
               'B_2_0','B_0_2','B_1_1',
               'AP_1_0','AP_0_1','AP_2_0','AP_0_2','AP_1_1',
               'BP_1_0','BP_0_1','BP_2_0','BP_0_2','BP_1_1',
               'A_DMAX','B_DMAX',
               'CHECKSUM']

specImgKwdTypes = [np.float32,np.float32,
                   'S24','S24',
                   np.float32,
                   np.int32,np.int32,np.int32,
                   'S12','S12',
                   np.float32,np.float32,
                   np.float32,np.float32,
                   np.float32,np.float32,np.float32,np.float32,
                   np.float32,np.float32,np.float32,
                   np.float32,np.float32,np.float32,
                   np.float32,np.float32,np.float32,np.float32,np.float32,
                   np.float32,np.float32,np.float32,np.float32,np.float32,
                   np.float32,np.float32,
                   'S16']


def make_cube(file_list, cube_file="cube.fits", verbose=True):
    """
    Turns a list of fits image files into one large data-cube.
    Input images must all have the same footprint and resolution.
    The resulting datacube is transposed for quite cutouts.
    This function can take some time to run and requires enough 
    memory to hold the entire cube in memory.

    Parameters
    ----------
    file_list : array
        The list of fits image files to cube.  
        Assumed to have the structure of TESS FFI files.
    cube_path : string
        Optional.  The file to save the output cube in. 
    verbose : bool
        Optional. If true intermediate information is printed. 

    Returns
    -------
    response: string or None
        If successfull, returns the path to the cube fits file, 
        if unsuccessfull returns None.
    """

    if verbose:
        startTime = time()
    
    # these will be the arrays and headers
    imgCube = None
    imgInfoTable = None
    primaryHeader = None
    secondaryHeader = None

    # Loop through files
    for i,fle in enumerate(file_list):
        
        ffiData = fits.open(fle)

        # if the arrays/headers aren't initialized do it now
        if imgCube is None:

            # The primary and secondary headers will be from the headers
            # of the first ffi with keywords unique to specific ffis removed
            # relevant individual ffi specific keywords will be captured
            # in the imgInfoTable
            primaryHeader = ffiData[0].header
            primaryHeader.remove('DATE-OBS')
            primaryHeader.remove('DATE-END')
            primaryHeader.remove('CHECKSUM')

            secondaryHeader = ffiData[1].header
            
            ffiImg = ffiData[1].data

            # set up the cube array
            imgCube = np.full((ffiImg.shape[0],ffiImg.shape[1],len(ffiFiles)),
                              np.nan, dtype=np.float32)

            # set up the img info table
            cols = []
            print(len(specImgKwds),len(specImgKwdTypes))
            for kwd,tpe in zip(specImgKwds[:-1],specImgKwdTypes[:-1]):
                cols.append(Column(name=kwd,dtype=tpe,length=len(ffiFiles)))
            imgInfoTable = Table(cols)
            print(imgInfoTable.columns)

        # add the image and info to the arrays
        imgCube[:,:,i] = ffiData[1].data
        for kwd in imgInfoTable.columns:
            imgInfoTable[kwd][i] = ffiData[1].header[kwd]

        # close fits file
        ffiData.close()

        if verbose:
            print("Completed file {}".format(i))

        for kwd in specImgKwds:
            secondaryHeader.remove(kwd)

        # put it all in a fits file 
        primaryHdu = fits.PrimaryHDU(header=primaryHeader)
        cubeHdu = fits.ImageHDU(data=imgCube,header=secondaryHeader)

        # make table hdu with the img info array
        cols = []
        for kwd in imgInfoTable.columns:
            tpe = 'F' if (imgInfoTable[kwd].dtype == np.float32) else 'A24'
            cols.append(fits.Column(name=kwd, format=tpe, array=imgInfoTable[kwd]))
        tcDef = fits.ColDefs(cols)
        timeHdu = fits.BinTableHDU.from_columns(tcDef)

        hduList = fits.HDUList([primaryHdu,cubeHdu,timeHdu])

        if verbose:
            writeTime = time()
    
        hduList.writeto(cube_file, overwrite=True) 

        if verbose:
            endTime = time()
            print("Total time elapsed:", endTime - beginTime)
            print("File write time:", endTime - writeTime)
