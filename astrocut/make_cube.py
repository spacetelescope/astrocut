import numpy as np

from astropy.io import fits
from astropy.table import Table,Column

from time import time
from datetime import date
from os import path

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
               'DQUALITY',
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
                   np.int32,
                   'S16']


def make_primary_header(ffi_main_header, ffi_img_header, sector=-1):
    
    header = ffi_main_header

    header.remove('CHECKSUM')

    header['ORIGIN'] = 'STScI/MAST'
    header['DATE'] = str(date.today())
    header['SECTOR'] = (sector, "Observing sector")
    header['CAMERA'] = (ffi_img_header['CAMERA'], ffi_img_header.comments['CAMERA'])
    header['CCD'] = (ffi_img_header['CCD'], ffi_img_header.comments['CCD'])

    return header
    

def make_cube(file_list, cube_file="img-cube.fits", verbose=True):
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

    # Getting the time sorted indices for the files
    file_list = np.array(file_list)
    start_times = np.zeros(len(file_list))
    
    for i,ffi in enumerate(file_list):
        start_times[i] = fits.getval(ffi, 'TSTART', 1)
        
    sorted_indices = np.argsort(start_times)      
    
    # these will be the arrays and headers
    imgCube = None
    imgInfoTable = None
    primaryHeader = None
    secondaryHeader = None

    # Loop through files
    for i,fle in enumerate(file_list[sorted_indices]):
        
        ffiData = fits.open(fle)

        # if the arrays/headers aren't initialized do it now
        if imgCube is None:

            # The primary and secondary headers will be from the headers
            # of the first ffi with keywords unique to specific ffis removed
            # relevant individual ffi specific keywords will be captured
            # in the imgInfoTable
            primaryHeader = make_primary_header(ffiData[0].header, ffiData[1].header)
            
            secondaryHeader = ffiData[1].header
            
            ffiImg = ffiData[1].data

            # set up the cube array
            imgCube = np.full((ffiImg.shape[0],ffiImg.shape[1],len(file_list),2),
                              np.nan, dtype=np.float32)

            # set up the img info table
            cols = []
            for kwd,tpe in zip(specImgKwds[:-1],specImgKwdTypes[:-1]):
                cols.append(Column(name=kwd,dtype=tpe,length=len(file_list)))
            cols.append(Column(name="FFI_FILE",dtype="S38",length=len(file_list)))
            imgInfoTable = Table(cols)

        # add the image and info to the arrays
        imgCube[:,:,i,0] = ffiData[1].data
        imgCube[:,:,i,1] = ffiData[2].data
        for kwd in imgInfoTable.columns:
            if kwd == "FFI_FILE":
                imgInfoTable[kwd][i] = path.basename(fle)
            else:
                imgInfoTable[kwd][i] = ffiData[1].header[kwd]

        if fle == file_list[-1]:
            primaryHeader['DATE-END'] = ffiData[1].header['DATE-END']

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
        print("Total time elapsed: {:.2f} sec".format(endTime - startTime))
        print("File write time: {:.2f} sec".format(endTime - writeTime))

    return cube_file
