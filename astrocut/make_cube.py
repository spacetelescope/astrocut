"""
Makes the cube!
"""

import numpy as np

from astropy.io import fits
from astropy.table import Table,Column
from astropy.wcs import WCS

from time import time
from datetime import date
from os import path


class CubeFactory():
    """
    Makes a cube!

    TODO: Document better.
    """
                
    def make_primary_header(self, ffi_main_header, ffi_img_header, sector=-1):
        """
        Given the primary and image headers from an input FFI, and the sector number, 
        build the cube's primary header.

        Parameters
        ----------
        ffi_main_header
        ffi_img_header
        sector
        """
    
        header = ffi_main_header
        header.remove('CHECKSUM', ignore_missing=True)

        header['ORIGIN'] = 'STScI/MAST'
        header['DATE'] = str(date.today())
        header['SECTOR'] = (sector, "Observing sector")
        header['CAMERA'] = (ffi_img_header['CAMERA'], ffi_img_header.comments['CAMERA'])
        header['CCD'] = (ffi_img_header['CCD'], ffi_img_header.comments['CCD'])

        return header
    
    def make_cube(self, file_list, cube_file="img-cube.fits", verbose=True):
        """
        Turns a list of fits image files into one large data-cube.
        Input images must all have the same footprint and resolution.
        The resulting datacube is transposed for quicker cutouts.
        This function can take some time to run and requires enough 
        memory to hold the entire cube in memory. 
        (For full TESS sectors this is about 40 GB)

        Parameters
        ----------
        file_list : array
            The list of fits image files to cube.  
            Assumed to have the format of a TESS FFI:
            - A primary HDU consisting only of a primary header
            - An image HDU containing the image
            - A second image HDU containing the uncertainty image
        cube_file : string
            Optional.  The filename/path to save the output cube in. 
        verbose : bool
            Optional. If true intermediate information is printed. 

        Returns
        -------
        response: string or None
            If successful, returns the path to the cube fits file, 
            if unsuccessful returns None.
        """

        if verbose:
            startTime = time()

        # Getting the time sorted indices for the files
        file_list = np.array(file_list)
        start_times = np.zeros(len(file_list))
    
        for i,ffi in enumerate(file_list):
            start_times[i] = fits.getval(ffi, 'TSTART', 1) # TODO: optionally pass this in?
            
        sorted_indices = np.argsort(start_times)      
    
        # these will be the arrays and headers
        img_cube = None
        img_info_table = None
        primary_header = None
        secondary_header = None

        # Loop through files
        for i,fle in enumerate(file_list[sorted_indices]):
        
            ffi_data = fits.open(fle)

            # if the arrays/headers aren't initialized do it now
            if img_cube is None:

                # We use the primary header from the first file as the cube primary header
                # and will add in information about the time of the final observation at the end
                primary_header = self.make_primary_header(ffi_data[0].header, ffi_data[1].header)

                # The image specific header information will be saved in a table in the second extension
                secondary_header = ffi_data[1].header
            
                ffi_img = ffi_data[1].data

                # set up the cube array
                img_cube = np.full((ffi_img.shape[0],ffi_img.shape[1],len(file_list),2),
                                   np.nan, dtype=np.float32)

                # set up the image info table
                cols = []
                for kwd,val,_ in secondary_header.cards: # not using comments
                    if type(val) == str:
                        tpe = "S"+str(len(val))
                    elif type(val) == int:
                        tpe = np.int32
                    else:
                        tpe = np.float32
                    
                    cols.append(Column(name=kwd,dtype=tpe,length=len(file_list))) 
                
                cols.append(Column(name="FFI_FILE",dtype="S38",length=len(file_list)))
            
                img_info_table = Table(cols)

            # add the image and info to the arrays
            img_cube[:,:,i,0] = ffi_data[1].data
            img_cube[:,:,i,1] = ffi_data[2].data
            for kwd in img_info_table.columns:
                if kwd == "FFI_FILE":
                    img_info_table[kwd][i] = path.basename(fle)
                else:
                    img_info_table[kwd][i] = ffi_data[1].header[kwd]

            if fle == file_list[-1]:
                primary_header['DATE-END'] = ffi_data[1].header['DATE-END']

            # close fits file
            ffi_data.close()

            if verbose:
                print("Completed file {}".format(i))

        # put it all in a fits file 
        primary_hdu = fits.PrimaryHDU(header=primary_header)
        cube_hdu = fits.ImageHDU(data=img_cube)

        # make table hdu with the img info array
        cols = []
        for kwd in img_info_table.columns:
            if img_info_table[kwd].dtype == np.float32:
                tpe = 'D'
            elif img_info_table[kwd].dtype == np.int32:
                tpe = 'J'
            else:
                tpe = "A24"
        
            cols.append(fits.Column(name=kwd, format=tpe, array=img_info_table[kwd]))
        
        col_def = fits.ColDefs(cols)
        table_hdu = fits.BinTableHDU.from_columns(col_def)

        hdu_list = fits.HDUList([primary_hdu,cube_hdu,table_hdu])

        if verbose:
            writeTime = time()
    
        hdu_list.writeto(cube_file, overwrite=True) 

        if verbose:
            endTime = time()
            print("Total time elapsed: {:.2f} sec".format(endTime - startTime))
            print("File write time: {:.2f} sec".format(endTime - writeTime))

        return cube_file
