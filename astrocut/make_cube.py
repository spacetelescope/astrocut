# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module implements the functionality to create image cubes for the purposes of
creating cutout target pixel files.
"""

import numpy as np

from astropy.io import fits
from astropy.table import Table,Column

from time import time
from datetime import date
from os import path
from copy import deepcopy


class CubeFactory():
    """
    Class for creating image cubes.
    
    This class emcompasses all of the cube making functionality.  
    In the current version this means creating image cubes fits files from TESS full frame image sets.
    Future versions will include more generalized cubing functionality.
    """
                
    def _make_primary_header(self, ffi_main_header, ffi_img_header, sector=-1):
        """
        Given the primary and image headers from an input FFI, and the sector number, 
        build the cube's primary header.
        This is a TESS specific function

        Parameters
        ----------
        ffi_main_header : `~astropy.io.fits.Header`
            The primary header from a TESS FFI fits file.
        ffi_img_header : `~astropy.io.fits.Header`
            The seconday (image) header from a TESS FFI fits file.
        sector : int
            TESS mission sector number

        Returns
        -------
        response :  `~astropy.io.fits.Header`
            Primary header for the image cube fits file.
        """
    
        header = deepcopy(ffi_main_header)
        header.remove('CHECKSUM', ignore_missing=True)

        header['ORIGIN'] = 'STScI/MAST'
        header['DATE'] = str(date.today())
        header['SECTOR'] = (sector, "Observing sector")
        header['CAMERA'] = (ffi_img_header['CAMERA'], ffi_img_header.comments['CAMERA'])
        header['CCD'] = (ffi_img_header['CCD'], ffi_img_header.comments['CCD'])

        

        return header
    
    def make_cube(self, file_list, cube_file="img-cube.fits", sector=None, verbose=True):
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
        sector : int
            Optional.  TESS sector to add as header keyword (not present in FFI files).
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
                primary_header = self._make_primary_header(ffi_data[0].header, ffi_data[1].header, sector=sector)

                # The image specific header information will be saved in a table in the second extension
                secondary_header = ffi_data[1].header
            
                ffi_img = ffi_data[1].data

                # set up the cube array
                img_cube = np.full((ffi_img.shape[0],ffi_img.shape[1],len(file_list),2),
                                   np.nan, dtype=np.float32)

                # set up the image info table
                cols = []
                for kwd,val,cmt in secondary_header.cards: 
                    if type(val) == str:
                        tpe = "S"+str(len(val))
                    elif type(val) == int:
                        tpe = np.int32
                    else:
                        tpe = np.float32
                    
                    cols.append(Column(name=kwd,dtype=tpe,length=len(file_list), meta={"comment":cmt}))
                    
                cols.append(Column(name="FFI_FILE",dtype="S" + str(len(path.basename(fle))), length=len(file_list)))
            
                img_info_table = Table(cols)

            # add the image and info to the arrays
            img_cube[:,:,i,0] = ffi_data[1].data
            img_cube[:,:,i,1] = ffi_data[2].data
            for kwd in img_info_table.columns:
                if kwd == "FFI_FILE":
                    img_info_table[kwd][i] = path.basename(fle)
                else:
                    img_info_table[kwd][i] = ffi_data[1].header[kwd]

            #if fle == file_list[-1]:
            if i == (len(file_list) - 1):
                primary_header['DATE-END'] = ffi_data[0].header['DATE-END']
                primary_header['TSTOP'] = ffi_data[0].header.get('TSTOP', 0)
                
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
                tpe = str(img_info_table[kwd].dtype).replace("S","A").strip("|")
        
            cols.append(fits.Column(name=kwd, format=tpe, array=img_info_table[kwd]))
        
        col_def = fits.ColDefs(cols)
        table_hdu = fits.BinTableHDU.from_columns(col_def)

        # Adding the comments to the header
        for kwd in img_info_table.columns:
            if kwd in ['XTENSION', 'BITPIX', 'NAXIS','NAXIS1','NAXIS2','PCOUNT','GCOUNT','TFIELDS']:
                continue # skipping the keyword already in use
            table_hdu.header[kwd] = img_info_table[kwd].meta.get("comment","")

        hdu_list = fits.HDUList([primary_hdu,cube_hdu,table_hdu])

        if verbose:
            writeTime = time()
    
        hdu_list.writeto(cube_file, overwrite=True) 

        if verbose:
            endTime = time()
            print("Total time elapsed: {:.2f} sec".format(endTime - startTime))
            print("File write time: {:.2f} sec".format(endTime - writeTime))

        return cube_file
