# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module implements the functionality to create image cubes for the purposes of
creating cutout target pixel files.
"""

import numpy as np
import os

from astropy.io import fits
from astropy.table import Table, Column

from time import time
from datetime import date
from copy import deepcopy


class CubeFactory():
    """
    Class for creating image cubes.
    
    This class emcompasses all of the cube making functionality.  
    In the current version this means creating image cubes fits files from TESS full frame image sets.
    Future versions will include more generalized cubing functionality.
    """

    def __init__(self, max_memory=50):
        """ Setting up the class members """

        self.max_memory = max_memory # in GB
        self.block_size = None # Number of rows
        self.num_blocks = None
        self.cube_shape = None
        
        self.time_keyword = 'TSTART' # TESS-specific
        self.last_file_keywords = ['DATE-END', 'TSTOP'] # TESS-specific (assumed to be in extension 0)
        self.image_header_keywords = ['CAMERA', 'CCD'] # TESS-specific
        self.template_requirements = {"WCSAXES": 2} # TESS-specific (assumed to be in extension 1)

        self.file_list = None
        self.template_file = None
        
        self.primary_header = None
        self.info_table = None
        self.cube_file = None

        
    def _configure_cube(self, file_list, **extra_keywords):
        """ 
        Run through all the files and set up the  basic parameters for the cube.
        Set up the cube primary header.
        """
        
        file_list = np.array(file_list)
        image_shape = None
        start_times = np.zeros(len(file_list))
        for i, ffi in enumerate(file_list):
            ffi_data = fits.open(ffi, mode='denywrite', memmap=True)
            
            start_times[i] = ffi_data[1].header.get(self.time_keyword)

            if image_shape is None:  # Only need to fill this once
                image_shape = ffi_data[1].data.shape
            
            if self.template_file is None:  # Only check this if we don't already have it

                is_template = True
                for key,value in self.template_requirements.items():
                    if ffi_data[1].header.get(key) != value:  # Checking for a good image header
                        is_template = False
                        
                if is_template:
                    self.template_file = ffi
                    
            ffi_data.close()
                
        self.file_list = file_list[np.argsort(start_times)]

        # Working out the block size and number of blocks needed for writing the cube
        # without using too much memory
        cube_size = np.prod(image_shape) * len(self.file_list) * 2 * 4 # in bytes (float32)
        slice_size = image_shape[1] * len(self.file_list) * 2 * 4 # in bytes (float32)
        self.block_size = int((self.max_memory * 1e9)//slice_size)
        self.num_blocks = int(image_shape[0]/self.block_size + 1)
        self.cube_shape = (image_shape[0], image_shape[1], len(self.file_list), 2)

        # Making the primary header
        with fits.open(self.file_list[0], mode='denywrite', memmap=True) as first_file:
            header = deepcopy(first_file[0].header)
            header.remove('CHECKSUM', ignore_missing=True)

            # Adding standard keywords
            header['ORIGIN'] = 'STScI/MAST'
            header['DATE'] = str(date.today())

            # Adding factory specific keywords
            for kwd in self.image_header_keywords:
                header[kwd] = (first_file[1].header[kwd], first_file[1].header.comments[kwd])

            # Adding the extra keywords passed in
            for kwd, value in extra_keywords.items():
                header[kwd] = (value[0], value[1])
                
        # Adding the keywords from the last file
        with fits.open(self.file_list[-1], mode='denywrite', memmap=True) as last_file:
            for kwd in self.last_file_keywords:
                header[kwd] = (last_file[0].header[kwd], last_file[0].header.comments[kwd])

        header["EXTNAME"] = "PRIMARY"

        self.primary_header = header


    def _build_info_table(self):
        """
        Reading the keywords and setting up the table to hold the image headers (extension 1)
        from every input file.
        """

        with fits.open(self.template_file, mode='denywrite', memmap=True) as ffi_data:
            
            # The image specific header information will be saved in a table in the second extension
            secondary_header = ffi_data[1].header

            # set up the image info table
            cols = []
            for kwd, val, cmt in secondary_header.cards: 
                if type(val) == str:
                    tpe = "S" + str(len(val))  # TODO: Maybe switch to U?
                elif type(val) == int:
                    tpe = np.int32
                else:
                    tpe = np.float32
                    
                cols.append(Column(name=kwd, dtype=tpe, length=len(self.file_list), meta={"comment": cmt}))
                    
            cols.append(Column(name="FFI_FILE", dtype="S" + str(len(os.path.basename(self.template_file))),
                               length=len(self.file_list)))
            self.info_table = Table(cols)

        
    def _build_cube_file(self, cube_file):
        """
        Build the cube file on disk with primary header, cube extension header,
        and space for the cube, filled with zeros.

        Note, this will overwrite the file if it already exists.
        """
        
        # Making sure the output directory exists
        direc, _ = os.path.split(cube_file)
        if direc and not os.path.exists(direc):
            os.makedirs(direc)

        # Writing the primary header
        self.primary_header.tofile(cube_file, overwrite=True)

        # Making the cube header and writing it
        data = np.zeros((100, 100, 10, 2), dtype=np.float32)
        hdu = fits.ImageHDU(data)
        header = hdu.header
        header["NAXIS4"], header["NAXIS3"], header["NAXIS2"], header["NAXIS1"] = self.cube_shape

        with open(cube_file,'ab') as CUBE:
            CUBE.write(bytearray(header.tostring(),encoding="utf-8"))

        # Expanding the file to fit the full data cube
        # fits requires all blocks to be a multiple of 2880
        cubesize_in_bytes = int((np.prod(self.cube_shape) * 4 / 2880) + 1) * 2880
        filelen = os.path.getsize(cube_file)
        with open(cube_file, 'r+b') as CUBE:
            CUBE.seek(filelen + cubesize_in_bytes - 1)
            CUBE.write(b'\0')

        self.cube_file = cube_file


    def _write_block(self, cube_hdu, start_row=0, end_row=None, fill_info_table=False, verbose=False):
        """
        Do one pass through the image files and write a block of the cube.

        cube_hdu is am hdulist object opened in update mode
        """

         # Loop through files
        for i, fle in enumerate(self.file_list):
        
            with fits.open(fle, mode='denywrite', memmap=True) as ffi_data:

                # add the image and info to the arrays
                cube_hdu[1].data[start_row:end_row, :, i, 0]  = ffi_data[1].data[start_row:end_row, :]
                cube_hdu[1].data[start_row:end_row, :, i, 1]  = ffi_data[2].data[start_row:end_row, :]
                
                if fill_info_table: # Also save the header info in the info table

                    for kwd in self.info_table.columns:
                        if kwd == "FFI_FILE":
                            self.info_table[kwd][i] = os.path.basename(fle)
                        else:
                            nulval = None
                            if self.info_table[kwd].dtype.name == "int32":
                                nulval = 0
                            elif self.info_table[kwd].dtype.char == "S":  # hacky way to check if it's a string
                                nulval = ""
                            self.info_table[kwd][i] = ffi_data[1].header.get(kwd, nulval)
            
            if verbose:
                print("Completed file {}".format(i))

        # Flush the filled block to disk
        cube_hdu.flush()

        
    def _write_info_table(self):
        """
        Turn the info table into an HDU object and append it to the cube file
        """

        # Make table hdu 
        cols = []
        for kwd in self.info_table.columns:
            if self.info_table[kwd].dtype == np.float32:
                tpe = 'D'
            elif self.info_table[kwd].dtype == np.int32:
                tpe = 'J'
            else:
                tpe = str(self.info_table[kwd].dtype).replace("S", "A").strip("|")
        
            cols.append(fits.Column(name=kwd, format=tpe, array=self.info_table[kwd]))
        
        col_def = fits.ColDefs(cols)
        table_hdu = fits.BinTableHDU.from_columns(col_def)

        # Adding the comments to the header
        for kwd in self.info_table.columns:
            if kwd in ['XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'PCOUNT', 'GCOUNT', 'TFIELDS']:
                continue  # skipping the keyword already in use
            table_hdu.header[kwd] = self.info_table[kwd].meta.get("comment", "")

        # Appending to the cube file
        with fits.open(self.cube_file, mode='append', memmap=True) as cube_hdus:
            cube_hdus.append(table_hdu)


    
    def make_cube(self, file_list, cube_file="img-cube.fits", sector=None, max_memory=50, verbose=True):
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

        self.max_memory=max_memory
            
        # Set up the basic cube parameters
        sector = (sector, "Observing sector")
        self._configure_cube(file_list, sector=sector)
    
        
        if verbose:
            print("Using {} to initialize the image header table.".format(os.path.basename(self.template_file)))
            print(f"Cube will be made in {self.num_blocks} blocks of {self.block_size} rows each.")

        # Set up the table to old the individual image heades
        self._build_info_table()

        
        # Write the empty file, ready for the cube to be added
        self._build_cube_file(cube_file)

        # Fill the image cube
        with fits.open(self.cube_file, mode='update', memmap=True) as cube_hdu:
            for i in range(self.num_blocks):
                start_row = i * self.block_size
                end_row = start_row + self.block_size
                if end_row >= len(self.file_list):
                    end_row = None

                fill_info_table = True if (i == 0) else False
                self._write_block(cube_hdu, start_row, end_row, fill_info_table, verbose)
  
                if verbose:
                    print(f"Completed block {i+1} of {self.num_blocks}")


        # Add the info table to the cube file
        self._write_info_table()
        if verbose:
            endTime = time()
            print("Total time elapsed: {:.2f} sec".format(endTime - startTime))

        return self.cube_file
