# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module implements the functionality to create image cubes for the purposes of
creating cutout target pixel files.
"""

import os
from copy import deepcopy
from datetime import date
from sys import platform, version_info
from time import time

import numpy as np
from astropy.io import fits
from astropy.table import Column, Table

if (version_info >= (3, 8)) and (platform != "win32"):
    from mmap import MADV_SEQUENTIAL

__all__ = ['CubeFactory', 'TicaCubeFactory']
ERROR_MSG = "One or more incorrect file types were input. Please input TICA FFI files when using\
                   ``TicaCubeFactory``, and SPOC FFI files when using ``CubeFactory``."


class CubeFactory():
    """
    Class for creating image cubes.
    
    This class emcompasses all of the cube making functionality.  
    In the current version this means creating image cubes fits files from TESS full frame image sets.
    Future versions will include more generalized cubing functionality.
    """

    def __init__(self, max_memory=50):
        """ Setting up the class members """

        self.max_memory = max_memory  # in GB
        self.block_size = None  # Number of rows
        self.num_blocks = None
        self.cube_shape = None
        
        self.time_keyword = 'TSTART'  # TESS-specific
        self.last_file_keywords = ['DATE-END', 'TSTOP']  # TESS-specific (assumed to be in extension 1)
        self.image_header_keywords = ['CAMERA', 'CCD']  # TESS-specific
        self.template_requirements = {"WCSAXES": 2}  # TESS-specific (assumed to be in extension 1)

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
                for key, value in self.template_requirements.items():
                    if ffi_data[1].header.get(key) != value:  # Checking for a good image header
                        is_template = False
                        
                if is_template:
                    self.template_file = ffi
                    
            ffi_data.close()
                
        self.file_list = file_list[np.argsort(start_times)]

        # Working out the block size and number of blocks needed for writing the cube
        # without using too much memory
        try:
            slice_size = image_shape[1] * len(self.file_list) * 2 * 4  # in bytes (float32)
        except IndexError:
            raise ValueError(ERROR_MSG)
        max_block_size = int((self.max_memory * 1e9)//slice_size)
        
        self.num_blocks = int(image_shape[0]/max_block_size + 1)
        self.block_size = int(image_shape[0]/self.num_blocks + 1)
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
                header[kwd] = (last_file[1].header[kwd], last_file[1].header.comments[kwd])

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
                if isinstance(val, str):
                    tpe = "S" + str(len(val))  # TODO: Maybe switch to U?
                elif isinstance(val, int):
                    tpe = np.int32
                else:
                    tpe = np.float64

                # Adding columns one by one 
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

        with open(cube_file, 'ab') as CUBE:
            CUBE.write(bytearray(header.tostring(), encoding="utf-8"))

        # Expanding the file to fit the full data cube
        # fits requires all blocks to be a multiple of 2880
        cubesize_in_bytes = ((np.prod(self.cube_shape) * 4 + 2880 - 1) // 2880) * 2880
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

        # Initializing the sub-cube
        nrows = (self.cube_shape[0] - start_row) if (end_row is None) else (end_row - start_row)
        sub_cube = np.zeros((nrows, *self.cube_shape[1:]), dtype=np.float32)
        
        # Loop through files
        for i, fle in enumerate(self.file_list):

            if verbose:
                st = time()

            with fits.open(fle, mode='denywrite', memmap=True) as ffi_data:

                # add the image and info to the arrays
                sub_cube[:, :, i, 0] = ffi_data[1].data[start_row:end_row, :]
                sub_cube[:, :, i, 1] = ffi_data[2].data[start_row:end_row, :]

                del ffi_data[1].data
                del ffi_data[2].data
                
                if fill_info_table:  # Also save the header info in the info table

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
                print(f"Completed file {i} in {time()-st:.3} sec.")

        # Fill block and flush to disk
        cube_hdu[1].data[start_row:end_row, :, :, :] = sub_cube

        if (version_info <= (3, 8)) or (platform == "win32"):
            cube_hdu.flush()

        del sub_cube

        
    def _write_info_table(self):
        """
        Turn the info table into an HDU object and append it to the cube file
        """

        # Make table hdu 
        cols = []
        for kwd in self.info_table.columns:
            if self.info_table[kwd].dtype == np.float64:
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
        This function can take some time to run, exactly how much time will depend on the number
        of input files and the maximum allowed memory. The runtime will be fastest if the
        entire data cube can be held in memory, however that can be quite large (~40GB for a full
        TESS main mission sector, 3 times that for a TESS extended mission sector).

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
        max_memory : float
            Optional, default is 50. The maximum amount of memory to make available for building
            the data cube in GB. Note, this is the maximum amount of space to be used for the cube
            array only, so should not be set to the full amount of memory on the system.
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

        self.max_memory = max_memory

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

            if (version_info >= (3, 8)) and (platform != "win32"):
                mm = fits.util._get_array_mmap(cube_hdu[1].data)
                mm.madvise(MADV_SEQUENTIAL)

            for i in range(self.num_blocks):
                start_row = i * self.block_size
                end_row = start_row + self.block_size
                if end_row >= self.cube_shape[0]:
                    end_row = None

                fill_info_table = True if (i == 0) else False
                self._write_block(cube_hdu, start_row, end_row, fill_info_table, verbose)
                  
                if verbose:
                    print(f"Completed block {i+1} of {self.num_blocks}")

        # Add the info table to the cube file
        self._write_info_table()
        if verbose:
            print(f"Total time elapsed: {(time() - startTime)/60:.2f} min")

        return self.cube_file


class TicaCubeFactory():
    """
    Class for creating TICA image cubes.
    
    This class emcompasses all of the cube making functionality.  
    In the current version this means creating image cubes fits files from TESS full frame image sets.
    Future versions will include more generalized cubing functionality.

    The TESS Image CAlibrator (TICA) products are high level science products (HLSPs) 
    developed by the MIT Quick Look Pipeline (https://github.com/mmfausnaugh/tica). These 
    images are produced and delivered up to 4x sooner than their SPOC counterparts (as of TESS EM2),
    and can therefore be used to produce the most up-to-date cutouts of a target. 
    More information on TICA can be found here: https://archive.stsci.edu/hlsp/tica
    """

    def __init__(self, max_memory=50):
        """ Setting up the class members."""

        self.max_memory = max_memory  # in GB
        self.block_size = None  # Number of rows
        self.num_blocks = None
        self.cube_shape = None
        
        self.time_keyword = 'STARTTJD'  # Start time in TJD. TICA-specific.
        self.last_file_keywords = ['ENDTJD']  # Stop time in TJD. TICA-specific (assumed to be in extension 0)                  
        self.image_header_keywords = ['CAMNUM', 'CCDNUM']  # Camera number and CCD number 
        self.template_requirements = {'NAXIS': 2}  # Using NAXIS instead of WCSAXES. 
                                                   
        self.file_list = None
        self.template_file = None
        
        self.primary_header = None
        self.info_table = None
        self.cube_file = None

        # Used when updating an existing cube 
        self.old_cols = None
        self.update = False
        self.cube_append = None

    def _configure_cube(self, file_list, **extra_keywords):
        """ Run through all the files and set up the basic parameters for the cube.
        Set up the cube primary header.
        """
        
        file_list = np.array(file_list)
        image_shape = None
        start_times = np.zeros(len(file_list))
        for i, ffi in enumerate(file_list):
            ffi_data = fits.open(ffi, mode='denywrite', memmap=True)
            
            start_times[i] = ffi_data[0].header.get(self.time_keyword)

            if image_shape is None:  # Only need to fill this once
                try:
                    image_shape = ffi_data[0].data.shape
                except AttributeError:
                    raise ValueError(ERROR_MSG)
            
            if self.template_file is None:  # Only check this if we don't already have it

                is_template = True
                for key, value in self.template_requirements.items():
                    if ffi_data[0].header.get(key) != value:  # Checking for a good image header
                        is_template = False
                        
                if is_template:
                    self.template_file = ffi
                    
            ffi_data.close()

        self.file_list = file_list[np.argsort(start_times)]

        # Working out the block size and number of blocks needed for writing the cube
        # without using too much memory
        slice_size = image_shape[1] * len(self.file_list) * 2 * 4  # in bytes (float32)
        max_block_size = (self.max_memory * 1e9) // slice_size
        
        self.num_blocks = int(image_shape[0]/max_block_size + 1)
        self.block_size = int(image_shape[0]/self.num_blocks + 1)
        
        # Determining cube shape: 
        # If it's a new TICA cube, the shape is (nRows, nCols, nImages, 1).
        # Axis 4 is `1` instead of `2` because we do not work with error arrays for TICA.
        if not self.update:
            self.cube_shape = (image_shape[0], image_shape[1], len(self.file_list), 1)
            
        # Else, if it's an update to an existing cube, the shape is (nRows, nCols, nImages + nNewImages, 2)
        else: 
            self.cube_shape = self.cube_append.shape
            
        # Making the primary header if there's no cube_file yet
        if not self.update:
            with fits.open(self.file_list[0], mode='denywrite', memmap=True) as first_file:
                header = deepcopy(first_file[0].header)
                header.remove('CHECKSUM', ignore_missing=True)

                # Adding standard keywords
                header['ORIGIN'] = 'STScI/MAST'
                header['DATE'] = str(date.today())

                # Adding factory specific keywords
                for kwd in self.image_header_keywords:
                    # TICA file structure differs from SPOC in that factory-specific kwds
                    # are in the 0th extension, along with the science data.
                    header[kwd] = (first_file[0].header[kwd], first_file[0].header.comments[kwd])

                # Adding the extra keywords passed in
                for kwd, value in extra_keywords.items():
                    header[kwd] = (value[0], value[1])

        # Otherwise we're updating an existing cube file
        else:
            with fits.open(self.cube_file, mode='update', memmap=True) as cube_hdu:
                header = cube_hdu[0].header 
                header['HISTORY'] = f'Updated on {str(date.today())} with new FFI delivery.' 
                header['HISTORY'] = f'First FFI is {str(os.path.basename(self.file_list[0]))}'

        # Adding the keywords from the last file
        with fits.open(self.file_list[-1], mode='denywrite', memmap=True) as last_file:
            for kwd in self.last_file_keywords:
                # TICA file structure differs from SPOC in that factory-specific kwds
                # are in the 0th extension, along with the science data.
                header[kwd] = (last_file[0].header[kwd], last_file[0].header.comments[kwd])

        header["EXTNAME"] = "PRIMARY"

        self.primary_header = header

    def _build_info_table(self):
        """ Reading the keywords and setting up the table to hold the image headers (extension 1)
        from every input file.
        """

        with fits.open(self.template_file, mode='denywrite', memmap=True) as ffi_data:
            
            # The image specific header information will be saved in a table in the second extension
            primary_header = ffi_data[0].header

            # set up the image info table
            existing_cols = []
            cols = []
            length = len(self.file_list)
            for kwd, val, cmt in primary_header.cards: 
                if isinstance(val, str):  
                    tpe = "S" + str(len(val))  # TODO: Maybe switch to U?
                elif isinstance(val, int):
                    tpe = np.int32
                else:
                    tpe = np.float64
                    
                if kwd not in existing_cols:
                    existing_cols.append(kwd)
                    cols.append(Column(name=kwd, dtype=tpe, length=length, meta={"comment": cmt}))

            cols.append(Column(name="FFI_FILE", dtype="S" + str(len(os.path.basename(self.template_file))),
                               length=length))
            
            self.info_table = Table(cols)

    def _build_cube_file(self, cube_file):
        """ Build the cube file on disk with primary header, cube extension header,
        and space for the cube, filled with zeros.

        Note, this will overwrite the file if it already exists.

        Parameters
        ----------
        cube_file : str 
            Optional. The filename/path to save the output cube in. 
        """
        
        # Making sure the output directory exists
        direc, _ = os.path.split(cube_file)
        if direc and not os.path.exists(direc):
            os.makedirs(direc)

        # Writing the primary header
        hdu0 = fits.PrimaryHDU(header=self.primary_header)
        hdul = fits.HDUList([hdu0])
        hdul.writeto(cube_file, overwrite=True)

        # Making the cube header and writing it
        data = np.zeros((100, 100, 10, 2), dtype=np.float32)
        hdu = fits.ImageHDU(data)
        header = hdu.header
        header["NAXIS4"], header["NAXIS3"], header["NAXIS2"], header["NAXIS1"] = self.cube_shape

        # Writes the header into the cube as an array of bytes
        with open(cube_file, 'ab') as CUBE:
            CUBE.write(bytearray(header.tostring(), encoding="utf-8"))

        # Expanding the file to fit the full data cube
        # fits requires all blocks to be a multiple of 2880
        cubesize_in_bytes = ((np.prod(self.cube_shape) * 4 + 2880 - 1) // 2880) * 2880
        filelen = os.path.getsize(cube_file)
        
        # Seek to end of file and write null byte
        with open(cube_file, 'r+b') as CUBE:
            CUBE.seek(filelen + cubesize_in_bytes - 1)
            CUBE.write(b'\0')
        

        self.cube_file = cube_file

    def _write_block(self, cube_hdu, start_row=0, end_row=None, fill_info_table=False, verbose=False):
        """ Do one pass through the image files and write a block of the cube.

        cube_hdu is an hdulist object opened in update mode
        """

        # Initializing the sub-cube
        nrows = (self.cube_shape[0] - start_row) if (end_row is None) else (end_row - start_row)
        sub_cube = np.zeros((nrows, *self.cube_shape[1:]), dtype=np.float32)
        
        # Loop through files
        for i, fle in enumerate(self.file_list):

            if verbose:
                st = time()

            # In this section we will take the SCI data from self.file_list above
            # and "paste" a cutout of the full SCI array into a 4d array called 
            # sub_cube. We iterate this process until the full SCI array for each 
            # FFI is copied onto the cube. Usually the sub_cube is ~1/3 the size of 
            # the full SCI array, so there are 3 iterations. 
            # For TICA, the SCI data exists in the 0th extension. 
            with fits.open(fle, mode='denywrite', memmap=True) as ffi_data:

                # Add the image and info to the arrays
                sub_cube[:, :, i, 0] = ffi_data[0].data[start_row:end_row, :]

                del ffi_data[0].data
               
                # Also save the header info in the info table
                if fill_info_table:

                    # Iterate over every keyword in the TICA FFI primary header
                    for kwd in self.info_table.columns:
                        if kwd == "FFI_FILE":
                            self.info_table[kwd][i] = os.path.basename(fle)
                            
                        else:
                            nulval = None
                            if self.info_table[kwd].dtype.name == "int32":
                                nulval = 0
                            elif self.info_table[kwd].dtype.char == "S":  # hacky way to check if it's a string
                                nulval = ""
                            
                            # NOTE: 
                            # A header keyword ('COMMENT') in TICA returns a keyword 
                            # value in the form of a _HeaderCommentaryCard instead of a STRING. 
                            # This breaks the info table because the info table 
                            # doesn't understand what a FITS header commentary card is. 
                            # Adding a try/except is a way to catch when these instances happen 
                            # and turn the keyword value from a _HeaderCommentaryCard to a
                            # string, which is what it's meant to be in the info table. 
                            #
                            # TO-DO: Find a more elegant way to handle these stray 
                            # _HeaderCommentaryCards. 
                            
                            try:
                                self.info_table[kwd][i] = ffi_data[0].header.get(kwd, nulval)
                            except ValueError:
                                kwd_val = ffi_data[0].header.get(kwd)
                                if isinstance(kwd_val, fits.header._HeaderCommentaryCards):
                                    self.info_table[kwd][i] = str(kwd_val)
                                else:
                                    raise

            if verbose:
                print(f"Completed file {i} in {time()-st:.3} sec.")

        # Fill block and flush to disk
        if not self.update:
            cube_hdu[1].data[start_row:end_row, :, :, :] = sub_cube
        else:
            self.cube_append[start_row:end_row, :, :, :] = sub_cube

        if (version_info <= (3, 8)) or (platform == "win32"):
            cube_hdu.flush()

        del sub_cube
    
    def _update_info_table(self):
        """ Updating an existing info table with newly created
        """

        with fits.open(self.template_file, mode='denywrite', memmap=True) as ffi_data:
            
            # The image specific header information will be saved in a table in the second extension
            primary_header = ffi_data[0].header

            # set up the image info table
            cols = []
            for kwd, val, cmt in primary_header.cards: 
                if isinstance(val, str):
                    tpe = "S" + str(len(val))  # TODO: Maybe switch to U?
                elif isinstance(val, int):
                    tpe = np.int32
                else:
                    tpe = np.float64
                    
                # If there's already an info table, this means we are
                # updating a cube instead of making a new one, so expanding 
                # the length by the new FFIs (hence +self.file_list)

                with fits.open(self.cube_file, mode='readonly') as hdul:

                    og_table = hdul[2].data
                    appended_column = np.concatenate((og_table[kwd], self.info_table[kwd]))

                length = len(og_table)+len(self.info_table[kwd])  
                
                cols.append(Column(appended_column, name=kwd, dtype=tpe, length=length, meta={"comment": cmt}))

            with fits.open(self.cube_file, mode='readonly') as hdul:

                og_table = hdul[2].data
                appended_column = np.concatenate((og_table['FFI_FILE'], self.info_table['FFI_FILE']))

                str_length = str(len(os.path.basename(self.template_file)))
                cols.append(Column(appended_column, name="FFI_FILE", dtype="S" + str_length, length=length))
    
            self.info_table = Table(cols)


    def _write_info_table(self):
        """
        Turn the info table into an HDU object and append it to the cube file
        """

        # Make table hdu 
        cols = []
        for kwd in self.info_table.columns:

            if self.info_table[kwd].dtype == np.float64:
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
        with fits.open(self.cube_file, mode='update', memmap=True) as cube_hdus:
            # If we're updating the cube, get rid of the existing table 
            # so we can replace it with the new one. 
            if self.update:
                cube_hdus.pop(index=2)
            cube_hdus.append(table_hdu)


    def _update_cube(self, file_list, cube_file, sector=None, max_memory=50, verbose=True):
        """ Updates an existing cube file if one has already been made and a new delivery is being appended to it. 
        Same functionality as make_cube(...), but working on an already existing file rather than building a new one. 
        This function will: 

        1. Create a new cube consisting of the new FFIs that will be appended to the existing cube
        2. Update primary header keywords to reflect the update to the file
        3. Expand the file size of the FITS file containing the cube, to accomodate for the updated one 
        4. Rename the file accordingly(?)

        """

        self.update = True  # we're updating!

        self.max_memory = max_memory

        if verbose:
            startTime = time()

        # Next locate the existing cube file and assign it a variable
        err_msg = 'Location of the cube file was unsuccessful. Please ensure the correct path was provided.'
        assert os.path.exists(cube_file), err_msg
        self.cube_file = cube_file

        if verbose:
            print(f'Updating cube file: {cube_file}')

        # Ensure that none of the files in file_list are in the cube already, to avoid duplicates
        in_cube = list(fits.getdata(self.cube_file, 2)['FFI_FILE'])

        # TO-DO: Add warning message instead of this verbose print stmnt.
        filtered_file_list = []
        for idx, file in enumerate(file_list): 

            if os.path.basename(file) in in_cube:
                print('File removed from list:')
                print(os.path.basename(file))

            if os.path.basename(file) not in in_cube:
                filtered_file_list.append(file)

        noffis_err_msg = 'No new FFIs found for the given sector.'
        assert len(filtered_file_list) > 0, noffis_err_msg

        if verbose: 
            print(f'{len(filtered_file_list)} new FFIs found!')
        
        # Creating an empty cube that will be appended to the existing cube
        og_cube = fits.getdata(cube_file, 1)
        dimensions = list(og_cube.shape)
        dimensions[2] = len(filtered_file_list)
        self.cube_append = np.zeros(dimensions)

        # Set up the basic cube parameters
        sector = (sector, "Observing sector")
        self._configure_cube(filtered_file_list, sector=sector)

        if verbose:
            print(f"FFIs will be appended in {self.num_blocks} blocks of {self.block_size} rows each.")
        
        # Starting a new info table from scratch with new rows
        self._build_info_table()

        # Update the image cube 
        with fits.open(self.cube_file, mode='update', memmap=True) as cube_hdu:

            # mmap: "allows you to take advantage of lower-level operating system 
            # functionality to read files as if they were one large string or array. 
            # This can provide significant performance improvements in code that 
            # requires a lot of file I/O."
            if (version_info >= (3, 8)) and (platform != "win32"):
                mm = fits.util._get_array_mmap(cube_hdu[1].data)
                # madvise: "Send advice option to the kernel about the memory region 
                # beginning at start and extending length bytes.""
                mm.madvise(MADV_SEQUENTIAL)

            for i in range(self.num_blocks):
                start_row = i * self.block_size
                end_row = start_row + self.block_size

                if end_row >= self.cube_shape[0]:
                    end_row = None

                # filling in the cube file with the new FFIs
                # the info table also gets updated here 
                fill_info_table = True
                self._write_block(cube_hdu, start_row, end_row, fill_info_table, verbose)

                if verbose:
                    print(f"Completed block {i+1} of {self.num_blocks}")
       
        # Append the new cube to the existing cube
        new_cube = np.concatenate((og_cube, self.cube_append), axis=2)

        # Add it to the HDU list 
        with fits.open(self.cube_file, mode='update') as hdul:
            
            if verbose:
                print(f'Original cube of size: {str(og_cube.shape)}')
                print(f'will now be replaced with cube of size: {str(new_cube.shape)}')
                print(f'for file ``{cube_file}``')
            hdul[1].data = new_cube

        # Appending new info table to original 
        self._update_info_table()
        
        # Writing the info table to EXT2 of the FITS file 
        self._write_info_table()

        if verbose:
            print(f"Total time elapsed: {(time() - startTime)/60:.2f} min")

        return self.cube_file


    def make_cube(self, file_list, cube_file="img-cube.fits", sector=None, max_memory=50, verbose=True):
        """
        Analogous to CubeFactory.make_cube(...). 
        Turns a list of fits image files into one large data-cube.
        Input images must all have the same footprint and resolution.
        The resulting datacube is transposed for quicker cutouts.
        This function can take some time to run, exactly how much time will depend on the number
        of input files and the maximum allowed memory. The runtime will be fastest if the
        entire data cube can be held in memory, however that can be quite large (~40GB for a full
        TESS main mission sector, 3 times that for a TESS extended mission sector).

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
        max_memory : float
            Optional, default is 50. The maximum amount of memory to make available for building
            the data cube in GB. Note, this is the maximum amount of space to be used for the cube
            array only, so should not be set to the full amount of memory on the system.
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

        self.max_memory = max_memory

        # Set up the basic cube parameters
        sector = (sector, "Observing sector")
        self._configure_cube(file_list, sector=sector)

        if verbose:
            print("Using {} to initialize the image header table.".format(os.path.basename(self.template_file)))
            print(f"Cube will be made in {self.num_blocks} blocks of {self.block_size} rows each.")
        
        # Set up the table to hold the individual image headers
        self._build_info_table()
        
        # Write the empty file, ready for the cube to be added
        self._build_cube_file(cube_file)
        
        # Fill the image cube
        with fits.open(self.cube_file, mode='update', memmap=True) as cube_hdu:
            
            if (version_info >= (3, 8)) and (platform != "win32"):
                mm = fits.util._get_array_mmap(cube_hdu[1].data)
                mm.madvise(MADV_SEQUENTIAL)

            for i in range(self.num_blocks):
                start_row = i * self.block_size
                end_row = start_row + self.block_size

                if end_row >= self.cube_shape[0]:
                    end_row = None

                fill_info_table = True if (i == 0) else False
                self._write_block(cube_hdu, start_row, end_row, fill_info_table, verbose)
                  
                if verbose:
                    print(f"Completed block {i+1} of {self.num_blocks}")

        # Add the info table to the cube file
        self._write_info_table()
        if verbose:
            print(f"Total time elapsed: {(time() - startTime)/60:.2f} min")

        return self.cube_file
        

