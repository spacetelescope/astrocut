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
from sys import version_info, platform

if (version_info >= (3, 8)) and (platform != "win32"):
    from mmap import MADV_SEQUENTIAL

__all__ = ['CubeFactory', 'TicaCubeFactory']

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
        slice_size = image_shape[1] * len(self.file_list) * 2 * 4  # in bytes (float32)
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
                if type(val) == str:
                    tpe = "S" + str(len(val))  # TODO: Maybe switch to U?
                elif type(val) == int:
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

    def __init__(self, max_memory=50):
        """ Setting up the class members. NOTE: Later down the line 
        we might want to generalize CubeFactory so that we no longer need 
        a separate class for the TICA cubes. 
        
        Modifying the __init__ 
        in CubeFactory so that it can switch parameters based on whether 
        we want cubes for TICA or TESS will probably be the route I take.
        Or see how this is generalized when Zcut wraps CubeFactory? 
        """

        self.max_memory = max_memory  # in GB
        self.block_size = None  # Number of rows
        self.num_blocks = None
        self.cube_shape = None
        
        self.time_keyword = 'STARTTJD'  # Start time in TJD. TICA-specific.
        self.last_file_keywords = ['ENDTJD']  # Stop time in TJD. TICA-specific (assumed to be in extension 0)                  
        self.image_header_keywords = ['CAMNUM', 'CCDNUM'] # Camera number and CCD number being used for the sector observation. 
                                                       # TICA-specific (assumed to be in extension 0)
        self.template_requirements = {'NAXIS': 2} # Using NAXIS instead of WCSAXES because TICA headers dont have WCSAXES kw.
                                              # Assuming NAXIS and WCSAXES would always have same values.
        self.file_list = None
        self.template_file = None
        
        self.primary_header = None
        self.info_table = None
        self.cube_file = None
        self.old_cols = None

    def _configure_cube(self, file_list, **extra_keywords):
        """ Run through all the files and set up the  basic parameters for the cube.
        Set up the cube primary header.
        """

        file_list = np.array(file_list)
        image_shape = None
        start_times = np.zeros(len(file_list))
        for i, ffi in enumerate(file_list):
            ffi_data = fits.open(ffi, mode='denywrite', memmap=True)
            
            start_times[i] = ffi_data[0].header.get('STARTTJD')

            if image_shape is None:  # Only need to fill this once
                image_shape = ffi_data[0].data.shape
            
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
        max_block_size = int((self.max_memory * 1e9)//slice_size)
        
        self.num_blocks = int(image_shape[0]/max_block_size + 1)
        self.block_size = int(image_shape[0]/self.num_blocks + 1)
        #                  NROWS,           NCOLS,         Num. Images, 
        if self.cube_file is None:
            self.cube_shape = (image_shape[0], image_shape[1], len(self.file_list), 2)
        else:  
            cube_shape = list(fits.getdata('tica-s0027-1-2-cube.fits', 1).shape)
            cube_shape[2] = cube_shape[2] + len(self.file_list)
            self.cube_shape = cube_shape  
            
        # Making the primary header if there's no cube_file yet
        if self.cube_file is None:
            with fits.open(self.file_list[0], mode='denywrite', memmap=True) as first_file:
                header = deepcopy(first_file[0].header)
                header.remove('CHECKSUM', ignore_missing=True)

                # Adding standard keywords
                header['ORIGIN'] = 'STScI/MAST'
                header['DATE'] = str(date.today())

                # Adding factory specific keywords
                for kwd in self.image_header_keywords:
                    header[kwd] = (first_file[0].header[kwd], first_file[0].header.comments[kwd])

                # Adding the extra keywords passed in
                for kwd, value in extra_keywords.items():
                    header[kwd] = (value[0], value[1])

        # Otherwise we're updating an existing cube file
        else:
            with fits.open(self.cube_file, mode='update', memmap=True) as cube_hdu:
                header = cube_hdu[0].header 
                header['DATE'] = str(date.today()) # Update

        # Adding the keywords from the last file
        with fits.open(self.file_list[-1], mode='denywrite', memmap=True) as last_file:
            for kwd in self.last_file_keywords:
                header[kwd] = (last_file[0].header[kwd], last_file[0].header.comments[kwd])

        header["EXTNAME"] = "PRIMARY"

        self.primary_header = header

    def _build_info_table(self):
        """ Reading the keywords and setting up the table to hold the image headers (extension 1)
        from every input file.
        """

        with fits.open(self.template_file, mode='denywrite', memmap=True) as ffi_data:
            
            # The image specific header information will be saved in a table in the second extension
            secondary_header = ffi_data[0].header

            # set up the image info table
            cols = []
            for kwd, val, cmt in secondary_header.cards: 
                if type(val) == str:
                    tpe = "S" + str(len(val))  # TODO: Maybe switch to U?
                elif type(val) == int:
                    tpe = np.int32
                else:
                    tpe = np.float64
                    
                # If there's already an info table, this means we are
                # updating a cube instead of making a new one, so expanding 
                # the length by the new FFIs (hence +self.file_list)

                if self.info_table is not None: 
                    length = len(self.info_table)+len(self.file_list)
                else: 
                    length = len(self.file_list)

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

        # I think this part increases the buffer size of ext1 and writes 
        # the cube in there?
        with open(cube_file, 'ab') as CUBE:
            CUBE.write(bytearray(header.tostring(), encoding="utf-8"))

        # Expanding the file to fit the full data cube
        # fits requires all blocks to be a multiple of 2880
        cubesize_in_bytes = ((np.prod(self.cube_shape) * 4 + 2880 - 1) // 2880) * 2880
        filelen = os.path.getsize(cube_file)
        
        with open(cube_file, 'r+b') as CUBE:
            # What is this .seek?
            # Looks like it's expanding the buffer size of the file?
            CUBE.seek(filelen + cubesize_in_bytes - 1)
            # b for buffer?
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

            if self.old_cols: 
                i = len(self.old_cols['SIMPLE']) + i 

            with fits.open(fle, mode='denywrite', memmap=True) as ffi_data:

                # add the image and info to the arrays
                sub_cube[:, :, i, 0] = ffi_data[0].data[start_row:end_row, :]

                del ffi_data[0].data
               
                if fill_info_table:  # Also save the header info in the info table

                    for kwd in self.info_table.columns: # Iterate over every keyword in the TICA FFI primary header
                        if kwd == "FFI_FILE":
                            self.info_table[kwd][i] = os.path.basename(fle)
                        else:
                            nulval = None
                            if self.info_table[kwd].dtype.name == "int32":
                                nulval = 0
                            elif self.info_table[kwd].dtype.char == "S":  # hacky way to check if it's a string
                                nulval = ""
                            
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

                elif self.old_cols:

                    for kwd in self.info_table.columns:
                        if kwd == "FFI_FILE":
                            self.info_table[kwd] = self.old_cols[kwd].append(fle)
                        else:
                            nulval = None
                            if self.info_table[kwd].dtype.name == "int32":
                                nulval = 0
                            elif self.info_table[kwd].dtype.char == "S":  # hacky way to check if it's a string
                                nulval = ""
                            
                            try:
                                self.info_table[kwd] = self.old_cols[kwd].append(ffi_data[0].header.get(kwd, nulval))
                            except ValueError:
                                kwd_val = ffi_data[0].header.get(kwd)
                                if isinstance(kwd_val, fits.header._HeaderCommentaryCards):
                                    self.info_table[kwd] = self.old_cols[kwd].append(str(kwd_val))

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


    def update_cube(self, file_list, cube_file, verbose=True):
        """ Updates an existing cube file if one has already been made and a new delivery is being appended to it. 
        Same functionality as make_cube(...), but working on an already existing file rather than building a new one. 
        """

        if verbose:
            startTime = time()
            
        # First locate the existing cube file and assign it a variable
        assert os.path.exists(cube_file), 'Location of the cube file was unsuccessful. Please ensure the correct path was provided. If file does not exist, create a new cube using ``~TicaCubeFactory.make_cube()``.'
        self.cube_file = cube_file

        if verbose:
            print(f'Updating cube file: {cube_file}')

        # Ensure that none of the files in file_list are in the cube already, to avoid duplicates
        in_cube = list(fits.getdata(self.cube_file, 2)['FFI_FILE'])

        for file in file_list: 
            assert file not in in_cube, f'FFI {file} is already in the cube. Removing it from ``file_list``.'
            file_list.remove(file)
        
        self._configure_cube(file_list)

        if verbose:
            print(f"FFIs will be appended in {self.num_blocks} blocks of {self.block_size} rows each.")
        
        # Expanding the length of the info table to accomodate new FFIs
        self.info_table = fits.getdata(self.cube_file, 2)
        self.old_cols = {}
        for column in self.info_table.columns:
            
            col = list(self.info_table[column.name])
            self.old_cols[column.name] = col

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
                fill_info_table = False
                self._write_block(cube_hdu, start_row, end_row, fill_info_table, verbose)

                if verbose:
                    print(f"Completed block {i+1} of {self.num_blocks}")

        # Add the info table to the cube file
        self._write_info_table()
        if verbose:
            print(f"Total time elapsed: {(time() - startTime)/60:.2f} min")

        return self.cube_file


    def make_cube(self, file_list, cube_file='img-cube.fits', verbose=True, max_memory=50):

        if verbose:
            startTime = time()

        self.max_memory = max_memory

        # Set up the basic cube parameters
        self._configure_cube(file_list)

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
        

