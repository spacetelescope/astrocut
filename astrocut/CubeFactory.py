# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module implements the functionality to create image cubes for the purposes of
creating cutout target pixel files.
"""

import os
from copy import deepcopy
from datetime import date
from sys import platform, version_info
from time import monotonic

import numpy as np
from astropy.io import fits
from astropy.table import Column, Table

from . import log
from .utils.utils import _handle_verbose

if (version_info >= (3, 8)) and (platform != "win32"):
    from mmap import MADV_SEQUENTIAL

ERROR_MSG = ("One or more incorrect file types were input. Please input TICA FFI files when using "
             "``TicaCubeFactory``, and SPOC FFI files when using ``CubeFactory``.")


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
        self.template_requirements = {'WCSAXES': 2}  # TESS-specific (assumed to be in extension 1)

        self.file_list = None
        self.template_file = None
        
        self.primary_header = None
        self.info_table = None
        self.cube_file = None
        self.img_ext = 1  # The extension of the image data in the input files

        # Used when updating an existing cube 
        self.old_cols = None
        self.update = False
        self.cube_append = None

    def _get_img_start_time(self, img_data):
        try:
            return img_data[self.img_ext].header.get(self.time_keyword)
        except IndexError:
            raise ValueError(ERROR_MSG)
        
    def _get_img_shape(self, img_data):
        return img_data[self.img_ext].data.shape
    
    def _get_cube_shape(self, image_shape):
        return (image_shape[0], image_shape[1], len(self.file_list), 2)

    def _configure_cube(self, file_list, **extra_keywords):
        """ 
        Run through all the files and set up the basic parameters for the cube.

        Set up the cube primary header.
        """
        
        file_list = np.array(file_list)
        image_shape = None
        start_times = np.zeros(len(file_list))
        for i, image in enumerate(file_list):

            img_data = fits.open(image, mode='denywrite', memmap=True)
            
            start_times[i] = self._get_img_start_time(img_data)

            if image_shape is None:  # Only need to fill this once
                image_shape = self._get_img_shape(img_data)
            
            if self.template_file is None:  # Only check this if we don't already have it

                is_template = True
                for key, value in self.template_requirements.items():
                    if img_data[self.img_ext].header.get(key) != value:  # Checking for a good image header
                        is_template = False
                        
                if is_template:
                    self.template_file = image
                    
            img_data.close()
                
        self.file_list = file_list[np.argsort(start_times)]

        # Working out the block size and number of blocks needed for writing the cube
        # without using too much memory
        slice_size = image_shape[1] * len(self.file_list) * 2 * 4  # in bytes (float32)
        max_block_size = (self.max_memory * 1e9) // slice_size
        
        self.num_blocks = int(image_shape[0] / max_block_size + 1)
        self.block_size = int(image_shape[0] / self.num_blocks + 1)

        # Determine cube shape
        if not self.update:
            # New cube
            self.cube_shape = self._get_cube_shape(image_shape)
        else:
            # Update to an existing cube
            self.cube_shape = self.cube_append.shape

        # Making the primary header
        if not self.update:
            with fits.open(self.file_list[0], mode='denywrite', memmap=True) as first_file:
                header = deepcopy(first_file[0].header)
                header.remove('CHECKSUM', ignore_missing=True)

                # Adding standard keywords
                header['ORIGIN'] = 'STScI/MAST'
                header['DATE'] = str(date.today())

                # Adding factory specific keywords
                for kwd in self.image_header_keywords:
                    header[kwd] = (first_file[self.img_ext].header[kwd], first_file[self.img_ext].header.comments[kwd])

                # Adding the extra keywords passed in
                for kwd, value in extra_keywords.items():
                    header[kwd] = (value[0], value[1])
        else:
            # Updating an existing cube
            with fits.open(self.cube_file, mode='update', memmap=True) as cube_hdu:
                header = cube_hdu[0].header
                header['HISTORY'] = f'Updated on {str(date.today())} with new image delivery.'             
                header['HISTORY'] = f'First image is {str(os.path.basename(self.file_list[0]))}'
                
        # Adding the keywords from the last file
        with fits.open(self.file_list[-1], mode='denywrite', memmap=True) as last_file:
            for kwd in self.last_file_keywords:
                header[kwd] = (last_file[self.img_ext].header[kwd], last_file[self.img_ext].header.comments[kwd])

        header["EXTNAME"] = "PRIMARY"

        self.primary_header = header


    def _build_info_table(self):
        """
        Reading the keywords and setting up the table to hold the image headers (extension 1)
        from every input file.
        """
        with fits.open(self.template_file, mode='denywrite', memmap=True) as ffi_data:
            
            # The image specific header information will be saved in a table in the second extension
            img_header = ffi_data[self.img_ext].header

            # Set up the image info table
            cols = []
            existing_cols = []
            length = len(self.file_list)
            for kwd, val, cmt in img_header.cards: 
                if isinstance(val, str):
                    tpe = "S" + str(len(val))  # TODO: Maybe switch to U?
                elif isinstance(val, int):
                    tpe = np.int32
                else:
                    tpe = np.float64

                # Adding columns one by one 
                if kwd not in existing_cols:
                    existing_cols.append(kwd)
                    cols.append(Column(name=kwd, dtype=tpe, length=length, meta={"comment": cmt}))
                    
            cols.append(Column(name="FFI_FILE", 
                               dtype="S" + str(len(os.path.basename(self.template_file))),
                               length=length))
            self.info_table = Table(cols)

        
    def _build_cube_file(self, cube_file):
        """
        Build the cube file on disk with primary header, cube extension header,
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


    def _write_to_sub_cube(self, sub_cube, idx, img_data, start_row, end_row):
        # add the image and info to the arrays
        sub_cube[:, :, idx, 0] = img_data[1].data[start_row:end_row, :]
        sub_cube[:, :, idx, 1] = img_data[2].data[start_row:end_row, :]

        del img_data[1].data
        del img_data[2].data


    def _get_header_keyword(self, kwd, idx, img_data, nulval):
        self.info_table[kwd][idx] = img_data[1].header.get(kwd, nulval)


    def _write_block(self, cube_hdu, start_row=0, end_row=None, fill_info_table=False, verbose=False):
        """
        Do one pass through the image files and write a block of the cube.

        cube_hdu is an hdulist object opened in update mode
        """
        # Log messages based on verbosity
        _handle_verbose(verbose)

        # Initializing the sub-cube
        nrows = (self.cube_shape[0] - start_row) if (end_row is None) else (end_row - start_row)
        sub_cube = np.zeros((nrows, *self.cube_shape[1:]), dtype=np.float32)
        
        # Loop through files
        for i, fle in enumerate(self.file_list):
            st = monotonic()
            with fits.open(fle, mode='denywrite', memmap=True) as img_data:

                # Add the image and info to the arrays
                self._write_to_sub_cube(sub_cube, i, img_data, start_row, end_row)
                
                # Also save the header info in the info table
                if fill_info_table:
                    # Iterate over every keyword in the primary header
                    for kwd in self.info_table.columns:
                        if kwd == "FFI_FILE":
                            self.info_table[kwd][i] = os.path.basename(fle)
                        else:
                            nulval = None
                            if self.info_table[kwd].dtype.name == "int32":
                                nulval = 0
                            elif self.info_table[kwd].dtype.char == "S":  # hacky way to check if it's a string
                                nulval = ""
                            self._get_header_keyword(kwd, i, img_data, nulval)

            log.debug("Completed file %d in %.3f sec.", i, monotonic() - st)

        # Fill block and flush to disk
        if not self.update:
            cube_hdu[1].data[start_row:end_row, :, :, :] = sub_cube
        else:
            self.cube_append[start_row:end_row, :, :, :] = sub_cube

        if (version_info <= (3, 8)) or (platform == "win32"):
            cube_hdu.flush()

        del sub_cube


    def _update_info_table(self):
        """ 
        Updating an existing info table with newly created
        """

        with fits.open(self.template_file, mode='denywrite', memmap=True) as img_data:
            
            # The image specific header information will be saved in a table in the second extension
            img_header = img_data[self.img_ext].header

            # set up the image info table
            cols = []
            for kwd, val, cmt in img_header.cards: 
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


    def _append_table_to_cube(self, table_hdu):
        with fits.open(self.cube_file, mode='append', memmap=True) as cube_hdus:
            cube_hdus.append(table_hdu)

        
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
        self._append_table_to_cube(table_hdu)


    def _update_cube(self, file_list, cube_file, sector=None, max_memory=50, verbose=True):
        """ Updates an existing cube file if one has already been made and a new delivery is being appended to it. 
        Same functionality as make_cube(...), but working on an already existing file rather than building a new one. 
        This function will: 

        1. Create a new cube consisting of the new FFIs that will be appended to the existing cube
        2. Update primary header keywords to reflect the update to the file
        3. Expand the file size of the FITS file containing the cube, to accomodate for the updated one 
        4. Rename the file accordingly(?)

        """
        # Log messages based on verbosity
        _handle_verbose(verbose)
        startTime = monotonic()
        self.update = True  # we're updating!
        self.max_memory = max_memory

        # Next locate the existing cube file and assign it a variable
        err_msg = 'Location of the cube file was unsuccessful. Please ensure the correct path was provided.'
        assert os.path.exists(cube_file), err_msg
        self.cube_file = cube_file

        log.debug('Updating cube file: %s', cube_file)

        # Ensure that none of the files in file_list are in the cube already, to avoid duplicates
        in_cube = list(fits.getdata(self.cube_file, 2)['FFI_FILE'])

        # TO-DO: Add warning message instead of this verbose print stmnt.
        filtered_file_list = []
        for idx, file in enumerate(file_list): 

            if os.path.basename(file) in in_cube:
                log.info('File removed from list:')
                log.info(os.path.basename(file))

            if os.path.basename(file) not in in_cube:
                filtered_file_list.append(file)

        noffis_err_msg = 'No new FFIs found for the given sector.'
        assert len(filtered_file_list) > 0, noffis_err_msg

        log.debug('%d new FFIs found!', len(filtered_file_list))
        
        # Creating an empty cube that will be appended to the existing cube
        og_cube = fits.getdata(cube_file, 1)
        dimensions = list(og_cube.shape)
        dimensions[2] = len(filtered_file_list)
        self.cube_append = np.zeros(dimensions)

        # Set up the basic cube parameters
        sector = (sector, "Observing sector")
        self._configure_cube(filtered_file_list, sector=sector)

        log.debug("FFIs will be appended in %d blocks of %d rows each.", self.num_blocks, self.block_size)
        
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
                log.debug("Completed block %d of %d", i + 1, self.num_blocks)
       
        # Append the new cube to the existing cube
        new_cube = np.concatenate((og_cube, self.cube_append), axis=2)

        # Add it to the HDU list 
        with fits.open(self.cube_file, mode='update') as hdul:
            log.debug('Original cube of size: %s', og_cube.shape)
            log.debug('will now be replaced with cube of size: %s', new_cube.shape)
            log.debug('for file ``%s``', cube_file)
            hdul[1].data = new_cube

        # Appending new info table to original 
        self._update_info_table()
        
        # Writing the info table to EXT2 of the FITS file 
        self._write_info_table()
        log.debug("Total time elapsed: %.2f min", (monotonic() - startTime) / 60)

        return self.cube_file


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
        # Log messages based on verbosity
        _handle_verbose(verbose)
        startTime = monotonic()

        self.max_memory = max_memory

        # Set up the basic cube parameters
        sector = (sector, "Observing sector")

        self._configure_cube(file_list, sector=sector)
    
        log.debug("Using %s to initialize the image header table.", os.path.basename(self.template_file))
        log.debug("Cube will be made in %d blocks of %d rows each.", self.num_blocks, self.block_size)

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
                  
                log.debug("Completed block %d of %d", i + 1, self.num_blocks)

        # Add the info table to the cube file
        self._write_info_table()
        log.debug("Total time elapsed: %.2f min", (monotonic() - startTime) / 60)

        return self.cube_file
