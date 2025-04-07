# Licensed under a 3-clause BSD style license - see LICENSE.rst

from copy import deepcopy
from datetime import date
from pathlib import Path
from time import monotonic
from typing import List, Optional, Union
import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Column, Table

# Try to import the MADV_SEQUENTIAL constant from the mmap module
# May fail on older Python versions or on Windows
try:
    from mmap import MADV_SEQUENTIAL
    mmap_imported = True
except ImportError:
    mmap_imported = False

from . import log
from .exceptions import DataWarning, InvalidInputError
from .utils.utils import _handle_verbose


class CubeFactory():
    """
    Class for creating image cubes. This class is built to accept TESS SPOC FFI files,
    but can be extended to work with other types of image files.

    Parameters
    ----------
    max_memory : int
        The maximum amount of memory to make available for building the data cube in GB.
        Note, this is the maximum amount of space to be used for the cube array only,
        so should not be set to the full amount of memory on the system.
    
    Methods
    -------
    make_cube(file_list, cube_file, sector, max_memory, verbose)
        Turns a list of FITS image files into one large data cube.
    update_cube(file_list, cube_file, sector, max_memory, verbose)
        Updates an existing cube file with new FITS images.
    """

    ERROR_MSG = ('One or more incorrect file types were input. Please input TICA FFI files when using '
                 '``TicaCubeFactory``, and SPOC FFI files when using ``CubeFactory``.')

    def __init__(self, max_memory: int = 50):
        """ Setting up the class members """

        self._max_memory = max_memory  # in GB
        self._block_size = None  # Number of rows
        self._num_blocks = None
        self._cube_shape = None
        
        self._time_keyword = 'TSTART'  # TESS-specific
        self._last_file_keywords = ['DATE-END', 'TSTOP']  # TESS-specific (assumed to be in extension 1)
        self._image_header_keywords = ['CAMERA', 'CCD']  # TESS-specific
        self._template_requirements = {'WCSAXES': 2}  # TESS-specific (assumed to be in extension 1)
        self._file_keyword = 'FFI_FILE'  # TESS-specific, used to build the info table
        self._keywords_in_use = ['XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'PCOUNT', 'GCOUNT', 'TFIELDS']

        self._file_list = None
        self._template_file = None
        
        self._primary_header = None
        self._info_table = None
        self._cube_file = None
        self._img_ext = 1  # The extension of the image data in the input files
        # Number of axes in the first dimension of image data, SPOC has both data and error values
        self._naxis1 = 2

        # Used when updating an existing cube 
        self._old_cols = None
        self._update = False
        self._cube_append = None

    def _get_img_start_time(self, img_data: fits.HDUList) -> float:
        """
        Get the start time of the image.

        Parameters
        ----------
        img_data : HDUList
            The image data.

        Returns
        -------
        float
            The start time of the image.

        Raises
        ------
        ValueError
            If any of the input files are not in the expected format.
        """
        try:
            return img_data[self._img_ext].header.get(self._time_keyword)
        except IndexError:
            # If image does not have a second extension, raise an error
            raise ValueError(self.ERROR_MSG)
        
    def _get_img_shape(self, img_data: fits.HDUList) -> tuple:
        """
        Get the shape of the image data.

        Parameters
        ----------
        img_data : HDUList
            The image data.

        Returns
        -------
        tuple
            The shape of the image data.
        """
        return img_data[self._img_ext].data.shape

    def _configure_cube(self, file_list: List[str], **extra_keywords: dict):
        """ 
        Iterate through the input files and set up the basic parameters and primary header for the data cube.

        Parameters
        ----------
        file_list : list
            List of input files.
        extra_keywords : dict
            Extra keywords to add to the primary header.

        Raises
        ------
        ValueError
            If any of the input files are not in the expected format.
        """
        
        file_list = np.array(file_list)
        image_shape = None
        start_times = np.zeros(len(file_list))

        # Iterate through files
        for i, image in enumerate(file_list):
            with fits.open(image, mode='denywrite', memmap=True) as img_data:
                # Add the image start time to array
                start_times[i] = self._get_img_start_time(img_data)

                if image_shape is None:  # Only need to fill this once
                    image_shape = self._get_img_shape(img_data)
                
                if self._template_file is None:  # Only check for template if one doesn't exist
                    # Template file must match the template requirements
                    if all(img_data[self._img_ext].header.get(key) == value for key, value 
                           in self._template_requirements.items()):
                        self._template_file = image
                
        # Sort the files by start time
        self._file_list = file_list[np.argsort(start_times)]

        # Working out the block size and number of blocks needed for writing the cube
        # without using too much memory
        slice_size = image_shape[1] * len(self._file_list) * 2 * 4  # in bytes (float32)
        max_block_size = (self._max_memory * 1e9) // slice_size
        self._num_blocks = int(image_shape[0] / max_block_size + 1)
        self._block_size = int(image_shape[0] / self._num_blocks + 1)

        # Determine cube shape
        if not self._update:
            # Making a new cube
            self._cube_shape = (image_shape[0], image_shape[1], len(self._file_list), self._naxis1)

            # Set up the primary header with earliest file
            with fits.open(self._file_list[0], mode='denywrite', memmap=True) as first_file:
                # Make a copy of the header and remove the checksum
                header = deepcopy(first_file[0].header)
                header.remove('CHECKSUM', ignore_missing=True)

                # Adding standard keywords
                header['ORIGIN'] = 'STScI/MAST'
                header['DATE'] = str(date.today())

                # Adding factory specific keywords
                for kwd in self._image_header_keywords:
                    header[kwd] = (first_file[self._img_ext].header[kwd], 
                                   first_file[self._img_ext].header.comments[kwd])

                # Adding the extra keywords passed in
                for kwd, value in extra_keywords.items():
                    header[kwd] = (value[0], value[1])
        else:
            # Update to an existing cube
            self._cube_shape = self._cube_append.shape

            # Update the primary header with new history
            with fits.open(self._cube_file, mode='update', memmap=True) as cube_hdu:
                header = cube_hdu[0].header
                header['HISTORY'] = f'Updated on {str(date.today())} with new image delivery.'             
                header['HISTORY'] = f'First image is {Path(self._file_list[0]).name}.'
                
        # Adding the keywords from the last file
        with fits.open(self._file_list[-1], mode='denywrite', memmap=True) as last_file:
            for kwd in self._last_file_keywords:
                header[kwd] = (last_file[self._img_ext].header[kwd], last_file[self._img_ext].header.comments[kwd])

        # Set extension name
        header['EXTNAME'] = 'PRIMARY'

        self._primary_header = header


    def _build_info_table(self):
        """
        Read the keywords and set up the table to hold the image headers from every input file.
        """
        with fits.open(self._template_file, mode='denywrite', memmap=True) as template_data:
            
            # The image specific header information will be saved in a table in the image extension
            img_header = template_data[self._img_ext].header

            # Set up the image info table
            cols = []
            existing_cols = []
            length = len(self._file_list)
            for kwd, val, cmt in img_header.cards:
                # Determine column type
                if isinstance(val, str):
                    tpe = 'S' + str(len(val))  # TODO: Maybe switch to U?
                elif isinstance(val, int):
                    tpe = np.int32
                else:
                    tpe = np.float64

                # Add column if it doesn't already exist
                if kwd not in existing_cols:
                    existing_cols.append(kwd)
                    cols.append(Column(name=kwd, dtype=tpe, length=length, meta={'comment': cmt}))
                    
            # Adding a column for the input file names
            cols.append(Column(name=self._file_keyword,
                               dtype=f'S{len(Path(self._template_file).name)}',
                               length=length))
            
            # Build Table from columns
            self._info_table = Table(cols)
        
    def _build_cube_file(self, cube_file: str):
        """
        Build the cube file on disk with the primary header, cube extension header,
        and space for the cube, filled with zeros.

        Note: This will overwrite the file if it already exists.

        Parameters
        ----------
        cube_file : str 
            The filename/path to save the output cube in. 
        """
        # Ensure that the output directory exists
        dir = Path(cube_file).parent
        if dir and not dir.exists():
            dir.mkdir(parents=True)

        # Write the primary header
        hdu0 = fits.PrimaryHDU(header=self._primary_header)            
        hdul = fits.HDUList([hdu0])            
        hdul.writeto(cube_file, overwrite=True)

        # Make the cube header and write it
        data = np.zeros((100, 100, 10, 2), dtype=np.float32)
        hdu = fits.ImageHDU(data)
        header = hdu.header
        header['NAXIS4'], header['NAXIS3'], header['NAXIS2'], header['NAXIS1'] = self._cube_shape

        # Write the header into the cube as an array of bytes
        with open(cube_file, 'ab') as CUBE:
            CUBE.write(bytearray(header.tostring(), encoding='utf-8'))

        # Expand the file to fit the full data cube
        # FITS requires all blocks to be a multiple of 2880
        cubesize_in_bytes = ((np.prod(self._cube_shape) * 4 + 2880 - 1) // 2880) * 2880
        file_len = Path(cube_file).stat().st_size

        # Seek to end of file and write null byte
        with open(cube_file, 'r+b') as CUBE:
            CUBE.seek(file_len + cubesize_in_bytes - 1)
            CUBE.write(b'\0')

        self._cube_file = cube_file

    def _write_to_sub_cube(self, sub_cube: np.ndarray, idx: int, img_data: fits.HDUList, start_row: int, end_row: int):
        """
        Write data from an input image to a sub-cube.

        Parameters
        ----------
        sub_cube : numpy.ndarray
            The sub-cube to write to.
        idx : int
            The index of the input file.
        img_data : HDUList
            The image data.
        start_row : int
            The starting row of the block.
        end_row : int
            The ending row of the block.
        """
        # Add image and uncertainty data to the sub-cube
        sub_cube[:, :, idx, 0] = img_data[1].data[start_row:end_row, :]
        sub_cube[:, :, idx, 1] = img_data[2].data[start_row:end_row, :]

        # Remove the data from the input image to save memory
        del img_data[1].data
        del img_data[2].data

    def _get_header_keyword(self, kwd: str, img_data: fits.HDUList, nulval: Optional[Union[int, str]]):
        """
        Get a header keyword from an input image and save it to the info table.

        Parameters
        ----------
        kwd : str
            The keyword to get.
        img_data : HDUList
            The image data.
        nulval : int or str
            The null value for the keyword.
        """
        return img_data[1].header.get(kwd, nulval)

    def _write_block(self, cube_hdu: fits.HDUList, start_row: int = 0, end_row: Optional[int] = None, 
                     fill_info_table: bool = False):
        """
        Write a block of the cube with data from input images.

        Parameters
        ----------
        cube_hdu : HDUList
            The cube FITS to write to.
        start_row : int
            The starting row of the block.
        end_row : int
            Optional. The ending row of the block.
        fill_info_table : bool
            If True, fill the info table with the header keywords.
        """
        # Initializing the sub-cube
        nrows = (self._cube_shape[0] - start_row) if (end_row is None) else (end_row - start_row)
        sub_cube = np.zeros((nrows, *self._cube_shape[1:]), dtype=np.float32)
        
        # Loop through input files
        for i, fle in enumerate(self._file_list):
            st = monotonic()
            with fits.open(fle, mode='denywrite', memmap=True) as img_data:
                # Write data from input image to sub-cube
                self._write_to_sub_cube(sub_cube, i, img_data, start_row, end_row)
                
                if fill_info_table:  # Also save the header info in the info table
                    # Iterate over every keyword in the primary header
                    for kwd in self._info_table.columns:
                        if kwd == self._file_keyword:
                            # Assign this keyword to the name of the file
                            self._info_table[kwd][i] = Path(fle).name
                        else:
                            # Determine null value based on dtype
                            nulval = None
                            if self._info_table[kwd].dtype.name == 'int32':
                                nulval = 0
                            elif self._info_table[kwd].dtype.char == 'S':  # hacky way to check if it's a string
                                nulval = ''
                            # Assign the keyword value from the image header
                            self._info_table[kwd][i] = self._get_header_keyword(kwd, img_data, nulval)

            log.debug('Completed file %d in %.3f sec.', i, monotonic() - st)

        # Fill block and flush to disk
        if not self._update:
            cube_hdu[1].data[start_row:end_row, :, :, :] = sub_cube
        else:
            self._cube_append[start_row:end_row, :, :, :] = sub_cube

        if not mmap_imported:
            # Need to flush with older Python versions (< 3.8) and on Windows because
            # memory-mapped files may not properly save changes
            cube_hdu.flush()

        # Delete the sub-cube to save memory
        del sub_cube

    def _write_info_table(self):
        """
        Append the info table to the cube file as a binary table.
        """
        # Iterate through info table keywords to create an array of fits.Column objects
        cols = []
        for kwd in self._info_table.columns:
            # Determine column type
            if self._info_table[kwd].dtype == np.float64:
                tpe = 'D'
            elif self._info_table[kwd].dtype == np.int32:
                tpe = 'J'
            else:
                tpe = str(self._info_table[kwd].dtype).replace('S', 'A').strip('|')

            # Create Column object and add to array
            cols.append(fits.Column(name=kwd, format=tpe, array=self._info_table[kwd]))
        
        # Create the table HDU
        col_def = fits.ColDefs(cols)
        table_hdu = fits.BinTableHDU.from_columns(col_def)

        # Add comments to the header
        for kwd in self._info_table.columns:
            if kwd in self._keywords_in_use:
                continue  # skipping the keyword already in use
            table_hdu.header[kwd] = self._info_table[kwd].meta.get('comment', '')

        # Append to the cube file
        with fits.open(self._cube_file, mode='update', memmap=True) as cube_hdus:
            if self._update:
                # If we're updating the cube, get rid of the existing table 
                # so we can replace it with the new one. 
                cube_hdus.pop(index=2)
            cube_hdus.append(table_hdu)

    def _update_info_table(self):
        """ 
        Update an existing info table with rows from a newly created info table.
        """
        # Extract header information from the template file
        with fits.open(self._template_file, mode='denywrite', memmap=True) as template_data:
            img_header = template_data[self._img_ext].header

        # Open the existing cube file to extract the original table
        with fits.open(self._cube_file, mode='readonly') as hdul:
            original_table = hdul[2].data

        # Prepare columns for the updated table
        cols = []
        for kwd, val, cmt in img_header.cards:
            # Determine dtype
            if isinstance(val, str):
                dtype = f'S{len(val)}'  # Using `S` type for binary FITS tables
            elif isinstance(val, int):
                dtype = np.int32
            else:
                dtype = np.float64

            # Append new data to the existing column
            updated_column = np.concatenate((original_table[kwd], self._info_table[kwd]))
            cols.append(Column(updated_column, name=kwd, dtype=dtype, meta={'comment': cmt}))

        # Handle file column separately
        file_updated_column = np.concatenate((original_table[self._file_keyword], self._info_table[self._file_keyword]))
        str_length = len(Path(self._template_file).name)
        cols.append(Column(file_updated_column, name=self._file_keyword, dtype=f'S{str_length}'))

        # Create the updated info table
        self._info_table = Table(cols)

    def make_cube(self, file_list: List[str], cube_file: str = 'img-cube.fits', sector: Optional[int] = None, 
                  max_memory: int = 50, verbose: bool = True):
        """
        Turns a list of FITS image files into one large data cube. Input images must all have the same 
        footprint and resolution. The resulting data cube is transposed for quicker cutouts. This function 
        can take some time to run, exactly how much time will depend on the number
        of input files and the maximum allowed memory. The runtime will be fastest if the
        entire data cube can be held in memory, however that can be quite large (~40GB for a full
        TESS main mission sector, 3 times that for a TESS extended mission sector).

        Parameters
        ----------
        file_list : array
            The list of FITS image files to cube.
        cube_file : str
            Optional.  The filename/path to save the output cube in. 
        sector : int
            Optional.  TESS sector to add as header keyword (not present in FFI files).
        max_memory : float
            Optional, default is 50. The maximum amount of memory to make available for building
            the data cube in GB. Note, this is the maximum amount of space to be used for the cube
            array only, so should not be set to the full amount of memory on the system.
        verbose : bool
            Optional. If True, intermediate information is printed. 

        Returns
        -------
        response: string or None
            If successful, returns the path to the cube FITS file, 
            if unsuccessful returns None.
        """
        # Log messages based on verbosity
        _handle_verbose(verbose)
        start_time = monotonic()

        self._max_memory = max_memory
        self._update = False

        # Set up the basic cube parameters
        sector = (sector, 'Observing sector')

        # Configure the cube and set up the primary header
        self._configure_cube(file_list, sector=sector)
    
        log.debug('Using %s to initialize the image header table.', Path(self._template_file).name)
        log.debug('Cube will be made in %d blocks of %d rows each.', self._num_blocks, self._block_size)

        # Set up the table to hold the individual image headers
        self._build_info_table()
        
        # Write the empty file, ready for the cube to be added
        self._build_cube_file(cube_file)

        # Fill the image cube
        with fits.open(self._cube_file, mode='update', memmap=True) as cube_hdu:
            # mmap: "allows you to take advantage of lower-level operating system 
            # functionality to read files as if they were one large string or array. 
            # This can provide significant performance improvements in code that 
            # requires a lot of file I/O."
            if mmap_imported:
                mm = fits.util._get_array_mmap(cube_hdu[1].data)
                # madvise: "Send advice option to the kernel about the memory region 
                # beginning at start and extending length bytes."
                mm.madvise(MADV_SEQUENTIAL)

            # Write blocks of data to the cube file
            for i in range(self._num_blocks):
                start_row = i * self._block_size
                end_row = start_row + self._block_size
                if end_row >= self._cube_shape[0]:  # Last block
                    end_row = None

                fill_info_table = True if (i == 0) else False
                self._write_block(cube_hdu, start_row, end_row, fill_info_table)
                  
                log.debug('Completed block %d of %d', i + 1, self._num_blocks)

        # Add the info table to the cube file
        self._write_info_table()
        log.debug('Total time elapsed: %.2f min', (monotonic() - start_time) / 60)

        return self._cube_file
    
    def update_cube(self, file_list: List[str], cube_file: str, sector: Optional[int] = None, max_memory: int = 50,
                    verbose: bool = True):
        """ 
        Updates an existing cube file with new FITS images. Same functionality as `CutoutFactory.make_cube`, 
        but working on an already existing file rather than building a new one. This function will:

        1. Create a new cube consisting of the new images that will be appended to the existing cube
        2. Update primary header keywords to reflect the update to the file
        3. Expand the file size of the FITS file containing the cube, to accomodate for the updated one

        Parameters
        ----------
        file_list : list
            The list of FITS image files to add to the cube.
        cube_file : str
            The filename/path to the existing cube FITS file.
        sector : int
            Optional.  TESS sector to add as header keyword (not present in FFI files).
        max_memory : float
            Optional, default is 50. The maximum amount of memory to make available for building
            the data cube in GB. Note, this is the maximum amount of space to be used for the cube
            array only, so should not be set to the full amount of memory on the system.
        verbose : bool
            Optional. If True, intermediate information is printed.  

        Returns
        -------
        response: string or None
            If successful, returns the path to the updated cube FITS file, 
            if unsuccessful returns None.
        """
        # Log messages based on verbosity
        _handle_verbose(verbose)
        start_time = monotonic()
        self._update = True
        self._max_memory = max_memory

        # If the cube file is not found, raise an error
        cube_path = Path(cube_file)
        if not cube_path.exists():
            raise InvalidInputError('Cube file was not found at the location provided. Please ensure the '
                                    'correct path was provided.')
        self._cube_file = cube_file
        log.debug('Updating cube file: %s', cube_file)

        # Extract existing image filenames from the cube to prevent duplicates
        existing_files = set(fits.getdata(self._cube_file, 2)[self._file_keyword])
        filtered_file_list = [file for file in file_list if Path(file).name not in existing_files]

        # Warn about and remove duplicates
        removed_files = set(file_list) - set(filtered_file_list)
        for file in removed_files:
            warnings.warn(f'Removed duplicate file: {Path(file).name}', DataWarning)

        # If no new images are found, raise an error
        if not filtered_file_list:
            raise InvalidInputError('No new images were found in the provided file list.')

        log.debug('%d new images found!', len(filtered_file_list))
        
        # Creating an empty cube that will be appended to the existing cube
        original_cube = fits.getdata(cube_file, 1)
        new_cube_shape = list(original_cube.shape)
        new_cube_shape[2] = len(filtered_file_list)
        self._cube_append = np.zeros(new_cube_shape)

        # Set up the basic cube parameters
        self._configure_cube(filtered_file_list, sector=(sector, 'Observing sector'))
        log.debug('Images will be appended in %d blocks of %d rows each.', self._num_blocks, self._block_size)
        
        # Starting a new info table from scratch with new rows
        self._build_info_table()

        # Update the image cube 
        with fits.open(self._cube_file, mode='update', memmap=True) as cube_hdu:
            # mmap: "allows you to take advantage of lower-level operating system 
            # functionality to read files as if they were one large string or array. 
            # This can provide significant performance improvements in code that 
            # requires a lot of file I/O."
            if mmap_imported:
                mm = fits.util._get_array_mmap(cube_hdu[1].data)
                # madvise: "Send advice option to the kernel about the memory region 
                # beginning at start and extending length bytes."
                mm.madvise(MADV_SEQUENTIAL)

            # Write blocks of data to the cube file
            for i in range(self._num_blocks):
                start_row = i * self._block_size
                end_row = start_row + self._block_size

                if end_row >= self._cube_shape[0]:
                    end_row = None

                # Filling in the cube file with data from new images
                # The info table also gets updated here
                self._write_block(cube_hdu, start_row, end_row, fill_info_table=True)
                log.debug('Completed block %d of %d', i + 1, self._num_blocks)
       
        # Append the new cube to the existing cube
        new_cube = np.concatenate((original_cube, self._cube_append), axis=2)

        # Replace cube data in FITS file
        with fits.open(self._cube_file, mode='update') as hdul:
            log.debug('Original cube of size: %s', original_cube.shape)
            log.debug('will now be replaced with cube of size: %s', new_cube.shape)
            log.debug('for file ``%s``', cube_file)
            hdul[1].data = new_cube

        # Update and write the info table
        self._update_info_table()
        self._write_info_table()
        log.debug('Total time elapsed: %.2f min', (monotonic() - start_time) / 60)

        return self._cube_file
