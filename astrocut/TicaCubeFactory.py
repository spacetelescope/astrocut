# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import Optional, Union

import numpy as np
from astropy.io import fits

from .CubeFactory import CubeFactory


class TicaCubeFactory(CubeFactory):
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

    def __init__(self, max_memory: int = 50):
        """ Setting up the class members."""
        super().__init__(max_memory=max_memory)

        self._time_keyword = 'STARTTJD'  # Start time in TJD. TICA-specific.
        self._last_file_keywords = ['ENDTJD']  # Stop time in TJD. TICA-specific (assumed to be in extension 0)                  
        self._image_header_keywords = ['CAMNUM', 'CCDNUM']  # Camera number and CCD number 
        self._template_requirements = {'NAXIS': 2}  # Using NAXIS instead of WCSAXES. 
        self._img_ext = 0  # TICA has image data in the primary extension
        self._naxis1 = 1  # TICA has data values only

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
        """       
        return img_data[self._img_ext].header.get(self._time_keyword)
        
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
        try:            
            return img_data[self._img_ext].data.shape          
        except AttributeError:
            # If data is not found in the image extension, raise an error          
            raise ValueError(self.ERROR_MSG)
    
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
        # Add image data to the sub-cube
        sub_cube[:, :, idx, 0] = img_data[0].data[start_row:end_row, :]

        # Remove the data from the input image to save memory
        del img_data[0].data

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
        # NOTE: 
        # A header keyword ('COMMENT') in TICA returns a keyword 
        # value in the form of a _HeaderCommentaryCard instead of a STRING. 
        # This breaks the info table because the info table 
        # doesn't understand what a FITS header commentary card is. 
        # Adding a try/except is a way to catch when these instances happen 
        # and turn the keyword value from a _HeaderCommentaryCard to a
        # string, which is what it's meant to be in the info table.
        # TODO: Find a more elegant way to handle these stray 
        # _HeaderCommentaryCards. 
        try:
            return img_data[0].header.get(kwd, nulval)
        except ValueError:
            kwd_val = img_data[0].header.get(kwd)
            if isinstance(kwd_val, fits.header._HeaderCommentaryCards):
                return str(kwd_val)
            else:
                raise
