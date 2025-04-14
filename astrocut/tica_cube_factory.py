# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import Optional, Union

import numpy as np
from astropy.io import fits

from .cube_factory import CubeFactory


class TicaCubeFactory(CubeFactory):
    """
    Class for creating TICA image cubes. 

    The TESS Image CAlibrator (TICA) products are high level science products (HLSPs) 
    developed by the MIT Quick Look Pipeline (https://github.com/mmfausnaugh/tica). These 
    images are produced and delivered up to 4x sooner than their SPOC counterparts (as of TESS EM2),
    and can therefore be used to produce the most up-to-date cutouts of a target. 
    More information on TICA can be found here: https://archive.stsci.edu/hlsp/tica

    Parameters
    ----------
    max_memory : int
        The maximum amount of memory to make available for building the data cube in GB.
        Note, this is the maximum amount of space to be used for the cube array only,
        so should not be set to the full amount of memory on the system.
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
        val = img_data[0].header.get(kwd, nulval)

        # The "COMMENT" keyword is in the form of a _HeaderCommentaryCard instead of a string
        return str(val) if isinstance(val, fits.header._HeaderCommentaryCards) else val
