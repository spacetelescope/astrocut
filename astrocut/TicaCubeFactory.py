"""
This module implements the functionality to create image cubes for the purposes of
creating cutout target pixel files.
"""

from astropy.io import fits

from .CubeFactory import CubeFactory

ERROR_MSG = ("One or more incorrect file types were input. Please input TICA FFI files when using "
             "``TicaCubeFactory``, and SPOC FFI files when using ``CubeFactory``.")


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

    def __init__(self, max_memory=50):
        """ Setting up the class members."""
        super().__init__(max_memory=max_memory)

        
        self.time_keyword = 'STARTTJD'  # Start time in TJD. TICA-specific.
        self.last_file_keywords = ['ENDTJD']  # Stop time in TJD. TICA-specific (assumed to be in extension 0)                  
        self.image_header_keywords = ['CAMNUM', 'CCDNUM']  # Camera number and CCD number 
        self.template_requirements = {'NAXIS': 2}  # Using NAXIS instead of WCSAXES. 
        self.img_ext = 0

    def _get_img_start_time(self, img_data):        
        return img_data[self.img_ext].header.get(self.time_keyword)
        
    def _get_img_shape(self, img_data):    
        try:            
            return img_data[self.img_ext].data.shape          
        except AttributeError:            
            raise ValueError(ERROR_MSG)
    
    def _get_cube_shape(self, image_shape):
        return (image_shape[0], image_shape[1], len(self.file_list), 1)
    
    def _write_to_sub_cube(self, sub_cube, idx, img_data, start_row, end_row):
        sub_cube[:, :, idx, 0] = img_data[0].data[start_row:end_row, :]
        del img_data[0].data

    def _get_header_keyword(self, kwd, idx, img_data, nulval):
        # NOTE: 
        # A header keyword ('COMMENT') in TICA returns a keyword 
        # value in the form of a _HeaderCommentaryCard instead of a STRING. 
        # This breaks the info table because the info table 
        # doesn't understand what a FITS header commentary card is. 
        # Adding a try/except is a way to catch when these instances happen 
        # and turn the keyword value from a _HeaderCommentaryCard to a
        # string, which is what it's meant to be in the info table. 
        #
        # TODO: Find a more elegant way to handle these stray 
        # _HeaderCommentaryCards. 
        
        try:
            self.info_table[kwd][idx] = img_data[0].header.get(kwd, nulval)
        except ValueError:
            kwd_val = img_data[0].header.get(kwd)
            if isinstance(kwd_val, fits.header._HeaderCommentaryCards):
                self.info_table[kwd][idx] = str(kwd_val)
            else:
                raise


    def _append_table_to_cube(self, table_hdu):
        with fits.open(self.cube_file, mode='update', memmap=True) as cube_hdus:
            if self.update:
                cube_hdus.pop(index=2)
            cube_hdus.append(table_hdu)
