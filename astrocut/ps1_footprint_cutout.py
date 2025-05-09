from typing import Union, List, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scheduler.filename_resolver.ps1 import PS1FilenameResolver

from .fits_cutout import FITSCutout


class PS1FootprintCutout(FITSCutout):
    
    def __init__(self, 
                 coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, u.Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, 
                 limit_rounding_method: str = 'round', 
                 sequence: Union[int, List[int], None] = None, 
                 img_filters: list[str] = ['g', 'r', 'i', 'z', 'y'],
                 verbose: bool = False):

        # Replace this with a call to the service
        filename_resolver = PS1FilenameResolver()
        filenames = filename_resolver.get_filenames(coordinates.ra.value, coordinates.dec.value, img_filters)

        super().__init__(filenames, coordinates, cutout_size, fill_value, limit_rounding_method, 
                         single_outfile=False, verbose=verbose)
