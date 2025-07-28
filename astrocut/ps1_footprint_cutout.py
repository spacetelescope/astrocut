import requests
from typing import Union, List, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from .fits_cutout import FITSCutout

class PS1FootprintCutout(FITSCutout):
    
    def __init__(self, 
                 coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, u.Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, 
                 limit_rounding_method: str = 'round',
                 survey: Union[str, list[str]] = ['3pi'],
                 filter: list[str] = ['g', 'r', 'i', 'z', 'y'],
                 file_stage: list[str] = ['stack'],
                 auxilary_data: list[str] = ['data'],
                 verbose: bool = False):
        
        # Survey validation
        survey = self._validate_str_list_input('survey', survey, ['3pi', 'md'])
        filter = self._validate_str_list_input('filter', filter, ['g', 'r', 'i', 'z', 'y'])
        file_stage = self._validate_str_list_input('file_stage', file_stage, ['stack', 'warp'])
        auxilary_data = self._validate_str_list_input('auxilary_data', auxilary_data, ['data', 'mask', 'wt', 'exp', 'expwt', 'num'])

        service_url = 'http://rest-api-alb-457210841.us-east-1.elb.amazonaws.com/filenames/ps1'
        # Get coordinates as a SkyCoord object
        if not isinstance(coordinates, SkyCoord):
            coordinates = SkyCoord(coordinates, unit='deg')

        # Replace this with a call to the service
        service_url = 'http://rest-api-alb-457210841.us-east-1.elb.amazonaws.com/filenames/ps1'
        data = {
            "position": [f'{coordinates.ra.value}, {coordinates.dec.value}'],
            #"size": 20,
            "survey": survey,
            "filter": filter,
            "file_stage": file_stage
        }
        response = requests.post(service_url, json=data)
        response.raise_for_status()
        filenames = response.json().get('result', [])

        super().__init__(filenames, coordinates, cutout_size, fill_value, limit_rounding_method, 
                         single_outfile=False, verbose=verbose)
