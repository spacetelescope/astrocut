from pathlib import Path
from typing import Union, List, Optional
from concurrent.futures import ProcessPoolExecutor

from s3path import S3Path

from . import __version__
from .asdf_spectral_cutout import ASDFSpectralCutout

class RomanSpectralCutout(ASDFSpectralCutout):
    """
    Class for creating spectral cutouts specifically from Roman ASDF data.
    Inherits from ASDFSpectralCutout.
    """
    def __init__(self, 
                 file: Union[str, Path, S3Path], 
                 source_ids: Union[str, int, List[Union[str, int]]], 
                 wl_range: Union[tuple, list],
                 lite: Optional[bool] = False,
                 verbose: bool = False):
        super().__init__(file, source_ids, wl_range, lite, verbose)
        self._mission_keyword = 'roman'

        # Make cutouts
        self.cutout()

    def with_params(self, *, wl_range=None, source_ids=None, lite=None):
        """
        Create a new RomanSpectralCutout instance based on this one with modified parameters.
        
        Parameters
        ----------
        wl_range : tuple or list, optional
            New wavelength range for the cutout. If None, uses the existing range.
        source_ids : list, optional
            New list of source IDs for the cutout. If None, uses the existing source IDs.
        lite : bool, optional
            Whether to create a lite version of the cutout. If None, uses the existing setting.

        Returns
        -------
        RomanSpectralCutout
            A new RomanSpectralCutout instance with the specified parameters.
        """
        return RomanSpectralCutout(
            self._file,
            source_ids or self._source_ids,
            wl_range or self._wl_range,
            lite or self._lite,
            verbose=self._verbose
        )