from abc import ABC
from pathlib import Path
from typing import Union, List, Optional

from s3path import S3Path

from .cutout import BaseCutout


class SpectralCutout(BaseCutout, ABC):
    """
    Class for creating spectral cutouts from multi-dimensional data.
    This class inherits from BaseCutout and implements methods specific to spectral data.
    """

    def __init__(self, 
                 file: Union[str, Path, S3Path], 
                 source_ids: Union[str, int, List[Union[str, int]]], 
                 wl_range: Union[tuple, list] = None,
                 lite: Optional[bool] = False,
                 verbose: bool = False):
        super().__init__(verbose=verbose)

        self._file = file
        self._source_ids = source_ids if isinstance(source_ids, list) else [source_ids]
        self._wl_range = wl_range
        self._lite = lite

        self.cutout_data = dict()  # To store cutout data for each source ID
