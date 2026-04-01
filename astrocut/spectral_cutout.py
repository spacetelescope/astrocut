from abc import ABC
from pathlib import Path
from typing import List, Optional, Union

from s3path import S3Path

from .cutout import BaseCutout


class SpectralCutout(BaseCutout, ABC):
    """
    Class for creating spectral cutouts from multi-dimensional data.
    This class inherits from BaseCutout and implements methods specific to spectral data.
    """

    def __init__(
        self,
        spectral_files: Union[str, Path, S3Path, List[Union[str, Path, S3Path]]],
        source_ids: Union[str, int, List[Union[str, int]]],
        wl_range: Union[tuple, list] = None,
        lite: Optional[bool] = False,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)

        spectral_files = spectral_files if isinstance(spectral_files, list) else [spectral_files]
        self._spectral_files = [str(file) for file in spectral_files]  # Ensure all file paths are strings
        self._source_ids = source_ids if isinstance(source_ids, list) else [source_ids]
        self._wl_range = wl_range
        self._lite = lite

        self.cutout_data = dict()
