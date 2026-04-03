from abc import ABC
from pathlib import Path
from typing import List, Optional, Union

from s3path import S3Path

from .cutout import BaseCutout


class SpectralCutout(BaseCutout, ABC):
    """
    Abstract class for creating cutouts from spectral data.

    Parameters
    ----------
    spectral_files : str, Path, S3Path, or list
        Path(s) to the input spectral files. Can be a single file or a list of files.
    source_ids : str, int, or list
        Source ID(s) to cut out. Can be a single ID or a list of IDs.
    wl_range : tuple or list, optional
        Wavelength range to cut out, specified as (min_wavelength, max_wavelength). If None, the full wavelength
        range will be used.
    lite : bool, optional
        If True, only a subset of the data and metadata will be included in the cutouts to reduce memory usage.
        Default is False (include all data and metadata).
    verbose : bool, optional
        If True, log messages will be printed during cutout generation. Default is False.
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
