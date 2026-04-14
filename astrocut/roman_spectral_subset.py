from pathlib import Path
from typing import List, Optional, Union

from s3path import S3Path

from .asdf_spectral_subset import ASDFSpectralSubset


class RomanSpectralSubset(ASDFSpectralSubset):
    """
    Class for creating subsets from Roman spectral data. Inherits from `ASDFSpectralSubset`
    and implements the same interface, but is designed for Roman data.

    Parameters
    ----------
    spectral_files : str, Path, S3Path, or list
        Path(s) to the input spectral files. Can be a single file or a list of files.
    source_ids : str, int, or list
        Source ID(s) to cut out. Can be a single ID or a list of IDs.
    wl_range : tuple or list, optional
        Wavelength range to cut out, specified as (min_wavelength, max_wavelength). If None,
        the full wavelength range will be used.
    lite : bool, optional
        If True, only a subset of the data and metadata will be included in the subsets to
        reduce memory usage. Default is True.
    max_workers : int, optional
        Maximum number of worker processes to use when generating subsets in parallel. Default is 1 (no parallelism).
        If None, the number of workers will be set based on the number of CPUs and input files. If an
        integer is provided, the number of workers used will be the minimum of that value and the number of
        input files. It is recommended to use parallel processing when generating subsets from multiple
        large input files. For a single input file, or for multiple small input files, multiprocessing may
        not provide a significant speedup and may even slow down execution due to the overhead of parallelization.
    verbose : bool, optional
        If True, log messages will be printed during subset generation. Default is False.
    """

    def __init__(
        self,
        spectral_files: Union[str, Path, S3Path, List[Union[str, Path, S3Path]]],
        source_ids: Union[str, int, List[Union[str, int]]],
        wl_range: Union[tuple, list] = None,
        lite: Optional[bool] = True,
        max_workers: Optional[int] = 1,
        verbose: bool = False,
    ):
        super().__init__(spectral_files, source_ids, wl_range, lite, max_workers, verbose)
        self._mission_keyword = "roman"

        # Make subsets
        self.subset()
