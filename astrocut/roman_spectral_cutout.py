import os
from pathlib import Path
from typing import Union, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from s3path import S3Path

from .asdf_spectral_cutout import ASDFSpectralCutout


class RomanSpectralCutout(ASDFSpectralCutout):
    """
    Class for creating spectral cutouts specifically from Roman ASDF data.
    Inherits from ASDFSpectralCutout.
    """
    def __init__(self, 
                 file: Union[str, Path, S3Path], 
                 source_ids: Union[str, int, List[Union[str, int]]], 
                 wl_range: Union[tuple, list] = None,
                 lite: Optional[bool] = True,
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
    

def _parallel_roman_spectral_cutout_worker(
    args: Tuple[
        Union[str, Path, S3Path],  # filepath
        List[str],                 # source_ids
        tuple,                     # wl_range
        bool,                      # lite
        Path                       # output_dir
    ]
) -> List[str]:
    """
    Worker function for parallel Roman spectral cutouts.
    Opens the file, generates cutouts, writes them to disk, and returns paths.
    """
    filepath, source_ids, wl_range, lite, output_dir = args

    cutout = RomanSpectralCutout(
        file=filepath,
        source_ids=source_ids,
        wl_range=wl_range,
        lite=lite,
        verbose=False,
    )

    return cutout.write_as_asdf(output_dir)


def roman_spectral_cut(
    file: Union[str, Path, S3Path],
    source_ids: Union[List[str], List[int]],
    wl_range: tuple,
    *,
    lite: bool = True,
    output_dir: Union[str, Path] = ".",
    batch_size: int = 128,
    workers: Optional[int] = None,
) -> List[str]:
    """
    Extract spectral cutouts in parallel and write them to ASDF files.

    Parameters
    ----------
    file : str or Path or S3Path
        Input spectral ASDF file.
    source_ids : list
        Source IDs to extract.
    wl_range : tuple
        Wavelength range.
    lite : bool, optional
        Whether to produce lite cutouts (default True).
    output_dir : str or Path, optional
        Output directory for cutout files.
    batch_size : int, optional
        Number of sources per worker batch.
    workers : int, optional
        Number of worker processes.

    Returns
    -------
    list of str
        Paths to written ASDF files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # source_ids = [str(sid) for sid in source_ids]

    if workers is None:
        workers = max(1, os.cpu_count() - 1)

    jobs = [
        (
            file,
            source_ids[i: i + batch_size],
            wl_range,
            lite,
            output_dir,
        )
        for i in range(0, len(source_ids), batch_size)
    ]

    written_paths: List[str] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_parallel_roman_spectral_cutout_worker, job): job
            for job in jobs
        }

        for future in as_completed(futures):
            written_paths.extend(future.result())

    return written_paths

