import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Union

from s3path import S3Path

from .asdf_spectral_cutout import ASDFSpectralCutout
from .exceptions import InvalidQueryError


class RomanSpectralCutout(ASDFSpectralCutout):
    """
    Class for creating cutouts from Roman spectral data. Inherits from `ASDFSpectralCutout`
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
        If True, only a subset of the data and metadata will be included in the cutouts to
        reduce memory usage. Default is True.
    verbose : bool, optional
        If True, log messages will be printed during cutout generation. Default is False.
    """

    def __init__(
        self,
        spectral_files: Union[str, Path, S3Path, List[Union[str, Path, S3Path]]],
        source_ids: Union[str, int, List[Union[str, int]]],
        wl_range: Union[tuple, list] = None,
        lite: Optional[bool] = True,
        verbose: bool = False,
    ):
        super().__init__(spectral_files, source_ids, wl_range, lite, verbose)
        self._mission_keyword = "roman"

        # Make cutouts
        self.cutout()


def _parallel_roman_spectral_cutout_worker(
    args: Tuple[
        Union[str, Path, S3Path],  # filepath
        List[str],  # source_ids
        tuple,  # wl_range
        bool,  # lite
        Path,  # output_dir
    ],
) -> List[str]:
    """
    Worker function for parallel Roman spectral cutouts.
    Opens the file, generates cutouts, writes them to disk, and returns paths.
    """
    filepath, source_id_batch, wl_range, lite, output_dir = args

    cutout = RomanSpectralCutout(
        spectral_files=filepath, source_ids=source_id_batch, wl_range=wl_range, lite=lite, verbose=False
    )

    return cutout.write_as_asdf(output_dir)


def roman_spectral_cut(
    spectral_files: Union[str, Path, S3Path, List[Union[str, Path, S3Path]]],
    source_ids: Union[str, int, List[Union[str, int]]],
    wl_range: Union[tuple, list] = None,
    *,
    lite: bool = True,
    output_dir: Union[str, Path] = ".",
    batch_size: int = 128,
    workers: Optional[int] = None,
) -> List[str]:
    """
    Generate and write Roman spectral cutouts in parallel using multiprocessing.

    This function divides the source IDs into batches and processes them in parallel across multiple worker processes.
    Cutouts are grouped by source ID and file so that each output ASDF file contains cutouts for a single source ID
    from a single input file.

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
        If True, only a subset of the data and metadata will be included in the cutouts to reduce memory usage.
        Default is True.
    output_dir : str or Path, optional
        Directory where the output ASDF files will be saved. Default is the current directory.
    batch_size : int, optional
        Number of source IDs to process in each batch for parallel cutout generation. Default is 128.
    workers : int, optional
        Number of worker processes to use for parallel cutout generation. If None, the number of CPU
        cores minus one will be used. Default is None.

    Returns
    -------
    list of str
        Paths to written ASDF files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectral_files = spectral_files if isinstance(spectral_files, list) else [spectral_files]

    if workers is None:
        cpu_count = os.cpu_count() or 1
        workers = max(1, cpu_count - 1)

    # Create jobs for parallel processing
    jobs = []
    for filepath in spectral_files:
        for i in range(0, len(source_ids), batch_size):
            jobs.append((filepath, source_ids[i : i + batch_size], wl_range, lite, output_dir))

    written_paths = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_parallel_roman_spectral_cutout_worker, job) for job in jobs]

        for future in as_completed(futures):
            try:
                written_paths.extend(future.result())
            except InvalidQueryError:
                continue

    return written_paths
