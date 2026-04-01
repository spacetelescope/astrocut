import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from s3path import S3Path

from .asdf_spectral_cutout import ASDFSpectralCutout
from .exceptions import InvalidQueryError


class RomanSpectralCutout(ASDFSpectralCutout):
    """
    Class for creating spectral cutouts specifically from Roman ASDF data.
    Inherits from ASDFSpectralCutout.
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
        str,  # group_by
    ],
) -> List[str]:
    """
    Worker function for parallel Roman spectral cutouts.
    Opens the file, generates cutouts, writes them to disk, and returns paths.
    """
    filepath, source_id_batch, wl_range, lite, output_dir, group_by = args

    cutout = RomanSpectralCutout(
        spectral_files=filepath, source_ids=source_id_batch, wl_range=wl_range, lite=lite, verbose=False
    )

    return cutout.write_as_asdf(output_dir, group_by=group_by)


def roman_spectral_cut(
    spectral_files: Union[str, Path, S3Path, List[Union[str, Path, S3Path]]],
    source_ids: Union[List[str], List[int]],
    wl_range: tuple,
    *,
    lite: bool = True,
    output_dir: Union[str, Path] = ".",
    group_by: Literal["source_file", "file", "combined"] = "file",
    batch_size: int = 128,
    workers: Optional[int] = None,
) -> List[str]:
    """
    Extract spectral cutouts in parallel and write them to ASDF files.

    Parameters
    ----------
    spectral_files : str or Path or S3Path or List[Union[str, Path, S3Path]]
        Input spectral ASDF file(s).
    source_ids : list
        Source IDs to extract.
    wl_range : tuple
        Wavelength range.
    lite : bool, optional
        Whether to produce lite cutouts (default True).
    output_dir : str or Path, optional
        Output directory for cutout files.
    group_by : {'source_file', 'file', 'combined'}, optional
        How to group cutouts into output files (default 'file').
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

    spectral_files = spectral_files if isinstance(spectral_files, list) else [spectral_files]

    if workers is None:
        workers = max(1, os.cpu_count() - 1)

    jobs = []
    for filepath in spectral_files:
        for i in range(0, len(source_ids), batch_size):
            jobs.append((filepath, source_ids[i : i + batch_size], wl_range, lite, output_dir, group_by))

    written_paths = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_parallel_roman_spectral_cutout_worker, job) for job in jobs]

        for future in as_completed(futures):
            try:
                written_paths.extend(future.result())
            except InvalidQueryError:
                continue

    return written_paths
