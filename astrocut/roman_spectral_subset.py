import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from s3path import S3Path

from . import log
from .asdf_spectral_subset import ASDFSpectralSubset
from .exceptions import InvalidInputError, InvalidQueryError


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
    verbose : bool, optional
        If True, log messages will be printed during subset generation. Default is False.
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

        # Make subsets
        self.subset()


def _roman_spectral_subset_worker(
    args: Tuple[
        Union[str, Path, S3Path],  # filepath
        List[Union[str, int]],  # source_ids
        tuple,  # wl_range
        bool,  # lite
        Path,  # output_dir
        Literal["source_file", "file", "combined"],  # group_by
    ],
) -> List[str]:
    """
    Worker function for parallel Roman spectral subsets.
    Opens the file, generates subsets, writes them to disk, and returns paths.
    """
    filepath, source_id_batch, wl_range, lite, output_dir, group_by = args

    subset = RomanSpectralSubset(
        spectral_files=filepath, source_ids=source_id_batch, wl_range=wl_range, lite=lite, verbose=False
    )

    return subset.write_as_asdf(output_dir, group_by=group_by)


def roman_spectral_subset(
    spectral_files: Union[str, Path, S3Path, List[Union[str, Path, S3Path]]],
    source_ids: Union[str, int, List[Union[str, int]]],
    wl_range: Union[tuple, list] = None,
    *,
    lite: bool = True,
    output_dir: Union[str, Path] = ".",
    batch_size: int = 256,
    workers: Optional[int] = 8,
    group_by: Literal["source_file", "file", "combined"] = "source_file",
) -> List[str]:
    """
    Generate and write Roman spectral subsets in parallel using multiprocessing.

    This function divides the source IDs into batches and processes them in parallel across multiple worker processes.

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
        If True, only a subset of the data and metadata will be included in the subsets to reduce memory usage.
        Default is True.
    output_dir : str or Path, optional
        Directory where the output ASDF files will be saved. Default is the current directory.
    batch_size : int, optional
        Batch size used to coalesce source IDs into at most ``workers`` chunks per file when
        ``group_by='source_file'``. Default is 256.
    workers : int or None, optional
        Number of worker processes to use for parallel subset generation. If None, the number of CPU
        cores minus one will be used. Default is 8.
    group_by : {'source_file', 'file', 'combined'}, optional
        Output grouping strategy passed to ``write_as_asdf``. Default is 'source_file'.
        - 'source_file': Each worker processes a chunk of source IDs for a single file, resulting in
        more parallelism and more output files.
        - 'file': Each worker processes all source IDs for a single file, resulting in fewer output files
        but less parallelism.
        - 'combined': All workers need all source IDs and files to generate each output file, so this mode
        is not parallelized and will run in a single process.

    Returns
    -------
    list of str
        Paths to written ASDF files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectral_files = spectral_files if isinstance(spectral_files, list) else [spectral_files]
    if not spectral_files:
        raise InvalidInputError("No spectral files provided. Please provide at least one spectral file.")

    source_ids = source_ids if isinstance(source_ids, list) else [source_ids]
    if not source_ids:
        raise InvalidInputError("No source IDs provided. Please provide at least one source ID.")

    if group_by == "combined":
        # For combined mode, all source IDs and files are needed to generate each output file, so we can't parallelize.
        # Just run the worker function directly in this case.
        return _roman_spectral_subset_worker(
            (list(spectral_files), list(source_ids), wl_range, lite, output_dir, group_by)
        )

    # Validate batch_size and workers parameters
    batch_size = max(1, int(batch_size))
    if workers is None:
        cpu_count = os.cpu_count() or 1
        workers = cpu_count - 1
    workers = max(1, int(workers))

    # Create jobs for parallel processing
    jobs = []
    if group_by == "source_file":
        # Coalesce source IDs into batches for each file, so each worker processes a
        # chunk of source IDs for a single file
        for filepath in spectral_files:
            bucket_count = min(workers, len(source_ids))
            # Coalesce source IDs into buckets for this file
            if source_ids and bucket_count > 0:
                effective_batch = max(1, int(batch_size))
                batches = [source_ids[i : i + effective_batch] for i in range(0, len(source_ids), effective_batch)]
                buckets = [[] for _ in range(bucket_count)]

                for idx, batch in enumerate(batches):
                    buckets[idx % bucket_count].extend(batch)

                chunks = [bucket for bucket in buckets if bucket]
                for source_id_chunk in chunks:
                    jobs.append((filepath, source_id_chunk, wl_range, lite, output_dir, group_by))
    elif group_by == "file":
        # Each job processes all source IDs for a single file, so we can still parallelize across files
        for filepath in spectral_files:
            jobs.append((filepath, list(source_ids), wl_range, lite, output_dir, group_by))
    else:
        raise InvalidInputError(
            f"Invalid group_by value: '{group_by}'. Must be one of 'source_file', 'file', or 'combined'."
        )

    written_paths = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_roman_spectral_subset_worker, job) for job in jobs]

        for future in as_completed(futures):
            try:
                written_paths.extend(future.result())
            except InvalidQueryError as e:
                log.warning(e)
                continue

    return written_paths
