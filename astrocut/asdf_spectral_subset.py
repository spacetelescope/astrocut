import os
import warnings
from abc import ABC
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import blake2s
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import asdf
import asdf.schema as asdf_schema
import numpy as np
from s3path import S3Path

from . import __version__, log
from .exceptions import DataWarning, InvalidInputError, InvalidQueryError
from .spectral_subset import SpectralSubset


def _make_pickle_safe(value):
    """
    Convert a value to a form that can be pickled and sent between processes. This is needed for the results returned
    by the worker function when using ProcessPoolExecutor, since ASDF trees can contain complex objects that may not
    be directly pickleable. This function recursively processes the data structure, converting numpy arrays to lists,
    and handling other non-pickleable types as needed.

    Parameters
    ----------
    value : any
        The value to convert to a pickle-safe form.

    Returns
    -------
    any
        A version of the input value that can be safely pickled and sent between processes.
    """
    if isinstance(value, Mapping):
        return {key: _make_pickle_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_pickle_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_make_pickle_safe(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _subset_file_worker(
    file: str,
    mission_keyword: str,
    source_ids,
    wl_range,
    lite: bool,
    verbose: bool,
) -> Tuple[str, Dict, Dict, Dict, Dict, List[str]]:
    """
    Worker function to subset a single ASDF file for the specified source IDs and wavelength range. This function is
    designed to be run in a separate process when using ProcessPoolExecutor for parallel processing of multiple files.

    Parameters
    ----------
    file : str
        The path to the ASDF file to subset.
    mission_keyword : str
        The keyword for the mission-specific data in the ASDF file.
    source_ids : list
        The list of source IDs to include in the subset.
    wl_range : tuple
        The wavelength range to include in the subset.
    lite : bool
        Whether to include only essential data in the subset.
    verbose : bool
        Whether to emit warnings for sources that are not found or out of bounds.

    Returns
    -------
    tuple
        (file, base_tree, base_mission, subset_data_for_file, out_trees_for_file, emitted_warnings)
    """
    base_tree, base_mission, subset_data_for_file, out_trees_for_file, emitted_warnings = _subset_single_file(
        file=file,
        mission_keyword=mission_keyword,
        source_ids=source_ids,
        wl_range=wl_range,
        lite=lite,
        verbose=verbose,
    )

    return (
        file,
        _make_pickle_safe(base_tree),
        _make_pickle_safe(base_mission),
        _make_pickle_safe(subset_data_for_file),
        _make_pickle_safe(out_trees_for_file),
        emitted_warnings,
    )


def _subset_single_file(
    file: str,
    mission_keyword: str,
    source_ids,
    wl_range,
    lite: bool,
    verbose: bool,
) -> Tuple[Dict, Dict, Dict, Dict, List[str]]:
    """
    Subset a single ASDF file for the specified source IDs and wavelength range. This function is called by the worker
    function and performs the actual subsetting logic for one file.

    Parameters
    ----------
    file : str
        The path to the ASDF file to subset.
    mission_keyword : str
        The keyword for the mission-specific data in the ASDF file.
    source_ids : list
        The list of source IDs to include in the subset.
    wl_range : tuple
        The wavelength range to include in the subset.
    lite : bool
        Whether to include only essential data in the subset.
    verbose : bool
        Whether to emit warnings for sources that are not found or out of bounds.

    Returns
    -------
    tuple
        (base_tree, base_mission, subset_data_for_file, out_trees_for_file, emitted_warnings)
    """
    subset_data_for_file = {}
    out_trees_for_file = {}
    emitted_warnings = []

    with asdf.open(file) as af:
        in_tree = af.tree
        mission_data = in_tree[mission_keyword]["data"]

        base_tree = {k: v for k, v in in_tree.items() if k not in {mission_keyword, "history"}}
        base_mission = {k: v for k, v in in_tree[mission_keyword].items() if k != "data"}

        for sid in source_ids:
            sid = str(sid)
            mission_key = sid

            # First try the source ID as a string key, then as an integer key if the string key is not found
            if mission_key not in mission_data:
                try:
                    sid_int = int(sid)
                except (TypeError, ValueError):
                    sid_int = None

                if sid_int in mission_data:
                    mission_key = sid_int
                else:
                    if verbose:
                        emitted_warnings.append(
                            f"Source ID {sid} not found in file {file}. Skipping this source for this file."
                        )
                    continue

            source = mission_data[mission_key]
            wl = source["wl"]

            # Cut out the wavelength range if specified, and emit warnings if the source is out of bounds or not found
            if wl_range is not None:
                wl_min = wl.min()
                wl_max = wl.max()

                if wl_range[0] > wl_max or wl_range[1] < wl_min:
                    emitted_warnings.append(
                        f"Wavelength range {wl_range} is out of bounds for source ID {sid} and file {file}. "
                        f"Available wavelength range: [{wl_min}, {wl_max}]. Skipping this source for this file."
                    )
                    continue

                mask = (wl >= wl_range[0]) & (wl <= wl_range[1])
            else:
                mask = slice(None)

            keys = ["wl", "flux", "flux_error"] if lite else source.keys()
            spectral_keys = {k for k, v in source.items() if hasattr(v, "__len__") and len(v) == len(wl)}

            new_source = {}
            for key in keys:
                value = source[key]
                if key in spectral_keys:
                    # For array-like data that matches the wavelength dimension, apply the wavelength mask to
                    # cut out the specified range
                    arr = np.asarray(value)
                    new_source[key] = arr[mask].copy()
                else:
                    try:
                        new_source[key] = deepcopy(value)
                    except Exception:
                        new_source[key] = value

            subset_data_for_file[sid] = new_source

            meta = {**base_mission.get("meta", {}), "source_id": sid}
            if lite:
                out_tree = {mission_keyword: {"data": new_source, "meta": meta}}
            else:
                out_tree = {
                    **base_tree,
                    mission_keyword: {
                        **base_mission,
                        "data": new_source,
                        "meta": meta,
                    },
                }

            out_trees_for_file[sid] = out_tree

    return base_tree, base_mission, subset_data_for_file, out_trees_for_file, emitted_warnings


def _append_history_entry(history_tree: Optional[Mapping], description: str) -> Dict[str, object]:
    """
    Append a new entry to an ASDF history tree with the given description and metadata about the software used.

    Parameters
    ----------
    history_tree : Mapping or None
        The existing ASDF history tree to which the new entry will be appended. If None, a new history
        tree will be created.
    description : str
        A description of the operation that generated the new history entry, which will be included
        in the entry's metadata.

    Returns
    -------
    dict
        A new ASDF history tree containing all existing entries from the input history tree (if any)
        plus a new entry with the provided description and software metadata.
    """
    merged_history = {}
    entries = []
    if isinstance(history_tree, Mapping):
        entries = list(history_tree.get("entries", []))

    # Create a new history entry for this subset operation and append it to the existing history entries
    history_entry = {
        "description": description,
        "time": datetime.now(timezone.utc),
        "software": {
            "name": "astrocut",
            "author": "Space Telescope Science Institute",
            "version": __version__,
            "homepage": "https://astrocut.readthedocs.io/en/latest/",
        },
    }
    entries.append(history_entry)

    merged_history["entries"] = entries
    return merged_history


@contextmanager
def _disabled_asdf_validation():
    """Temporarily disable ASDF schema validation for trusted bulk writes."""
    original_validate = asdf_schema.validate
    asdf_schema.validate = lambda *args, **kwargs: None
    try:
        yield
    finally:
        asdf_schema.validate = original_validate


def _write_tree_batch_worker(tree_jobs: List[Tuple[Dict, str]], validate_output: bool = False) -> List[str]:
    """
    Worker function to write a batch of ASDF trees to files.

    Parameters
    ----------
    tree_jobs : list of tuples
        A list of tuples, where each tuple contains an ASDF tree (as a dictionary) and
        the corresponding output file path.
    validate_output : bool, optional
        Whether to validate the output ASDF files after writing. Default is False, which can improve performance
        when writing large batches of files that are expected to be valid.

    Returns
    -------
    list of str
        A list of the output file paths that were written by this worker.
    """
    subset_paths = []
    # Write the ASDF trees to files, optionally disabling validation for performance

    with nullcontext() if validate_output else _disabled_asdf_validation():
        for tree, subset_path in tree_jobs:
            af = asdf.AsdfFile(tree)
            af.write_to(subset_path)
            subset_paths.append(subset_path)
    return subset_paths


class ASDFSpectralSubset(SpectralSubset, ABC):
    """
    Abstract class for creating subsets from ASDF spectral data. This class is designed to
    handle subsets from ASDF files that follow a specific structure, where the spectral data
    is organized under a mission-specific keyword.

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
        If True, only metadata and a subset of the data will be included in the subsets to
        reduce memory usage. Default is True.
    max_workers : int, optional
        Maximum number of worker processes to use when generating subsets in parallel. Default is None.
        If None, the number of workers will be set based on the number of CPUs and input files. If an
        integer is provided, the number of workers used will be the minimum of that value and the number of
        input files.

        It is recommended to use parallel processing when generating subsets from multiple
        large input files. For a single input file, or for multiple small input files, multiprocessing may
        not provide a significant speedup and may even slow down execution due to the overhead of parallelization.
    verbose : bool, optional
        If True, log messages will be printed during subset generation. Default is False.

    Attributes
    ----------
    subset_data : dict
        A dictionary to store the subset data for each source ID and input file combination.

    Methods
    -------
    get_asdf_subsets(group_by=group_by, source_ids=source_ids, spectral_files=spectral_files)
        Get ASDF subset(s) for specified source IDs and input files, grouped by source and file,
        file, or combined.
    write_as_asdf(output_dir=output_dir, group_by=group_by, source_ids=source_ids, spectral_files=spectral_files)
        Write the ASDF subset(s) to files in the specified output directory, grouped by source
        and file, file, or combined.
    subset()
        Generate the spectral subset(s) from the input ASDF files based on the specified
        source IDs and wavelength range.
    """

    _invalid_group_by_msg = "Invalid group_by value: '{}'. Must be one of 'source_file', 'file', or 'combined'."

    def __init__(
        self,
        spectral_files: Union[str, Path, S3Path, List[Union[str, Path, S3Path]]],
        source_ids: Union[str, int, List[Union[str, int]]],
        wl_range: Union[tuple, list] = None,
        lite: Optional[bool] = True,
        max_workers: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__(spectral_files, source_ids, wl_range, lite, verbose)

        self._mission_keyword = None  # To be set in subclass
        self._out_trees = {}  # Store ASDF tree for the subsets
        self._base_trees = {}  # To store base ASDF tree without mission data
        self._base_missions = {}  # To store base mission data without source-specific data

        self._max_workers = max_workers

        self.subset_data = dict()  # Store subset data for each source ID and input file combination

    def subset(self):
        """
        Generate the spectral subset(s) from the input ASDF files based on the specified
        source IDs and wavelength range.

        Raises
        ------
        InvalidQueryError
            If no subsets were created, which may indicate that the source IDs are not present in the spectral files
            or that the wavelength range does not overlap with the data.
        """
        # Reset outputs in case subset() is called more than once
        self._out_trees = {}
        self._base_trees = {}
        self._base_missions = {}
        self.subset_data = {}

        files = [str(f) for f in self._spectral_files]

        # If max_workers is not specified, default to min(file count, CPU count)
        cpu_count = os.cpu_count() or 1
        max_workers = min(len(files), cpu_count) if self._max_workers is None else min(len(files), self._max_workers)

        def _apply_worker_result(result):
            # Helper function to apply the results from the worker function to the internal attributes of the class
            (
                file,
                base_tree,
                base_mission,
                subset_data_for_file,
                out_trees_for_file,
                emitted_warnings,
            ) = result

            for warning_message in emitted_warnings:
                warnings.warn(warning_message, DataWarning)

            if out_trees_for_file:
                self._base_trees[file] = base_tree
                self._base_missions[file] = base_mission
                self.subset_data[file] = subset_data_for_file
                self._out_trees[file] = out_trees_for_file

        if len(files) <= 1 or max_workers == 1:
            # Process files sequentially
            for file in files:
                _apply_worker_result(
                    _subset_file_worker(
                        file=file,
                        mission_keyword=self._mission_keyword,
                        source_ids=self._source_ids,
                        wl_range=self._wl_range,
                        lite=self._lite,
                        verbose=self._verbose,
                    )
                )
        else:
            # Process files in parallel using ProcessPoolExecutor
            log.debug(f"Processing files in parallel with up to {max_workers} workers...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _subset_file_worker,
                        file,
                        self._mission_keyword,
                        self._source_ids,
                        self._wl_range,
                        self._lite,
                        self._verbose,
                    )
                    for file in files
                ]

                for future in as_completed(futures):
                    _apply_worker_result(future.result())

        if not self._out_trees:
            raise InvalidQueryError(
                "No subsets were created. Please verify that the source IDs are present in "
                "the spectral files and that the wavelength range overlaps with the data."
            )

    def get_source_file_keys(
        self,
        *,
        spectral_files: Union[str, Path, S3Path, List[Union[str, Path, S3Path]]] = None,
        source_ids: Union[str, int, List[Union[str, int]]] = None,
    ) -> Dict[str, Tuple[str, str]]:
        """
        Get valid string keys for source/file subset selection.

        Parameters
        ----------
        spectral_files : str or Path or S3Path or list, optional
            Specific spectral files to include. If None, all available files are included.
        source_ids : str or int or list, optional
            Specific source IDs to include. If None, all available source IDs are included.

        Returns
        -------
        dict
            Mapping from a user-friendly key string to ``(file, source_id)`` tuples.
        """
        files_to_include, sources_to_include = self._resolve_selection(
            spectral_files=spectral_files, source_ids=source_ids
        )

        source_file_pairs = []
        for file in files_to_include:
            for sid in sources_to_include:
                if sid not in self._out_trees[file]:
                    log.debug(f"Source ID {sid} not found in file {file}. Skipping this source for this file.")
                    continue
                source_file_pairs.append((file, sid))

        keys = self._build_source_file_keys(source_file_pairs)

        return keys

    def get_asdf_subsets(
        self,
        *,
        group_by: Literal["source_file", "file", "combined"] = "combined",
        spectral_files: Union[str, Path, S3Path, List[Union[str, Path, S3Path]]] = None,
        source_ids: Union[str, int, List[Union[str, int]]] = None,
    ) -> dict:
        """
        Get ASDF subset objects for specified source IDs and input files, grouped by source and file, file, or combined.

        Parameters
        ----------
        group_by : {'source_file', 'file', 'combined'}, optional
            Determines how the subsets are grouped in the output ASDF objects. Default is 'combined'.
            - 'source_file': Separate ASDF object for each source ID and input file combination.
            - 'file': One ASDF object per input file, containing all specified source IDs from that file.
            - 'combined': A single ASDF object containing all specified source IDs from all input files.
        spectral_files : str or Path or S3Path or list, optional
            Specific spectral files to include in the output. If None, all input spectral files will be included.
            Can be a single file or a list of files.
        source_ids : str or int or list, optional
            Specific source IDs to include in the output. If None, all source IDs from the subset
            results will be included. Can be a single ID or a list of IDs.

        Returns
        -------
        dict or asdf.AsdfFile
            Depending on the value of `group_by`, this method returns either a dictionary of ASDF subset objects keyed
            by source ID and input file combination ('source_file'), a dictionary of ASDF subset objects
            keyed by input file ('file'), or a single ASDF subset object containing all subsets ('combined').
        """
        asdf_subsets = {}
        if group_by == "source_file":
            # Build deterministic keys for the source/file combinations and disambiguate if needed
            source_file_keys = self.get_source_file_keys(spectral_files=spectral_files, source_ids=source_ids)

            # Create separate ASDF objects for each source/file combination
            for string_key, (file, sid) in source_file_keys.items():
                tree = deepcopy(self._out_trees[file][sid])
                af = asdf.AsdfFile(tree)
                af.tree["history"] = _append_history_entry(
                    tree.get("history", {}),
                    f"Spectral subset created for source ID {sid} from file {file}"
                    f"{f' with wavelength range {self._wl_range is not None}' if self._wl_range else ''}.",
                )
                asdf_subsets[string_key] = af

            return asdf_subsets

        files_to_include, sources_to_include = self._resolve_selection(
            spectral_files=spectral_files, source_ids=source_ids
        )
        if group_by == "file":
            # Group by file, combining all sources in each file
            for file in files_to_include:
                source_ids = [sid for sid in sources_to_include if sid in self._out_trees[file]]
                if self._lite:
                    combined_tree = {
                        self._mission_keyword: {
                            "data": {
                                sid: self._out_trees[file][sid][self._mission_keyword]["data"]
                                for sid in sources_to_include
                                if sid in self._out_trees[file]
                            },
                            "meta": {**self._base_missions[file]["meta"], "source_ids": source_ids},
                        }
                    }
                else:
                    base_tree_copy = deepcopy(self._base_trees[file])
                    combined_tree = {
                        **base_tree_copy,
                        self._mission_keyword: {
                            **self._base_missions[file],  # shallow copy mission-level
                            "data": {
                                sid: self._out_trees[file][sid][self._mission_keyword]["data"]
                                for sid in sources_to_include
                                if sid in self._out_trees[file]
                            },
                            "meta": {**self._base_missions[file]["meta"], "source_ids": source_ids},
                        },
                    }
                af = asdf.AsdfFile(combined_tree)
                af.tree["history"] = _append_history_entry(
                    combined_tree.get("history", {}),
                    f"Spectral subset created for source IDs {source_ids} from file {file}"
                    f"{f' with wavelength range {self._wl_range}' if self._wl_range is not None else ''}.",
                )
                asdf_subsets[file] = af
        elif group_by == "combined":
            # Group all sources and spectral files into a single ASDF file
            # ASDF data should be keyed by file and then source_id to avoid key collisions
            # meta should also be keyed by file
            meta_by_file = {}
            for file in files_to_include:
                meta_by_file[str(file)] = {
                    **self._base_missions[file].get("meta", {}),
                    "source_ids": [sid for sid in sources_to_include if sid in self._out_trees[file]],
                }

            if self._lite:
                combined_tree = {
                    self._mission_keyword: {
                        "data": {
                            str(file): {
                                sid: self._out_trees[file][sid][self._mission_keyword]["data"]
                                for sid in sources_to_include
                                if sid in self._out_trees[file]
                            }
                            for file in files_to_include
                        },
                        "meta": meta_by_file,
                    }
                }
            else:
                # TODO: Right now, this doesn't work with the original key (asdf_library, history, etc.) because of ASDF
                # validation errors due to required keys not being present. So, we rename the keys and append
                # "_combined" to indicate that they are combined from multiple files. This is not ideal, but
                # it allows us to include all the data in a single ASDF file without validation errors.
                combined_keys = set()
                for file in files_to_include:
                    combined_keys.update(self._base_trees[file].keys())

                base_tree = {}
                for key in combined_keys:
                    key_by_file = {
                        str(file): self._base_trees[file][key]
                        for file in files_to_include
                        if key in self._base_trees[file]
                    }
                    base_tree[f"{key}_combined"] = key_by_file

                combined_tree = {
                    **base_tree,
                    self._mission_keyword: {
                        "data": {
                            str(file): {
                                sid: self._out_trees[file][sid][self._mission_keyword]["data"]
                                for sid in sources_to_include
                                if sid in self._out_trees[file]
                            }
                            for file in files_to_include
                        },
                        "meta": meta_by_file,
                    },
                }

            af = asdf.AsdfFile(combined_tree)
            af.tree["history"] = _append_history_entry(
                combined_tree.get("history", {}),
                f"Spectral subset created for source IDs {sources_to_include} from files {', '.join(files_to_include)}"
                f"{f' with wavelength range {self._wl_range}' if self._wl_range is not None else ''}.",
            )
            asdf_subsets = af
        else:
            raise InvalidInputError(self._invalid_group_by_msg.format(group_by))

        return asdf_subsets

    def _build_write_jobs(
        self,
        output_dir: Union[str, Path],
        group_by: Literal["source_file", "file", "combined"],
        spectral_files: List[Union[str, Path]],
        source_ids: Union[str, int, List[Union[str, int]]],
    ) -> List[Tuple[asdf.AsdfFile, str]]:
        """
        Build a list of ASDF subset objects and corresponding output file paths to write, based
        on the specified grouping.

        Parameters
        ----------
        output_dir : str or Path
            The directory where the output ASDF files will be written.
        group_by : {'source_file', 'file', 'combined'}
            Determines how the subsets are grouped in the output ASDF objects. Must be
            one of 'source_file', 'file', or 'combined'.
        spectral_files : list of str or Path
            The list of input spectral files to include in the output subsets.
        source_ids : str or int or list
            Specific source IDs to include in the output. Can be a single ID or a list of IDs.

        Returns
        -------
        list of tuples
            A list of tuples, where each tuple contains an ASDF subset object and the corresponding
            output file path to which it should be written.
        """
        write_jobs = []  # List of tuples: (asdf.AsdfFile, output_file_path)
        if group_by == "source_file":
            # Write separate ASDF files for each source/file combination, using the deterministic keys
            # for the source/file combinations
            af_by_source_file = self.get_asdf_subsets(
                group_by="source_file",
                source_ids=source_ids,
                spectral_files=spectral_files,
            )
            source_file_keys = self.get_source_file_keys(
                spectral_files=spectral_files,
                source_ids=source_ids,
            )

            for key, af in af_by_source_file.items():
                file, sid = source_file_keys[key]
                filename = f'{Path(file).stem}_subset_{sid}{"_lite" if self._lite else ""}.asdf'
                write_jobs.append((af, str(output_dir / filename)))

        elif group_by == "file":
            # Write one ASDF file per input file, containing all specified source IDs from that file
            af_by_file = self.get_asdf_subsets(
                group_by="file",
                source_ids=source_ids,
                spectral_files=spectral_files,
            )
            for file, af in af_by_file.items():
                filename = f'{Path(file).stem}_subset{"_lite" if self._lite else ""}.asdf'
                write_jobs.append((af, str(output_dir / filename)))

        elif group_by == "combined":
            # Write a single ASDF file containing all specified source IDs from all input files
            af = self.get_asdf_subsets(
                group_by="combined",
                source_ids=source_ids,
                spectral_files=spectral_files,
            )
            filename = f'combined_spectral_subset{"_lite" if self._lite else ""}.asdf'
            write_jobs.append((af, str(output_dir / filename)))

        else:
            raise InvalidInputError(self._invalid_group_by_msg.format(group_by))

        return write_jobs

    def write_as_asdf(
        self,
        output_dir: Union[str, Path] = ".",
        *,
        group_by: Literal["source_file", "file", "combined"] = "combined",
        spectral_files: List[Union[str, Path]] = None,
        source_ids: Union[str, int, List[Union[str, int]]] = None,
        max_workers: Optional[int] = None,
        validate_output: bool = True,
    ) -> List[str]:
        """
        Write the ASDF subset(s) to files in the specified output directory, grouped by source
        and file, file, or combined.

        Parameters
        ----------
        output_dir : str or Path, optional
            Output directory for subset files. Default is the current directory.
        group_by : {'source_file', 'file', 'combined'}, optional
            Determines how the subsets are grouped in the output ASDF files. Default is 'combined'.
            - 'source_file': Separate ASDF file for each source ID and input file combination.
            - 'file': One ASDF file per input file, containing all specified source IDs from that file.
            - 'combined': A single ASDF file containing all specified source IDs from all input files.
        spectral_files : str or Path or S3Path or list, optional
            Specific spectral files to include in the output. If None, all input spectral files will be included.
            Can be a single file or a list of files.
        source_ids : str or int or list, optional
            Specific source IDs to include in the output. If None, all source IDs from the
            subset results will be included. Can be a single ID or a list of IDs.
        max_workers : int or None, optional
            Maximum number of worker processes to use when writing files in parallel. Default is None.
            If None, the number of workers will be set based on the number of write jobs and CPU count.
            It is recommended to use parallel processing when writing large batches of files (>5000).
        validate_output : bool, optional
            If True, run ASDF schema validation during each file write.
            If False, validate only a single output file during its write, then skip schema
            validation for all remaining writes. Default is True. Consider setting to False
            for improved performance when writing large batches of files that are expected to be valid.

        Returns
        -------
        List[str]
            List of file paths to the written ASDF subset files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        write_jobs = self._build_write_jobs(
            output_dir=output_dir,
            group_by=group_by,
            spectral_files=spectral_files,
            source_ids=source_ids,
        )

        if not write_jobs:
            return []

        subset_paths = []
        if not validate_output:
            # Validate exactly one output file during write, before writing the remaining files.
            validation_job = write_jobs[0]
            validation_af, validation_target_path = validation_job
            validation_af.write_to(validation_target_path)
            subset_paths.append(validation_target_path)
            write_jobs = write_jobs[1:]

            if not write_jobs:
                return subset_paths

        # If max_workers is not specified, default to CPU count, but never more than the number of write jobs
        cpu_count = os.cpu_count() or 1
        worker_count = min(len(write_jobs), cpu_count) if max_workers is None else min(len(write_jobs), max_workers)
        log.debug(f"Writing {len(write_jobs)} ASDF subset(s) to disk with up to {worker_count} worker(s)...")

        # Single-worker path: keep it simple and avoid process overhead
        if worker_count == 1:
            with nullcontext() if validate_output else _disabled_asdf_validation():
                for af, subset_path in write_jobs:
                    af.write_to(subset_path)
                    subset_paths.append(subset_path)
            return subset_paths

        # Multi-worker path: split jobs as evenly as possible and write batches in processes
        batch_count = max(1, min(worker_count, len(write_jobs)))
        batches = [[] for _ in range(batch_count)]
        for idx, item in enumerate(write_jobs):
            batches[idx % batch_count].append(item)
        write_job_batches = [batch for batch in batches if batch]
        tree_job_batches = [
            [(_make_pickle_safe(af.tree), subset_path) for af, subset_path in batch] for batch in write_job_batches
        ]

        batch_results = [None] * len(tree_job_batches)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(_write_tree_batch_worker, batch, validate_output): idx
                for idx, batch in enumerate(tree_job_batches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                batch_results[idx] = future.result()

        paths = list(subset_paths)
        for batch_paths in batch_results:
            paths.extend(batch_paths)
        return paths

    def _build_source_file_keys(self, source_file_pairs: List[Tuple[str, str]]) -> Dict[str, Tuple[str, str]]:
        """
        Build deterministic source/file keys and disambiguate collisions when needed.

        Parameters
        ----------
        source_file_pairs : list of tuples
            List of (file, source_id) pairs for which to build keys.

        Returns
        -------
        dict
            Mapping from string keys to (file, source_id) tuples.
        """
        key_candidates = []
        base_counts = {}
        for file, sid in source_file_pairs:
            base_key = self._make_source_file_key(file, sid)
            key_candidates.append((base_key, file, sid))
            base_counts[base_key] = base_counts.get(base_key, 0) + 1

        keys = {}
        for base_key, file, sid in key_candidates:
            if base_counts[base_key] == 1 and base_key not in keys:
                keys[base_key] = (file, sid)
                continue

            file_hash = blake2s(str(file).encode("utf-8"), digest_size=4).hexdigest()
            disambiguated_key = self._make_source_file_key(
                file,
                sid,
                disambiguator=file_hash,
            )
            keys[disambiguated_key] = (file, sid)

        return keys

    def _resolve_selection(
        self,
        spectral_files: Union[str, Path, S3Path, List[Union[str, Path, S3Path]]] = None,
        source_ids: Union[str, int, List[Union[str, int]]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Resolve and validate file/source selections against available subset results.

        Parameters
        ----------
        spectral_files : str or Path or S3Path or list, optional
            Specific spectral files to include. If None, all available files are included.
        source_ids : str or int or list, optional
            Specific source IDs to include. If None, all available source IDs are included.

        Returns
        -------
        tuple
            A tuple containing two lists: (files_to_include, sources_to_include), where each list contains the
            validated string keys for the selected files and source IDs, respectively.
        """
        if spectral_files is None:
            files_to_include = list(self._out_trees.keys())
        else:
            files_to_include = spectral_files if isinstance(spectral_files, list) else [spectral_files]
            files_to_include = [str(file) for file in files_to_include]

            for file in files_to_include:
                if file not in self._out_trees:
                    raise InvalidQueryError(f"Spectral file {file} not found in subset results.")

        all_source_ids = set()
        for file in files_to_include:
            all_source_ids.update(str(sid) for sid in self._out_trees[file].keys())
        all_source_ids = sorted(all_source_ids)

        if source_ids is None:
            sources_to_include = all_source_ids
        else:
            sources_to_include = source_ids if isinstance(source_ids, list) else [source_ids]
            sources_to_include = [str(sid) for sid in sources_to_include]

            for sid in sources_to_include:
                if sid not in all_source_ids:
                    raise InvalidQueryError(f"Source ID {sid} not found in subset results.")

        return files_to_include, sources_to_include

    @staticmethod
    def _make_source_file_key(
        file: Union[str, Path, S3Path],
        source_id: Union[str, int],
        disambiguator: Optional[str] = None,
    ) -> str:
        """
        Create a stable string key for a source ID and input spectral file combination.

        Parameters
        ----------
        file : str or Path or S3Path
            The input spectral file path.
        source_id : str or int
            The source ID.
        disambiguator : str, optional
            Additional suffix used only when needed to disambiguate duplicate keys.

        Returns
        -------
        str
            A string key combining the source ID and the input file name, suitable for use as a dictionary key
        """
        components = [str(source_id), Path(str(file)).stem]

        if disambiguator:
            components.append(disambiguator)

        return "_".join(components)
