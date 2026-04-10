import warnings
from abc import ABC
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import asdf
import numpy as np
from s3path import S3Path

from . import __version__
from .exceptions import DataWarning, InvalidInputError, InvalidQueryError
from .spectral_subset import SpectralSubset


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
        verbose: bool = False,
    ):
        super().__init__(spectral_files, source_ids, wl_range, lite, verbose)

        self._mission_keyword = None  # To be set in subclass
        self._out_trees = {}  # Store ASDF tree for the subsets
        self._base_trees = {}  # To store base ASDF tree without mission data
        self._base_missions = {}  # To store base mission data without source-specific data

        self.subset_data = dict()  # Store subset data for each source ID and input file combination
        self.source_file_keys = {}  # Map string source/file keys to (file, source_id) tuples

    @staticmethod
    def make_source_file_key(file: Union[str, Path, S3Path], source_id: Union[str, int]) -> str:
        """
        Create a stable string key for a source ID and input spectral file combination.

        Parameters
        ----------
        file : str or Path or S3Path
            The input spectral file path.
        source_id : str or int
            The source ID.

        Returns
        -------
        str
            A string key combining the source ID and the input file name, suitable for use as a dictionary key
        """
        return f"{source_id}_{Path(str(file)).stem}"

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

        keys = {}
        for file in files_to_include:
            for sid in sources_to_include:
                if sid not in self._out_trees[file]:
                    continue
                keys[self.make_source_file_key(file, sid)] = (file, sid)

        self.source_file_keys = keys
        return keys

    def subset(self):
        """
        Generate the spectral subset(s) from the input ASDF files based on the specified
        source IDs and wavelength range.

        Raises
        ------
        InvalidQueryError
            If no subsets were created, which may occur if the specified source IDs are not present in
            the input spectral files or if the specified wavelength range does not overlap with the data.
        """
        for file in self._spectral_files:
            self._subset_file(file)

        if not self._out_trees:
            raise InvalidQueryError(
                "No subsets were created. Please verify that the source IDs are present in "
                "the spectral files and that the wavelength range overlaps with the data."
            )

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
        files_to_include, sources_to_include = self._resolve_selection(
            spectral_files=spectral_files, source_ids=source_ids
        )

        asdf_subsets = {}
        if group_by == "source_file":
            # Group by source and file
            self.source_file_keys = {}
            for file in files_to_include:
                for sid in sources_to_include:
                    if sid not in self._out_trees[file]:
                        warnings.warn(
                            f"Source ID {sid} not found in file {file}. " "Skipping this source for this file.",
                            DataWarning,
                        )
                        continue
                    tree = deepcopy(self._out_trees[file][sid])
                    af = asdf.AsdfFile(tree)
                    af.add_history_entry(
                        f"Spectral subset created for source ID {sid} from file {file} "
                        f"with wavelength range {self._wl_range}.",
                        software={
                            "name": "astrocut",
                            "author": "Space Telescope Science Institute",
                            "version": __version__,
                            "homepage": "https://astrocut.readthedocs.io/en/latest/",
                        },
                    )
                    string_key = self.make_source_file_key(file, sid)
                    self.source_file_keys[string_key] = (file, sid)
                    asdf_subsets[string_key] = af
        elif group_by == "file":
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
                af.add_history_entry(
                    f"Spectral subset created for source IDs {source_ids} from file {file} "
                    f"with wavelength range {self._wl_range}.",
                    software={
                        "name": "astrocut",
                        "author": "Space Telescope Science Institute",
                        "version": __version__,
                        "homepage": "https://astrocut.readthedocs.io/en/latest/",
                    },
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
            af.add_history_entry(
                f"Spectral subset created for source IDs {sources_to_include} from files {', '.join(files_to_include)} "
                f"with wavelength range {self._wl_range}.",
                software={
                    "name": "astrocut",
                    "author": "Space Telescope Science Institute",
                    "version": __version__,
                    "homepage": "https://astrocut.readthedocs.io/en/latest/",
                },
            )
            asdf_subsets = af
        else:
            raise InvalidInputError(self._invalid_group_by_msg.format(group_by))

        return asdf_subsets

    def write_as_asdf(
        self,
        output_dir: Union[str, Path] = ".",
        *,
        group_by: Literal["source_file", "file", "combined"] = "combined",
        spectral_files: List[Union[str, Path]] = None,
        source_ids: Union[str, int, List[Union[str, int]]] = None,
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

        Returns
        -------
        List[str]
            List of file paths to the written ASDF subset files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        subset_paths = []  # List to store paths to subset files

        if group_by == "source_file":
            af_by_source_file = self.get_asdf_subsets(
                group_by="source_file",
                source_ids=source_ids,
                spectral_files=spectral_files,
            )
            for source_file_key, af in af_by_source_file.items():
                file, sid = self.source_file_keys[source_file_key]
                filename = f'{Path(file).stem}_subset_{sid}{"_lite" if self._lite else ""}.asdf'
                subset_path = output_dir / filename
                af.write_to(subset_path)
                subset_paths.append(str(subset_path))
        elif group_by == "file":
            af_by_file = self.get_asdf_subsets(group_by="file", source_ids=source_ids, spectral_files=spectral_files)
            for file, af in af_by_file.items():
                filename = f'{Path(file).stem}_subset{"_lite" if self._lite else ""}.asdf'
                subset_path = output_dir / filename
                af.write_to(subset_path)
                subset_paths.append(str(subset_path))
        elif group_by == "combined":
            af = self.get_asdf_subsets(group_by="combined", source_ids=source_ids, spectral_files=spectral_files)
            filename = f'combined_spectral_subset{"_lite" if self._lite else ""}.asdf'
            subset_path = output_dir / filename
            af.write_to(subset_path)
            subset_paths.append(str(subset_path))
        else:
            raise InvalidInputError(self._invalid_group_by_msg.format(group_by))

        return subset_paths

    def _subset_file(self, file: str):
        """
        Generate subsets for a single input spectral file and store the results in the
        internal subset_data and _out_trees attributes.

        Parameters
        ----------
        file : str
            The path to the input spectral file for which to generate subsets.
        """
        with asdf.open(file) as af:
            in_tree = af.tree
            mission_data = in_tree[self._mission_keyword]["data"]

            self._base_trees[file] = {k: v for k, v in in_tree.items() if k != self._mission_keyword}
            self._base_missions[file] = {k: v for k, v in in_tree[self._mission_keyword].items() if k != "data"}

            for sid in self._source_ids:
                sid = str(sid)
                mission_key = sid
                if mission_key not in mission_data:
                    try:
                        sid_int = int(sid)
                    except (TypeError, ValueError):
                        sid_int = None

                    if sid_int in mission_data:
                        mission_key = sid_int
                    else:
                        if self._verbose:
                            warnings.warn(
                                f"Source ID {sid} not found in file {file}. " "Skipping this source for this file.",
                                DataWarning,
                            )
                        continue

                source = mission_data[mission_key]
                wl = source["wl"]

                # Apply wavelength range filter if specified
                if self._wl_range is not None:
                    # Validate that wl_range overlaps with available wavelength data
                    wl_min = wl.min()
                    wl_max = wl.max()

                    if self._wl_range[0] > wl_max or self._wl_range[1] < wl_min:
                        warnings.warn(
                            f"Wavelength range {self._wl_range} is out of bounds for source ID {sid} and file {file}. "
                            f"Available wavelength range: [{wl_min}, {wl_max}]. Skipping this source for this file.",
                            DataWarning,
                        )
                        continue

                    mask = (wl >= self._wl_range[0]) & (wl <= self._wl_range[1])
                else:
                    mask = slice(None)

                # Identify spectral keys (arrays with length equal to n_wl)
                keys = ["wl", "flux", "flux_error"] if self._lite else source.keys()
                spectral_keys = {k for k, v in source.items() if hasattr(v, "__len__") and len(v) == len(wl)}

                new_source = {}
                for key in keys:
                    value = source[key]
                    if key in spectral_keys:
                        arr = np.asarray(value)
                        new_source[key] = arr[mask].copy()
                    else:
                        try:
                            new_source[key] = deepcopy(value)
                        except Exception:
                            new_source[key] = value

                # Add to subset data dictionary
                self.subset_data.setdefault(file, {})
                self.subset_data[file][sid] = new_source

                # Build one output tree per source
                meta = {**self._base_missions[file].get("meta", {}), "source_id": sid}
                if self._lite:
                    out_tree = {self._mission_keyword: {"data": new_source, "meta": meta}}
                else:
                    out_tree = {
                        **self._base_trees[file],  # shallow copy top-level
                        self._mission_keyword: {
                            **self._base_missions[file],  # shallow copy mission-level
                            "data": new_source,
                            "meta": meta,
                        },
                    }
                self._out_trees.setdefault(file, {})
                self._out_trees[file][sid] = out_tree
