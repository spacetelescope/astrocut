from abc import ABC
from pathlib import Path
from typing import Union, List, Optional, Literal
from copy import deepcopy
import warnings

import asdf
from s3path import S3Path

from . import __version__
from .spectral_cutout import SpectralCutout
from .exceptions import InvalidQueryError, DataWarning


class ASDFSpectralCutout(SpectralCutout, ABC):
    """
    Class for creating spectral cutouts from multi-dimensional data.
    This class inherits from BaseCutout and implements methods specific to spectral data.
    """

    def __init__(self, 
                 spectral_files: Union[str, Path, S3Path, List[Union[str, Path, S3Path]]], 
                 source_ids: Union[str, int, List[Union[str, int]]], 
                 wl_range: Union[tuple, list] = None,
                 lite: Optional[bool] = True,
                 verbose: bool = False):
        super().__init__(spectral_files, source_ids, wl_range, lite, verbose)

        self._mission_keyword = None  # To be set in subclass
        self._out_trees = {}  # Store ASDF tree for the cutouts
        self._base_trees = {}  # To store base ASDF tree without mission data
        self._base_missions = {}  # To store base mission data without source-specific data

    def get_asdf_cutouts(self,
                         *,
                         group_by: Literal['source_file', 'file', 'combined'] = 'file',
                         source_ids: Union[str, int, List[Union[str, int]]] = None,
                         spectral_files: Union[str, Path, S3Path, List[Union[str, Path, S3Path]]] = None) -> dict:
        """
        Get a combined ASDF cutout containing all sources or a subset of sources.

        Parameters
        ----------
        source_ids : str or int or list, optional
            Specific source IDs to include in the combined cutout.
            If None, all sources will be included.

        Returns
        -------
        asdf.AsdfFile
            The ASDF file containing cutouts for the specified sources.
        """
        # Determine which spectral files to include
        if spectral_files is None:
            files_to_include = self._out_trees.keys()
        else:
            files_to_include = spectral_files if isinstance(spectral_files, list) else [spectral_files]

            for file in files_to_include:
                if file not in self._out_trees:
                    raise InvalidQueryError(f'Spectral file {file} not found in cutout results.')

        # Determine which sources to include
        all_source_ids = set()
        for file in files_to_include:
            all_source_ids.update(self._out_trees[file].keys())
    
        if source_ids is None:
            sources_to_include = all_source_ids
        else:
            sources_to_include = source_ids if isinstance(source_ids, list) else [source_ids]

            for sid in sources_to_include:
                if sid not in all_source_ids:
                    raise InvalidQueryError(f'Source ID {sid} not found in cutout results.')

        asdf_cutouts = {}
        if group_by == 'source_file':
            # Group by source and file
            for file in files_to_include:
                for sid in sources_to_include:
                    if sid not in self._out_trees[file]:
                        warnings.warn(f'Source ID {sid} not found in file {file}. '
                                      'Skipping this source for this file.', DataWarning)
                        continue
                    tree = deepcopy(self._out_trees[file][sid])
                    af = asdf.AsdfFile(tree)
                    print(af.tree['history'])
                    af.add_history_entry(
                        f"Spectral cutout created for source ID {sid} from file {file} "
                        f"with wavelength range {self._wl_range}.",
                        software={
                            "name": "astrocut",
                            "author": "Space Telescope Science Institute",
                            "version": __version__,
                            "homepage": "https://astrocut.readthedocs.io/en/latest/",
                        },
                    )
                    asdf_cutouts[(file, sid)] = af
        elif group_by == 'file':
            # Group by file, combining all sources in each file
            for file in files_to_include:
                source_ids = [sid for sid in sources_to_include if sid in self._out_trees[file]]
                if self._lite:
                    combined_tree = {
                        self._mission_keyword: {
                            'data': {sid: self._out_trees[file][sid][self._mission_keyword]['data'] 
                                     for sid in sources_to_include if sid in self._out_trees[file]}, 
                            'meta': {'source_ids': source_ids}
                        }
                    }
                else:
                    base_tree_copy = deepcopy(self._base_trees[file])
                    combined_tree = {
                        **base_tree_copy,
                        self._mission_keyword: {
                            **self._base_missions[file],  # shallow copy mission-level
                            'data': {sid: self._out_trees[file][sid][self._mission_keyword]['data'] 
                                     for sid in sources_to_include if sid in self._out_trees[file]},
                            'meta': {**self._base_missions[file]['meta'], 'source_ids': source_ids}
                        }
                    }
                af = asdf.AsdfFile(combined_tree)
                af.add_history_entry(
                    f"Spectral cutout created for source IDs {source_ids} from file {file} "
                    f"with wavelength range {self._wl_range}.",
                    software={
                        "name": "astrocut",
                        "author": "Space Telescope Science Institute",
                        "version": __version__,
                        "homepage": "https://astrocut.readthedocs.io/en/latest/",
                    },
                )
                asdf_cutouts[file] = af
        else:
            # Group all sources and spectral files into a single ASDF file
            # ASDF data should be keyed by file and then source_id to avoid key collisions
            # meta should also be keyed by file
            if self._lite:
                combined_tree = {
                    self._mission_keyword: {
                        'data': {file: {sid: self._out_trees[file][sid][self._mission_keyword]['data'] 
                                        for sid in sources_to_include if sid in self._out_trees[file]} 
                                 for file in files_to_include},
                        'meta': {'source_ids': [sid for sid in sources_to_include if sid in all_source_ids]}
                    }
                }
            else:
                # TODO: Right now, this doesn't work with the original key because of ASDF validation errors 
                # due to required keys not being present
                base_tree = {}
                for key in self._base_trees[file].keys():
                    key_by_file = {file: self._base_trees[file][key] for file in files_to_include}
                    base_tree[f'{key}_combined'] = key_by_file

                meta_by_file = {}
                for file in files_to_include:
                    meta_by_file[file] = {
                        **self._base_missions[file].get('meta', {}), 
                        'source_ids': [sid for sid in sources_to_include if sid in self._out_trees[file]]
                    }

                combined_tree = {
                    **base_tree,
                    self._mission_keyword: {
                        'data': {file: {sid: self._out_trees[file][sid][self._mission_keyword]['data'] 
                                        for sid in sources_to_include if sid in self._out_trees[file]} 
                                 for file in files_to_include},
                        'meta': meta_by_file
                    }
                }

            af = asdf.AsdfFile(combined_tree)
            af.add_history_entry(
                f"Spectral cutout created for source IDs {sources_to_include} from files {files_to_include} "
                f"with wavelength range {self._wl_range}.",
                software={
                    "name": "astrocut",
                    "author": "Space Telescope Science Institute",
                    "version": __version__,
                    "homepage": "https://astrocut.readthedocs.io/en/latest/",
                },
            )
            asdf_cutouts = af

        return asdf_cutouts
    
    def _cutout_file(self, file: Union[str, Path, S3Path]):
        """
        Internal method to perform cutout on a single file.
        This is a placeholder for potential future use if we want to support multiple files.
        """
        with asdf.open(file) as af:
            in_tree = af.tree  # Load the entire tree lazily
            mission_data = in_tree[self._mission_keyword]['data']

            self._base_trees[file] = {
                k: v for k, v in in_tree.items()
                if k != self._mission_keyword
            }
            self._base_missions[file] = {
                k: v for k, v in in_tree[self._mission_keyword].items()
                if k != "data"
            }

            for sid in self._source_ids:
                # sid = str(sid)
                if sid not in mission_data:
                    # Skip if the source is not found in this file
                    print(f'Warning: Source ID {sid} not found in file {file}. Skipping this source for this file.')
                    continue  
                
                source = mission_data[sid]
                wl = source['wl']

                # Apply wavelength range filter if specified
                if self._wl_range is not None:
                    # Validate that wl_range overlaps with available wavelength data
                    wl_min = wl.min()
                    wl_max = wl.max()
                    
                    if self._wl_range[0] > wl_max or self._wl_range[1] < wl_min:
                        warnings.warn(
                            f'Wavelength range {self._wl_range} is out of bounds for source ID {sid} and file {file}. '
                            f'Available wavelength range: [{wl_min}, {wl_max}]. Skipping this source for this file.',
                            DataWarning
                        )
                        continue
                    
                    mask = (wl >= self._wl_range[0]) & (wl <= self._wl_range[1])
                else:
                    mask = slice(None)

                # Identify spectral keys (arrays with length equal to n_wl)
                keys = ['flux', 'wl'] if self._lite else source.keys()
                spectral_keys = {
                    k for k, v in source.items()
                    if hasattr(v, '__len__') and len(v) == len(wl)
                }

                new_source = {}
                for key in keys:
                    value = source[key]
                    if key in spectral_keys:
                        new_source[key] = value[mask]  # sliced, materialized
                    else:
                        new_source[key] = value  # reused as-is (lazy / scalar)

                # Build one output tree per source
                if self._lite:
                    out_tree = {self._mission_keyword: {'data': new_source, 'meta': {'source_id': sid}}}
                else:
                    out_tree = {
                        **self._base_trees[file],  # shallow copy top-level
                        self._mission_keyword: {
                            ** self._base_missions[file],  # shallow copy mission-level
                            'data': new_source,
                            'meta': {**self._base_missions[file].get('meta', {}), 'source_id': sid}
                        }
                    }

                self._out_trees.setdefault(file, {})
                self._out_trees[file][sid] = out_tree

    def cutout(self):
        """
        Generate the spectral cutout(s).
        This method should be implemented to handle spectral data specifically.
        """
        # TODO: If Astrocut will be responsible for locating and opening cloud files, we may want to implement 
        # caching of open file handlers.
        for file in self._spectral_files:
            self._cutout_file(file)
            
        if not self._out_trees:
            raise InvalidQueryError(
                'No cutouts were created. Please verify that the source IDs are present in '
                'the spectral files and that the wavelength range overlaps with the data.'
            )

    def write_as_asdf(self, 
                      output_dir: Union[str, Path] = '.',
                      *,
                      group_by: Literal['source_file', 'file', 'combined'] = 'file',
                      source_ids: Union[str, int, List[Union[str, int]]] = None,
                      spectral_files: List[Union[str, Path]] = None) -> List[str]:
        """
        Write the ASDF cutout(s) to files in the specified output directory.

        Parameters
        ----------
        output_dir : str or Path
            Directory where the ASDF cutout files will be saved.
        source_ids : str or int or list, optional
            Specific source IDs to write. If None, all source IDs will be written.
            When single_outfile is True, these sources will be combined into one file.
            When single_outfile is False, separate files will be created for each source.
        single_outfile : bool, optional
            If True (default), writes all sources to a single combined file.
            If False, writes separate files for each source.

        Returns
        -------
        List[str]
            List of file paths where the ASDF cutouts were saved.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cutout_paths = []  # List to store paths to cutout files

        if group_by == 'source_file':
            af_by_source_file = self.get_asdf_cutouts(group_by='source_file', 
                                                      source_ids=source_ids, spectral_files=spectral_files)
            for (file, sid), af in af_by_source_file.items():
                filename = f'{Path(file).stem}_cutout_{sid}{"_lite" if self._lite else ""}.asdf'
                cutout_path = output_dir / filename
                af.write_to(cutout_path)
                cutout_paths.append(str(cutout_path))
        elif group_by == 'file':
            af_by_file = self.get_asdf_cutouts(group_by='file', source_ids=source_ids, spectral_files=spectral_files)
            for file, af in af_by_file.items():
                filename = f'{Path(file).stem}_cutout{"_lite" if self._lite else ""}.asdf'
                cutout_path = output_dir / filename
                af.write_to(cutout_path)
                cutout_paths.append(str(cutout_path))
        else:
            af = self.get_asdf_cutouts(group_by='combined', source_ids=source_ids, spectral_files=spectral_files)
            filename = f'combined_spectral_cutout{"_lite" if self._lite else ""}.asdf'
            cutout_path = output_dir / filename
            af.write_to(cutout_path)
            cutout_paths.append(str(cutout_path))

        return cutout_paths
