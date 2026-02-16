from abc import ABC
from pathlib import Path
from typing import Union, List, Optional
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
                 file: Union[str, Path, S3Path], 
                 source_ids: Union[str, int, List[Union[str, int]]], 
                 wl_range: Union[tuple, list] = None,
                 lite: Optional[bool] = True,
                 verbose: bool = False):
        super().__init__(file, source_ids, wl_range, lite, verbose)

        self._out_trees = {}  # Store ASDF tree for the cutout
        self._mission_keyword = None  # To be set in subclass
        self._base_tree = None  # To store base ASDF tree without mission data
        self._base_mission = None  # To store base mission data without source-specific data

    def get_asdf_cutout(self, source_ids: Union[str, int, List[Union[str, int]]] = None) -> asdf.AsdfFile:
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
        # Determine which sources to include
        if source_ids is None:
            sources_to_include = self._out_trees.keys()
        else:
            if not isinstance(source_ids, list):
                source_ids = [source_ids]
            sources_to_include = source_ids
        
        # Combine specified sources into a single tree
        all_sources = {}
        for sid in sources_to_include:
            if sid not in self._out_trees:
                raise InvalidQueryError(f'Source ID {sid} not found in cutout results.')
            tree = self._out_trees[sid]
            source_data = tree[self._mission_keyword]['data']
            # Individual trees have data directly, not keyed by source_id
            all_sources[sid] = source_data
        
        if self._lite:
            combined_tree = {self._mission_keyword: {'data': all_sources}}
        else:
            # Build combined tree using base metadata (which doesn't include source_id)
            combined_tree = {
                **self._base_tree,
                self._mission_keyword: {
                    **self._base_mission,
                    'data': all_sources
                }
            }
        
        af = asdf.AsdfFile(combined_tree)
        af.add_history_entry(
            f"Spectral cutout created for source IDs {list(all_sources.keys())} "
            f"with wavelength range {self._wl_range}.",
            software={
                "name": "astrocut",
                "author": "Space Telescope Science Institute",
                "version": __version__,
                "homepage": "https://astrocut.readthedocs.io/en/latest/",
            },
        )
        return af

    def get_asdf_cutout_by_source(self, source_id: Union[str, int]) -> asdf.AsdfFile:
        """
        Get the ASDF cutout for a specific source ID.

        Parameters
        ----------
        source_id : str or int
            The source ID for which to retrieve the ASDF cutout.

        Returns
        -------
        asdf.AsdfFile
            The ASDF file containing the cutout for the specified source ID.
        """
        tree = self._out_trees[source_id]
        af = asdf.AsdfFile(tree)
        af.add_history_entry(
            f"Spectral cutout created for source ID {source_id} "
            f"with wavelength range {self._wl_range}.",
            software={
                "name": "astrocut",
                "author": "Space Telescope Science Institute",
                "version": __version__,
                "homepage": "https://astrocut.readthedocs.io/en/latest/",
            },
        )
        return af

    def cutout(self):
        """
        Generate the spectral cutout(s).
        This method should be implemented to handle spectral data specifically.
        """
        # TODO: If Astrocut will be responsible for locating and opening cloud files, we may want to implement 
        # caching of open file handlers.
        with asdf.open(self._file) as af:
            in_tree = af.tree  # Load the entire tree lazily
            mission_data = in_tree[self._mission_keyword]['data']

            self._base_tree = {
                k: v for k, v in in_tree.items()
                if k != self._mission_keyword
            }
            self._base_mission = {
                k: v for k, v in in_tree[self._mission_keyword].items()
                if k != "data"
            }

            for sid in self._source_ids:
                # sid = str(sid)
                if sid not in mission_data:
                    raise InvalidQueryError(f'Source ID {sid} not found in the data.')
                
                source = mission_data[sid]
                wl = source['wl']

                if self._wl_range is not None:
                    # Validate that wl_range overlaps with available wavelength data
                    wl_min = wl.min()
                    wl_max = wl.max()
                    
                    if self._wl_range[0] > wl_max or self._wl_range[1] < wl_min:
                        warnings.warn(
                            f'Wavelength range {self._wl_range} is out of bounds for source ID {sid}. '
                            f'Available wavelength range: [{wl_min}, {wl_max}]. Skipping this source.',
                            DataWarning
                        )
                        continue
                    
                    mask = (wl >= self._wl_range[0]) & (wl <= self._wl_range[1])
                else:
                    mask = slice(None)

                n_wl = len(wl)
                keys = ['flux', 'wl'] if self._lite else source.keys()

                # Identify spectral keys (arrays with length equal to n_wl)
                spectral_keys = {
                    k for k, v in source.items()
                    if hasattr(v, '__len__') and len(v) == n_wl
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
                        **self._base_tree,  # shallow copy top-level
                        self._mission_keyword: {
                            **self._base_mission,
                            'data': new_source,
                            'meta': {**self._base_mission.get('meta', {}), 'source_id': sid}
                        }
                    }

                self._out_trees[sid] = out_tree
            
            if not self._out_trees:
                raise InvalidQueryError(
                    'No cutouts were created. All source IDs were out of bounds for the requested wavelength range.'
                )

    def write_as_asdf(self, output_dir: Union[str, Path] = '.', 
                      source_ids: Union[str, int, List[Union[str, int]]] = None,
                      single_outfile: bool = True) -> List[str]:
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
        
        if single_outfile:
            # Write single combined file
            filename = f'{Path(self._file).stem}_cutout{"_lite" if self._lite else ""}.asdf'
            cutout_path = output_dir / filename
            af = self.get_asdf_cutout(source_ids=source_ids)
            af.write_to(cutout_path)
            cutout_paths.append(str(cutout_path))
        else:
            # Write separate files for each source
            out_trees = self._out_trees
            if source_ids is not None:
                if not isinstance(source_ids, list):
                    source_ids = [source_ids]
                out_trees = {sid: self._out_trees[sid] for sid in source_ids}

            for sid in out_trees.keys():
                filename = f'{Path(self._file).stem}_cutout_{sid}{"_lite" if self._lite else ""}.asdf'
                cutout_path = output_dir / filename
                af = self.get_asdf_cutout_by_source(sid)
                af.write_to(cutout_path)
                cutout_paths.append(str(cutout_path))

        return cutout_paths
