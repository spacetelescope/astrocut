from pathlib import Path
from typing import Union, List, Optional

import asdf
from s3path import S3Path

from . import __version__
from .cutout import BaseCutout
from .exceptions import InvalidQueryError


class SpectralCutout(BaseCutout):
    """
    Class for creating spectral cutouts from multi-dimensional data.
    This class inherits from BaseCutout and implements methods specific to spectral data.
    """

    def __init__(self, 
                 file: Union[str, Path, S3Path], 
                 source_ids: Union[str, int, List[Union[str, int]]], 
                 wl_range: Union[tuple, list],
                 lite: Optional[bool] = False,
                 verbose: bool = False):
        super().__init__(verbose=verbose)

        self._file = file
        self._source_ids = source_ids if isinstance(source_ids, list) else [source_ids]
        self._wl_range = wl_range
        self._lite = lite

        self.cutout_data = dict()  # To store cutout data for each source ID
        self._asdf_cutout = None
        self._out_tree = None  # Store ASDF tree for the cutout

        # Make cutouts
        self.cutout()

    @property
    def asdf_cutout(self) -> asdf.AsdfFile:
        """
        Return the ASDF file containing the spectral cutout(s).
        """
        if not self._asdf_cutout:
            af = asdf.AsdfFile(self._out_tree)
            history_msg = (
                f'Spectral cutout created for source IDs {self._source_ids} '
                f'with wavelength range {self._wl_range}.'
            )

            af.add_history_entry(
                history_msg,
                software={
                    'name': 'astrocut',
                    'author': 'Space Telescope Science Institute',
                    'version': __version__,
                    'homepage': 'https://astrocut.readthedocs.io/en/latest/',
                },
            )

            self._asdf_cutout = af
        return self._asdf_cutout
    
    def with_params(self, *, wl_range=None, source_ids=None, lite=None):
        """
        Create a new SpectralCutout instance based on this one with modified parameters.
        
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
        SpectralCutout
            A new SpectralCutout instance with the specified parameters.
        """
        return SpectralCutout(
            self._file,
            source_ids or self._source_ids,
            wl_range or self._wl_range,
            lite or self._lite,
            verbose=self._verbose
        )

    def cutout(self):
        """
        Generate the spectral cutout(s).
        This method should be implemented to handle spectral data specifically.
        """
        # TODO: Should we allow users to pass in a file handler to avoid reopening the file multiple times?
        # If Astrocut will be responsible for locating and opening cloud files, we may want to implement 
        # caching of open file handlers.
        with asdf.open(self._file) as af:
            in_tree = af.tree  # Load the entire tree lazily
            roman_data = in_tree['roman']['data']

            # Build output tree once
            if self._lite:
                out_tree = {'roman': {'data': {}}}
            else:
                out_tree = {
                    **in_tree,  # shallow copy top-level
                    'roman': {
                        **in_tree['roman'],
                        'data': {}
                    }
                }

            for sid in self._source_ids:
                sid = str(sid)

                if sid not in roman_data:
                    raise InvalidQueryError(f'Source ID {sid} not found in the data.')
                
                source = roman_data[sid]
                wl = source['wl']

                # Determine wavelength mask
                if self._wl_range is not None:
                    mask = (wl >= self._wl_range[0]) & (wl <= self._wl_range[1])
                else:
                    mask = slice(None)

                # Copy *all* keys, but slice only spectral arrays
                n_wl = len(wl)
                keys = ['wl', 'flux'] if self._lite else source.keys()
                spectral_keys = [
                    k for k, v in source.items()
                    if hasattr(v, '__len__') and len(v) == n_wl
                ]

                new_source = {}
                for key in keys:
                    value = source[key]
                    if key in spectral_keys:
                        new_source[key] = value[mask]  # sliced, materialized
                    else:
                        new_source[key] = value  # reused as-is (lazy / scalar)

                out_tree['roman']['data'][sid] = new_source

            # Store final tree
            self._out_tree = out_tree
