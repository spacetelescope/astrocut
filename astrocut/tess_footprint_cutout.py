import re
from pathlib import Path
from typing import Union, List, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io.fits import HDUList
from astropy.table import Table

from . import log
from .exceptions import InvalidQueryError, InvalidInputError
from .footprint_cutout import FootprintCutout, get_ffis, ra_dec_crossmatch
from .tess_cube_cutout import TessCubeCutout


class TessFootprintCutout(FootprintCutout):
    """
    Class for generating cutouts from TESS Full Frame Images (FFIs) hosted on the cloud 
    based on the footprint of the cutout.

    Parameters
    ----------
    coordinates : str | `~astropy.coordinates.SkyCoord`
        Coordinates of the center of the cutout.
    cutout_size : int | array | list | tuple | `~astropy.units.Quantity`
        Size of the cutout array.
    fill_value : int | float
        Value to fill the cutout with if the cutout is outside the image.
    limit_rounding_method : str
        Method to use for rounding the cutout limits. Options are 'round', 'ceil', and 'floor'.
    sequence : int | list | None
        Default None. Sequence(s) from which to generate cutouts. Can provide a single
        sequence number as an int or a list of sequence numbers. If not specified, 
        cutouts will be generated from all sequences that contain the cutout.
        For the TESS mission, this parameter corresponds to sectors.
    product : str, optional
        Default 'SPOC'. The product type to make the cutouts from.
    verbose : bool
        If True, log messages are printed to the console.

    Attributes
    ----------
    tess_cube_cutout : `~astrocut.TessCubeCutout`
        Object containing the cutouts from the TESS cubes.
    cutouts_by_file : dict
        Dictionary where each key is an input cube S3 URI and its corresponding value is the resulting cutout as a 
        ``CubeCutoutInstance`` object.
    cutouts : list
        List of cutout objects.
    tpf_cutouts_by_file : dict
        Dictionary where each key is an input cube S3 URI and its corresponding value is the resulting cutout target
        pixel file object.
    tpf_cutouts : list
        List of cutout target pixel file objects.

    Methods
    -------
    cutout()
        Generate the cutouts from the cloud FFIs that intersect the cutout footprint.
    write_as_tpf(output_dir)
        Write the cutouts as Target Pixel Files (TPFs) to the specified directory.
    """

    def __init__(self, coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, u.Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, limit_rounding_method: str = 'round', 
                 sequence: Union[int, List[int], None] = None, product: str = 'SPOC', verbose: bool = False):
        super().__init__(coordinates, cutout_size, fill_value, limit_rounding_method, sequence, verbose)
        
        # Validate and set the product
        if product.upper() not in ['SPOC', 'TICA']:
            raise InvalidInputError('Product for TESS cube cutouts must be either "SPOC" or "TICA".')
        self._product = product.upper()
        self._arcsec_per_px = 21  # Number of arcseconds per pixel in a TESS image

        # Set S3 URIs to footprint cache file and base file path based on product
        if self._product == 'SPOC':
            self._s3_footprint_cache = 's3://stpubdata/tess/public/footprints/tess_ffi_footprint_cache.json'
            self._s3_base_file_path = 's3://stpubdata/tess/public/mast/'
        else:
            self._s3_footprint_cache = 's3://stpubdata/tess/public/footprints/tica_ffi_footprint_cache.json'
            self._s3_base_file_path = 's3://stpubdata/tess/public/mast/tica/'

        # Make the cutouts upon initialization
        self.cutout()
    
    def _extract_sequence_information(self, sector_name: str) -> dict:
        """
        Extract the sector, camera, and ccd information from the sector name.

        Parameters
        ----------
        sector_name : str
            The name of the sector.

        Returns
        -------
        dict
            A dictionary containing the sector name, sector number, camera number, and CCD number.
        """
        # Define the pattern based on the product
        if self._product == 'SPOC':
            pattern = re.compile(r"(tess-s)(?P<sector>\d{4})-(?P<camera>\d{1,4})-(?P<ccd>\d{1,4})")
        elif self._product == 'TICA':
            pattern = re.compile(r"(hlsp_tica_s)(?P<sector>\d{4})-(cam)(?P<camera>\d{1,4})-(ccd)(?P<ccd>\d{1,4})")
        else:
            # Return an empty dictionary if the product is not recognized
            return {}
        sector_match = re.match(pattern, sector_name)

        if not sector_match:
            # Return an empty dictionary if the name does not match the product pattern
            return {}

        # Extract the sector, camera, and ccd information
        sector = sector_match.group("sector")
        camera = sector_match.group("camera")
        ccd = sector_match.group("ccd")

        # Rename the TICA sector because of the naming convention in Astrocut
        if self._product == 'TICA':
            sector_name = f"tica-s{sector}-{camera}-{ccd}"

        return {"sectorName": sector_name, "sector": sector, "camera": camera, "ccd": ccd}
    
    def _create_sequence_list(self, observations: Table) -> List[dict]:
        """
        Extracts sequence information from a list of observations.

        Parameters
        ----------
        observations : `~astropy.table.Table`
            A table of FFI observations.

        Returns
        -------
        list of dict
            A list of dictionaries, each containing the sector name, sector number, camera number, and CCD number.
        """
        # Filter observations by target name to get only the FFI observations
        target_name = "TESS FFI" if self._product == 'SPOC' else "TICA FFI"
        obs_filtered = [obs for obs in observations if obs["target_name"].upper() == target_name]

        sequence_results = []
        for row in obs_filtered:
            # Extract the sector information for each FFI observation
            sequence_extraction = self._extract_sequence_information(row["obs_id"])
            if sequence_extraction:
                sequence_results.append(sequence_extraction)

        return sequence_results
    
    def _get_files_from_cone_results(self, cone_results: Table) -> List[dict]:
        """
        Converts a `~astropy.table.Table` of cone search results to a list of dictionaries containing 
        information for each cloud cube file that intersects with the cutout.

        Parameters
        ----------
        cone_results : `~astropy.table.Table`
            A table containing observation results, including sector information.

        Returns
        -------
        cube_files : list of dict
            A list of dictionaries, each containing:
            - "folder": The folder name corresponding to the sector, prefixed with 's' and zero-padded to 4 digits.
            - "cube": The expected filename for the cube FITS file in the format "{sectorName}-cube.fits".
            - "sectorName": The sector name.
        """
        # Create a list of dictionaries containing the sector information
        seq_list = self._create_sequence_list(cone_results)

        # Create a list of dictionaries containing the cube file information
        cube_files = [
            {
                "folder": "s" + sector["sector"].rjust(4, "0"),
                "cube": sector["sectorName"] + "-cube.fits",
                "sectorName": sector["sectorName"],
            }
            for sector in seq_list
        ]
        return cube_files

    def cutout(self):
        """
        Generate the cutouts from the cloud FFIs that intersect the cutout footprint.

        Raises
        ------
        InvalidQueryError
            If no files are found for the given sequences.
            If the given coordinates are not found within the specified sequence(s).
        """
        # Get footprints from the cloud
        all_ffis = get_ffis(self._s3_footprint_cache)
        log.debug('Found %d footprint files.', len(all_ffis))

        # Filter footprints by sequence
        if self._sequence:
            all_ffis = all_ffis[np.isin(all_ffis['sequence_number'], self._sequence)]

            if len(all_ffis) == 0:
                raise InvalidQueryError('No files were found for sequences: ' +
                                        ', '.join(str(s) for s in self._sequence))
            
            log.debug('Filtered to %d footprints for sequences: %s', len(all_ffis), 
                      ', '.join(str(s) for s in self._sequence))

        # Get sequence names and files that contain the cutout
        cone_results = ra_dec_crossmatch(all_ffis, self._coordinates, self._cutout_size, self._arcsec_per_px)
        if not cone_results:
            raise InvalidQueryError('The given coordinates were not found within the specified sequence(s).')
        files_mapping = self._get_files_from_cone_results(cone_results)
        log.debug('Found %d matching files.', len(files_mapping))

        # Generate the cube cutouts
        log.debug('Generating cutouts...')
        input_files = [f"{self._s3_base_file_path}{file['cube']}" for file in files_mapping]
        tess_cube_cutout = TessCubeCutout(input_files, self._coordinates, self._cutout_size, 
                                          self._fill_value, self._limit_rounding_method, threads=8, 
                                          product=self._product, verbose=self._verbose)
        
        # Assign attributes from the TessCubeCutout object
        self.tess_cube_cutout = tess_cube_cutout
        self.cutouts_by_file = tess_cube_cutout.cutouts_by_file
        self.cutouts = tess_cube_cutout.cutouts
        self.tpf_cutouts_by_file = tess_cube_cutout.tpf_cutouts_by_file
        self.tpf_cutouts = tess_cube_cutout.tpf_cutouts
        
    def write_as_tpf(self, output_dir: Union[str, Path] = '.') -> List[str]:
        """
        Write the cutouts to disk as target pixel files.

        Parameters
        ----------
        output_dir : str | Path
            The output directory where the cutout files will be saved.

        Returns
        -------
        cutout_paths : list
            List of file paths to cutout target pixel files.
        """
        return self.tess_cube_cutout.write_as_tpf(output_dir)
    

def cube_cut_from_footprint(coordinates: Union[str, SkyCoord], cutout_size, 
                            sequence: Union[int, List[int], None] = None, product: str = 'SPOC',
                            memory_only=False, output_dir: str = '.', 
                            verbose: bool = False) -> Union[List[str], List[HDUList]]:
    """
    Generates cutouts around `coordinates` of size `cutout_size` from image cube files hosted on the S3 cloud.

    Parameters
    ----------
    coordinates : str or `astropy.coordinates.SkyCoord` object
        The position around which to cutout.
        It may be specified as a string ("ra dec" in degrees)
        or as the appropriate `~astropy.coordinates.SkyCoord` object.
    cutout_size : int, array-like, `~astropy.units.Quantity`
        The size of the cutout array. If ``cutout_size``
        is a scalar number or a scalar `~astropy.units.Quantity`,
        then a square cutout of ``cutout_size`` will be created.  If
        ``cutout_size`` has two elements, they should be in ``(ny, nx)``
        order.  Scalar numbers in ``cutout_size`` are assumed to be in
        units of pixels. `~astropy.units.Quantity` objects must be in pixel or
        angular units.
    sequence : int, List[int], optional
        Default None. Sequence(s) from which to generate cutouts. Can provide a single
        sequence number as an int or a list of sequence numbers. If not specified, 
        cutouts will be generated from all sequences that contain the cutout.
        For the TESS mission, this parameter corresponds to sectors.
    product : str, optional
        Default 'SPOC'. The product type to make the cutouts from.
    memory_only : bool, optional
        Default False. If True, the cutouts are stored in memory and not written to disk.
    output_dir : str, optional
        Default '.'. The path to which output files are saved.
        The current directory is default.
    verbose : bool, optional
        Default False. If True, intermediate information is printed.

    Returns
    -------
    cutout_files : list
        List of paths to the cutout files if ``memory_only`` is False.
        If ``memory_only`` is True, returns a list of cutouts as memory objects.

    Examples
    --------
    >>> from astrocut import cube_cut_from_footprint
    >>> cube_cut_from_footprint(  #doctest: +SKIP
    ...         coordinates='83.40630967798376 -62.48977125108528',
    ...         cutout_size=64,
    ...         sequence=[1, 2],  # TESS sectors
    ...         product='SPOC',
    ...         output_dir='./cutouts')
    ['./cutouts/tess-s0001-4-4/tess-s0001-4-4_83.406310_-62.489771_64x64_astrocut.fits',
     './cutouts/tess-s0002-4-1/tess-s0002-4-1_83.406310_-62.489771_64x64_astrocut.fits']
    """

    cutouts = TessFootprintCutout(coordinates, cutout_size, sequence=sequence, product=product, verbose=verbose)

    if memory_only:
        return cutouts.tpf_cutouts
    
    # Write cutouts
    return cutouts.write_as_tpf(output_dir)
