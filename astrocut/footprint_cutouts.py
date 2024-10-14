# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module creates cutouts from data cubes found in the cloud."""

import os
import re
from typing import List, Union
import warnings
from threading import Lock

from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from cachetools import TTLCache, cached
import numpy as np
import requests
from spherical_geometry.polygon import SphericalPolygon

from astrocut.exceptions import InvalidQueryError
from astrocut.cube_cut import CutoutFactory

from . import log
from .utils.utils import parse_size_input, _handle_verbose

TESS_ARCSEC_PER_PX = 21  # Number of arcseconds per pixel in a TESS image
FFI_TTLCACHE = TTLCache(maxsize=10, ttl=900)  # Cache for FFI footprint files
CUBE_CUT_THREADS = 8  # Number of threads to use in `cube_cut` function


def _s_region_to_polygon(s_region: Column):
    """
    Takes in a s_region string of type POLYGON and returns it as a spherical_region Polygon.

    Example input:
    'POLYGON 229.80771900 -75.17048500 241.67788000 -63.95992300 269.94872000 -64.39276400 277.87862300 -75.57754400'
    """

    def ind_sregion_to_polygon(s_reg):
        sr_list = s_reg.strip().split()
        reg_type = sr_list[0].upper()

        if reg_type == 'POLYGON':
            ras = np.array(sr_list[1::2], dtype=float)
            ras[ras < 0] = ras[ras < 0] + 360
            decs = np.array(sr_list[2::2], dtype=float)
            return SphericalPolygon.from_radec(ras, decs)
        else:
            raise ValueError('Unsupported S_Region type.')

    return np.vectorize(ind_sregion_to_polygon)(s_region)


@cached(cache=FFI_TTLCACHE, lock=Lock())
def get_caom_ffis(product: str = 'SPOC'):
    """
    Fetches footprints for Full Frame Images (FFIs) from the Common Archive Observation Model. The resulting
    table contains each (FFI) and a 'polygon' column that describes the image's footprints as polygon points
    and vectors.

    Parameters
    ----------
    product : str, optional
        Default 'SPOC'. The product type for which to fetch footprints.

    Returns
    -------
    all_ffis : `~astropy.table.Table`
        Table containing information about FFIs and their footprints.
    """
    # Define the URL and parameters for the query
    obs_collection = 'TESS' if product == 'SPOC' else 'HLSP'
    target_name = 'TESS FFI' if product == 'SPOC' else 'TICA FFI'
    url = 'https://mast.stsci.edu/vo-tap/api/v0.1/caom/sync'
    params = {
        'FORMAT': 'csv',
        'LANG': 'ADQL',
        'QUERY': 'SELECT obs_id, t_min, t_max, s_region, target_name, sequence_number '
                'FROM dbo.ObsPointing '
                f"WHERE obs_collection='{obs_collection}' AND dataproduct_type='image' AND target_name='{target_name}'"
    }

    # Send the GET request
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise HTTPError if response is not 200

    # Load CSV data into a Table
    all_ffis = Table.read(response.text, format='csv')
    all_ffis.sort('obs_id')

    # Convert regions to polygons
    all_ffis['polygon'] = _s_region_to_polygon(all_ffis['s_region'])

    return all_ffis


def _ffi_intersect(ffi_list: Table, polygon: SphericalPolygon):
    """
    Vectorizing the spherical_coordinate intersects_polygon function
    """
    def single_intersect(ffi, polygon):
        return ffi.intersects_poly(polygon)

    return np.vectorize(single_intersect)(ffi_list['polygon'], polygon)


def ra_dec_crossmatch(all_ffis: Table, coordinates: SkyCoord, cutout_size, arcsec_per_px: int = TESS_ARCSEC_PER_PX):
    """
    Returns the Full Frame Images (FFIs) whose footprints overlap with a cutout of a given position and size.

    Parameters
    ----------
    all_ffis : `~astropy.table.Table`
        Table of FFIs to crossmatch with the cutout.
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
    arcsec_per_px : int, optional
        Default 21. The number of arcseconds per pixel in an image. Used to determine
        the footprint of the cutout. Default is the number of arcseconds per pixel in
        a TESS image.

    Returns
    -------
    matching_ffis : `~astropy.table.Table`
        Table containing information about FFIs whose footprints overlap those of the cutout.
    """
    # Convert coordinates to SkyCoord
    if not isinstance(coordinates, SkyCoord):
        coordinates = SkyCoord(coordinates, unit='deg')

    # Parse cutout size
    cutout_size = parse_size_input(cutout_size)

    ra, dec = coordinates.ra, coordinates.dec
    ffi_inds = []

    # Create polygon for intersection
    # Convert dimensions from pixels to arcseconds and divide by 2 to get offset from center
    ra_offset = ((cutout_size[0] * arcsec_per_px) / 2) * u.arcsec
    dec_offset = ((cutout_size[1] * arcsec_per_px) / 2) * u.arcsec

    # Calculate RA and Dec boundaries
    ra_bounds = [ra - ra_offset, ra + ra_offset]
    dec_bounds = [dec - dec_offset, dec + dec_offset]

    # Get RA and Dec for four corners of rectangle
    ras = [ra_bounds[0].value, ra_bounds[1].value, ra_bounds[1].value, ra_bounds[0].value]
    decs = [dec_bounds[0].value, dec_bounds[0].value, dec_bounds[1].value, dec_bounds[1].value]

    # Create SphericalPolygon for comparison
    cutout_fp = SphericalPolygon.from_radec(ras, decs, center=(ra, dec))
    ffi_inds = _ffi_intersect(all_ffis, cutout_fp)

    return all_ffis[ffi_inds]


def _extract_sequence_information(sector_name: str, product: str):
    """Extract the sector, camera, and ccd information from the sector name"""
    if product == 'SPOC':
        pattern = re.compile(r"(tess-s)(?P<sector>\d{4})-(?P<camera>\d{1,4})-(?P<ccd>\d{1,4})")
    elif product == 'TICA':
        pattern = re.compile(r"(hlsp_tica_s)(?P<sector>\d{4})-(cam)(?P<camera>\d{1,4})-(ccd)(?P<ccd>\d{1,4})")
    else:
        return {}
    sector_match = re.match(pattern, sector_name)

    if not sector_match:
        return {}

    sector = sector_match.group("sector")
    camera = sector_match.group("camera")
    ccd = sector_match.group("ccd")

    # Rename the TICA sector because of the naming convention in Astrocut
    if product == 'TICA':
        sector_name = f"tica-s{sector}-{camera}-{ccd}"

    return {"sectorName": sector_name, "sector": sector, "camera": camera, "ccd": ccd}


def _create_sequence_list(observations: Table, product: str):
    """Extracts sequence information from a list of observations"""
    target_name = "TESS FFI" if product == 'SPOC' else "TICA FFI"
    obs_filtered = [obs for obs in observations if obs["target_name"].upper() == target_name]

    sequence_results = []
    for row in obs_filtered:
        sequence_extraction = _extract_sequence_information(row["obs_id"], product=product)
        if sequence_extraction:
            sequence_results.append(sequence_extraction)

    return sequence_results


def _get_cube_files_from_sequence_obs(sequences: list):
    """Convert obs_id sequence information into cube file names"""
    cube_files = [
        {
            "folder": "s" + sector["sector"].rjust(4, "0"),
            "cube": sector["sectorName"] + "-cube.fits",
            "sectorName": sector["sectorName"],
        }
        for sector in sequences
    ]
    return cube_files


def cube_cut_from_footprint(coordinates: Union[str, SkyCoord], cutout_size, 
                            sequence: Union[int, List[int], None] = None, product: str = 'SPOC', 
                            output_dir: str = '.', verbose: bool = False):
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
    output_dir : str, optional
        Default '.'. The path to which output files are saved.
        The current directory is default.
    verbose : bool, optional
        Default False. If True, intermediate information is printed.

    Returns
    -------
    cutout_files : list
        List of paths to cutout files.

    Examples
    --------
    >>> from astrocut.footprint_cutouts import cube_cut_from_footprint
    >>> cube_cut_from_footprint(  #doctest: +SKIP
    ...         coordinates='83.40630967798376 -62.48977125108528',
    ...         cutout_size=64,
    ...         sequence=[1, 2],  # TESS sectors
    ...         product='SPOC',
    ...         output_dir='./cutouts')
    ['./cutouts/tess-s0001-4-4/tess-s0001-4-4_83.406310_-62.489771_64x64_astrocut.fits',
     './cutouts/tess-s0002-4-1/tess-s0002-4-1_83.406310_-62.489771_64x64_astrocut.fits']
    """
    # Log messages based on verbosity
    _handle_verbose(verbose)
    
    # Convert to SkyCoord
    if not isinstance(coordinates, SkyCoord):
        coordinates = SkyCoord(coordinates, unit='deg')
    log.debug(f'Coordinates: {coordinates}')

    # Parse cutout size
    cutout_size = parse_size_input(cutout_size)
    log.debug(f'Cutout size: {cutout_size}')

    # Get FFI footprints from the cloud
    # s3_uri = 's3://tesscut-ops-footprints/tess_ffi_footprint_cache.json' if product == 'SPOC' \
    #     else 's3://tesscut-ops-footprints/tica_ffi_footprint_cache.json'
    # all_ffis = _get_s3_ffis(s3_uri=s3_uri, as_table=True, load_polys=True)
    all_ffis = get_caom_ffis(product)
    log.debug(f'Found {len(all_ffis)} footprint files.')

    # Filter FFIs by provided sectors
    if sequence:
        # Convert to list
        if isinstance(sequence, int):
            sequence = [sequence]
        all_ffis = all_ffis[np.isin(all_ffis['sequence_number'], sequence)]

        if len(all_ffis) == 0:
            raise InvalidQueryError('No FFI cube files were found for sequences: ' +
                                    ', '.join(str(s) for s in sequence))
        
        
        log.debug(f'Filtered to {len(all_ffis)} footprints for sequences: {", ".join(str(s) for s in sequence)}')

    # Get sector names and cube files that contain the cutout
    cone_results = ra_dec_crossmatch(all_ffis, coordinates, cutout_size, TESS_ARCSEC_PER_PX)
    if not cone_results:
        raise InvalidQueryError('The given coordinates were not found within the specified sequence(s).')
    seq_list = _create_sequence_list(cone_results, product)
    cube_files_mapping = _get_cube_files_from_sequence_obs(seq_list)
    log.debug(f'Found {len(cube_files_mapping)} matching cube files.')
    base_file_path = "s3://stpubdata/tess/public/mast/" if product == 'SPOC' \
        else "s3://stpubdata/tess/public/mast/tica/"

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Executor function to generate cutouts from a cube file
    def process_file(file):
        try:
            factory = CutoutFactory()
            file_path = os.path.join(base_file_path, file['cube'])
            output_path = os.path.join(output_dir, file['sectorName'])
            cutout = factory.cube_cut(
                file_path,
                coordinates,
                cutout_size=cutout_size,
                product=product,
                output_path=output_path,
                threads=CUBE_CUT_THREADS,
                verbose=verbose
            )
            return cutout
        except Exception as e:
            warnings.warn(f'Unable to generate cutout from {file_path}: {e}', AstropyWarning)
            return None
        
    # Generate cutout from each cube file
    log.debug('Generating cutouts...')
    cutout_files = [process_file(file) for file in cube_files_mapping]

    return cutout_files
