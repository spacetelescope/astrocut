# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module creates cutouts from data cubes found in the cloud."""

import json
import re
import os
from typing import List, Literal, Union
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
import numpy as np
import fsspec
from spherical_geometry.polygon import SphericalPolygon
from spherical_geometry.vector import radec_to_vector

from astrocut.exceptions import InvalidQueryError
from astrocut.cube_cut import CutoutFactory

from .utils.utils import parse_size_input

TESS_ARCSEC_PER_PX = 21


def _s_region_to_polygon(s_region: Column):
    """
    Takes in a s_region string of type POLYGON or CIRCLE and returns it as
    a spherical_region Polygon.

    Example inputs:
    'POLYGON 229.80771900 -75.17048500 241.67788000 -63.95992300 269.94872000 -64.39276400 277.87862300 -75.57754400'
    'CIRCLE ICRS 244.38324081 -75.86611807 0.625'
    """

    def ind_sregion_to_polygon(s_reg):
        sr_list = s_reg.strip().split()
        reg_type = sr_list[0].upper()

        if reg_type == "POLYGON":
            ras = np.array(sr_list[1::2], dtype=float)
            ras[ras < 0] = ras[ras < 0] + 360
            decs = np.array(sr_list[2::2], dtype=float)
            return SphericalPolygon.from_radec(ras, decs)
        elif reg_type == "CIRCLE":
            ra, dec, rad = np.array(sr_list[-3:], dtype=float)
            return SphericalPolygon.from_cone(ra, dec, rad)
        else:
            raise ValueError("Unsupported S_Region type.")

    return np.vectorize(ind_sregion_to_polygon)(s_region)


def _get_s3_ffis(s3_uri, as_table: bool = False, load_polys: bool = False):
    """
    Fetch the S3 footprint file containing a dict of all FFIs and a polygon column
    that holds the s_regions as polygon points and vectors.

    Optional Parameters:
        as_table: Return the footprint file as an Astropy Table
        load_polys: Convert the s_region column to an array of SphericalPolygon objects
    """
    # Open footprint file with fsspec
    # Use caching to help performance, but check that remote UID matches the local
    # Expiry time is 1 week by default
    with fsspec.open('filecache::' + s3_uri, s3={'anon': True}, 
                     filecache={'cache_storage': 's3_cache', 'check_files': True}) as f:
        ffis = json.loads(f.read())

    if load_polys:
        ffis['polygon'] = _s_region_to_polygon(ffis['s_region'])

    if as_table:
        ffis = Table(ffis)

    return ffis


def _ffi_intersect(ffi_list: Table, polygon: SphericalPolygon):
    """
    Vectorizing the spherical_coordinate intersects_polygon function
    """
    def single_intersect(ffi, polygon):
        return ffi.intersects_poly(polygon)

    return np.vectorize(single_intersect)(ffi_list["polygon"], polygon)


def _ra_dec_crossmatch(all_ffis: Table, coord: SkyCoord, cutout_size, arcsec_per_px: int):
    """Determine which sector/camera/ccd(s) contain the given ra/dec."""    
    ra, dec = coord.ra, coord.dec

    # Performing the crossmatch
    ffi_inds = []
    if (cutout_size == 0).all():
        vector_coord = radec_to_vector(ra, dec)
        for sequence in np.unique(all_ffis["sequence_number"]):
            # np returns a 2-long array where the first element is indexes and the 2nd element is empty
            seq_ffi_inds = np.where(all_ffis["sequence_number"] == sequence)[0]

            for ind in seq_ffi_inds:
                if all_ffis[ind]["polygon"].contains_point(vector_coord):
                    ffi_inds.append(ind)
                    break  # the ra/dec will only be on one ccd per sector
    else:
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
                            sequences: Union[int, List[int], None] = None, product: str = 'SPOC', 
                            output_dir: str = '.', threads: Union[int, Literal['auto']] = 1,
                            verbose: bool = False):
    """
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
    sequences : int, List[int], optional
        Default None. Sequence(s) from which to generate cutouts. Can provide a single
        sequence number as an int or a list of sequence numbers. If not specified, 
        cutouts will be generated from all sequences that contain the cutout.
    product : str, optional
        Default 'SPOC'. The product type to make the cutouts from.
    output_dir : str, optional
        Default '.'. The path to which output files are saved.
        The current directory is default.
    threads : int, 'auto', optional
        Default 1. Number of threads to use when generating cutouts. Setting <=1 disables the threadpool, 
        >1 sets threadpool to the specified number of threads, and "auto" uses `concurrent.futures.ThreadPoolExecutor`'s
        default: cpu_count + 4, limit to max of 32
    verbose : bool, optional
        Default False. If True, intermediate information is printed.

    Returns
    -------
    cutout_files : list
        List of paths to cutout files.
    """
    
    # Convert to SkyCoord
    if not isinstance(coordinates, SkyCoord):
        coordinates = SkyCoord(coordinates, unit='deg')
    if verbose:
        print('Coordinates: ', coordinates)

    # Parse cutout size
    cutout_size = parse_size_input(cutout_size)
    if verbose:
        print('Cutout size: ', cutout_size)

    # Get FFI footprints from the cloud
    s3_uri = 's3://tesscut-ops-footprints/tess_ffi_footprint_cache.json' if product == 'SPOC' \
        else 's3://tesscut-ops-footprints/tica_ffi_footprint_cache.json'
    all_ffis = _get_s3_ffis(s3_uri=s3_uri, as_table=True, load_polys=True)
    if verbose:
        print(f'Found {len(all_ffis)} footprint files.')

    # Filter FFIs by provided sectors
    if sequences and product:
        # Convert to list
        if isinstance(sequences, int):
            sequences = [sequences]
        all_ffis = all_ffis[np.isin(all_ffis['sequence_number'], sequences)]

        if len(all_ffis) == 0:
            raise InvalidQueryError(f'No FFI cube files were found for sequences: \
                                    {", ".join(str(s) for s in sequences)}')
        
        if verbose:
            print(f'Filtered to {len(all_ffis)} footprints for sequences: {", ".join(str(s) for s in sequences)}')

    # Get sector names and cube files that contain the cutout
    cone_results = _ra_dec_crossmatch(all_ffis, coordinates, cutout_size, TESS_ARCSEC_PER_PX)
    if not cone_results:
        raise InvalidQueryError('The given coordinates were not found within the specified sector.')
    seq_list = _create_sequence_list(cone_results, product)
    cube_files_mapping = _get_cube_files_from_sequence_obs(seq_list)
    if verbose:
        print(f'Found {len(cube_files_mapping)} matching cube files.')
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
                threads='auto',
                verbose=verbose
            )
            return cutout
        except Exception as e:
            warnings.warn(f'Unable to generate cutout from {file_path}: {e}', AstropyWarning)
            return None
        
    # Generate cutout from each cube file
    cutout_files = []
    if threads == 'auto' or threads > 1:
        # Parallel processing
        if verbose:
            print(f'Generating cutouts in parallel with max threads: {threads}')

        # Use ThreadPoolExecutor for parallelization
        max_workers = None if threads == "auto" else threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file, file) for file in cube_files_mapping]

            # Collect results as they are completed
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    cutout_files.append(result)
    else:
        # Sequential Processing
        if verbose:
            print('Generating cutouts in sequence.')
        cutout_files = [process_file(file) for file in cube_files_mapping]

    return cutout_files
