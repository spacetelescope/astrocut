import json
from abc import ABC, abstractmethod
from threading import Lock
from typing import List, Tuple, Union

import astropy.units as u
import fsspec
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from cachetools import TTLCache, cached
from spherical_geometry.polygon import SphericalPolygon
from spherical_geometry.vector import radec_to_vector

from .cutout import Cutout

FFI_TTLCACHE = TTLCache(maxsize=10, ttl=900)  # Cache for FFI footprint files


class FootprintCutout(Cutout, ABC):
    """
    Abstract class that creates cutouts from data files hosted on the S3 cloud.

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
    verbose : bool
        If True, log messages are printed to the console.

    Methods
    -------
    cutout()
        Fetch the cloud files that contain the cutout and generate the cutouts.
    """

    def __init__(self, coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, u.Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, limit_rounding_method: str = 'round', 
                 sequence: Union[int, List[int], None] = None, verbose: bool = False):
        super().__init__([], coordinates, cutout_size, fill_value, limit_rounding_method, verbose)

        # Assigning other attributes
        if isinstance(sequence, int):
            sequence = [sequence]  # Convert to list
        self._sequence = sequence

    @abstractmethod
    def cutout(self):
        """
        Generate cutouts from the cloud files that contain the cutout's footprint.

        This method is abstract and should be implemented in subclasses.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    @staticmethod
    def _s_region_to_polygon(s_region: Column) -> Column:
        """
        Returns a column of `~spherical_geometry.polygon.SphericalPolygon` objects from a column of
        s_region strings.

        Parameters
        ----------
        s_region : `~astropy.table.Column`
            Column containing the s_region string. Example input: 'POLYGON 229.80771900 -75.17048500 
            241.67788000 -63.95992300 269.94872000 -64.39276400 277.87862300 -75.57754400'

        Returns
        -------
        polygon : `~astropy.table.Column`
            Column containing `~spherical_geometry.polygon.SphericalPolygon` objects representing each s_region.
        """
        def ind_sregion_to_polygon(s_reg):
            """
            Helper function to convert s_region string to a `~spherical_geometry.polygon.SphericalPolygon` object.

            Parameters
            ----------
            s_reg : str
                A string defining a spatial region, expected to be in the 'POLYGON' format.

            Returns
            -------
            `~spherical_geometry.polygon.SphericalPolygon`
                A SphericalPolygon object created from the provided coordinates.
            
            Raises
            ------
            ValueError
                If the S_REGION type is not 'POLYGON'.
            
            """
            # Split input string into individual components
            sr_list = s_reg.strip().split()

            # Extract the region type (first element of list)
            reg_type = sr_list[0].upper()

            if reg_type == 'POLYGON':
                # Extract RA and Dec values
                # RAs are at odd indices
                ras = np.array(sr_list[1::2], dtype=float)

                # Convert negative RAs to the 0-360 range
                ras[ras < 0] = ras[ras < 0] + 360

                # Decs are at even indices
                decs = np.array(sr_list[2::2], dtype=float)

                # Create SphericalPolygon object
                return SphericalPolygon.from_radec(ras, decs)
            else:
                raise ValueError(f'Unsupported s_region type: {reg_type}.')

        return np.vectorize(ind_sregion_to_polygon)(s_region)
    
    @staticmethod
    def _ffi_intersect(ffi_list: Table, polygon: SphericalPolygon) -> np.ndarray:
        """
        Vectorizing the spherical_coordinate intersects_polygon function.

        Parameters
        ----------
        ffi_list : `~astropy.table.Table`
            Table containing information about FFIs and their footprints.
        polygon : `~spherical_geometry.polygon.SphericalPolygon`
            SphericalPolygon object representing the cutout's footprint.

        Returns
        -------
        intersect : `~numpy.ndarray`
            Boolean array indicating whether each FFI intersects with the cutout.
        """
        def single_intersect(ffi, polygon):
            return ffi.intersects_poly(polygon)

        return np.vectorize(single_intersect)(ffi_list['polygon'], polygon)
    

@cached(cache=FFI_TTLCACHE, lock=Lock())
def get_ffis(s3_footprint_cache: str) -> Table:
    """
    Fetches footprints for Full Frame Images (FFIs) from S3. The resulting
    table contains each (FFI) and a 'polygon' column that describes the image's footprints as polygon points
    and vectors.

    This method is outside the class definition to allow for caching.

    Parameters
    ----------
    s3_footprint_cache : str
        S3 URI to the footprint cache file.

    Returns
    -------
    ffis : `~astropy.table.Table`
        Table containing information about FFIs and their footprints.
    """
    # Open footprint file with fsspec
    with fsspec.open(s3_footprint_cache, s3={'anon': True}) as f:
        ffis = json.load(f)

    # Compute spherical polygons
    ffis['polygon'] = FootprintCutout._s_region_to_polygon(ffis['s_region'])

    # Convert to Astropy table
    ffis = Table(ffis)

    return ffis


def _crossmatch_point(ra: SkyCoord, dec: SkyCoord, all_ffis: Table) -> np.ndarray:
    """
    Returns the indices of the Full Frame Images (FFIs) that contain the given RA and
    Dec coordinates by checking which FFI polygons contain the point.
    
    Parameters
    ----------
    ra : SkyCoord
        Right Ascension in degrees.
    dec : SkyCoord
        Declination in degrees.
    all_ffis : `~astropy.table.Table`
        Table of FFIs to crossmatch with the point.
        
    Returns
    -------
    ffi_inds : `~numpy.ndarray`
        Indices of FFIs that contain the given RA and Dec coordinates.
    """
    ffi_inds = []
    vector_coord = radec_to_vector(ra, dec)
    for sector in np.unique(all_ffis['sequence_number']):
        # Returns a 2-long array where the first element is indexes and the 2nd element is empty
        sector_ffi_inds = np.where(all_ffis['sequence_number'] == sector)[0]

        for ind in sector_ffi_inds:
            if all_ffis[ind]["polygon"].contains_point(vector_coord):
                ffi_inds.append(ind)
                break  # the ra/dec will only be on one ccd per sector
    return np.array(ffi_inds, dtype=int)


def _crossmatch_polygon(ra: SkyCoord, dec: SkyCoord, all_ffis: Table, px_size: np.ndarray,
                        arcsec_per_px: int = 21) -> np.ndarray:
    """
    Returns the indices of the Full Frame Images (FFIs) that intersect with the given cutout footprint
    by checking which FFI polygons intersect with the cutout polygon.

    Parameters
    ----------
    ra : SkyCoord
        Right Ascension in degrees.
    dec : SkyCoord
        Declination in degrees.
    all_ffis : `~astropy.table.Table`
        Table of FFIs to crossmatch with the point.
    px_size : array-like
        Size of the cutout in pixels, in the form [ny, nx].
    arcsec_per_px : int, optional
        Default 21. The number of arcseconds per pixel in an image. Used to determine
        the footprint of the cutout. Default is the number of arcseconds per pixel in
        a TESS image.

    Returns
    -------
    ffi_inds : `~numpy.ndarray`
        Boolean array indicating whether each FFI intersects with the cutout.
    """
    # Create polygon for intersection
    # Convert dimensions from pixels to arcseconds and divide by 2 to get offset from center
    # If one of the dimensions is 0, use a very small value to avoid issues with SphericalPolygon
    min_offset = 0.1  # pixels
    ra_offset = ((max(px_size[0], min_offset) * arcsec_per_px) / 2) * u.arcsec
    dec_offset = ((max(px_size[1], min_offset) * arcsec_per_px) / 2) * u.arcsec

    # Calculate RA and Dec boundaries
    ra_bounds = [ra - ra_offset, ra + ra_offset]
    dec_bounds = [dec - dec_offset, dec + dec_offset]

    # Get RA and Dec for four corners of rectangle
    ras = [ra_bounds[0].value, ra_bounds[1].value, ra_bounds[1].value, ra_bounds[0].value]
    decs = [dec_bounds[0].value, dec_bounds[0].value, dec_bounds[1].value, dec_bounds[1].value]

    # Create SphericalPolygon for comparison
    cutout_fp = SphericalPolygon.from_radec(ras, decs, center=(ra, dec))

    # Find indices of FFIs that intersect with the cutout
    ffi_inds = np.vectorize(lambda ffi: ffi.intersects_poly(cutout_fp))(all_ffis['polygon'])
    ffi_inds = FootprintCutout._ffi_intersect(all_ffis, cutout_fp)

    return ffi_inds


def ra_dec_crossmatch(all_ffis: Table, coordinates: Union[SkyCoord, str], cutout_size, 
                      arcsec_per_px: int = 21) -> Table:
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
        then a square cutout of ``cutout_size`` will be used.  If
        ``cutout_size`` has two elements, they should be in ``(ny, nx)``
        order.  Scalar numbers in ``cutout_size`` are assumed to be in
        units of pixels. `~astropy.units.Quantity` objects must be in pixel or
        angular units.

        If a cutout size of zero is provided, the function will return FFIs that contain 
        the exact RA and Dec position. If a non-zero cutout size is provided, the function 
        will return FFIs whose footprints overlap with the cutout area.
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
    ra, dec = coordinates.ra, coordinates.dec

    px_size = np.zeros(2, dtype=object)
    for axis, size in enumerate(Cutout.parse_size_input(cutout_size, allow_zero=True)):
        if isinstance(size, u.Quantity):  # If Quantity, convert to pixels
            if size.unit == u.pixel:
                px_size[axis] = size.value
            else:  # Angular size
                # Convert angular size to pixels
                px_size[axis] = (size.to_value(u.arcsec)) / arcsec_per_px
        else:  # Assume pixels
            px_size[axis] = size

    if np.all(px_size == 0):
        # Cross match with point
        ffi_inds = _crossmatch_point(ra, dec, all_ffis)
    else:
        # Cross match with polygon
        ffi_inds = _crossmatch_polygon(ra, dec, all_ffis, px_size, arcsec_per_px)

    return all_ffis[ffi_inds]
