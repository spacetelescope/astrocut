# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module creates cutouts from data cubes found in the cloud."""

from typing import List, Union

from astropy.coordinates import SkyCoord
from astropy.io.fits import HDUList
from astropy.table import Table

from .TessFootprintCutout import TessFootprintCutout


def ra_dec_crossmatch(all_ffis: Table, coordinates: SkyCoord, cutout_size, arcsec_per_px: int = 21) -> Table:
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
    return TessFootprintCutout.ra_dec_crossmatch(all_ffis, coordinates, cutout_size, arcsec_per_px)


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

    cutouts = TessFootprintCutout(coordinates, cutout_size, sequence=sequence, product=product, verbose=verbose)

    if memory_only:
        return cutouts.tpf_cutouts
    
    # Write cutouts
    return cutouts.write_as_tpf(output_dir)
