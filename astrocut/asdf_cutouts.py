# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements cutout functionality similar to fitscut, but for the ASDF file format."""
import copy
import pathlib
from typing import Union, Tuple
import requests

import asdf
import astropy
import gwcs
import numpy as np
import s3fs
from s3path import S3Path

from astropy.coordinates import SkyCoord
from astropy.modeling import models


def _get_cloud_http(s3_uri: Union[str, S3Path], verbose: bool = False) -> str:
    """ 
    Get the HTTP URI of a cloud resource from an S3 URI.

    Parameters
    ----------
    s3_uri : string | S3Path
        the S3 URI of the cloud resource
    verbose : bool
        Default False. If true intermediate information is printed.
    """

    # check if public or private by sending an HTTP request
    s3_path = S3Path.from_uri(s3_uri) if isinstance(s3_uri, str) else s3_uri
    url = f'https://{s3_path.bucket}.s3.amazonaws.com/{s3_path.key}'
    resp = requests.head(url, timeout=10)
    is_anon = False if resp.status_code == 403 else True
    if verbose and not is_anon:
        print(f'Attempting to access private S3 bucket: {s3_path.bucket}')

    # create file system and get URL of file
    fs = s3fs.S3FileSystem(anon=is_anon)
    with fs.open(s3_uri, 'rb') as f:
        return f.url()


def get_center_pixel(gwcsobj: gwcs.wcs.WCS, ra: float, dec: float) -> tuple:
    """ 
    Get the center pixel from a Roman 2D science image.

    For an input RA, Dec sky coordinate, get the closest pixel location
    on the input Roman image.

    Parameters
    ----------
    gwcsobj : gwcs.wcs.WCS
        The Roman GWCS object.
    ra : float
        The input right ascension.
    dec : float
        The input declination.

    Returns
    -------
    tuple
        The pixel position, FITS wcs object
    """

    # Convert the gwcs object to an astropy FITS WCS header
    header = gwcsobj.to_fits_sip()

    # Update WCS header with some keywords that it's missing.
    # Otherwise, it won't work with astropy.wcs tools (TODO: Figure out why. What are these keywords for?)
    for k in ['cpdis1', 'cpdis2', 'det2im1', 'det2im2', 'sip']:
        if k not in header:
            header[k] = 'na'

    # New WCS object with updated header
    wcs_updated = astropy.wcs.WCS(header)

    # Turn input RA, Dec into a SkyCoord object
    coordinates = SkyCoord(ra, dec, unit='deg')

    # Map the coordinates to a pixel's location on the Roman 2d array (row, col)
    row, col = gwcsobj.invert(coordinates)

    return (row, col), wcs_updated


def _get_cutout(data: asdf.tags.core.ndarray.NDArrayType, coords: Union[tuple, SkyCoord],
                wcs: astropy.wcs.wcs.WCS = None, size: int = 20, outfile: str = "example_roman_cutout.fits",
                write_file: bool = True, fill_value: Union[int, float] = np.nan,
                gwcsobj: gwcs.wcs.WCS = None) -> astropy.nddata.Cutout2D:
    """ 
    Get a Roman image cutout.

    Cut out a square section from the input image data array.  The ``coords`` can either be a tuple of x, y
    pixel coordinates or an astropy SkyCoord object, in which case, a wcs is required.  Writes out a
    new output file containing the image cutout of the specified ``size``.  Default is 20 pixels.

    Parameters
    ----------
    data : asdf.tags.core.ndarray.NDArrayType
        the input Roman image data array
    coords : Union[tuple, SkyCoord]
        the input pixel or sky coordinates
    wcs : astropy.wcs.wcs.WCS, Optional
        the astropy FITS wcs object
    size : int, optional
        the image cutout pizel size, by default 20
    outfile : str, optional
        the name of the output cutout file, by default "example_roman_cutout.fits"
    write_file : bool, by default True
        Flag to write the cutout to a file or not
    fill_value: int | float, by default np.nan
        The fill value for pixels outside the original image.
    gwcsobj : gwcs.wcs.WCS, Optional
        the original gwcs object for the full image, needed only when writing cutout as asdf file

    Returns
    -------
    astropy.nddata.Cutout2D:
        an image cutout object

    Raises
    ------
    ValueError:
        when a wcs is not present when coords is a SkyCoord object
    RuntimeError:
        when the requested cutout does not overlap with the original image
    ValueError:
        when no gwcs object is provided when writing to an asdf file
    """

    # check for correct inputs
    if isinstance(coords, SkyCoord) and not wcs:
        raise ValueError('wcs must be input if coords is a SkyCoord.')

    # create the cutout
    try:
        cutout = astropy.nddata.Cutout2D(data, position=coords, wcs=wcs, size=(size, size), mode='partial',
                                         fill_value=fill_value)
    except astropy.nddata.utils.NoOverlapError as e:
        raise RuntimeError('Could not create 2d cutout.  The requested cutout does not overlap with the '
                           'original image.') from e

    # check if the data is a quantity and get the array data
    if isinstance(cutout.data, astropy.units.Quantity):
        data = cutout.data.value
    else:
        data = cutout.data

    # write the cutout to the output file
    if write_file:
        # check the output file type
        out = pathlib.Path(outfile)
        write_as = out.suffix or '.fits'
        outfile = outfile if out.suffix else str(out) + write_as

        # write out the file
        if write_as == '.fits':
            _write_fits(cutout, outfile)
        elif write_as == '.asdf':
            if not gwcsobj:
                raise ValueError('The original gwcs object is needed when writing to asdf file.')
            _write_asdf(cutout, gwcsobj, outfile)

    return cutout


def _write_fits(cutout: astropy.nddata.Cutout2D, outfile: str = "example_roman_cutout.fits"):
    """ 
    Write cutout as FITS file.

    Parameters
    ----------
    cutout : astropy.nddata.Cutout2D
        the 2d cutout
    outfile : str, optional
        the name of the output cutout file, by default "example_roman_cutout.fits"
    """
    # check if the data is a quantity and get the array data
    if isinstance(cutout.data, astropy.units.Quantity):
        data = cutout.data.value
    else:
        data = cutout.data

    astropy.io.fits.writeto(outfile, data=data, header=cutout.wcs.to_header(relax=True), overwrite=True)


def _slice_gwcs(gwcsobj: gwcs.wcs.WCS, slices: Tuple[slice, slice]) -> gwcs.wcs.WCS:
    """ 
    Slice the original gwcs object.

    "Slices" the original gwcs object down to the cutout shape.  This is a hack
    until proper gwcs slicing is in place a la fits WCS slicing.  The ``slices``
    keyword input is a tuple with the x, y cutout boundaries in the original image
    array, e.g. ``cutout.slices_original``.  Astropy Cutout2D slices are in the form
    ((ymin, ymax, None), (xmin, xmax, None))

    Parameters
    ----------
    gwcsobj : gwcs.wcs.WCS
        the original gwcs from the input image
    slices : Tuple[slice, slice]
        the cutout x, y slices as ((ymin, ymax), (xmin, xmax))

    Returns
    -------
    gwcs.wcs.WCS
        The sliced gwcs object
    """
    tmp = copy.deepcopy(gwcsobj)

    # get the cutout array bounds and create a new shift transform to the cutout
    # add the new transform to the gwcs
    xmin, xmax = slices[1].start, slices[1].stop
    ymin, ymax = slices[0].start, slices[0].stop
    shape = (ymax - ymin, xmax - xmin)
    offsets = models.Shift(xmin, name='cutout_offset1') & models.Shift(ymin, name='cutout_offset2')
    tmp.insert_transform('detector', offsets, after=True)

    # modify the gwcs bounding box to the cutout shape
    tmp.bounding_box = ((0, shape[0] - 1), (0, shape[1] - 1))
    tmp.pixel_shape = shape[::-1]
    tmp.array_shape = shape
    return tmp


def _write_asdf(cutout: astropy.nddata.Cutout2D, gwcsobj: gwcs.wcs.WCS, outfile: str = "example_roman_cutout.asdf"):
    """ 
    Write cutout as ASDF file.

    Parameters
    ----------
    cutout : astropy.nddata.Cutout2D
        the 2d cutout
    gwcsobj : gwcs.wcs.WCS
        the original gwcs object for the full image
    outfile : str, optional
        the name of the output cutout file, by default "example_roman_cutout.asdf"
    """
    # slice the origial gwcs to the cutout
    sliced_gwcs = _slice_gwcs(gwcsobj, cutout.slices_original)

    # create the asdf tree
    tree = {'roman': {'meta': {'wcs': sliced_gwcs}, 'data': cutout.data}}
    af = asdf.AsdfFile(tree)

    # Write the data to a new file
    af.write_to(outfile)


def asdf_cut(input_file: Union[str, pathlib.Path, S3Path], ra: float, dec: float, cutout_size: int = 20,
             output_file: Union[str, pathlib.Path] = "example_roman_cutout.fits",
             write_file: bool = True, fill_value: Union[int, float] = np.nan,
             verbose: bool = False) -> astropy.nddata.Cutout2D:
    """ 
    Takes a single ASDF input file (`input_file`) and generates a cutout of designated size `cutout_size`
    around the given coordinates (`coordinates`).

    Preliminary proof-of-concept functionality.

    Parameters
    ----------
    input_file : str | Path | S3Path
        The input ASDF file.
    ra : float
        The right ascension of the central cutout.
    dec : float
        The declination of the central cutout.
    cutout_size : int
        Optional, default 20. The image cutout pixel size.
    output_file : str | Path
        Optional, default "example_roman_cutout.fits". The name of the output cutout file.
    write_file : bool
        Optional, default True. Flag to write the cutout to a file or not.
    fill_value: int | float
        Optional, default `np.nan`. The fill value for pixels outside the original image.
    verbose : bool
        Default False. If true intermediate information is printed.

    Returns
    -------
    astropy.nddata.Cutout2D:
        An image cutout object.
    """

    # if file comes from AWS cloud bucket, get HTTP URL to open with asdf
    file = input_file
    if (isinstance(input_file, str) and input_file.startswith('s3://')) or isinstance(input_file, S3Path):
        file = _get_cloud_http(input_file, verbose)

    # get the 2d image data
    with asdf.open(file) as f:
        data = f['roman']['data']
        gwcsobj = f['roman']['meta']['wcs']

        # get the center pixel
        pixel_coordinates, wcs = get_center_pixel(gwcsobj, ra, dec)

        # create the 2d image cutout
        return _get_cutout(data, pixel_coordinates, wcs, size=cutout_size, outfile=output_file,
                           write_file=write_file, fill_value=fill_value, gwcsobj=gwcsobj)
