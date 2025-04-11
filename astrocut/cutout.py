from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Union, Tuple
import warnings

from astropy import wcs
import astropy.units as u
from s3path import S3Path
from astropy.coordinates import SkyCoord
import numpy as np

from astrocut.exceptions import InputWarning, InvalidInputError, InvalidQueryError

from . import log
from .utils.utils import _handle_verbose


class Cutout(ABC):
    """
    Abstract class for creating cutouts. This class defines attributes and methods that are common to all
    cutout classes.

    Parameters
    ----------
    input_files : list
        List of input image files.
    coordinates : str | `~astropy.coordinates.SkyCoord`
        Coordinates of the center of the cutout.
    cutout_size : int | array | list | tuple | `~astropy.units.Quantity`
        Size of the cutout array.
    fill_value : int | float
        Value to fill the cutout with if the cutout is outside the image.
    limit_rounding_method : str
        Method to use for rounding the cutout limits. Options are 'round', 'ceil', and 'floor'.
    verbose : bool
        If True, log messages are printed to the console.

    Methods
    -------
    cutout()
        Generate the cutouts.
    """

    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, u.Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, limit_rounding_method: str = 'round', 
                 verbose: bool = False):
        
        # Log messages according to verbosity
        _handle_verbose(verbose)

        # Ensure that input files are in a list
        if isinstance(input_files, str) or isinstance(input_files, Path):
            input_files = [input_files]
        self._input_files = input_files

        # Get coordinates as a SkyCoord object
        if not isinstance(coordinates, SkyCoord):
            coordinates = SkyCoord(coordinates, unit='deg')
        self._coordinates = coordinates
        log.debug('Coordinates: %s', self._coordinates)

        # Turning the cutout size into an array of two values
        self._cutout_size = self._parse_size_input(cutout_size)
        log.debug('Cutout size: %s', self._cutout_size)

        # Assigning other attributes
        valid_rounding = ['round', 'ceil', 'floor']
        if not isinstance(limit_rounding_method, str) or limit_rounding_method.lower() not in valid_rounding:
            raise InvalidInputError(f'Limit rounding method {limit_rounding_method} is not recognized. '
                                    'Valid options are {valid_rounding}.')
        self._limit_rounding_method = limit_rounding_method
        
        if not isinstance(fill_value, int) and not isinstance(fill_value, float):
            raise InvalidInputError('Fill value must be an integer or a float.')
        self._fill_value = fill_value
    
        self._verbose = verbose

        # Initialize cutout dictionary
        self.cutouts_by_file = {}

    def _get_cutout_limits(self, img_wcs: wcs.WCS) -> np.ndarray:
        """
        Returns the x and y pixel limits for the cutout.

        Note: This function does no bounds checking, so the returned limits are not 
            guaranteed to overlap the original image.

        Parameters
        ----------
        img_wcs : `~astropy.wcs.WCS`
            The WCS for the image or cube that the cutout is being cut from.

        Returns
        -------
        response : `numpy.array`
            The cutout pixel limits in an array of the form [[xmin,xmax],[ymin,ymax]]
        """
        # Calculate pixel corresponding to coordinate
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='All-NaN slice encountered')
                center_pixel = self._coordinates.to_pixel(img_wcs)
        except wcs.NoConvergence:  # If wcs can't converge, center coordinate is far from the footprint
            raise InvalidQueryError('Cutout location is not in image footprint!')

        # We may get nans without a NoConvergence error
        if np.isnan(center_pixel).any():
            raise InvalidQueryError('Cutout location is not in image footprint!')
        
        lims = np.zeros((2, 2), dtype=int)
        for axis, size in enumerate(self._cutout_size):
            
            if not isinstance(size, u.Quantity):  # assume pixels
                dim = size / 2
            elif size.unit == u.pixel:  # also pixels
                dim = size.value / 2
            elif size.unit.physical_type == 'angle':  # angular size
                pixel_scale = u.Quantity(wcs.utils.proj_plane_pixel_scales(img_wcs)[axis],
                                         img_wcs.wcs.cunit[axis])
                dim = (size / pixel_scale).decompose() / 2
            else:
                raise InvalidInputError(f'Cutout size unit {size.unit.aliases[0]} is not supported.')

            # Round the limits according to the requested method
            rounding_funcs = {
                'round': np.round,
                'ceil': np.ceil,
                'floor': np.floor
            }
            round_func = rounding_funcs[self._limit_rounding_method]

            lims[axis, 0] = int(round_func(center_pixel[axis] - dim))
            lims[axis, 1] = int(round_func(center_pixel[axis] + dim))

            # The case where the requested area is so small it rounds to zero
            if lims[axis, 0] == lims[axis, 1]:
                lims[axis, 0] = int(np.floor(center_pixel[axis]))
                lims[axis, 1] = lims[axis, 0] + 1
        return lims

    @abstractmethod
    def cutout(self):
        """
        Generate the cutout(s).

        This method is abstract and should be defined in subclasses.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    @staticmethod
    def _parse_size_input(cutout_size):
        """
        Makes the given cutout size into a length 2 array.

        Parameters
        ----------
        cutout_size : int, array-like, `~astropy.units.Quantity`
            The size of the cutout array. If ``cutout_size`` is a scalar number or a scalar 
            `~astropy.units.Quantity`, then a square cutout of ``cutout_size`` will be created.  
            If ``cutout_size`` has two elements, they should be in ``(ny, nx)`` order.  Scalar numbers 
            in ``cutout_size`` are assumed to be in units of pixels. `~astropy.units.Quantity` objects 
            must be in pixel or angular units.

        Returns
        -------
        response : array
            Length two cutout size array, in the form [ny, nx].
        """

        # Making size into an array [ny, nx]
        if np.isscalar(cutout_size):
            cutout_size = np.repeat(cutout_size, 2)

        if isinstance(cutout_size, u.Quantity):
            cutout_size = np.atleast_1d(cutout_size)
            if len(cutout_size) == 1:
                cutout_size = np.repeat(cutout_size, 2)

        if len(cutout_size) > 2:
            warnings.warn('Too many dimensions in cutout size, only the first two will be used.',
                          InputWarning)
            cutout_size = cutout_size[:2]

        
        for dim in cutout_size:
            # Raise error if either dimension is not a positive number
            if dim <= 0:
                raise InvalidInputError('Cutout size dimensions must be greater than zero. '
                                        f'Provided size: ({cutout_size[0]}, {cutout_size[1]})')
            
            # Raise error if either dimension is not an pixel or angular Quantity
            if isinstance(dim, u.Quantity) and dim.unit != u.pixel and dim.unit.physical_type != 'angle':
                raise InvalidInputError(f'Cutout size unit {dim.unit.aliases[0]} is not supported.')

        return cutout_size
