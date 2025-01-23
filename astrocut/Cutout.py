from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Union, Tuple

from astropy import wcs
import astropy.units as u
from s3path import S3Path
from astropy.coordinates import SkyCoord
import numpy as np

from astrocut.exceptions import InvalidInputError, InvalidQueryError

from . import log
from .utils.utils import _handle_verbose, parse_size_input


class Cutout(ABC):
    """
    Abstract class for creating cutouts. This class defines attributes and methods that are common to all
    cutout classes.

    Attributes
    ----------
    input_files : list
        List of input image files.
    coordinates : str | `~astropy.coordinates.SkyCoord`
        Coordinates of the center of the cutout.
    cutout_size : int | array | list | tuple | `~astropy.units.Quantity`
        Size of the cutout array.
    fill_value : int | float
        Value to fill the cutout with if the cutout is outside the image.
    memory_only : bool
        If True, the cutout is written to memory instead of disk.
    output_dir : str | Path
        Directory to write the cutout file(s) to.
    limit_rounding_method : str
        Method to use for rounding the cutout limits. Options are 'round', 'ceil', and 'floor'.
    verbose : bool
        If True, log messages are printed to the console.

    Methods
    -------
    get_cutout_limits(img_wcs)
        Returns the x and y pixel limits for the cutout.
    cutout()
        Generate the cutouts.
    """

    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, u.Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, memory_only: bool = False,
                 output_dir: Union[str, Path] = '.', limit_rounding_method: str = 'round', verbose: bool = True):
        
        # Log messages according to verbosity
        _handle_verbose(verbose)

        # Ensure that input files are in a list
        if isinstance(input_files, str) or isinstance(input_files, Path):
            input_files = [input_files]
        self._input_files = input_files

        # Get coordinates as a SkyCoord object
        if coordinates and not isinstance(coordinates, SkyCoord):
            coordinates = SkyCoord(coordinates, unit='deg')
        self._coordinates = coordinates
        log.debug('Coordinates: %s', self._coordinates)

        # Turning the cutout size into an array of two values
        self._cutout_size = parse_size_input(cutout_size)
        log.debug('Cutout size: %s', self._cutout_size)

        # Assigning other attributes
        valid_rounding = ['round', 'ceil', 'floor']
        if not isinstance(limit_rounding_method, str) or limit_rounding_method.lower() not in valid_rounding:
            raise InvalidInputError(f'Limit rounding method {limit_rounding_method} is not recognized. '
                                    'Valid options are {valid_rounding}.')
        self._limit_rounding_method = limit_rounding_method
        self._fill_value = fill_value
        self._memory_only = memory_only
        self._output_dir = output_dir
        self._verbose = verbose

    def _get_cutout_limits(self, img_wcs: wcs.WCS) -> np.ndarray:
        """
        Returns the x and y pixel limits for the cutout.

        Note: This function does no bounds checking, so the returned limits are not 
            guaranteed to overlap the original image.

        Parameters
        ----------
        img_wcs : `~astropy.wcs.WCS`
            The WCS for the image that the cutout is being cut from.

        Returns
        -------
        response : `numpy.array`
            The cutout pixel limits in an array of the form [[xmin,xmax],[ymin,ymax]]
        """
        # Calculate pixel corresponding to coordinate
        try:
            center_pixel = self._coordinates.to_pixel(img_wcs)
        except wcs.NoConvergence:  # If wcs can't converge, center coordinate is far from the footprint
            raise InvalidQueryError("Cutout location is not in image footprint!")

        # Sometimes, we may get nans without a NoConvergence error
        if np.isnan(center_pixel).any():
            raise InvalidQueryError("Cutout location is not in image footprint!")
        
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
                raise InvalidInputError(f'Cutout size units {size.unit} are not supported.')

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
            if self._limit_rounding_method == 'round' and lims[axis, 0] == lims[axis, 1]:
                lims[axis, 0] = int(np.floor(center_pixel[axis] - 1))
                lims[axis, 1] = lims[axis, 0] + 1

        return lims

    @abstractmethod
    def cutout(self):
        """
        Generate the cutout(s).

        This method is abstract and should be defined in subclasses.
        """
        pass
