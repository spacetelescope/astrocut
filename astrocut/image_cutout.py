from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Optional, Union, Tuple
import warnings

from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.visualization import (SqrtStretch, LogStretch, AsinhStretch, SinhStretch, LinearStretch,
                                   MinMaxInterval, ManualInterval, AsymmetricPercentileInterval)

import numpy as np
from PIL import Image
from s3path import S3Path

from . import log
from .exceptions import DataWarning, InputWarning, InvalidInputError
from .cutout import Cutout


class ImageCutout(Cutout, ABC):
    """
    Abstract class for creating cutouts from images. This class defines attributes and methods that are common to all
    image cutout classes.

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

    Attributes
    ----------
    cutouts_by_file : dict
        Dictionary containing the cutouts for each input file.
    image_cutouts : list
        List of `~PIL.Image` objects representing the cutouts.

    Methods
    -------
    get_image_cutouts(stretch, minmax_percent, minmax_value, invert, colorize)
        Get the cutouts as `~PIL.Image` objects.
    cutout()
        Generate the cutouts.
    write_as_img(stretch, minmax_percent, minmax_value, invert, colorize, output_format, output_dir, cutout_prefix)
        Write the cutouts to a file in an image format.
    normalize_img(stretch, minmax_percent, minmax_value, invert)
        Apply given stretch and scaling to an image array.
    """

    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, limit_rounding_method: str = 'round',
                 verbose: bool = False):
        super().__init__(input_files, coordinates, cutout_size, fill_value, limit_rounding_method, verbose)

        # Stores the image cutouts as PIL.Image objects
        self._image_cutouts = None

    @property
    def image_cutouts(self) -> List[Image.Image]:
        """
        Return the cutouts as a list of `PIL.Image` objects.

        If the image objects have not been generated yet, they will be generated with default
        normalization parameters.
        """
        if not self._image_cutouts:
            self._image_cutouts = self.get_image_cutouts()
        return self._image_cutouts
    
    def get_image_cutouts(self, stretch: Optional[str] = 'asinh', minmax_percent: Optional[List[int]] = None, 
                          minmax_value: Optional[List[int]] = None, invert: Optional[bool] = False, 
                          colorize: Optional[bool] = False) -> List[Image.Image]:
        """
        Get the cutouts as `~PIL.Image` objects given certain normalization parameters. This method also sets
        the `image_cutouts` attribute.

        Parameters
        ----------
        stretch : str
            Optional, default 'asinh'. The stretch to apply to the image array.
            Valid values are: asinh, sinh, sqrt, log, linear
        minmax_percent : array
            Optional. Interval based on a keeping a specified fraction of pixels (can be asymmetric) 
            when scaling the image. The format is [lower percentile, upper percentile], where pixel
            values below the lower percentile and above the upper percentile are clipped.
            Only one of minmax_percent and minmax_value should be specified.
        minmax_value : array
            Optional. Interval based on user-specified pixel values when scaling the image.
            The format is [min value, max value], where pixel values below the min value and above
            the max value are clipped.
            Only one of minmax_percent and minmax_value should be specified.
        invert : bool
            Optional, default False.  If True the image is inverted (light pixels become dark and vice versa).
        colorize : bool
            Optional, default False. If True, the first three cutouts will be combined into a single RGB image.

        Returns
        -------
        image_cutouts : list
            List of `~PIL.Image` objects representing the cutouts.
        """
        # Validate the stretch parameter
        valid_stretches = ['asinh', 'sinh', 'sqrt', 'log', 'linear']
        if not isinstance(stretch, str) or stretch.lower() not in valid_stretches:
            raise InvalidInputError(f'Stretch {stretch} is not recognized. Valid options are {valid_stretches}.')
        stretch = stretch.lower()

        # Apply default scaling for image outputs
        if (minmax_percent is None) and (minmax_value is None):
            minmax_percent = [0.5, 99.5]

        if colorize:  # color cutout
            all_cutouts = [x for fle in self._input_files for x in self.cutouts_by_file.get(fle, [])]

            # Check for the correct number of cutouts
            if len(all_cutouts) < 3:
                raise InvalidInputError(('Color cutouts require 3 input images (RGB).'
                                         'If you supplied 3 images one of the cutouts may have been empty.'))
            if len(all_cutouts) > 3:
                warnings.warn('Too many inputs for a color cutout, only the first three will be used.', InputWarning)
                all_cutouts = all_cutouts[:3]

            img_arrs = []
            for cutout in all_cutouts:
                # Image output, applying the appropriate normalization parameters
                img_arrs.append(self.normalize_img(cutout.data, stretch, minmax_percent, minmax_value, invert))

            # Combine the three cutouts into a single RGB image
            self._image_cutouts = [Image.fromarray(np.dstack([img_arrs[0], img_arrs[1], img_arrs[2]]).astype(np.uint8))]
        else:  # one image per cutout
            image_cutouts = []
            for file, cutout_list in self.cutouts_by_file.items():
                for i, cutout in enumerate(cutout_list):
                    # Apply the appropriate normalization parameters
                    img_arr = self.normalize_img(cutout.data, stretch, minmax_percent, minmax_value, invert)
                    image_cutouts.append(Image.fromarray(img_arr))

            self._image_cutouts = image_cutouts

        return self._image_cutouts
    
    @abstractmethod
    def _cutout_file(self, file: Union[str, Path, S3Path]):
        """
        Cutout an image file.

        This method is abstract and should be defined in subclasses.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def cutout(self):
        """
        Generate the cutout(s).

        This method is abstract and should be defined in subclasses.
        """
        raise NotImplementedError('Subclasses must implement this method.')
    
    def _parse_output_format(self, output_format: str) -> str:
        """
        Parse the output format string and return it in a standardized format.

        Parameters
        ----------
        output_format : str
            The output format string.

        Returns
        -------
        out_format : str
            The output format string in a standardized format.
        """
        # Put format in standard format
        out_lower = output_format.lower()
        output_format = f'.{out_lower}' if not output_format.startswith('.') else out_lower
    
        # Error if the output format is not supported
        if output_format not in Image.registered_extensions().keys():
            raise InvalidInputError(f'Output format {output_format} is not supported.')
        
        return output_format

    def _save_img_to_file(self, im: Image, file_path: str) -> bool:
        """
        Save a `~PIL.Image` object to a file.

        Parameters
        ----------
        im : `~PIL.Image`
            The image to save.
        file_path : str
            The path to save the image to.

        Returns
        -------
        success : bool
            True if the image was saved successfully, False otherwise.
        """
        try:
            im.save(file_path)
            return True
        except ValueError as e:
            output_format = Path(file_path).suffix
            warnings.warn(f'Cutout could not be saved in {output_format} format: {e}. '
                          'Please try a different output format.', DataWarning)
            return False
        except KeyError as e:
            output_format = Path(file_path).suffix
            warnings.warn(f'Cutout could not be saved in {output_format} format due to a KeyError: {e}. '
                          'Please try a different output format.', DataWarning)
            return False
        except OSError as e:
            warnings.warn(f'Cutout could not be saved: {e}', DataWarning)
            return False

    def write_as_img(self, stretch: Optional[str] = 'asinh', minmax_percent: Optional[List[int]] = None, 
                     minmax_value: Optional[List[int]] = None, invert: Optional[bool] = False, 
                     colorize: Optional[bool] = False, output_format: str = '.jpg', 
                     output_dir: Union[str, Path] = '.', cutout_prefix: str = 'cutout') -> Union[str, List[str]]:
        """
        Write the cutout to memory or to a file in an image format. If colorize is set, the first 3 cutouts 
        will be combined into a single RGB image. Otherwise, each cutout will be written to a separate file.

        Parameters
        ----------
        stretch : str
            Optional, default 'asinh'. The stretch to apply to the image array.
            Valid values are: asinh, sinh, sqrt, log, linear
        minmax_percent : array
            Optional. Interval based on a keeping a specified fraction of pixels (can be asymmetric) 
            when scaling the image. The format is [lower percentile, upper percentile], where pixel
            values below the lower percentile and above the upper percentile are clipped.
            Only one of minmax_percent and minmax_value shoul be specified.
        minmax_value : array
            Optional. Interval based on user-specified pixel values when scaling the image.
            The format is [min value, max value], where pixel values below the min value and above
            the max value are clipped.
            Only one of minmax_percent and minmax_value should be specified.
        invert : bool
            Optional, default False.  If True the image is inverted (light pixels become dark and vice versa).
        colorize : bool
            Optional, default False. If True, the first three cutouts will be combined into a single RGB image.
        output_format : str
            Optional, default '.jpg'. The output format for the cutout image(s).
        output_dir : str | `~pathlib.Path`
            Optional, default '.'. The directory to write the cutout image(s) to.
        cutout_prefix : str
            Optional, default 'cutout'. The prefix to add to the cutout image file name.

        Returns
        -------
        cutout_path : List[Path]
            Path(s) to the written cutout files.

        Raises
        ------
        InvalidInputError
            If less than three inputs were provided for a colorized cutout.
        """
        # Parse the output format
        output_format = self._parse_output_format(output_format)

        # Get the image cutouts with the given normalization parameters
        image_cutouts = self.get_image_cutouts(stretch, minmax_percent, minmax_value, invert, colorize)

        # Create the output directory if it does not exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set up output files and write them
        if colorize:  # Combine first three cutouts into a single RGB image
            # Write the colorized cutout to disk
            filename = '{}_{:.7f}_{:.7f}_{}-x-{}_astrocut{}'.format(
                cutout_prefix,
                self._coordinates.ra.value,
                self._coordinates.dec.value,
                str(self._cutout_size[0]).replace(' ', ''), 
                str(self._cutout_size[1]).replace(' ', ''),
                output_format
            )

            # Attempt to write image to file
            cutout_paths = Path(output_dir, filename).as_posix()
            success = self._save_img_to_file(image_cutouts[0], cutout_paths)
            if not success:
                cutout_paths = None

        else:  # Write each cutout to a separate image file
            cutout_paths = []  # Store the paths of the written cutout files
            for i, file in enumerate(self.cutouts_by_file):
                # Write individual cutouts to disk
                filename = '{}_{:.7f}_{:.7f}_{}-x-{}_astrocut_{}{}'.format(
                    Path(file).stem,
                    self._coordinates.ra.value,
                    self._coordinates.dec.value,
                    str(self._cutout_size[0]).replace(' ', ''), 
                    str(self._cutout_size[1]).replace(' ', ''),
                    i,
                    output_format)
                
                # Attempt to write image to file
                cutout_path = Path(output_dir, filename).as_posix()
                success = self._save_img_to_file(image_cutouts[i], cutout_path)

                # Append the path to the written file or the memory object
                # If the image could not be written, append None
                if not success:
                    cutout_path = None
                cutout_paths.append(cutout_path)

        log.debug('Cutout filepaths: {}'.format(cutout_paths))
        return cutout_paths
    
    @staticmethod
    def normalize_img(img_arr: np.ndarray, stretch: str = 'asinh', minmax_percent: Optional[List[int]] = None, 
                      minmax_value: Optional[List[int]] = None, invert: bool = False) -> np.ndarray:
        """
        Apply given stretch and scaling to an image array.

        Parameters
        ----------
        img_arr : array
            The input image array.
        stretch : str
            Optional, default 'asinh'. The stretch to apply to the image array.
            Valid values are: asinh, sinh, sqrt, log, linear
        minmax_percent : array
            Optional. Interval based on a keeping a specified fraction of pixels (can be asymmetric) 
            when scaling the image. The format is [lower percentile, upper percentile], where pixel
            values below the lower percentile and above the upper percentile are clipped.
            Only one of minmax_percent and minmax_value shoul be specified.
        minmax_value : array
            Optional. Interval based on user-specified pixel values when scaling the image.
            The format is [min value, max value], where pixel values below the min value and above
            the max value are clipped.
            Only one of minmax_percent and minmax_value should be specified.
        invert : bool
            Optional, default False.  If True the image is inverted (light pixels become dark and vice versa).

        Returns
        -------
        response : array
            The normalized image array, in the form in an integer arrays with values in the range 0-255.

        Raises
        ------
        InvalidInputError
            If the stretch is not supported.
        """

        # Check if the input image array is empty
        if img_arr.size == 0:
            raise InvalidInputError('Input image array is empty.')

        # Setting up the transform with the stretch
        if stretch == 'asinh':
            transform = AsinhStretch()
        elif stretch == 'sinh':
            transform = SinhStretch()
        elif stretch == 'sqrt':
            transform = SqrtStretch()
        elif stretch == 'log':
            transform = LogStretch()
        elif stretch == 'linear':
            transform = LinearStretch()
        else:
            raise InvalidInputError(f'Stretch {stretch} is not supported!'
                                    'Valid options are: asinh, sinh, sqrt, log, linear.')

        # Adding the scaling to the transform
        if minmax_percent is not None:
            transform += AsymmetricPercentileInterval(*minmax_percent)
            
            if minmax_value is not None:
                warnings.warn('Both minmax_percent and minmax_value are set, minmax_value will be ignored.', 
                              InputWarning)
        elif minmax_value is not None:
            transform += ManualInterval(*minmax_value)
        else:  # Default, scale the entire image range to [0,1]
            transform += MinMaxInterval()

        # Performing the transform and then putting it into the integer range 0-255
        norm_img = transform(img_arr)
        np.multiply(255, norm_img, out=norm_img)
        norm_img = norm_img.astype(np.uint8)

        # Applying invert if requested
        np.subtract(255, norm_img, out=norm_img, where=invert)

        return norm_img
    

def normalize_img(img_arr: np.ndarray, stretch: str = 'asinh', minmax_percent: Optional[List[int]] = None, 
                  minmax_value: Optional[List[int]] = None, invert: bool = False) -> np.ndarray:
    """
    Apply given stretch and scaling to an image array.

    Parameters
    ----------
    img_arr : array
        The input image array.
    stretch : str
        Optional, default 'asinh'. The stretch to apply to the image array.
        Valid values are: asinh, sinh, sqrt, log, linear
    minmax_percent : array
        Optional. Interval based on a keeping a specified fraction of pixels (can be asymmetric) 
        when scaling the image. The format is [lower percentile, upper percentile], where pixel
        values below the lower percentile and above the upper percentile are clipped.
        Only one of minmax_percent and minmax_value shoul be specified.
    minmax_value : array
        Optional. Interval based on user-specified pixel values when scaling the image.
        The format is [min value, max value], where pixel values below the min value and above
        the max value are clipped.
        Only one of minmax_percent and minmax_value should be specified.
    invert : bool
        Optional, default False.  If True the image is inverted (light pixels become dark and vice versa).

    Returns
    -------
    response : array
        The normalized image array, in the form in an integer arrays with values in the range 0-255.
    """
    return ImageCutout.normalize_img(img_arr=img_arr,
                                     stretch=stretch,
                                     minmax_percent=minmax_percent,
                                     minmax_value=minmax_value,
                                     invert=invert)
