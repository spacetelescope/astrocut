from abc import abstractmethod, ABC
from pathlib import Path
from time import monotonic
from typing import List, Optional, Union, Tuple
import warnings

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.units import Quantity
from astropy.visualization import (SqrtStretch, LogStretch, AsinhStretch, SinhStretch, LinearStretch,
                                   MinMaxInterval, ManualInterval, AsymmetricPercentileInterval)
import numpy as np
from PIL import Image
from s3path import S3Path

from . import log
from .exceptions import DataWarning, InputWarning, InvalidInputError, InvalidQueryError
from .Cutout import Cutout


class ImageCutout(Cutout, ABC):
    """
    Abstract class for creating cutouts from images. This class defines attributes and methods that are common to all
    image cutout classes.

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
    stretch : str
        Optional, default 'asinh'. The stretch to apply to the image array.
        Valid values are: asinh, sinh, sqrt, log, linear.
    minmax_percent : list
        Optional. Interval based on a keeping a specified fraction of pixels (can be asymmetric) 
        when scaling the image. The format is [lower percentile, upper percentile], where pixel
        values below the lower percentile and above the upper percentile are clipped.
        Only one of minmax_percent and minmax_value should be specified.
    minmax_value : list
        Optional. Interval based on user-specified pixel values when scaling the image.
        The format is [min value, max value], where pixel values below the min value and above
        the max value are clipped.
        Only one of minmax_percent and minmax_value should be specified.
    invert : bool
        Optional, default False.  If True the image is inverted (light pixels become dark and vice versa).
    colorize : bool
        Optional, default False.  If True a single color image is produced as output, and it is expected
        that three files are given as input.
    output_format : str
        Optional, default '.jpg'. The format of the output image file.
    cutout_prefix : str
        Optional, default 'cutout'. The prefix to use for the output file name.
    verbose : bool
        If True, log messages are printed to the console.

    Methods
    -------
    _get_cutout_data()
        Get the cutout data from the input image.
    _cutout_file()
        Cutout an image file.
    _write_to_memory()
        Write the cutouts to memory.
    _write_as_fits()
        Write the cutouts to a file in FITS format.
    _write_as_asdf()
        Write the cutouts to a file in ASDF format.
    _write_as_img()
        Write the cutouts to a file in an image format.
    _write_cutouts()
        Write the cutouts according to the specified location and output format.
    cutout()
        Generate the cutouts.
    normalize_img()
        Apply given stretch and scaling to an image array.
    """

    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, memory_only: bool = False,
                 output_dir: Union[str, Path] = '.', limit_rounding_method: str = 'round', 
                 stretch: Optional[str] = None, minmax_percent: Optional[List[int]] = None, 
                 minmax_value: Optional[List[int]] = None, invert: Optional[bool] = None, 
                 colorize: Optional[bool] = None, output_format: str = 'jpg', 
                 cutout_prefix: str = 'cutout', verbose: bool = False):
        super().__init__(input_files, coordinates, cutout_size, fill_value, memory_only, output_dir, 
                         limit_rounding_method, verbose)
        # Output format should be lowercase and begin with a dot
        out_lower = output_format.lower()
        self._output_format = f'.{out_lower}' if not output_format.startswith('.') else out_lower
        
        # Warn if image processing parameters are provided for FITS output
        if (self._output_format == '.fits') and (stretch or minmax_percent or 
                                                 minmax_value or invert or colorize):
            warnings.warn('Stretch, minmax_percent, minmax_value, invert, and colorize are not supported '
                          'for FITS output and will be ignored.', InputWarning)

        # Assign attributes with defaults if not provided
        stretch = stretch or 'asinh'
        valid_stretches = ['asinh', 'sinh', 'sqrt', 'log', 'linear']
        if not isinstance(stretch, str) or stretch.lower() not in valid_stretches:
            raise InvalidInputError(f'Stretch {stretch} is not recognized. Valid options are {valid_stretches}.')
        self._stretch = stretch.lower()
        self._invert = invert or False
        self._colorize = colorize or False
        self._minmax_percent = minmax_percent
        self._minmax_value = minmax_value
        self._cutout_prefix = cutout_prefix

        # Initialize cutout dictionary and counters
        self._cutout_dict = {}
        self._num_empty = 0
        self._num_cutouts = 0

        # Apply default scaling for image outputs
        if (self._minmax_percent is None) and (self._minmax_value is None):
            self._minmax_percent = [0.5, 99.5]

    @abstractmethod
    def _cutout_file(self, file: Union[str, Path, S3Path]):
        """
        Cutout an image file.

        This method is abstract and should be defined in subclasses.
        """
        pass


    @abstractmethod
    def _write_as_fits(self):
        """
        Write the cutouts to a file in FITS format.

        This method is abstract and should be defined in the subclass.
        """
        pass

    @abstractmethod
    def _write_as_asdf(self):
        """
        Write the cutouts to a file in ASDF format.

        This method is abstract and should be defined in the subclass.
        """
        pass

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
            warnings.warn(f'Cutout could not be saved in {self._output_format} format: {e}. '
                          'Please try a different output format.', DataWarning)
            return False
        except KeyError as e:
            warnings.warn(f'Cutout could not be saved in {self._output_format} format due to a KeyError: {e}. '
                          'Please try a different output format.', DataWarning)
            return False
        except OSError as e:
            warnings.warn(f'Cutout could not be saved: {e}', DataWarning)
            return False

    def _write_as_img(self) -> Union[str, List[str]]:
        """
        Write the cutout to memory or to a file in an image format. If colorize is set, the first 3 cutouts 
        will be combined into a single RGB image. Otherwise, each cutout will be written to a separate file.

        Returns
        -------
        cutout_path : List[Path]
            Path(s) to the written cutout files.

        Raises
        ------
        InvalidInputError
            If less than three inputs were provided for a colorized cutout.
        """
        # Set up output files and write them
        if self._colorize:  # Combine first three cutouts into a single RGB image
            cutouts = [x for fle in self._input_files for x in self._cutout_dict.get(fle, [])]

            # Check for the correct number of cutouts
            if len(cutouts) < 3:
                raise InvalidInputError(('Color cutouts require 3 input images (RGB).'
                                         'If you supplied 3 images one of the cutouts may have been empty.'))
            if len(cutouts) > 3:
                warnings.warn('Too many inputs for a color cutout, only the first three will be used.', InputWarning)
                cutouts = cutouts[:3]

            im = Image.fromarray(np.dstack([cutouts[0], cutouts[1], cutouts[2]]).astype(np.uint8))

            if self._memory_only:
                return [im]

            # Write the colorized cutout to disk
            cutout_path = '{}_{:.7f}_{:.7f}_{}-x-{}_astrocut{}'.format(
                self._cutout_prefix,
                self._coordinates.ra.value,
                self._coordinates.dec.value,
                str(self._cutout_size[0]).replace(' ', ''), 
                str(self._cutout_size[1]).replace(' ', ''),
                self._output_format
            )
            cutout_path = Path(self._output_dir, cutout_path).as_posix()
            success = self._save_img_to_file(im, cutout_path)
            if not success:
                return 

        else:  # Write each cutout to a separate image file
            cutout_path = []  # Store the paths of the written cutout files
            for file, cutout_list in self._cutout_dict.items():
                if not cutout_list:
                    warnings.warn(f'Cutout of {file} contains no data and will not be written.', DataWarning)
                    continue
                for i, cutout in enumerate(cutout_list):

                    im = Image.fromarray(cutout)
                    if self._memory_only:
                        cutout_path.append(im)
                        continue

                    # Write individual cutouts to disk
                    file_path = '{}_{:.7f}_{:.7f}_{}-x-{}_astrocut_{}{}'.format(
                        Path(file).stem,
                        self._coordinates.ra.value,
                        self._coordinates.dec.value,
                        str(self._cutout_size[0]).replace(' ', ''), 
                        str(self._cutout_size[1]).replace(' ', ''),
                        i,
                        self._output_format)
                    file_path = Path(self._output_dir, file_path).as_posix()
                    success = self._save_img_to_file(im, file_path)
                    if success:
                        cutout_path.append(file_path)

        return cutout_path

    def _write_cutouts(self) -> Union[str, List]:
        """
        Write the cutout to a file according to the specified output format.

        Returns
        -------
        cutout_path : Path | list
            Cutouts as memory objects or path(s) to the written cutout files.

        Raises
        ------
        InvalidInputError
            If the output format is not supported.
        """
        if self._memory_only:
            # Write only to memory if specified
            log.info('Writing cutouts to memory only. No output files will be created.')
        else:
            # If writing to disk, ensure that output directory exists
            Path(self._output_dir).mkdir(parents=True, exist_ok=True)

        if self._output_format == '.fits':
            return self._write_as_fits()
        elif self._output_format == '.asdf':
            return self._write_as_asdf()
        elif self._output_format in Image.registered_extensions().keys():
            return self._write_as_img()
        else:
            raise InvalidInputError(f'Output format {self._output_format} is not supported.')
        
    def cutout(self) -> Union[str, List[str], List[fits.HDUList]]:
        """
        Generate cutouts from a list of input images.

        Returns
        -------
        cutout_path : Path | list
            Cutouts as memory objects or path(s) to the written cutout files.

        Raises
        ------
        InvalidQueryError
            If no cutouts contain data.
        """
        # Track start time
        start_time = monotonic()

        # Cutout each input file
        for file in self._input_files:
            self._cutout_file(file)

        # If no cutouts contain data, raise exception
        if self._num_cutouts == self._num_empty:
            raise InvalidQueryError('Cutout contains no data! (Check image footprint.)')

        # Write cutout(s)
        cutout_path = self._write_cutouts()

        # Log cutout path and total time elapsed
        log.debug('Cutout fits file(s): %s', cutout_path)
        log.debug('Total time: %.2f sec', monotonic() - start_time)

        return cutout_path
    
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
