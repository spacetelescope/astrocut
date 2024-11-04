from pathlib import Path
from time import monotonic
from typing import List, Union
import warnings

from astropy.visualization import (SqrtStretch, LogStretch, AsinhStretch, SinhStretch, LinearStretch,
                                   MinMaxInterval, ManualInterval, AsymmetricPercentileInterval)
from astropy.nddata import NoOverlapError
import numpy as np
from PIL import Image
from s3path import S3Path

from . import log
from .exceptions import DataWarning, InputWarning, InvalidInputError, InvalidQueryError
from .FITSCutout import FITSCutout


class ImageCutout(FITSCutout):

    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates, cutout_size: int = 25,
                 fill_value: Union[int, float] = np.nan, extension=None, cutout_prefix: str = 'cutout', 
                 output_dir: str = '.', stretch: str = 'asinh', minmax_percent: List[int] = None, minmax_value: List[int] = None,
                 invert: bool = False, colorize: bool = False, img_format: str = 'jpg', verbose: bool = True):
        super().__init__(input_files, coordinates, cutout_size, fill_value, extension, cutout_prefix=cutout_prefix, 
                         output_dir=output_dir, verbose=verbose)
        
        # Attributes for normalizing image cutouts
        self.stretch = stretch
        self.minmax_percent = minmax_percent
        self.minmax_value = minmax_value
        self.invert = invert
        self.colorize = colorize
        self.img_format = img_format

    def normalize_img(self, img_arr):
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


        # Setting up the transform with the stretch
        if self.stretch == 'asinh':
            transform = AsinhStretch()
        elif self.stretch == 'sinh':
            transform = SinhStretch()
        elif self.stretch == 'sqrt':
            transform = SqrtStretch()
        elif self.stretch == 'log':
            transform = LogStretch()
        elif self.stretch == 'linear':
            transform = LinearStretch()
        else:
            raise InvalidInputError("Stretch {} is not supported!".format(self.stretch))

        # Adding the scaling to the transform
        if self.minmax_percent is not None:
            transform += AsymmetricPercentileInterval(*self.minmax_percent)
            
            if self.minmax_value is not None:
                warnings.warn("Both minmax_percent and minmax_value are set, minmax_value will be ignored.",
                                InputWarning)
        elif self.minmax_value is not None:
            transform += ManualInterval(*self.minmax_value)
        else:  # Default, scale the entire image range to [0,1]
            transform += MinMaxInterval()

        # Performing the transform and then putting it into the integer range 0-255
        norm_img = transform(img_arr)
        norm_img = np.multiply(255, norm_img, out=norm_img)
        norm_img = norm_img.astype(np.uint8)

        # Applying invert if requested
        if self.invert:
            norm_img = 255 - norm_img

        return norm_img
        
    
    def cutout(self):

        start_time = monotonic()

        # Applying the default scaling
        if (self.minmax_percent is None) and (self.minmax_value is None):
            self.minmax_percent = [0.5, 99.5]

        cutout_hdu_dict = {}
        for file in self.input_files:
            # Load data
            hdulist, cutout_inds = self._load_data(file)

            # create HDU cutouts
            cutout_hdus = []
            for ind in cutout_inds:
                try:
                    cutout = self._hducut(hdulist[ind])

                    # We only need the data array for images
                    cutout = cutout.data

                    # Apply the appropriate normalization parameters
                    normalized_cutout = self.normalize_img(cutout)

                    # Check that there is data in the cutout
                    if not (cutout == 0).all():
                        cutout_hdus.append(normalized_cutout)

                except OSError as err:
                    warnings.warn((f"Error {err} encountered when performing cutout on {file}, "
                                    f"extension {ind}, skipping..."),
                                    DataWarning)
                    self.num_empty += 1
                except NoOverlapError as err:
                    warnings.warn((f"Cutout footprint does not overlap with data in {file}, "
                                    f"extension {ind}, skipping..."),
                                    DataWarning)
                    self.num_empty += 1
                    
            hdulist.close()
            cutout_hdu_dict[file] = cutout_hdus

        # If no cutouts contain data, raise exception
        if not cutout_hdu_dict:
            raise InvalidQueryError("Cutout contains no data! (Check image footprint.)")
        
        # Make sure that output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Set up output files and write them
        if self.colorize:
            cutouts = [x for fle in self.input_files for x in cutout_hdu_dict.get(fle, [])]

            # Doing checks correct number of cutouts
            if len(cutouts) < 3:
                raise InvalidInputError(("Color cutouts require 3 input images (RGB)."
                                            "If you supplied 3 images one of the cutouts may have been empty."))
            if len(cutouts) > 3:
                warnings.warn("Too many inputs for a color cutout, only the first three will be used.",
                                InputWarning)
                cutouts = cutouts[:3]

            cutout_path = "{}_{:7f}_{:7f}_{}-x-{}_astrocut.{}".format(self.cutout_prefix,
                                                                      self.coordinates.ra.value,
                                                                      self.coordinates.dec.value,
                                                                      str(self.cutout_size[0]).replace(' ', ''), 
                                                                      str(self.cutout_size[1]).replace(' ', ''),
                                                                      self.img_format.lower()) 
            cutout_path = Path(self.output_dir, cutout_path)
            Image.fromarray(np.dstack([cutouts[0], cutouts[1], cutouts[2]]).astype(np.uint8)).save(cutout_path)

        else:
            cutout_path = []
            for file, cutout_list in cutout_hdu_dict.items():

                if not cutout_list:
                    warnings.warn("Cutout of {} contains no data and will not be written.".format(file),
                                DataWarning)
                    continue

                for i, cutout in enumerate(cutout_list):
                    file_path = "{}_{:7f}_{:7f}_{}-x-{}_astrocut_{}.{}".format(Path(file).name.rstrip('.fits'),
                                                                               self.coordinates.ra.value,
                                                                               self.coordinates.dec.value,
                                                                               str(self.cutout_size[0]).replace(' ', ''), 
                                                                               str(self.cutout_size[1]).replace(' ', ''),
                                                                               i,
                                                                               self.img_format.lower())
                    file_path = Path(self.output_dir, file_path)
                    cutout_path.append(file_path)
                
                    Image.fromarray(cutout).save(file_path)

        log.debug("Cutout fits file(s): %s", cutout_path)
        log.debug("Total time: %.2f sec", monotonic() - start_time)
        return cutout_path