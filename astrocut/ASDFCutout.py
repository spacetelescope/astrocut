import copy
from pathlib import Path
from typing import List, Tuple, Union, Optional
import warnings

import asdf
import gwcs
import numpy as np
import requests
import s3fs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import models
from astropy.nddata.utils import Cutout2D, NoOverlapError
from astropy.units import Quantity
from astropy.wcs import WCS
from s3path import S3Path

from . import log
from .ImageCutout import ImageCutout
from .exceptions import DataWarning


class ASDFCutout(ImageCutout):
    """
    Class for creating cutouts from ASDF files.

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
    key : str
        Optional, default None. Access key ID for S3 file system.
    secret : str
        Optional, default None. Secret access key for S3 file system.
    token : str
        Optional, default None. Security token for S3 file system.
    verbose : bool
        If True, log messages are printed to the console.

    Methods
    -------
    _get_cloud_http()
        Get the HTTP URL of a cloud resource from an S3 URI.
    _load_file_data()
        Load the data from an input file.
    _get_cutout_data()
        Get the cutout data from the input image.
    _slice_gwcs()
        Slice the original gwcs object to fit the cutout.
    _cutout_file()
        Create a cutout from an input file.
    _write_as_format()
        Write the cutout to disk or memory in the specified format.
    _write_as_fits()
        Write the cutouts to disk or memory in FITS format.
    _write_as_asdf()
        Write the cutouts to disk or memory in ASDF format.
    get_center_pixel()
        Get the closest pixel location on an input image for a given set of coordinates.
    """
        
    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, memory_only: bool = False,
                 output_dir: Union[str, Path] = '.', limit_rounding_method: str = 'round', 
                 stretch: Optional[str] = None, minmax_percent: Optional[List[int]] = None, 
                 minmax_value: Optional[List[int]] = None, invert: Optional[bool] = None, 
                 colorize: Optional[bool] = None, output_format: str = '.asdf', 
                 key: Optional[str] = None, secret: Optional[str] = None,
                 token: Optional[str] = None, verbose: bool = False):
        # Superclass constructor 
        super().__init__(input_files, coordinates, cutout_size, fill_value, memory_only, output_dir, 
                         limit_rounding_method, stretch, minmax_percent, minmax_value, invert, colorize, 
                         output_format, 'cutout', verbose=verbose)

        # Assign AWS credential attributes
        self._key = key
        self._secret = secret
        self._token = token
        self._mission_kwd = 'roman'

    def _get_cloud_http(self, input_file: Union[str, S3Path]) -> str:
        """ 
        Get the HTTP URL of a cloud resource from an S3 URI.

        Parameters
        ----------
        input_file : str | S3Path
            The input file S3 URI.

        Returns
        -------
        str
            The HTTP URL of the cloud resource.
        """
        # Check if public or private by sending an HTTP request
        s3_path = S3Path.from_uri(input_file) if isinstance(input_file, str) else input_file
        url = f'https://{s3_path.bucket}.s3.amazonaws.com/{s3_path.key}'
        resp = requests.head(url, timeout=10)
        is_anon = False if resp.status_code == 403 else True
        if not is_anon:
            log.debug('Attempting to access private S3 bucket: %s', s3_path.bucket)

        # Create file system and get URL of file
        fs = s3fs.S3FileSystem(anon=is_anon, key=self._key, secret=self._secret, token=self._token)
        with fs.open(input_file, 'rb') as f:
            return f.url()

    def _load_file_data(self, input_file):
        """
        Load relevant data from an input file.

        Parameters
        ----------
        input_file : str | Path | S3Path
            The input file to load data from.

        Returns
        -------
        data : array
            The image data.
        wcs : `~astropy.wcs.WCS`
            The FITS WCS of the image. This is approximated from the GWCS object.
        gwcs : `~gwcs.wcs.WCS`
            The GWCS of the image.
        pixel_coords : tuple
            The pixel coordinates closest to the center of the cutout.
        """
        # If file comes from AWS cloud bucket, get HTTP URL to open with asdf
        if (isinstance(input_file, str) and input_file.startswith('s3://')) or isinstance(input_file, S3Path):
            input_file = self._get_cloud_http(input_file)

        # Get data and GWCS object from ASDF input file
        with asdf.open(input_file) as af:
            data = af[self._mission_kwd]['data']
            gwcs = af[self._mission_kwd]['meta'].get('wcs', None)

        return (data, gwcs)
    
    def _get_cutout_data(self, data: np.ndarray, wcs: WCS, pixel_coords: Tuple[int, int]) -> Cutout2D:
        """
        Get the cutout data from the input image.

        Parameters
        ----------
        data : array
            The input image data.
        wcs : `~astropy.wcs.WCS`
            The approximated WCS of the input image.
        pixel_coords : tuple
            The pixel coordinates closest to the center of the cutout.

        Returns
        -------
        img_cutout : `~astropy.nddata.Cutout2D`
            The cutout object.
        """
        log.debug('Original image shape: %s', data.shape)

        # Using `~astropy.nddata.Cutout2D` to get the cutout data and handle WCS
        # Passing in pixel coordinates that were calculated using the original GWCS object,
        # so the approximate WCS object will not be used to calculate the pixel coordinates 
        # of the cutout center. Approximate WCS will be used in calculation of cutout bounds 
        # if cutout size is given in angular units.
        img_cutout = Cutout2D(data,
                              position=pixel_coords,
                              wcs=wcs,
                              size=(self._cutout_size[1], self._cutout_size[0]),
                              mode='partial',
                              fill_value=self._fill_value)
        log.debug('Image cutout shape: %s', img_cutout.shape)

        return img_cutout

    def _slice_gwcs(self, cutout: Cutout2D, gwcs: gwcs.wcs.WCS) -> gwcs.wcs.WCS:
        """ 
        Slice the original gwcs object.

        "Slices" the original gwcs object down to the cutout shape.  This is a hack
        until proper gwcs slicing is in place a la fits WCS slicing.  The ``slices``
        keyword input is a tuple with the x, y cutout boundaries in the original image
        array, e.g. ``cutout.slices_original``.  Astropy Cutout2D slices are in the form
        ((ymin, ymax, None), (xmin, xmax, None))

        Parameters
        ----------
        cutout : astropy.nddata.Cutout2D
            The cutout object.
        gwcs : gwcs.wcs.WCS
            The original GWCS from the input image.

        Returns
        -------
        gwcs.wcs.WCS
            The sliced GWCS object.
        """
        # Create copy of original gwcs object
        tmp = copy.deepcopy(gwcs)

        # Get the cutout array bounds and create a new shift transform to the cutout
        # Add the new transform to the gwcs
        slices = cutout.slices_original
        xmin, xmax = slices[1].start, slices[1].stop
        ymin, ymax = slices[0].start, slices[0].stop
        shape = (ymax - ymin, xmax - xmin)
        offsets = models.Shift(xmin, name='cutout_offset1') & models.Shift(ymin, name='cutout_offset2')
        tmp.insert_transform('detector', offsets, after=True)

        # Modify the gwcs bounding box to the cutout shape
        tmp.bounding_box = ((0, shape[0] - 1), (0, shape[1] - 1))
        tmp.pixel_shape = shape[::-1]
        tmp.array_shape = shape
        return tmp
    
    def _cutout_file(self, file: Union[str, Path, S3Path]):
        """
        Create a cutout from a single input file.

        Parameters
        ----------
        file : str | Path | S3Path
            The input file to create a cutout from.
        """
        # Load the data from the input file
        data, gwcs = self._load_file_data(file)

        # Skip if the file does not contain a GWCS object
        if gwcs is None:
            warnings.warn(f'File {file} does not contain a GWCS object. Skipping...',
                          DataWarning)
            self._num_empty += 1
            return

        # Get closest pixel coordinates and approximated WCS
        pixel_coords, wcs = self.get_center_pixel(gwcs, self._coordinates.ra.value, self._coordinates.dec.value)
        self._num_cutouts += 1

        # Create the cutout
        try:
            cutout2D = self._get_cutout_data(data, wcs, pixel_coords)
        except NoOverlapError:
            warnings.warn(f'Cutout footprint does not overlap with data in {file}, skipping...',
                          DataWarning)
            self._num_empty += 1
            return
        
        # Check that there is data in the cutout image
        if (cutout2D.data == 0).all() or (np.isnan(cutout2D.data)).all():
            warnings.warn(f'Cutout of {file} contains no data, skipping...',
                          DataWarning)
            self._num_empty += 1
            return

        # Convert Quantity data to ndarray
        if isinstance(cutout2D.data, Quantity):
            cutout2D.data = cutout2D.data.value

        if self._output_format == '.asdf':            
            # Slice the origial gwcs to the cutout
            sliced_gwcs = self._slice_gwcs(cutout2D, gwcs)

            # Create the asdf tree
            tree = {self._mission_kwd: {'meta': {'wcs': sliced_gwcs}, 'data': cutout2D.data}}
            cutout = asdf.AsdfFile(tree)

        elif self._output_format == '.fits':
            # TODO: Create a FITS object with ASDF extension
            # Create a primary FITS header to hold data and WCS
            primary_hdu = fits.PrimaryHDU(data=cutout2D.data, header=cutout2D.wcs.to_header(relax=True))

            # Write to HDUList
            cutout = fits.HDUList([primary_hdu])
            
        else:
            # Image output, apply the appropriate normalization parameters
            cutout = [self.normalize_img(cutout2D.data, self._stretch, self._minmax_percent, self._minmax_value,
                                         self._invert)]

        # Add cutout to dictionary
        self._cutout_dict[file] = cutout

    def _write_as_format(self):
        """
        Write the cutout to disk or memory in the specified output format.

        Returns
        -------
        all_cutouts : str | list
            The path(s) to the cutout file(s) or the cutout memory objects.
        """
        if self._memory_only:
            return list(self._cutout_dict.values())
        
        all_cutouts = []
        for file, cutout in self._cutout_dict.items():
            filename = '{}_{:.7f}_{:.7f}_{}-x-{}_astrocut{}'.format(
                Path(file).stem,
                self._coordinates.ra.value,
                self._coordinates.dec.value,
                str(self._cutout_size[0]).replace(' ', ''), 
                str(self._cutout_size[1]).replace(' ', ''),
                self._output_format)
            cutout_path = Path(self._output_dir, filename)
            if self._output_format == '.fits':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore') 
                    cutout.writeto(cutout_path, overwrite=True, checksum=True)
            elif self._output_format == '.asdf':
                cutout.write_to(cutout_path)
            all_cutouts.append(cutout_path.as_posix())

        return all_cutouts if len(all_cutouts) > 1 else all_cutouts[0]
    
    def _write_as_fits(self):
        """
        Write the cutouts to disk or memory in FITS format.

        Returns
        -------
        str | list
            The path(s) to the cutout FITS file(s) or the cutout memory objects.
        """
        return self._write_as_format()

    def _write_as_asdf(self):
        """
        Write the cutouts to disk or memory in ASDF format.

        Returns
        -------
        str | list
            The path(s) to the cutout ASDF file(s) or the cutout memory objects.
        """
        return self._write_as_format()
    
    @staticmethod 
    def get_center_pixel(gwcsobj: gwcs.wcs.WCS, ra: float, dec: float) -> Tuple[Tuple[int, int], WCS]:
        """ 
        Get the closest pixel location on an input image for a given set of coordinates.

        Parameters
        ----------
        gwcsobj : gwcs.wcs.WCS
            The GWCS object.
        ra : float
            The right ascension of the input coordinates.
        dec : float
            The declination of the input coordinates.

        Returns
        -------
        pixel_position
            The pixel position of the input coordinates.
        wcs_updated : `~astropy.wcs.WCS`
            The approximated FITS WCS object.
        """

        # Convert the gwcs object to an astropy FITS WCS header
        header = gwcsobj.to_fits_sip()

        # Update WCS header with some keywords that it's missing.
        # Otherwise, it won't work with astropy.wcs tools (TODO: Figure out why. What are these keywords for?)
        for k in ['cpdis1', 'cpdis2', 'det2im1', 'det2im2', 'sip']:
            if k not in header:
                header[k] = 'na'

        # New WCS object with updated header
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wcs_updated = WCS(header)

        # Turn input RA, Dec into a SkyCoord object
        coordinates = SkyCoord(ra, dec, unit='deg')

        # Map the coordinates to a pixel's location on the Roman 2d array (row, col)
        gwcsobj.bounding_box = None
        row, col = gwcsobj.invert(coordinates)

        return (row, col), wcs_updated
