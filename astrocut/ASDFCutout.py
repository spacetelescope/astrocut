import copy
from pathlib import Path
from time import monotonic
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
from .exceptions import DataWarning, InvalidQueryError


class ASDFCutout(ImageCutout):
    """
    Class for creating cutouts from ASDF files.

    Args
    ----
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
    key : str
        Optional, default None. Access key ID for S3 file system.
    secret : str
        Optional, default None. Secret access key for S3 file system.
    token : str
        Optional, default None. Security token for S3 file system.
    verbose : bool
        If True, log messages are printed to the console.

    Attributes
    ----------
    cutouts : list
        The cutouts as a list of `astropy.nddata.Cutout2D` objects.
    cutouts_by_file : dict
        The cutouts as `astropy.nddata.Cutout2D` objects stored by input filename.
    fits_cutouts : list
        The cutouts as a list `astropy.io.fits.HDUList` objects.
    asdf_cutouts : list
        The cutouts as a list of `asdf.AsdfFile` objects.

    Methods
    -------
    _get_cloud_http(input_file)
        Get the HTTP URL of a cloud resource from an S3 URI.
    _load_file_data(input_file)
        Load the data from an input file.
    _get_cutout_data(data, wcs, pixel_coords)
        Get the cutout data from the input image.
    _slice_gwcs(cutout, gwcs)
        Slice the original gwcs object to fit the cutout.
    _cutout_file(file)
        Create a cutout from an input file.
    cutout()
        Generate cutouts from a list of input images.
    _write_as_format(output_format, output_dir)
        Write the cutout to disk or memory in the specified format.
    write_as_fits(output_dir)
        Write the cutouts to disk or memory in FITS format.
    write_as_asdf(output_dir)
        Write the cutouts to disk or memory in ASDF format.
    get_center_pixel(gwcsobj, ra, dec)
        Get the closest pixel location on an input image for a given set of coordinates.
    """
        
    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, limit_rounding_method: str = 'round',
                 key: Optional[str] = None, secret: Optional[str] = None,
                 token: Optional[str] = None, verbose: bool = False):
        # Superclass constructor 
        super().__init__(input_files, coordinates, cutout_size, fill_value, limit_rounding_method, verbose=verbose)

        # Assign AWS credential attributes
        self._key = key
        self._secret = secret
        self._token = token
        self._mission_kwd = 'roman'

        self.cutouts = []  # Public attribute to hold `Cutout2D` objects
        self._asdf_cutouts = None  # Store ASDF objects
        self._fits_cutouts = None  # Store FITS objects
        self._gwcs_objects = []  # Store original GWCS objects

        # Make cutouts
        self.cutout()

    @property
    def fits_cutouts(self) -> List[fits.HDUList]:
        """
        Return the cutouts as a list `astropy.io.fits.HDUList` objects.
        """
        if not self._fits_cutouts:
            fits_cutouts = []
            for cutout in self.cutouts:
                # TODO: Create a FITS object with ASDF extension
                # Create a primary FITS header to hold data and WCS
                primary_hdu = fits.PrimaryHDU(data=cutout.data, header=cutout.wcs.to_header(relax=True))

                # Write to HDUList
                fits_cutouts.append(fits.HDUList([primary_hdu]))
            self._fits_cutouts = fits_cutouts
        return self._fits_cutouts
    
    @property
    def asdf_cutouts(self) -> List[asdf.AsdfFile]:
        """
        Return the cutouts as a list of `asdf.AsdfFile` objects.
        """
        if not self._asdf_cutouts:
            asdf_cutouts = []
            for i, cutout in enumerate(self.cutouts):
                # Slice the origial gwcs to the cutout
                sliced_gwcs = self._slice_gwcs(cutout, self._gwcs_objects[i])

                # Create the asdf tree
                tree = {self._mission_kwd: {'meta': {'wcs': sliced_gwcs}, 'data': cutout.data}}
                asdf_cutouts.append(asdf.AsdfFile(tree))
            self._asdf_cutouts = asdf_cutouts
        return self._asdf_cutouts

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

    def _load_file_data(self, input_file: Union[str, Path, S3Path]) -> Tuple[np.ndarray, gwcs.wcs.WCS]:
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
        gwcs : `~gwcs.wcs.WCS`
            The GWCS of the image.
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

        # Data in the Cutout2D object is a view into the original data. We need to deep copy the 
        # data in the cutout object to ensure that it is fully independent of the original data 
        # and that it does not take up the same amount of storage.
        img_cutout.data = np.array(img_cutout.data, copy=True)

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
            return

        # Get closest pixel coordinates and approximated WCS
        pixel_coords, wcs = self.get_center_pixel(gwcs, self._coordinates.ra.value, self._coordinates.dec.value)

        # Create the cutout
        try:
            cutout2D = self._get_cutout_data(data, wcs, pixel_coords)
        except NoOverlapError:
            warnings.warn(f'Cutout footprint does not overlap with data in {file}, skipping...',
                          DataWarning)
            return
        
        # Check that there is data in the cutout image
        if (cutout2D.data == 0).all() or (np.isnan(cutout2D.data)).all():
            warnings.warn(f'Cutout of {file} contains no data, skipping...',
                          DataWarning)
            return

        # Convert Quantity data to ndarray
        if isinstance(cutout2D.data, Quantity):
            cutout2D.data = cutout2D.data.value

        # Store the Cutout2D object
        self.cutouts.append(cutout2D)

        # Store the original GWCS to use if creating asdf.AsdfFile objects
        self._gwcs_objects.append(gwcs)

        # Store cutout with filename
        self.cutouts_by_file[file] = [cutout2D]

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
        if not self.cutouts:
            raise InvalidQueryError('Cutout contains no data! (Check image footprint.)')

        # Log total time elapsed
        log.debug('Total time: %.2f sec', monotonic() - start_time)

        return self.cutouts

    def _write_as_format(self, output_format: str, output_dir: Union[str, Path] = '.') -> List[str]:
        """
        Write the cutout to disk in the specified output format.

        Parameters
        ----------
        output_format : str
            The output format to write the cutout to. Options are '.fits' and '.asdf'.
        output_dir : str | Path
            The output directory to write the cutouts to

        Returns
        -------
        cutout_paths : list
            The path(s) to the cutout file(s) or the cutout memory objects.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cutout_paths = []  # List to store paths to cutout files
        for i, file in enumerate(self.cutouts_by_file):
            # Determine the output path
            filename = '{}_{:.7f}_{:.7f}_{}-x-{}_astrocut{}'.format(
                Path(file).stem,
                self._coordinates.ra.value,
                self._coordinates.dec.value,
                str(self._cutout_size[0]).replace(' ', ''), 
                str(self._cutout_size[1]).replace(' ', ''),
                output_format)
            cutout_path = Path(output_dir, filename)

            if output_format == '.fits':
                cutout = self.fits_cutouts[i]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore') 
                    cutout.writeto(cutout_path, overwrite=True, checksum=True)

            elif output_format == '.asdf':
                cutout = self.asdf_cutouts[i]
                cutout.write_to(cutout_path)

            cutout_paths.append(cutout_path.as_posix())

        log.debug('Cutout filepaths: {}'.format(cutout_paths))
        return cutout_paths
    
    def write_as_fits(self, output_dir: Union[str, Path] = '.') -> List[str]:
        """
        Write the cutouts to disk or memory in FITS format.

        Parameters
        ----------
        output_dir : str | Path
            The output directory to write the cutouts to. Defaults to the current directory.

        Returns
        -------
        list
            A list of paths to the cutout FITS files.
        """
        return self._write_as_format(output_format='.fits', output_dir=output_dir)

    def write_as_asdf(self, output_dir: Union[str, Path] = '.') -> List[str]:
        """
        Write the cutouts to disk or memory in ASDF format.

        Parameters
        ----------
        output_dir : str | Path
            The output directory to write the cutouts to. Defaults to the current directory.

        Returns
        -------
        list
            A list of paths to the cutout ASDF files.
        """
        return self._write_as_format(output_format='.asdf', output_dir=output_dir)
    
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
