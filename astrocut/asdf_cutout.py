import sys
import warnings
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import List, Tuple, Union, Optional
from datetime import date

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
from astropy.utils.decorators import deprecated_renamed_argument
from astropy.wcs import WCS
from s3path import S3Path

from . import log, __version__
from .image_cutout import ImageCutout
from .exceptions import DataWarning, ModuleWarning, InvalidQueryError, InvalidInputError


class ASDFCutout(ImageCutout):
    """
    Class for creating cutouts from ASDF files.

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
    key : str
        Optional, default None. Access key ID for S3 file system.
    secret : str
        Optional, default None. Secret access key for S3 file system.
    token : str
        Optional, default None. Security token for S3 file system.
    lite : bool
        Optional, default False. By default, the class creates cutouts of all arrays in the input
        file (e.g., data, error, uncertainty, variance, etc.) where the last two dimensions match the 
        shape of the science data array. It also preserves all of the metadata from the input file.

        If this parameter is True, the cutout will be created in "lite" mode,
        which means that it will only contain the data and an updated world coordinate system.
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
    image_cutouts : list
        List of `~PIL.Image.Image` objects representing the cutouts.

    Methods
    -------
    cutout()
        Generate cutouts from a list of input images.
    write_as_fits(output_dir)
        Write the cutouts to disk or memory in FITS format.
    write_as_asdf(output_dir)
        Write the cutouts to disk or memory in ASDF format.
    """
        
    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, key: Optional[str] = None, secret: Optional[str] = None,
                 token: Optional[str] = None, lite: Optional[bool] = False, verbose: bool = False):
        # Superclass constructor 
        super().__init__(input_files, coordinates, cutout_size, fill_value, verbose=verbose)

        # Must be using Python 3.11 or higher to support stdatamodels and ASDF-in-FITS embedding
        self._py311_or_higher = sys.version_info >= (3, 11)

        # Assign AWS credential attributes
        self._key = key
        self._secret = secret
        self._token = token
        self._mission_kwd = 'roman'

        self.cutouts = []  # Public attribute to hold `Cutout2D` objects
        self._asdf_cutouts = None  # Store ASDF objects
        self._fits_cutouts = None  # Store FITS objects
        self._gwcs_objects = []  # Store original GWCS objects
        self._asdf_trees = []  # Store ASDF trees for each cutout
        self._lite = lite  # Flag for lite mode

        # Make cutouts
        self.cutout()

    @property
    def fits_cutouts(self) -> List[fits.HDUList]:
        """
        Return the cutouts as a list `astropy.io.fits.HDUList` objects.
        """
        if not self._fits_cutouts:
            # Try to import stdatamodels for ASDF-in-FITS embedding
            if self._py311_or_higher:
                try:
                    # Check version of stdatamodels
                    from stdatamodels import __version__ as stdata_version, asdf_in_fits
                    if stdata_version < '4.1.0':
                        warnings.warn(
                            'The `stdatamodels` package is not available in the correct version (>=4.1.0); '
                            'ASDF-in-FITS embedding will be skipped for these cutouts. Install the optional '
                            'dependency with: pip install "astrocut[all]" or pip install stdatamodels>=4.1.0',
                            ModuleWarning
                        )
                        self._can_embed_asdf_in_fits = False
                    else:
                        self._can_embed_asdf_in_fits = True
                except ImportError:
                    warnings.warn(
                        'The `stdatamodels` package cannot be imported; ASDF-in-FITS embedding will be '
                        'skipped for these cutouts. Install the optional dependency with: '
                        'pip install "astrocut[all]" or pip install stdatamodels>=4.1.0',
                        ModuleWarning
                    )
                    self._can_embed_asdf_in_fits = False
            else:
                warnings.warn('ASDF-in-FITS embedding requires Python 3.11 or higher. '
                              'Skipping embedding for these cutouts.', ModuleWarning)
                self._can_embed_asdf_in_fits = False

            fits_cutouts = []
            for i, (file, cutouts) in enumerate(self.cutouts_by_file.items()):                
                cutout = cutouts[0]
                if self._lite:
                    tree = {
                        # Tree should only include sliced WCS and original filename
                        self._mission_kwd: {
                            'meta': {'wcs': self._slice_gwcs(cutout, self._gwcs_objects[i]),
                                     'orig_file': str(file)}
                        }
                    }
                else:
                    tree = self._asdf_trees[i]
                    # Tree should only include meta
                    tree[self._mission_kwd] = {'meta': tree[self._mission_kwd]['meta']}

                # Build the PrimaryHDU with keywords
                primary_hdu = fits.PrimaryHDU()
                primary_hdu.header.extend([('ORIGIN', 'STScI/MAST', 'institution responsible for creating this file'),
                                           ('DATE', str(date.today()), 'file creation date'),
                                           ('PROCVER', __version__, 'software version'),
                                           ('RA_OBJ', self._coordinates.ra.deg, '[deg] right ascension'),
                                           ('DEC_OBJ', self._coordinates.dec.deg, '[deg] declination')])
                
                # Build ImageHDU with cutout data and WCS
                image_hdu = fits.ImageHDU(data=cutout.data, header=cutout.wcs.to_header(relax=True))
                image_hdu.header['ORIG_FLE'] = str(file)  # Add original file to header
                image_hdu.header['EXTNAME'] = 'CUTOUT'
                hdul = fits.HDUList([primary_hdu, image_hdu])

                if self._can_embed_asdf_in_fits:
                    hdul_embed = asdf_in_fits.to_hdulist(tree, hdul)
                else:
                    hdul_embed = hdul
                fits_cutouts.append(hdul_embed)
            self._fits_cutouts = fits_cutouts
        return self._fits_cutouts
    
    @property
    def asdf_cutouts(self) -> List[asdf.AsdfFile]:
        """
        Return the cutouts as a list of `asdf.AsdfFile` objects.
        """
        if not self._asdf_cutouts:
            asdf_cutouts = []
            for i, (file, cutouts) in enumerate(self.cutouts_by_file.items()):
                cutout = cutouts[0]
                if self._lite:
                    tree = self._get_lite_tree(str(file), cutout, self._gwcs_objects[i])
                else:
                    tree = self._asdf_trees[i]

                # Create the AsdfFile object and add history to it
                af = asdf.AsdfFile(tree)
                af.add_history_entry(
                    f'Cutout of size {cutout.shape} at sky coordinates '
                    f'({self._coordinates.ra.value}, {self._coordinates.dec.value})',
                    software={
                        'name': 'astrocut',
                        'author': 'Space Telescope Science Institute',
                        'version': __version__,
                        'homepage': 'https://astrocut.readthedocs.io/en/latest/'
                    }
                )
                asdf_cutouts.append(af)

            self._asdf_cutouts = asdf_cutouts
        return self._asdf_cutouts
    
    def _get_lite_tree(self, file_str: str, cutout: Cutout2D, gwcs: gwcs.wcs.WCS) -> dict:
        """
        Helper function to create an ASDF tree in lite mode.

        Parameters
        ----------
        file_str : str
            The input filename as a string.
        cutout : `~astropy.nddata.Cutout2D`
            The cutout object.
        gwcs : gwcs.wcs.WCS
            The original GWCS object.

        Returns
        -------
        tree : dict
            The ASDF tree in lite mode. The tree contains only the cutout data and the sliced GWCS.
        """
        return {
            self._mission_kwd: {
                'meta': {'wcs': self._slice_gwcs(cutout, gwcs),
                         'orig_file': file_str},
                'data': cutout.data
            }
        }

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

    def _load_file_data(self, input_file: Union[str, Path, S3Path]) -> dict:
        """
        Load relevant data from an input file.

        Parameters
        ----------
        input_file : str | Path | S3Path
            The input file to load data from.

        Returns
        -------
        tree : dict
            The ASDF tree of the input file.
        """
        # If file comes from AWS cloud bucket, get HTTP URL to open with asdf
        if (isinstance(input_file, str) and input_file.startswith('s3://')) or isinstance(input_file, S3Path):
            input_file = self._get_cloud_http(input_file)

        # Get data and GWCS object from ASDF input file
        with asdf.open(input_file) as af:
            tree = deepcopy(af.tree)

        return tree
    
    def _make_cutout(self, array: np.ndarray, position: tuple, wcs: WCS) -> Cutout2D:
        """
        Helper to generate a Cutout2D and return plain ndarray data.

        Parameters
        ----------
        array : np.ndarray
            The input data array.
        position : tuple
            The (x, y) position of the cutout center.
        wcs : WCS
            The WCS object associated with the input array.

        Returns
        -------
        cutout : Cutout2D
            The generated cutout.
        """
        cutout = Cutout2D(
            array,
            position=position,
            wcs=wcs,
            size=(self._cutout_size[1], self._cutout_size[0]),
            mode="partial",
            fill_value=self._fill_value,
            copy=True,
        )

        # Strip units if present
        if isinstance(cutout.data, Quantity):
            cutout.data = cutout.data.value

        return cutout
    
    def _get_cutout_data(self, tree: dict, wcs: WCS, pixel_coords: Tuple[int, int]) -> Cutout2D:
        """
        Get the cutout data from the input image.

        Parameters
        ----------
        tree : dict
            The ASDF tree of the input file.
        wcs : `~astropy.wcs.WCS`
            The approximated WCS of the input image.
        pixel_coords : tuple
            The pixel coordinates closest to the center of the cutout.

        Returns
        -------
        img_cutout : `~astropy.nddata.Cutout2D`
            The cutout object.
        """
        keys = list(tree[self._mission_kwd].keys()) if not self._lite else ['data']
        data_shape = tree[self._mission_kwd]['data'].shape

        data_cutout = None  # Initialize cutout variable
        for key in keys:
            obj = tree[self._mission_kwd][key]
            if not isinstance(obj, np.ndarray):
                continue

            shape = obj.shape
            is_data = key == 'data'

            if shape[-2:] != data_shape[-2:]:
                continue  # Skip arrays not aligned with science data

            log.debug(f'Original {key} shape: {shape}')

            if obj.ndim == 2:
                # Simple 2D cutout
                cutout = self._make_cutout(obj, pixel_coords, wcs if is_data else None)
                tree[self._mission_kwd][key] = cutout.data
                if is_data:
                    data_cutout = cutout
                log.debug(f'{key} cutout shape: {cutout.shape}')

            else:
                # Cube or higher dimension array
                new_shape = obj.shape[:-2] + (self._cutout_size[1], self._cutout_size[0])
                cutout_cube = np.full(new_shape, self._fill_value, dtype=obj.dtype)

                for idx in np.ndindex(obj.shape[:-2]):
                    cutout = self._make_cutout(obj[idx], pixel_coords, None)
                    cutout_cube[idx] = cutout.data

                tree[self._mission_kwd][key] = cutout_cube
                log.debug(f'{key} cutout shape: {cutout_cube.shape}')

        return data_cutout

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
        tmp = deepcopy(gwcs)

        # Get the cutout array bounds and create a new shift transform to the cutout
        # Add the new transform to the gwcs
        slices = cutout.slices_original
        xmin, xmax = slices[1].start, slices[1].stop
        ymin, ymax = slices[0].start, slices[0].stop
        shape = (xmax - xmin, ymax - ymin)
        offsets = models.Shift(xmin, name='cutout_offset1') & models.Shift(ymin, name='cutout_offset2')
        tmp.insert_transform('detector', offsets, after=True)

        # Modify the gwcs bounding box to the cutout shape
        tmp.bounding_box = ((0, shape[0] - 1), (0, shape[1] - 1))
        tmp.pixel_shape = shape
        tmp.array_shape = shape[::-1]
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
        tree = self._load_file_data(file)

        # Skip if the file does not contain a GWCS object
        gwcs = tree[self._mission_kwd]['meta'].get('wcs', None)
        if gwcs is None:
            warnings.warn(f'File {file} does not contain a GWCS object. Skipping...',
                          DataWarning)
            return

        # Get closest pixel coordinates and approximated WCS
        pixel_coords, wcs = get_center_pixel(gwcs, self._coordinates.ra.value, self._coordinates.dec.value)

        # Create the cutout
        try:
            data_cutout = self._get_cutout_data(tree, wcs, pixel_coords)
        except NoOverlapError:
            warnings.warn(f'Cutout footprint does not overlap with data in {file}, skipping...',
                          DataWarning)
            return
        
        # Check that there is data in the cutout image
        if (data_cutout.data == 0).all() or (np.isnan(data_cutout.data)).all():
            warnings.warn(f'Cutout of {file} contains no data, skipping...',
                          DataWarning)
            return

        # Store the Cutout2D object
        self.cutouts.append(data_cutout)

        # Store the original GWCS to use if creating asdf.AsdfFile objects
        self._gwcs_objects.append(gwcs)

        # Store the ASDF tree for this cutout
        file_str = str(file)
        if not self._lite:
            tree[self._mission_kwd]['meta']['wcs'] = self._slice_gwcs(data_cutout, gwcs)
            tree[self._mission_kwd]['meta']['orig_file'] = file_str
            self._asdf_trees.append(tree)

        # Store cutout with filename
        self.cutouts_by_file[file] = [data_cutout]

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
    
    def _make_cutout_filename(self, file: str, output_format: str) -> str:
        """
        Generate a standardized filename for the cutout.

        Overrides the superclass method to include the '_lite' tag if applicable and the output format.

        Parameters
        ----------
        file : str
            The input file name.
        output_format : str
            The output format to write the cutout to. Options are '.fits' and '.asdf'.

        Returns
        -------
        filename : str
            The generated filename for the cutout.
        """
        return '{}_{:.7f}_{:.7f}_{}-x-{}{}_astrocut{}'.format(
            Path(file).stem,
            self._coordinates.ra.value,
            self._coordinates.dec.value,
            str(self._cutout_size[0]).replace(' ', ''), 
            str(self._cutout_size[1]).replace(' ', ''),
            '_lite' if self._lite else '',
            output_format)

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
            filename = self._make_cutout_filename(file, output_format)
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

    def write_as_zip(self, output_dir: Union[str, Path] = '.', filename: Union[str, Path, None] = None,
                     *, output_format: str = '.asdf') -> str:
        """
        Package the ASDF or FITS cutouts into a zip archive without writing intermediates.

        Parameters
        ----------
        output_dir : str | Path, optional
            Directory where the zip will be created. Default '.'.
        filename : str | Path | None, optional
            Name (or path) of the output zip file. If not provided, defaults to
            'astrocut_{ra}_{dec}_{size}.zip'. If provided without a '.zip' suffix,
            the suffix is added automatically.
        output_format : str, optional
            Either '.asdf' (default) or '.fits'. Determines which in-memory representation is zipped.

        Returns
        -------
        str
            Path to the created zip file.
        """
        fmt = output_format.lower().strip()
        fmt = '.' + fmt if not fmt.startswith('.') else fmt
        if fmt not in ('.asdf', '.fits'):
            raise InvalidInputError("File format must be either '.asdf' or '.fits'")

        def build_entries():
            use_fits = fmt == '.fits'
            objs = self.fits_cutouts if use_fits else self.asdf_cutouts

            for i, file in enumerate(self.cutouts_by_file):
                arcname = self._make_cutout_filename(file, fmt)
                yield arcname, objs[i]

        return self._write_cutouts_to_zip(output_dir=output_dir, filename=filename, build_entries=build_entries)
    

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
    row, col = gwcsobj.invert(coordinates, with_bounding_box=False)

    return (row, col), wcs_updated


@deprecated_renamed_argument('output_file', None, '1.0.0', warning_type=DeprecationWarning,
                             message='`output_file` is non-operational and will be removed in a future version.')
def asdf_cut(input_files: List[Union[str, Path, S3Path]], 
             ra: float, 
             dec: float, 
             cutout_size: int = 25,
             output_file: Union[str, Path] = "example_roman_cutout.fits",
             write_file: bool = True, 
             fill_value: Union[int, float] = np.nan,
             output_dir: Union[str, Path] = '.',
             output_format: str = '.asdf', 
             key: str = None,
             secret: str = None, 
             token: str = None,
             lite: bool = False,
             verbose: bool = False) -> Cutout2D:
    """
    Takes one of more ASDF input files (`input_files`) and generates a cutout of designated size `cutout_size`
    around the given coordinates (`coordinates`). The cutout is written to a file or returned as an object.

    This function is maintained for backwards compatibility. For maximum flexibility, we recommend using the
    ``ASDFCutout`` class directly.

    Parameters
    ----------
    input_file : str | Path | S3Path
        The input ASDF file.
    ra : float
        The right ascension of the central cutout.
    dec : float
        The declination of the central cutout.
    cutout_size : int
        Optional, default 25. The image cutout pixel size.
        Note: Odd values for `cutout_size` generally result in a cutout that is more accurately 
        centered on the target coordinates compared to even values, due to the symmetry of the 
        pixel grid. 
    output_file : str | Path
        Optional, default "example_roman_cutout.fits". The name of the output cutout file.
        This parameter is deprecated and will be removed in a future version.
    write_file : bool
        Optional, default True. Flag to write the cutout to a file or not.
    fill_value: int | float
        Optional, default `np.nan`. The fill value for pixels outside the original image.
    output_dir : str | Path
        Optional, default ".". The directory to write the cutout file(s) to.
    output_format : str
        Optional, default ".asdf". The format of the output cutout file. If `write_file` is False,
        then cutouts will be returned as `asdf.AsdfFile` objects if `output_format` is ".asdf" or
        as `astropy.io.fits.HDUList` objects if `output_format` is ".fits".
    key : string
        Default None. Access key ID for S3 file system. Only applicable if `input_file` is a
        cloud resource.
    secret : string
        Default None. Secret access key for S3 file system. Only applicable if `input_file` is a
        cloud resource.
    token : string
        Default None. Security token for S3 file system. Only applicable if `input_file` is a
        cloud resource.
    lite : bool
        Optional, default False. By default, the class creates cutouts of all arrays in the input
        file (e.g., data, error, uncertainty, variance, etc.) where the last two dimensions match the 
        shape of the science data array. It also preserves all of the metadata from the input file.

        If this parameter is True, the cutout will be created in "lite" mode,
        which means that it will only contain the data and an updated world coordinate system.
    verbose : bool
        Default False. If True, intermediate information is printed.

    Returns
    -------
    response : str | list
        A list of cutout file paths if `write_file` is True, otherwise a list of cutout objects.
    """
    asdf_cutout = ASDFCutout(input_files, f'{ra} {dec}', cutout_size, fill_value, key=key, 
                             secret=secret, token=token, lite=lite, verbose=verbose)
    
    if not write_file:  # Returns as Cutout2D objects
        return asdf_cutout.cutouts
    
    # Get output format in standard form
    output_format = f'.{output_format}' if not output_format.startswith('.') else output_format
    output_format = output_format.lower()

    if output_format == '.asdf':
        return asdf_cutout.write_as_asdf(output_dir)
    elif output_format == '.fits':
        return asdf_cutout.write_as_fits(output_dir)
    else:
        # Error if output format not recognized
        raise InvalidInputError(f'Output format {output_format} is not recognized. '
                                'Valid options are ".asdf" and ".fits".')
