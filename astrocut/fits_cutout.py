from datetime import date
from pathlib import Path
from time import monotonic
from typing import List, Literal, Optional, Tuple, Union
import warnings

from astropy import log as astropy_log
from astropy.coordinates import SkyCoord
from astropy.nddata import NoOverlapError
from astropy.io import fits
from astropy.io.fits import HDUList
from astropy.units import Quantity
from astropy.utils.decorators import deprecated_renamed_argument
from astropy.wcs import WCS
import numpy as np
from s3path import S3Path

from .exceptions import DataWarning, InvalidQueryError
from .image_cutout import ImageCutout
from . import __version__, log


class FITSCutout(ImageCutout):
    """
    Class for creating cutouts from FITS files.

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
    extension : int | list | 'all'
        Optional, default None. The extension(s) to cutout from. If None, the first extension with data is used.
    single_outfile : bool
        Optional, default True. If True, all cutouts are written to a single file or HDUList.
    verbose : bool
        If True, log messages are printed to the console.

    Attributes
    ----------
    cutouts_by_file : dict
        The cutouts as a list of `FITSCutout.CutoutInstance` objects stored by input filename.
    fits_cutouts : list
        The cutouts as a list of `astropy.io.fits.HDUList` objects.
    hdu_cutouts_by_file : dict
        The cutouts as `astropy.io.fits.ImageHDU` objects stored by input filename.

    Methods
    -------
    cutout()
        Generate cutouts from a list of input images.
    write_as_fits(output_dir, cutout_prefix)
        Write the cutouts to files in FITS format.
    """
        
    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, limit_rounding_method: str = 'round',
                 extension: Optional[Union[int, List[int], Literal['all']]] = None, 
                 single_outfile: bool = True, verbose: bool = False):
        # Superclass constructor 
        super().__init__(input_files, coordinates, cutout_size, fill_value, limit_rounding_method, verbose)
               
        # If a single extension is given, make it a list
        if isinstance(extension, int):
            extension = [extension]
        self._extension = extension

        # Assigning other attributes
        self._single_outfile = single_outfile
        self._fits_cutouts = None
        self.hdu_cutouts_by_file = {}

        # Make the cutouts upon initialization
        self.cutout()

    def _construct_fits_from_hdus(self, cutout_hdus: List[fits.ImageHDU]) -> fits.HDUList:
        """
        Make one or more cutout HDUs into a single HDUList object.

        Parameters
        ----------
        cutout_hdus : list
            The `~astropy.io.fits.hdu.image.ImageHDU` object(s) to be written to the fits file.

        Returns
        -------
        response : `~astropy.io.fits.HDUList`
            The HDUList object.
        """        
        # Setting up the Primary HDU
        keywords = dict()
        if self._coordinates:
            keywords = {'RA_OBJ': (self._coordinates.ra.deg, '[deg] right ascension'),
                        'DEC_OBJ': (self._coordinates.dec.deg, '[deg] declination')}

        # Build the primary HDU with keywords
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header.extend([('ORIGIN', 'STScI/MAST', 'institution responsible for creating this file'),
                                   ('DATE', str(date.today()), 'file creation date'),
                                   ('PROCVER', __version__, 'software version')])
        for kwd in keywords:
            primary_hdu.header[kwd] = keywords[kwd]

        return fits.HDUList([primary_hdu] + cutout_hdus)

    @property
    def fits_cutouts(self):
        """
        Return the cutouts as a list `astropy.io.fits.HDUList` objects.
        """
        if not self._fits_cutouts:
            fits_cutouts = []
            if self._single_outfile:  # one output file for all input files
                cutout_hdus = [x for file in self.hdu_cutouts_by_file for x in self.hdu_cutouts_by_file[file]]
                fits_cutouts = [self._construct_fits_from_hdus(cutout_hdus)]
            else:  # one output file per input file
                for file, cutout_list in self.hdu_cutouts_by_file.items():
                    fits_cutouts.append(self._construct_fits_from_hdus(cutout_list))
            self._fits_cutouts = fits_cutouts
        return self._fits_cutouts

    def _parse_extensions(self, input_file: Union[str, Path, S3Path], infile_exts: np.ndarray) -> List[int]:
        """
        Given a list of image extensions available in the file with infile_name, cross-match with
        user input extensions to figure out which extensions to use for cutout.

        Parameters
        ----------
        input_file : str | Path | S3Path
            The path to the input file.
        infile_exts : list
            List of image extensions available in the file.

        Returns
        -------
        cutout_exts : list
            List of extensions to be cutout.
        """
        # Skip files with no image data
        if len(infile_exts) == 0:
            warnings.warn(f'No image extensions with data found in {input_file}, skipping...', DataWarning)
            return []
                
        if self._extension is None:
            cutout_exts = infile_exts[:1]  # Take the first image extension
        elif self._extension == 'all':
            cutout_exts = infile_exts  # Take all the extensions
        else:  # User input extentions
            cutout_exts = [x for x in infile_exts if x in self._extension]
            if len(cutout_exts) < len(self._extension):
                empty_exts = ','.join([str(x) for x in self._extension if x not in cutout_exts])
                warnings.warn(f'Not all requested extensions in {input_file} are image extensions or have '
                              f'data, extension(s) {empty_exts} will be skipped.', DataWarning)

        return cutout_exts

    def _load_file_data(self, input_file: Union[str, Path, S3Path]) -> Tuple[fits.HDUList, List[int]]:
        """
        Load the data from an input file and determine which extension(s) to cutout from.

        Parameters
        ----------
        input_file : str | Path | S3Path
            The path to the input file.

        Returns
        --------
        hdulist : `~astropy.io.fits.HDUList`
            The HDU list for the input file.
        cutout_inds : list
            The indices of the extension(s) to cutout from.
        """
        # Account for cloud-hosted files
        fsspec_kwargs = {'anon': True} if 's3://' in input_file else None

        # Open the file
        hdulist = fits.open(input_file, mode='denywrite', memmap=True, fsspec_kwargs=fsspec_kwargs)

        # Sorting out which extension(s) to cutout
        infile_exts = np.where([hdu.is_image and hdu.size > 0 for hdu in hdulist])[0]
        cutout_inds = self._parse_extensions(input_file, infile_exts)

        return (hdulist, cutout_inds)
    
    def _get_img_wcs(self, hdu_header: fits.Header) -> Tuple[WCS, bool]:
        """
        Get the WCS for an image.

        Parameters
        ----------
        hdu_header : `~astropy.io.fits.Header`
            The header for the image HDU.
        
        Returns
        --------
        img_wcs : `~astropy.wcs.WCS`
            The WCS for the image.
        no_sip : bool
            Whether the image WCS has no SIP information.
        """
        # We are going to reroute the logging to a string stream temporarily so we can
        # intercept any message from astropy, chiefly the "Inconsistent SIP distortion information"
        # INFO message which will indicate that we need to remove existing SIP keywords
        # from a WCS whose CTYPE does not include SIP. In this we are taking the CTYPE to be
        # correct and adjusting the header keywords to match.
        hdlrs = astropy_log.handlers
        astropy_log.handlers = []
        with astropy_log.log_to_list() as log_list:        
            img_wcs = WCS(hdu_header, relax=True)

        for hd in hdlrs:
            astropy_log.addHandler(hd)

        no_sip = False
        if (len(log_list) > 0):
            if ('Inconsistent SIP distortion information' in log_list[0].msg):
                # Remove sip coefficients
                img_wcs.sip = None
                no_sip = True
            else:  # Message(s) we didn't prepare for we want to go ahead and display
                for log_rec in log_list:
                    astropy_log.log(log_rec.levelno, log_rec.msg, extra={'origin': log_rec.name})

        return (img_wcs, no_sip)

    def _hducut(self, cutout_data: np.ndarray, cutout_wcs: WCS, hdu_header: fits.Header, no_sip: bool,
                ind: int, primary_filename: fits.Header, is_empty: bool) -> fits.ImageHDU:
        """
        Create a cutout HDU from an image HDU.

        Parameters
        ----------
        cutout_data : `numpy.ndarray`
            The cutout data.
        img_wcs : `~astropy.wcs.WCS`
            The WCS for the image.
        hdu_header : `~astropy.io.fits.Header`
            The header for the image HDU.
        no_sip : bool   
            Whether the image WCS has no SIP information.
        ind : int
            The index of the extension in the original file.
        primary_filename : str
            The filename in the header of the primary HDU.
        is_empty : bool
            Indicates if the cutout has no image data.

        Returns
        -------
        response : `~astropy.io.fits.ImageHDU`
            The cutout HDU.
        """
        # Updating the header with the new wcs info
        if no_sip:
            hdu_header.update(cutout_wcs.to_header(relax=False))
        else:
            hdu_header.update(cutout_wcs.to_header(relax=True))  # relax arg is for sip distortions if they exist

        # Naming the extension and preserving the original name
        hdu_header['O_EXT_NM'] = (hdu_header.get('EXTNAME'), 'Original extension name.')
        hdu_header['EXTNAME'] = 'CUTOUT'

        # Moving the filename, if present, into the ORIG_FLE keyword
        hdu_header['ORIG_FLE'] = (hdu_header.get('FILENAME'), 'Original image filename.')
        hdu_header.remove('FILENAME', ignore_missing=True)

        # Check that there is data in the cutout image
        if is_empty:
            hdu_header['EMPTY'] = (True, 'Indicates no data in cutout image.')

        # Create the cutout HDU
        cutout_hdu = fits.ImageHDU(header=hdu_header, data=cutout_data)

        # Adding a few more keywords
        cutout_hdu.header['ORIG_EXT'] = (ind, 'Extension in original file.')
        if not cutout_hdu.header.get('ORIG_FLE') and primary_filename:
            cutout_hdu.header['ORIG_FLE'] = primary_filename

        return cutout_hdu

    def _cutout_file(self, file: Union[str, Path, S3Path]):
        """
        Create cutouts from a single file.

        Parameters
        ----------
        file : str | Path | S3Path
            The path to the file.
        """
        # Load data
        hdulist, cutout_inds = self._load_file_data(file)

        if not len(cutout_inds):  # No image extensions with data were found
            hdulist.close()
            return

        # Create HDU cutouts
        cutouts = []
        fits_cutouts = []
        num_empty = 0
        for ind in cutout_inds:
            try:
                # Get HDU, header, and WCS
                img_hdu = hdulist[ind] 
                hdu_header = fits.Header(img_hdu.header, copy=True)
                img_wcs, no_sip = self._get_img_wcs(hdu_header)
                primary_filename = hdulist[0].header.get('FILENAME')

                # Create the cutout
                # Eventually, this will be replaced by a call to Cutout2D
                cutout = self.CutoutInstance(img_hdu.section, img_wcs, self)

                # Save the cutout data to use when outputting as an image
                # Eventually, the values here will be a list of Cutout2D objects
                is_empty = (cutout.data == 0).all() or (np.isnan(cutout.data)).all()
                if is_empty:
                    num_empty += 1
                else:
                    cutouts.append(cutout)

                # Also save the cutouts as ImageHDU objects for FITS output
                fits_cutouts.append(self._hducut(cutout.data, cutout.wcs, hdu_header, no_sip, ind, 
                                                 primary_filename, is_empty))

            except OSError as err:
                warnings.warn(f'Error {err} encountered when performing cutout on {file}, '
                              f'extension {ind}, skipping...', DataWarning)
                num_empty += 1
            except NoOverlapError:
                warnings.warn(f'Cutout footprint does not overlap with data in {file}, '
                              f'extension {ind}, skipping...', DataWarning)
                num_empty += 1
            except InvalidQueryError:
                warnings.warn(f'Cutout footprint does not overlap with data in {file}, '
                              f'extension {ind}, skipping...', DataWarning)
                num_empty += 1
            except ValueError as err:
                if 'Input position contains invalid values' in str(err):
                    warnings.warn(f'Cutout footprint does not overlap with data in {file}, '
                                  f'extension {ind}, skipping...', DataWarning)
                    num_empty += 1
                else:
                    raise
        
        # Close HDUList
        hdulist.close()

        if num_empty == len(cutout_inds):  # No extensions have cutout data
            warnings.warn(f'Cutout of {file} contains no data, skipping...', DataWarning)
        else:  # At least one extension has cutout data
            # Save cutouts
            self.cutouts_by_file[file] = cutouts
            self.hdu_cutouts_by_file[file] = fits_cutouts

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
        if not self.cutouts_by_file:
            raise InvalidQueryError('Cutout contains no data! (Check image footprint.)')

        # Log total time elapsed
        log.debug('Total time: %.2f sec', monotonic() - start_time)

        return self.fits_cutouts

    def write_as_fits(self, output_dir: Union[str, Path] = '.', cutout_prefix: str = 'cutout') -> List[str]:
        """
        Write the cutouts to memory or to a file in FITS format.

        Returns
        -------
        cutout_paths : list
            A list of paths to the cutout FITS files.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if self._single_outfile:  # one output file for all input files
            log.debug('Returning cutout as a single FITS file.')

            cutout_fits = self.fits_cutouts[0]
            filename = '{}_{:.7f}_{:.7f}_{}-x-{}_astrocut.fits'.format(
                cutout_prefix,
                self._coordinates.ra.value,
                self._coordinates.dec.value,
                str(self._cutout_size[0]).replace(' ', ''),
                str(self._cutout_size[1]).replace(' ', ''))
            cutout_path = Path(output_dir, filename)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore') 
                cutout_fits.writeto(cutout_path, overwrite=True, checksum=True)
            # Return file path or memory object
            return [cutout_path.as_posix()]
    
        else:  # one output file per input file
            log.debug('Returning cutouts as individual FITS files.')

            cutout_paths = []
            for i, file in enumerate(self.hdu_cutouts_by_file):
                cutout_fits = self.fits_cutouts[i]
                filename = '{}_{:.7f}_{:.7f}_{}-x-{}_astrocut.fits'.format(
                    Path(file).stem,
                    self._coordinates.ra.value,
                    self._coordinates.dec.value,
                    str(self._cutout_size[0]).replace(' ', ''), 
                    str(self._cutout_size[1]).replace(' ', ''))
                cutout_path = Path(output_dir, filename)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    cutout_fits.writeto(cutout_path, overwrite=True, checksum=True)
                # Append file path or memory object
                cutout_paths.append(cutout_path.as_posix())

            log.debug('Cutout filepaths: {}'.format(cutout_paths))
            return cutout_paths
        
    class CutoutInstance:
        """
        Represents an individual cutout with its own data and WCS. Eventually, this will be replaced
        by `astropy.nddata.Cutout2D` objects.

        Parameters
        ----------
        img_data : `~astropy.io.fits.Section`
            The data for the image.
        img_wcs : `~astropy.wcs.WCS`
            The WCS for the image.
        parent : `FITSCutout`
            The parent FITSCutout object.

        Attributes
        ----------
        data : `numpy.ndarray`
            The cutout data array.
        shape : tuple
            The shape of the cutout data array.
        shape_input : tuple
            The shape of the input image data array.
        slices_original : tuple
            Slices for the minimal bounding box of the cutout with respect to the original array.
        xmin_original : int
            The minimum x value of the cutout in the original array.
        xmax_original : int
            The maximum x value of the cutout in the original array.
        ymin_original : int
            The minimum y value of the cutout in the original array.
        ymax_original : int
            The maximum y value of the cutout in the original array.
        wcs : `~astropy.wcs.WCS`
            The WCS for the cutout.
        """

        def __init__(self, img_data: fits.Section, img_wcs: WCS, parent: 'FITSCutout'):
            # Calculate cutout limits
            cutout_lims = parent._get_cutout_limits(img_wcs)

            # Extract data from Section
            self.data = self._get_cutout_data(img_data, cutout_lims, parent)
            self.shape = self.data.shape
            self.shape_input = img_data.shape

            self.wcs = self._get_cutout_wcs(img_wcs, cutout_lims)
        
        def _get_cutout_data(self, data: fits.Section, cutout_lims: np.ndarray, 
                             parent: 'FITSCutout') -> np.ndarray:
            """
            Extract the cutout data from an image.

            Parameters
            ----------
            data : `~astropy.io.fits.Section`
                The data for the image.
            cutout_lims : `numpy.ndarray`
                The cutout pixel limits in an array of the form [[ymin,ymax],[xmin,xmax]]
            parent : `FITSCutout`
                The parent FITSCutout object. Needed for access to certain attributes.

            Returns
            --------
            cutout_data : `numpy.ndarray`
                The cutout data.
            """
            log.debug('Original image shape: %s', data.shape)

            # Get the limits for the cutout
            # These limits are not guaranteed to be within the image footprint
            (xmin, xmax), (ymin, ymax) = cutout_lims
            ymax_img, xmax_img = data.shape

            # Check the cutout is on the image
            if (xmax <= 0) or (xmin >= xmax_img) or (ymax <= 0) or (ymin >= ymax_img):
                raise InvalidQueryError('Cutout location is not in image footprint!')

            # Adjust limits to fit within image bounds
            xmin_clipped, xmax_clipped = max(0, xmin), min(xmax_img, xmax)
            ymin_clipped, ymax_clipped = max(0, ymin), min(ymax_img, ymax)

            # Compute padding required (before and after in x and y)
            padding = np.array([
                (max(0, -ymin), max(0, ymax - ymax_img)),  # (top, bottom)
                (max(0, -xmin), max(0, xmax - xmax_img))   # (left, right)
            ])
            
            # Extract the cutout
            img_cutout = data[ymin_clipped:ymax_clipped, xmin_clipped:xmax_clipped]

            # Assign relevant attributes
            self.slices_original = slice(ymin_clipped, ymax_clipped + 1), slice(xmin_clipped, xmax_clipped + 1)
            self.xmin_original, self.xmax_original = xmin_clipped, xmax_clipped
            self.ymin_original, self.ymax_original = ymin_clipped, ymax_clipped

            # Adding padding to the cutout so that it's the expected size
            if padding.any():  # only do if we need to pad
                img_cutout = np.pad(img_cutout, padding, 'constant', constant_values=parent._fill_value)

            log.debug('Image cutout shape: %s', img_cutout.shape)

            return img_cutout

        def _get_cutout_wcs(self, img_wcs: WCS, cutout_lims: np.ndarray) -> WCS:
            """
            Starting with the full image WCS and adjusting it for the cutout WCS.
            Adjusts CRPIX values and adds physical WCS keywords.

            Parameters
            ----------
            img_wcs : `~astropy.wcs.WCS`
                WCS for the image the cutout is being cut from.
            cutout_lims : `numpy.ndarray`
                The cutout pixel limits in an array of the form [[ymin,ymax],[xmin,xmax]]

            Returns
            --------
            response :  `~astropy.wcs.WCS`
                The cutout WCS object including SIP distortions if present.
            """
            # relax = True is important when the WCS has sip distortions, otherwise it has no effect
            wcs_header = img_wcs.to_header(relax=True) 

            # Adjusting the CRPIX values
            wcs_header['CRPIX1'] -= cutout_lims[0, 0]
            wcs_header['CRPIX2'] -= cutout_lims[1, 0]

            # Adding the physical WCS keywords
            wcs_header.set('WCSNAMEP', 'PHYSICAL', 'name of world coordinate system alternate P')
            wcs_header.set('WCSAXESP', 2, 'number of WCS physical axes')
            wcs_header.set('CTYPE1P', 'RAWX', 'physical WCS axis 1 type CCD col')
            wcs_header.set('CUNIT1P', 'PIXEL', 'physical WCS axis 1 unit')
            wcs_header.set('CRPIX1P', 1, 'reference CCD column')
            wcs_header.set('CRVAL1P', cutout_lims[0, 0] + 1, 'value at reference CCD column')
            wcs_header.set('CDELT1P', 1.0, 'physical WCS axis 1 step')
            wcs_header.set('CTYPE2P', 'RAWY', 'physical WCS axis 2 type CCD col')
            wcs_header.set('CUNIT2P', 'PIXEL', 'physical WCS axis 2 unit')
            wcs_header.set('CRPIX2P', 1, 'reference CCD row')
            wcs_header.set('CRVAL2P', cutout_lims[1, 0] + 1, 'value at reference CCD row')
            wcs_header.set('CDELT2P', 1.0, 'physical WCS axis 2 step')
            
            return WCS(wcs_header)
        

@deprecated_renamed_argument('correct_wcs', None, '1.0.0', warning_type=DeprecationWarning,
                             message='`correct_wcs` is non-operational and will be removed in a future version.')
def fits_cut(input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
             cutout_size: Union[int, np.ndarray, Quantity, List[int], Tuple[int]] = 25,
             correct_wcs: bool = False, extension: Optional[Union[int, List[int], Literal['all']]] = None,
             single_outfile: bool = True, cutout_prefix: str = 'cutout', output_dir: Union[str, Path] = '.', 
             memory_only: bool = False, fill_value: Union[int, float] = np.nan, limit_rounding_method: str = 'round',
             verbose=False) -> Union[str, List[str], List[HDUList]]:
    """
    Takes one or more FITS files with the same WCS/pointing, makes the same cutout in each file,
    and returns the result either in a single FITS file with one cutout per extension or in 
    individual fits files. The memory_only flag allows the cutouts to be returned as 
    `~astropy.io.fits.HDUList` objects rather than saving to disk.

    Note: No checking is done on either the WCS pointing or pixel scale. If images don't line up
    the cutouts will also not line up.

    This function is maintained for backwards compatibility. For maximum flexibility, we recommend using
    ``FITSCutout`` directly.

    Parameters
    ----------
    input_files : list
        List of fits image files to cutout from. The image is assumed to be in the first extension.
    coordinates : str or `~astropy.coordinates.SkyCoord` object
        The position around which to cutout. It may be specified as a string ("ra dec" in degrees) 
        or as the appropriate `~astropy.coordinates.SkyCoord` object.
    cutout_size : int, array-like, `~astropy.units.Quantity`
        The size of the cutout array. If ``cutout_size`` is a scalar number or a scalar 
        `~astropy.units.Quantity`, then a square cutout of ``cutout_size`` will be created.  
        If ``cutout_size`` has two elements, they should be in ``(ny, nx)`` order.  Scalar numbers 
        in ``cutout_size`` are assumed to be in units of pixels. `~astropy.units.Quantity` objects 
        must be in pixel or angular units.
    extension : int, list of ints, None, or 'all'
       Optional, default None. Default is to cutout the first extension that has image data.
       The user can also supply one or more extensions to cutout from (integers), or 'all'.
    single_outfile : bool 
        Default True. If true return all cutouts in a single fits file with one cutout per extension,
        if False return cutouts in individual fits files. If returing a single file the filename will 
        have the form: <cutout_prefix>_<ra>_<dec>_<size x>_<size y>.fits. If returning multiple files
        each will be named: <original filemame base>_<ra>_<dec>_<size x>_<size y>.fits.
    cutout_prefix : str 
        Default value "cutout". Only used if single_outfile is True. A prefix to prepend to the cutout 
        filename. 
    output_dir : str
        Default value '.'. The directory to save the cutout file(s) to.
    memory_only : bool
        Default value False. If set to true, instead of the cutout file(s) being written to disk
        the cutout(s) are returned as a list of `~astropy.io.fit.HDUList` objects. If set to
        True cutout_prefix and output_dir are ignored, however single_outfile can still be used to
        set the number of returned `~astropy.io.fits.HDUList` objects.
    fill_value : int | float
        Value to fill the cutout with if the cutout is outside the image.
    limit_rounding_method : str
        Method to use for rounding the cutout limits. Options are 'round', 'ceil', and 'floor'.
    verbose : bool
        Default False. If true intermediate information is printed.

    Returns
    -------
    response : str or list
        If single_outfile is True, returns the single output filepath. Otherwise, returns a list of all 
        the output filepaths.
        If memory_only is True, a list of `~astropy.io.fit.HDUList` objects is returned instead of
        file name(s).
    """
    fits_cutout = FITSCutout(input_files, coordinates, cutout_size, fill_value, limit_rounding_method, 
                             extension, single_outfile, verbose)
    
    if memory_only:
        return fits_cutout.fits_cutouts
    
    cutout_paths = fits_cutout.write_as_fits(output_dir, cutout_prefix)
    return cutout_paths[0] if len(cutout_paths) == 1 else cutout_paths


def img_cut(input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
            cutout_size: Union[int, np.ndarray, Quantity, List[int], Tuple[int]] = 25, stretch: str = 'asinh', 
            minmax_percent: Optional[List[int]] = None, minmax_value: Optional[List[int]] = None, 
            invert: bool = False, img_format: str = '.jpg', colorize: bool = False,
            cutout_prefix: str = 'cutout', output_dir: Union[str, Path] = '.', 
            extension: Optional[Union[int, List[int], Literal['all']]] = None, fill_value: Union[int, float] = np.nan,
            limit_rounding_method: str = 'round', verbose=False) -> Union[str, List[str]]:
    """
    Takes one or more fits files with the same WCS/pointing, makes the same cutout in each file,
    and returns the result either as a single color image or in individual image files.

    Note: No checking is done on either the WCS pointing or pixel scale. If images don't line up
    the cutouts will also not line up.

    This function is maintained for backwards compatibility. For maximum flexibility, we recommend using
    ``FITSCutout`` directly.

    Parameters
    ----------
    input_files : list
        List of fits image files to cutout from. The image is assumed to be in the first extension.
    coordinates : str or `~astropy.coordinates.SkyCoord` object
        The position around which to cutout. It may be specified as a string ("ra dec" in degrees) 
        or as the appropriate `~astropy.coordinates.SkyCoord` object.
    cutout_size : int, array-like, `~astropy.units.Quantity`
        The size of the cutout array. If ``cutout_size`` is a scalar number or a scalar 
        `~astropy.units.Quantity`, then a square cutout of ``cutout_size`` will be created.  
        If ``cutout_size`` has two elements, they should be in ``(ny, nx)`` order.  Scalar numbers 
        in ``cutout_size`` are assumed to be in units of pixels. `~astropy.units.Quantity` objects 
        must be in pixel or angular units.
    stretch : str
        Optional, default 'asinh'. The stretch to apply to the image array.
        Valid values are: asinh, sinh, sqrt, log, linear
    minmax_percent : array
        Optional, default [0.5,99.5]. Interval based on a keeping a specified fraction of pixels 
        (can be asymmetric) when scaling the image. The format is [lower percentile, upper percentile], 
        where pixel values below the lower percentile and above the upper percentile are clipped.
        Only one of minmax_percent and minmax_value should be specified.
    minmax_value : array
        Optional. Interval based on user-specified pixel values when scaling the image.
        The format is [min value, max value], where pixel values below the min value and above
        the max value are clipped.
        Only one of minmax_percent and minmax_value should be specified.
    invert : bool
        Optional, default False.  If True the image is inverted (light pixels become dark and vice versa).
    img_format : str
        Optional, default 'jpg'. The output image file type. Valid values are "jpg" and "png".
    colorize : bool
        Optional, default False.  If True a single color image is produced as output, and it is expected
        that three files are given as input.
    cutout_prefix : str 
        Default value "cutout". Only used when producing a color image. A prefix to prepend to the 
        cutout filename. 
    output_dir : str
        Defaul value '.'. The directory to save the cutout file(s) to.
    fill_value : int | float
        Value to fill the cutout with if the cutout is outside the image.
    limit_rounding_method : str
        Method to use for rounding the cutout limits. Options are 'round', 'ceil', and 'floor'.
    extension : int, list of ints, None, or 'all'
        Optional, default None. Default is to cutout the first extension that has image data.
        The user can also supply one or more extensions to cutout from (integers), or "all".
    verbose : bool
        Default False. If true intermediate information is printed.

    Returns
    -------
    response : str or list
        If colorize is True, returns the single output filepath. Otherwise, returns a list of all 
        the output filepaths.
    """

    fits_cutout = FITSCutout(input_files, coordinates, cutout_size, fill_value, limit_rounding_method, 
                             extension, verbose=verbose)
    
    cutout_paths = fits_cutout.write_as_img(stretch, minmax_percent, minmax_value, invert, colorize, img_format,
                                            output_dir, cutout_prefix)
    return cutout_paths
