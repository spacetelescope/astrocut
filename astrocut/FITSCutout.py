from datetime import date
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
import warnings

from astropy import log as astropy_log
from astropy.coordinates import SkyCoord
from astropy.nddata import NoOverlapError
from astropy.io import fits
from astropy.units import Quantity
from astropy.wcs import WCS
import numpy as np
from s3path import S3Path

from .exceptions import DataWarning, InputWarning, InvalidQueryError
from .ImageCutout import ImageCutout
from . import __version__, log


class FITSCutout(ImageCutout):
    """
    Class for creating cutouts from FITS files.

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
    extension : int | list | 'all'
        Optional, default None. The extension(s) to cutout from. If None, the first extension with data is used.
    single_outfile : bool
        Optional, default True. If True, all cutouts are written to a single file or HDUList.
    verbose : bool
        If True, log messages are printed to the console.

    Methods
    -------
    _parse_extensions()
        Determine which extension(s) to cutout from.
    _load_file_data()
        Load the data from an input file.
    _get_img_wcs()
        Get the WCS for an image.
    _get_cutout_data()
        Get the cutout data from an image.
    _get_cutout_wcs()
        Get the WCS for a cutout.
    _hducut()
        Create a cutout HDU from an image HDU.
    _cutout_file()
        Cutout an image file.
    _construct_fits_from_hdus()
        Make one or more cutout HDUs into a single HDUList object.
    _write_to_memory()
        Write the cutouts to memory.
    _write_as_fits()
        Write the cutouts to a file in FITS format.
    _write_as_asdf()
        Write the cutouts to a file in ASDF format.
    """

    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, memory_only: bool = False,
                 output_dir: Union[str, Path] = '.', limit_rounding_method: str = 'round', stretch: str = 'asinh', 
                 minmax_percent: Optional[List[int]] = None, minmax_value: Optional[List[int]] = None, 
                 invert: bool = False, colorize: bool = False, output_format: str = 'fits', 
                 cutout_prefix: str = 'cutout', extension: Optional[Union[int, List[int], Literal['all']]] = None, 
                 single_outfile: bool = True, verbose: bool = True):
        super().__init__(input_files, coordinates, cutout_size, fill_value, memory_only, output_dir, 
                         limit_rounding_method, stretch, minmax_percent, minmax_value, invert, colorize, 
                         output_format, cutout_prefix, verbose)        
        # If a single extension is given, make it a list
        if isinstance(extension, int):
            extension = [extension]
        self._extension = extension

        # Assigning other attributes
        self._single_outfile = single_outfile
        self._cutout_filenames = []

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
            warnings.warn(f"No image extensions with data found in {input_file}, skipping...",
                          DataWarning)
            return []
                
        if self._extension is None:
            cutout_exts = infile_exts[:1]  # Take the first image extension
        elif self._extension == 'all':
            cutout_exts = infile_exts  # Take all the extensions
        else:  # User input extentions
            cutout_exts = [x for x in infile_exts if x in self._extension]
            if len(cutout_exts) < len(self._extension):
                warnings.warn((f"Not all requested extensions in {input_file} are image extensions or have "
                               f"data, extension(s) {','.join([x for x in self._extension if x not in cutout_exts])}"
                               " will be skipped."), DataWarning)

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
            if ("Inconsistent SIP distortion information" in log_list[0].msg):
                # Remove sip coefficients
                img_wcs.sip = None
                no_sip = True
            else:  # Message(s) we didn't prepare for we want to go ahead and display
                for log_rec in log_list:
                    astropy_log.log(log_rec.levelno, log_rec.msg, extra={"origin": log_rec.name})

        return (img_wcs, no_sip)
    
    def _get_cutout_data(self, data: fits.Section, wcs: WCS) -> np.ndarray:
        """
        Get the cutout data from an image.

        Parameters
        ----------
        data : `~astropy.io.fits.Section`
            The data for the image.
        wcs : `~astropy.wcs.WCS`
            The WCS for the image.

        Returns
        --------
        cutout_data : `numpy.ndarray`
            The cutout data.
        """
        log.debug("Original image shape: %s", data.shape)

        # Get the limits for the cutout
        # These limits are not guaranteed to be within the image footprint
        cutout_lims = self._get_cutout_limits(wcs)
        xmin, xmax = cutout_lims[0]
        ymin, ymax = cutout_lims[1]
        ymax_img, xmax_img = data.shape

        # Check the cutout is on the image
        if (xmax <= 0) or (xmin >= xmax_img) or (ymax <= 0) or (ymin >= ymax_img):
            raise InvalidQueryError("Cutout location is not in image footprint!")

        # Adjust limits and figure out the padding
        padding = np.zeros((2, 2), dtype=int)
        if xmin < 0:
            padding[1, 0] = -xmin
            xmin = 0
        if ymin < 0:
            padding[0, 0] = -ymin
            ymin = 0
        if xmax > xmax_img:
            padding[1, 1] = xmax - xmax_img
            xmax = xmax_img
        if ymax > ymax_img:
            padding[0, 1] = ymax - ymax_img
            ymax = ymax_img  
            
        img_cutout = data[ymin:ymax, xmin:xmax]

        # Adding padding to the cutout so that it's the expected size
        if padding.any():  # only do if we need to pad
            img_cutout = np.pad(img_cutout, padding, 'constant', constant_values=self._fill_value)

        log.debug("Image cutout shape: %s", img_cutout.shape)

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
        wcs_header["CRPIX1"] -= cutout_lims[0, 0]
        wcs_header["CRPIX2"] -= cutout_lims[1, 0]

        # Adding the physical WCS keywords
        wcs_header.set("WCSNAMEP", "PHYSICAL", "name of world coordinate system alternate P")
        wcs_header.set("WCSAXESP", 2, "number of WCS physical axes")
        wcs_header.set("CTYPE1P", "RAWX", "physical WCS axis 1 type CCD col")
        wcs_header.set("CUNIT1P", "PIXEL", "physical WCS axis 1 unit")
        wcs_header.set("CRPIX1P", 1, "reference CCD column")
        wcs_header.set("CRVAL1P", cutout_lims[0, 0] + 1, "value at reference CCD column")
        wcs_header.set("CDELT1P", 1.0, "physical WCS axis 1 step")
        wcs_header.set("CTYPE2P", "RAWY", "physical WCS axis 2 type CCD col")
        wcs_header.set("CUNIT2P", "PIXEL", "physical WCS axis 2 unit")
        wcs_header.set("CRPIX2P", 1, "reference CCD row")
        wcs_header.set("CRVAL2P", cutout_lims[1, 0] + 1, "value at reference CCD row")
        wcs_header.set("CDELT2P", 1.0, "physical WCS axis 2 step")
        
        return WCS(wcs_header)

    def _hducut(self, img_hdu: fits.ImageHDU, img_wcs: WCS, hdu_header: fits.Header, no_sip: bool) -> fits.ImageHDU:
        """
        Create a cutout HDU from an image HDU.

        Parameters
        ----------
        img_hdu : `~astropy.io.fits.ImageHDU`
            The image HDU to cutout from.
        img_wcs : `~astropy.wcs.WCS`
            The WCS for the image.
        hdu_header : `~astropy.io.fits.Header`
            The header for the image HDU.
        no_sip : bool   
            Whether the image WCS has no SIP information.

        Returns
        -------
        response : `~astropy.io.fits.ImageHDU`
            The cutout HDU.
        """
        # Get the data for the cutout
        img_cutout = self._get_cutout_data(img_hdu.section, img_wcs)

        # Get the cutout WCS
        # cutout_wcs = img_cutout.wcs
        cutout_wcs = self._get_cutout_wcs(img_wcs, self._get_cutout_limits(img_wcs))

        # Updating the header with the new wcs info
        if no_sip:
            hdu_header.update(cutout_wcs.to_header(relax=False))
        else:
            hdu_header.update(cutout_wcs.to_header(relax=True))  # relax arg is for sip distortions if they exist

        # Naming the extension and preserving the original name
        hdu_header["O_EXT_NM"] = (hdu_header.get("EXTNAME"), "Original extension name.")
        hdu_header["EXTNAME"] = "CUTOUT"

        # Moving the filename, if present, into the ORIG_FLE keyword
        hdu_header["ORIG_FLE"] = (hdu_header.get("FILENAME"), "Original image filename.")
        hdu_header.remove("FILENAME", ignore_missing=True)

        # Check that there is data in the cutout image
        if (img_cutout == 0).all() or (np.isnan(img_cutout)).all():
            hdu_header["EMPTY"] = (True, "Indicates no data in cutout image.")
            self._num_empty += 1

        return fits.ImageHDU(header=hdu_header, data=img_cutout)

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

        # Create HDU cutouts
        cutouts = []
        self._num_cutouts += len(cutout_inds)
        for ind in cutout_inds:
            try:
                # Get HDU, header, and WCS
                img_hdu = hdulist[ind] 
                hdu_header = fits.Header(img_hdu.header, copy=True)
                img_wcs, no_sip = self._get_img_wcs(hdu_header)

                if self._output_format == '.fits':
                    # Make a cutout hdu
                    cutout = self._hducut(img_hdu, img_wcs, hdu_header, no_sip)

                    # Adding a few more keywords
                    cutout.header["ORIG_EXT"] = (ind, "Extension in original file.")
                    if not cutout.header.get("ORIG_FLE") and hdulist[0].header.get("FILENAME"):
                        cutout.header["ORIG_FLE"] = hdulist[0].header.get("FILENAME")
                else:
                    # We only need the data array for images
                    cutout = self._get_cutout_data(img_hdu.section, img_wcs)

                    # Apply the appropriate normalization parameters
                    cutout = self.normalize_img(cutout, self._stretch, self._minmax_percent, self._minmax_value, 
                                                self._invert)

                    if (cutout == 0).all():
                        continue

                cutouts.append(cutout)
            except OSError as err:
                warnings.warn((f"Error {err} encountered when performing cutout on {file}, "
                               f"extension {ind}, skipping..."), DataWarning)
                self._num_empty += 1
            except NoOverlapError:
                warnings.warn((f"Cutout footprint does not overlap with data in {file}, "
                               f"extension {ind}, skipping..."), DataWarning)
                self._num_empty += 1
            except ValueError as err:
                if "Input position contains invalid values" in str(err):
                    warnings.warn((f"Cutout footprint does not overlap with data in {file}, "
                                   f"extension {ind}, skipping..."), DataWarning)
                    self._num_empty += 1
                else:
                    raise
        
        # Close HDUList
        hdulist.close()

        # Save cutouts
        self._cutout_dict[file] = cutouts

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
            keywords = {"RA_OBJ": (self._coordinates.ra.deg, '[deg] right ascension'),
                        "DEC_OBJ": (self._coordinates.dec.deg, '[deg] declination')}

        # Build the primary HDU with keywords
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header.extend([("ORIGIN", 'STScI/MAST', "institution responsible for creating this file"),
                                   ("DATE", str(date.today()), "file creation date"),
                                   ('PROCVER', __version__, 'software version')])
        for kwd in keywords:
            primary_hdu.header[kwd] = keywords[kwd]

        return fits.HDUList([primary_hdu] + cutout_hdus)

    def _write_to_memory(self) -> List[fits.HDUList]:
        """
        Return the cutouts as a list of HDUList objects.

        Returns
        -------
        cutout_fits : list
            List of `~astropy.io.fits.HDUList` objects.
        """
        if self._single_outfile:
            # Collect all cutout HDUs into a single HDUList object
            cutout_hdus = [x for file in self._cutout_dict for x in self._cutout_dict[file]]
            return [self._construct_fits_from_hdus(cutout_hdus)]
        else:
            # Write cutouts for each input file to a separate HDUList object
            cutouts_fits = []

            for file, cutout_list in self._cutout_dict.items():
                if np.array([x.header.get("EMPTY") for x in cutout_list]).all():
                    # Skip files with no data in the cutout images
                    warnings.warn(f"Cutout of {file} contains no data and will not be returned.",
                                  DataWarning)
                    continue

                if not self._memory_only:
                    # If cutouts are being written to disk, we need to keep track of their
                    # input filenames to name the cutout files appropriately
                    self._cutout_filenames.append(file)

                cutouts_fits.append(self._construct_fits_from_hdus(cutout_list))

            return cutouts_fits

    def _write_as_fits(self) -> Union[str, List[str]]:
        """
        Write the cutouts to a file in FITS format.

        Returns
        -------
        cutout_paths : str | list
            The path(s) to the cutout file(s).
        """
        # Get cutouts as memory objects
        cutouts_fits = self._write_to_memory()

        cutout_paths = []  # Cutout file paths
        for i, cutout in enumerate(cutouts_fits):
            # Naming the cutout file
            if self._single_outfile:
                filename = "{}_{:.7f}_{:.7f}_{}-x-{}_astrocut.fits".format(
                    self._cutout_prefix,
                    self._coordinates.ra.value,
                    self._coordinates.dec.value,
                    str(self._cutout_size[0]).replace(' ', ''),
                    str(self._cutout_size[1]).replace(' ', ''))
            else:
                filename = "{}_{:.7f}_{:.7f}_{}-x-{}_astrocut.fits".format(
                    Path(self._cutout_filenames[i]).stem,
                    self._coordinates.ra.value,
                    self._coordinates.dec.value,
                    str(self._cutout_size[0]).replace(' ', ''), 
                    str(self._cutout_size[1]).replace(' ', ''))

            # Write the cutout file to disk
            cutout_path = Path(self._output_dir, filename)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") 
                cutout.writeto(cutout_path, overwrite=True, checksum=True)
            log.debug("Cutout fits file: %s", cutout_path)
            cutout_paths.append(cutout_path.as_posix())

        return cutout_paths if len(cutout_paths) > 1 else cutout_paths[0]

    def _write_as_asdf(self):
        """ASDF output is not yet implemented for FITS files."""
        warnings.warn("ASDF output not yet implemented for FITS files. "
                      "Returning cutout(s) as a list of ~astropy.io.fits.HDUList objects.",
                      InputWarning)
        return self._write_to_memory()
