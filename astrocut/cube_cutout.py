from abc import ABC, abstractmethod
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from time import monotonic
from typing import List, Literal, Tuple, Union

import astropy.units as u
import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from s3path import S3Path

from . import log
from .cutout import Cutout
from .exceptions import InvalidQueryError
from .utils.wcs_fitting import fit_wcs_from_points


class CubeCutout(Cutout, ABC):
    """
    Abstract class for creating cutouts from image cubes.

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
    threads : int | 'auto'
        The number of threads to use for making the cutouts. If 'auto', the number of threads will be set to the number
        of available CPUs.
    verbose : bool
        If True, log messages are printed to the console.

    Attributes
    ----------
    cutouts_by_file : dict
        Dictionary where each key is an input cube filename and its corresponding value is the resulting cutout as a 
        ``CubeCutoutInstance`` object.
    cutouts : list
        List of cutout objects.

    Methods
    -------
    cutout()
        Generate the cutouts.
    """
    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, u.Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, limit_rounding_method: str = 'round', 
                 threads: Union[int, Literal['auto']] = 1, verbose: bool = False):
        super().__init__(input_files, coordinates, cutout_size, fill_value, limit_rounding_method, verbose)

        # Assign the number of threads to use when making cutout
        self._threads = threads

        # Populate these in child classes
        self._has_uncertainty = False  # If the cube has uncertainty data
        self._img_kwds = {}  # Extra keywords to add to the TPF headers
        self._wcs_axes_keyword = None  # Keyword corresponding to WCS axis
        self._wcs_axes_value = None  # Expected value for the WCS axis keyword
        self._skip_kwds = []  # Keywords to skip when adding to the TPF headers

    @property
    def cutouts(self):
        """
        Return a list of cutouts as `CubeCutout.CubeCutoutInstance` objects.
        """
        return list(self.cutouts_by_file.values())

    def _load_file_data(self, file: Union[str, Path, S3Path]) -> fits.HDUList:
        """
        Load the data from an input cube file.

        Parameters
        ----------
        file : str, Path, S3Path
            The path to the cube file.

        Returns
        -------
        cube : `~astropy.io.fits.HDUList`
            The cube data.
        """
        # Suppress FITSFixedWarning from astropy
        warnings.filterwarnings('ignore', category=wcs.FITSFixedWarning)

        # Options when opening the cube file
        fits_options = {
            'mode': 'denywrite',
            'lazy_load_hdus': True
        }

        # Add options based on the file's location
        if isinstance(file, str) and file.startswith('s3://'):  # cloud-hosted file
            fits_options.update({
                'use_fsspec': True,
                'fsspec_kwargs': {'default_block_size': 10_000, 'anon': True}
            })
        else:
            fits_options['memmap'] = True
            self._threads = 1  # disable threading for local storage access

        return fits.open(file, **fits_options)

    def _parse_table_info(self, table_data: fits.fitsrec.FITS_rec) -> WCS:
        """
        Takes the header and the middle entry from the cube table (EXT 2) of image header data,
        builds a WCS object that encapsulates the given WCS information,
        and collects other keywords into a dictionary.

        Parameters
        ----------
        table_data : `~astropy.io.fits.fitsrec.FITS_rec`
            The cube image header data table.

        Returns
        -------
        wcs : `~astropy.wcs.WCS`
            The WCS object built from the table data.

        Raises
        ------
        ~astrocut.wcs.NoWcsKeywordsFoundError
            If no FFI rows contain valid WCS keywords.
        """
        # Find the middle row of the table
        data_ind = len(table_data) // 2
        table_row = None

        # Iterate to find a row containing valid WCS information
        while table_row is None:
            if data_ind == len(table_data):
                # Reset the index to the first row
                data_ind = 0
            elif data_ind == (len(table_data) // 2) - 1:
                # Error if all indices have been checked
                raise wcs.NoWcsKeywordsFoundError('No FFI rows contain valid WCS keywords.')

            # Making sure we have a row with wcs info.
            row = table_data[data_ind]
            if row[self._wcs_axes_keyword] == self._wcs_axes_value:
                table_row = row
            else:
                # If not found, move to the next
                data_ind += 1

        log.debug('Using WCS from row %s out of %s', data_ind, len(table_data))

        # Convert the row into a FITS header
        wcs_header = fits.Header()
        for col in table_data.columns:
            wcs_val = table_row[col.name]

            if (not isinstance(wcs_val, str)) and (np.isnan(wcs_val)):
                continue  # Skip NaN values

            wcs_header[col.name] = wcs_val

        # Populate the image keywords dictionary
        for kwd in self._img_kwds:
            self._img_kwds[kwd][0] = wcs_header.get(kwd)
        
        # Add the FFI file reference
        self._img_kwds['WCS_FFI'] = [table_row['FFI_FILE'],
                                     'FFI used for cutout WCS']
        
        return WCS(wcs_header, relax=True)

    def _add_column_wcs(self, table_header: fits.Header, wcs_dict: dict):
        """
        Adds WCS information for the array columns to the cutout table header.

        Parameters
        ----------
        table_header : `~astropy.io.fits.Header`
            The table header for the cutout table that will be modified in place to include 
            WCS information.
        wcs_dict : dict
            Dictionary of wcs keyword/value pairs to be added to each array column in the 
            cutout table header.
        """
        # Mapping of FITS table keywords to descriptions
        keyword_comments = {
            'TTYPE': 'column name',
            'TFORM': 'column format',
            'TUNIT': 'unit',
            'TDISP': 'display format',
            'TDIM': 'multi-dimensional array spec',
            'TNULL': 'null value'
        }

        for kwd in table_header:
            # Set header comment if the keyword matches
            for key, comment in keyword_comments.items():
                if key in kwd:
                    table_header.comments[kwd] = comment
                    break
            
            # If the column is a 2D array (indicated by 'TDIM'), add WCS info
            if 'TDIM' in kwd and kwd[:-1] == 'TTYPE':
                for wcs_key, (val, com) in wcs_dict.items():
                    table_header.insert(kwd, (wcs_key.format(int(kwd[-1]) - 1), val, com))

    def _add_img_kwds(self, table_header: fits.Header):
        """
        Adding extra keywords to the table header.

        Parameters
        ----------
        table_header : `~astropy.io.fits.Header`
            The table header to add keywords to.  It will be modified in place.
        """
        # Add image keywords to the table header
        for key, value in self._img_kwds.items():
            if key not in self._skip_kwds:
                table_header[key] = tuple(value)

    def _apply_header_inherit(self, hdu_list: fits.HDUList):
        """
        The INHERIT keyword indicated that keywords from the primary header should be duplicated in 
        the headers of all subsequent extensions.  This function performs this addition in place to 
        the given HDUList.
        
        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            The hdu list to apply the INHERIT keyword to.
        """
        primary_header = hdu_list[0].header
        reserved_kwds = {'COMMENT', 'SIMPLE', 'BITPIX', 'EXTEND', 'NEXTEND'}

        for hdu in hdu_list[1:]:
            header = hdu.header
            if header.get('INHERIT', False):
                for kwd, val in primary_header.items():
                    if (kwd not in header) and (kwd not in reserved_kwds):
                        header[kwd] = (val, primary_header.comments[kwd])
    
    @abstractmethod
    def _cutout_file(self, file: Union[str, Path, S3Path]):
        """
        Make a cutout from a single cube file.

        This method is abstract and should be defined in subclasses.
        """
        raise NotImplementedError('Subclasses must implement this method.')
    
    def cutout(self):
        """
        Generate cutouts from a list of input cube files.

        Returns
        -------
        cutouts : list
            List of cutout memory objects.
        """
        # Track start time
        start_time = monotonic()

        # Cutout each input cube file
        for file in self._input_files:
            self._cutout_file(file)

        if not self.cutouts_by_file:
            raise InvalidQueryError('Cube cutout contains no data! (Check image footprint.)')

        # Log total time elapsed
        log.debug('Total time: %.2f sec', (monotonic() - start_time))

        return self.cutouts
    
    class CubeCutoutInstance(ABC):
        """
        Represents an individual cutout with its own data, uncertainty, and aperture arrays.

        Parameters
        ----------
        cube : `~astropy.io.fits.HDUList`
            The input cube.
        file : str | Path | S3Path
            The input cube filename.
        cube_wcs : `~astropy.wcs.WCS`
            The WCS object for the input cube.
        has_uncert : bool
            If the cube has uncertainty data.
        parent : `CubeCutout`
            The parent `CubeCutout` object.

        Attributes
        ----------
        data : `numpy.array`
            The image data cutout.
        uncertainty : `numpy.array`
            The uncertainty data cutout if the cube has uncertainties. Otherwise, returns `None`.
        aperture : `numpy.array`
            The aperture array.
        wcs : `~astropy.wcs.WCS`
            The WCS object for the cutout.
        wcs_fit : dict
            Dictionary of WCS fit keywords.
        cutout_lims : `numpy.array`
            The limits of the cutout in pixel coordinates.
        cube_wcs : `~astropy.wcs.WCS`
            The WCS object for the input cube.
        cube_filename : str | Path | S3Path
            The input cube filename.
        """

        def __init__(self, cube: fits.HDUList, file: Union[str, Path, S3Path], cube_wcs: WCS, 
                     has_uncert: bool, parent: 'CubeCutout'):
            self._parent = parent
            self.cube_filename = file
            self.cube_wcs = cube_wcs
            self.wcs_fit = {
                'WCS_MSEP': [None, '[deg] Max offset between cutout WCS and FFI WCS'],
                'WCS_SIG': [None, '[deg] Error measurement of cutout WCS fit'],
            }

            # Get cutout limits
            self.cutout_lims = self._parent._get_cutout_limits(cube_wcs)

            # Get cutout data
            self.data, self.uncertainty, self.aperture = self._get_cutout_data(cube[1].section, 
                                                                               self._parent._threads, has_uncert)
            self.shape = self.data.shape
            
            # Get cutout WCS
            self.wcs = self._get_full_cutout_wcs(cube_wcs, cube[2].header)

            # Fit the cutout WCS
            self._fit_cutout_wcs(self.data.shape[1:])

        def _get_cutout_data(self, transposed_cube: fits.ImageHDU.section, threads: int, 
                             has_uncert: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Extract a cutout from an image/uncertainty cube that has been transposed to have time on the longest axis.

            Parameters
            ----------
            transposed_cube : `fits.ImageHDU.section`
                Transposed image/uncertainty array.
            threads : int
                The number of threads to use for the cutout.
            has_uncert : bool
                If the cube has uncertainty data.

            Returns
            -------
            img_cutout : `numpy.array`
                The untransposed image cutout array.
            uncert_cutout : `numpy.array` or `None`
                The untransposed uncertainty cutout array if `self._product` is 'SPOC'. Otherwise, returns `None`.
            aperture : `numpy.array`
                The aperture array. This is a 2D array that is the same size as a single cutout that is 1 where 
                there is image data and 0 where there isn't.
            """
            # Compute cutout limits based on WCS
            xmin, xmax = self.cutout_lims[1]
            ymin, ymax = self.cutout_lims[0]

            # Get the cube's shape limits
            xmax_cube, ymax_cube, _, _ = transposed_cube.shape

            # Adjust limits and compute padding
            padding = np.zeros((3, 2), dtype=int)
            xmin, padding[1, 0] = max(0, xmin), max(0, -xmin)
            ymin, padding[2, 0] = max(0, ymin), max(0, -ymin)
            xmax, padding[1, 1] = min(xmax, xmax_cube), max(0, xmax - xmax_cube)
            ymax, padding[2, 1] = min(ymax, ymax_cube), max(0, ymax - ymax_cube)    
            
            # Perform the cutout
            if threads == 'auto' or threads > 1:
                max_workers = None if threads == 'auto' else threads
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    # Increase download performance by making remote cutouts inside of a threadpool
                    # astropy.io.fits Section class executes a list comprehension, generating a sequence of http range
                    # requests (through fsspec) one per our slowest moving dimension (x).
                    # https://github.com/astropy/astropy/blob/0a71105fbaa71439206d496a44df11abc0e3ac96/astropy/io/fits/hdu/image.py#L1059
                    # By running inside a threadpool, these requests instead execute concurrently which increases cutout
                    # throughput. Extremely small cutouts (< 4px in x dim) are not likely to see an improvement
                    cutouts = list(pool.map(lambda x: transposed_cube[x, ymin:ymax, :, :], range(xmin, xmax)))
                # Stack the list of cutouts
                cutout = np.stack(cutouts)
            else:
                cutout = transposed_cube[xmin:xmax, ymin:ymax, :, :]

            # Extract image and uncertainty cutouts
            img_cutout = np.moveaxis(cutout[:, :, :, 0], 2, 0)
            uncert_cutout = np.moveaxis(cutout[:, :, :, 1], 2, 0) if has_uncert else None
            
            # Create aperture mask
            aperture = np.ones((xmax - xmin, ymax - ymin), dtype=np.int32)

            # Apply padding if needed
            if padding.any():
                img_cutout = np.pad(img_cutout, padding, 'constant', constant_values=self._parent._fill_value)
                if has_uncert:
                    uncert_cutout = np.pad(uncert_cutout, padding, 'constant', constant_values=self._parent._fill_value)
                aperture = np.pad(aperture, padding[1:], 'constant', constant_values=0)

            log.debug('Image cutout cube shape: %s', img_cutout.shape)
            if has_uncert:
                log.debug('Uncertainty cutout cube shape: %s', uncert_cutout.shape)
        
            return img_cutout, uncert_cutout, aperture

        @abstractmethod
        def _get_full_cutout_wcs(self, cube_wcs: WCS, cube_table_header: fits.Header) -> WCS:
            """
            Adjust the full FFI WCS for the cutout WCS. Modifies CRPIX values and adds physical WCS keywords.

            This method is abstract and should be defined in subclasses.
            """
            raise NotImplementedError('Subclasses must implement this method.')

        def _fit_cutout_wcs(self, cutout_shape: Tuple[int, int]) -> Tuple[float, float]:
            """
            Given a full (including SIP coefficients) WCS for the cutout, 
            calculate the best fit linear WCS and a measure of the goodness-of-fit.
            
            The new WCS is stored in ``self.wcs``.
            Goodness-of-fit measures are returned and stored in ``self.wcs_fit``.

            Parameters
            ----------
            cutout_wcs :  `~astropy.wcs.WCS`
                The full (including SIP coefficients) cutout WCS object 
            cutout_shape : tuple
                The shape of the cutout in the form (width, height).

            Returns
            -------
            max_dist : float
                The maximum separation between the original and fitted WCS in degrees.
            sigma : float
                The error measurement of the fit in degrees.
            """
            # Getting matched pixel, world coordinate pairs
            # We will choose no more than 100 pixels spread evenly throughout the image
            # Centered on the center pixel.
            # To do this we the appropriate "step size" between pixel coordinates
            # (i.e. we take every ith pixel in each row/column) [TODOOOOOO]
            # For example in a 5x7 cutout with i = 2 we get:
            #
            # xxoxoxx
            # ooooooo
            # xxoxoxx
            # ooooooo
            # xxoxoxx
            #
            # Where x denotes the indexes that will be used in the fit.
            height, width = cutout_shape

            # Determine step size for selecting up to 100 sample points
            step_size = 1
            while (width / step_size) * (height / step_size) > 100:
                step_size += 1
                
            # Create evenly spaced pixel indices along the x and y axes
            xvals = np.arange(0, width, step_size).tolist()
            if xvals[-1] != width - 1:
                xvals.append(width - 1)

            yvals = np.arange(0, height, step_size).tolist()
            if yvals[-1] != height - 1:
                yvals.append(height - 1)
            
            # Generate pixel coordinate pairs
            pix_inds = np.array(list(product(xvals, yvals)))

            # Convert pixel coordinates to world coordinates
            world_pix = SkyCoord(self.wcs.all_pix2world(pix_inds, 0), unit='deg')

            # Fit a linear WCS
            linear_wcs = fit_wcs_from_points([pix_inds[:, 0], pix_inds[:, 1]], world_pix, proj_point='center')
            self.wcs = linear_wcs

            # Evaluate fit using all the pixels in the cutout
            full_pix_inds = np.array(list(product(range(width), range(height))))
            world_pix_original = SkyCoord(self.wcs.all_pix2world(full_pix_inds, 0), unit='deg')
            world_pix_fitted = SkyCoord(linear_wcs.all_pix2world(full_pix_inds, 0), unit='deg')

            # Compute fit residuals
            dists = world_pix_original.separation(world_pix_fitted).to('deg')
            max_dist = dists.max().value
            sigma = np.sqrt(np.sum(dists.value**2))

            # Store fit quality metrics
            self.wcs_fit['WCS_MSEP'][0] = max_dist
            self.wcs_fit['WCS_SIG'][0] = sigma
