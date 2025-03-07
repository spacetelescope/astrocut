
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements the cutout functionality."""

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
from astropy.time import Time
from astropy.wcs import WCS
from s3path import S3Path

from . import __version__, log
from .Cutout import Cutout
from .exceptions import DataWarning, InvalidQueryError
from .utils.wcs_fitting import fit_wcs_from_points


class CubeCutout(Cutout):
    """
    Class for creating cutouts from full frame image cubes.

    This class is currently specific to TESS full frame image cubes, but can be extended to support other missions.

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
    product : str
        The product type to make the cutouts from.
        Can either be 'SPOC' or 'TICA' (default is 'SPOC').
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
    tpf_cutouts_by_file : dict
        Dictionary where each key is an input cube filename and its corresponding value is the resulting cutout target
        pixel file object.
    tpf_cutouts : list
        List of cutout target pixel file objects.

    Methods
    -------
    _load_file_data(file)
        Load the data from an input cube file.
    _parse_table_info(table_data)
        Takes the header and the middle entry from the cube table (EXT 2) of image header data,
        builds a WCS object that encapsulates the given WCS information,
        and collects other keywords into a dictionary.
    _get_cutout_wcs_dict(cutout_wcs, cutout_lims)
        Create a dictionary of WCS keywords for the cutout.
        Adds the physical keywords for transformation back from cutout to location on FFI.
    _update_primary_header(primary_header)
        Updates the primary header of the cutout target pixel file.
        This function sets the object's RA and Dec to the central coordinates of the cutout.
        Since other TESS target pixel file keywords are not available, they are set to 0 or empty strings
        as placeholders.
    _add_column_wcs(table_header, wcs_dict)
        Adds WCS information for the array columns to the cutout table header.
    _add_img_kwds(table_header)
        Adding extra keywords to the table header.
    _apply_header_inherit(hdu_list)
        Apply header inheritance to the cutout target pixel file.
    _build_tpf(cube, cutout, cutout_wcs_dict)
        Build the cutout target pixel file.
    _cutout_file(file)
        Make a cutout from a single cube file.
    cutout()
        Generate the cutouts.
    write_as_tpf(output_dir, output_file)
        Write the cutouts to target pixel files.

    """
    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates: Union[SkyCoord, str], 
                 cutout_size: Union[int, np.ndarray, u.Quantity, List[int], Tuple[int]] = 25,
                 fill_value: Union[int, float] = np.nan, limit_rounding_method: str = 'round', 
                 product: str = 'SPOC', threads: Union[int, Literal['auto']] = 1, verbose: bool = False):
        super().__init__(input_files, coordinates, cutout_size, fill_value, limit_rounding_method, verbose)
        # Assign input attributes
        self._threads = threads
        self._product = product.upper()

        # Extra keywords from the FFI image headers in SPOC (TESS-specific)
        # These are applied to both SPOC and TICA cutouts for consistency.
        self._img_kwds = {
            'BACKAPP': [None, 'background is subtracted'],
            'CDPP0_5': [None, 'RMS CDPP on 0.5-hr time scales'],
            'CDPP1_0': [None, 'RMS CDPP on 1.0-hr time scales'],
            'CDPP2_0': [None, 'RMS CDPP on 2.0-hr time scales'],
            'CROWDSAP': [None, 'Ratio of target flux to total flux in op. ap.'],
            'DEADAPP': [None, 'deadtime applied'],
            'DEADC': [None, 'deadtime correction'],
            'EXPOSURE': [None, '[d] time on source'],
            'FLFRCSAP': [None, 'Frac. of target flux w/in the op. aperture'],
            'FRAMETIM': [None, '[s] frame time [INT_TIME + READTIME]'],
            'FXDOFF': [None, 'compression fixed offset'],
            'GAINA': [None, '[electrons/count] CCD output A gain'],
            'GAINB': [None, '[electrons/count] CCD output B gain'],
            'GAINC': [None, '[electrons/count] CCD output C gain'],
            'GAIND': [None, '[electrons/count] CCD output D gain'],
            'INT_TIME': [None, '[s] photon accumulation time per frame'],
            'LIVETIME': [None, '[d] TELAPSE multiplied by DEADC'],
            'MEANBLCA': [None, '[count] FSW mean black level CCD output A'],
            'MEANBLCB': [None, '[count] FSW mean black level CCD output B'],
            'MEANBLCC': [None, '[count] FSW mean black level CCD output C'],
            'MEANBLCD': [None, '[count] FSW mean black level CCD output D'],
            'NREADOUT': [None, 'number of read per cadence'],
            'NUM_FRM': [None, 'number of frames per time stamp'],
            'READNOIA': [None, '[electrons] read noise CCD output A'],
            'READNOIB': [None, '[electrons] read noise CCD output B'],
            'READNOIC': [None, '[electrons] read noise CCD output C'],
            'READNOID': [None, '[electrons] read noise CCD output D'],
            'READTIME': [None, '[s] readout time per frame'],
            'TIERRELA': [None, '[d] relative time error'],
            'TIMEDEL': [None, '[d] time resolution of data'],
            'TIMEPIXR': [None, 'bin time beginning=0 middle=0.5 end=1'],
            'TMOFST11': [None, '(s) readout delay for camera 1 and ccd 1'],
            'VIGNAPP': [None, 'vignetting or collimator correction applied'],
        }

        self.tpf_cutouts_by_file = {}

        # Make the cutouts upon initialization
        self.cutout()

    @property
    def cutouts(self):
        """
        Return a list of cutouts as `CubeCutout.CubeCutoutInstance` objects.
        """
        return list(self.cutouts_by_file.values())

    @property
    def tpf_cutouts(self):
        """
        Return the cutouts as a list of `astropy.io.fits.HDUList` target pixel file objects.
        """
        return list(self.tpf_cutouts_by_file.values())

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
        wcsaxes_keyword = 'CTYPE2'
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
            if row[wcsaxes_keyword] == 'DEC--TAN-SIP':
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
    
    def _get_cutout_wcs_dict(self, cutout_wcs: WCS, cutout_lims: np.ndarray) -> dict:
        """
        Create a dictionary of WCS keywords for the cutout.
        Adds the physical keywords for transformation back from cutout to location on FFI.
        This is a TESS-specific function.

        Parameters
        ----------
        cutout_wcs : `~astropy.wcs.WCS`
            The WCS object for the cutout.
        cutout_lims : `~numpy.ndarray`
            The limits of the cutout in pixel coordinates.
        
        Returns
        -------
        cutout_wcs_dict : dict
            Cutout WCS column header keywords as dictionary of 
            ``{<kwd format string>: [value, desc]} pairs.``
        """
        wcs_header = cutout_wcs.to_header()
        wcs_axes = wcs_header.get('WCSAXES', 2)  # Default to 2 if missing

        # Create the cutout WCS dictionary
        cutout_wcs_dict = {
            # Number of axes
            'WCAX{}': [wcs_axes, 'Number of WCS axes'],
            # Celestial WCS Keywords
            '1CTYP{}': [wcs_header['CTYPE1'], 'right ascension coordinate type'],
            '2CTYP{}': [wcs_header['CTYPE2'], 'declination coordinate type'],
            '1CRPX{}': [wcs_header['CRPIX1'], '[pixel] reference pixel along image axis 1'],
            '2CRPX{}': [wcs_header['CRPIX2'], '[pixel] reference pixel along image axis 2'],
            '1CRVL{}': [wcs_header['CRVAL1'], '[deg] right ascension at reference pixel'],
            '2CRVL{}': [wcs_header['CRVAL2'], '[deg] declination at reference pixel'],
            '1CUNI{}': [wcs_header['CUNIT1'], 'physical unit in column dimension'],
            '2CUNI{}': [wcs_header['CUNIT2'], 'physical unit in row dimension'],
            '1CDLT{}': [wcs_header['CDELT1'], '[deg] pixel scale in RA dimension'],
            '2CDLT{}': [wcs_header['CDELT1'], '[deg] pixel scale in DEC dimension'],
            # Coordinate transformation matrix (PC) keywords
            '11PC{}': [wcs_header['PC1_1'], 'Coordinate transformation matrix element'],
            '12PC{}': [wcs_header['PC1_2'], 'Coordinate transformation matrix element'],
            '21PC{}': [wcs_header['PC2_1'], 'Coordinate transformation matrix element'],
            '22PC{}': [wcs_header['PC2_2'], 'Coordinate transformation matrix element'],
            # Physical keywords
            'WCSN{}P': ['PHYSICAL', 'table column WCS name'],
            'WCAX{}P': [2, 'table column physical WCS dimensions'],
            '1CTY{}P': ['RAWX', 'table column physical WCS axis 1 type, CCD col'],
            '2CTY{}P': ['RAWY', 'table column physical WCS axis 2 type, CCD row'],
            '1CUN{}P': ['PIXEL', 'table column physical WCS axis 1 unit'],
            '2CUN{}P': ['PIXEL', 'table column physical WCS axis 2 unit'],
            '1CRV{}P': [cutout_lims[0, 0] + 1, 'table column physical WCS ax 1 ref value'],
            '2CRV{}P': [cutout_lims[1, 0] + 1, 'table column physical WCS ax 2 ref value'],
            # TODO: can we calculate these? or are they fixed?
            '1CDL{}P': [1.0, 'table column physical WCS a1 step'],    
            '2CDL{}P': [1.0, 'table column physical WCS a2 step'],
            '1CRP{}P': [1, 'table column physical WCS a1 reference'],
            '2CRP{}P': [1, 'table column physical WCS a2 reference'],
        }

        return cutout_wcs_dict
    
    def _update_primary_header(self, primary_header: fits.Header):
        """
        Updates the primary header of the cutout target pixel file.
        This function sets the object's RA and Dec to the central coordinates of the cutout.
        Since other TESS target pixel file keywords are not available, they are set to 0 or empty strings
        as placeholders.
        This is a TESS-specific function.

        Parameters
        ----------
        primary_header : `~astropy.io.fits.Header`
            The primary header from the cube file that will be modified in place for the cutout.
        """
        # Add cutout-specific metadata
        primary_header.update({
            'CREATOR': ('astrocut', 'software used to produce this file'),
            'PROCVER': (__version__, 'software version'),
            'FFI_TYPE': (self._product, 'the FFI type used to make the cutouts'),
            'RA_OBJ': (self._coordinates.ra.deg, '[deg] right ascension'),
            'DEC_OBJ': (self._coordinates.dec.deg, '[deg] declination'),
            'TIMEREF': ('SOLARSYSTEM' if self._product == 'SPOC' else None, 
                        'barycentric correction applied to times'),
            'TASSIGN': ('SPACECRAFT' if self._product == 'SPOC' else None, 
                        'where time is assigned'),
            'TIMESYS': ('TDB', 'time system is Barycentric Dynamical Time (TDB)'),
            'BJDREFI': (2457000, 'integer part of BTJD reference date'),
            'BJDREFF': (0.00000000, 'fraction of the day in BTJD reference date'),
            'TIMEUNIT': ('d', 'time unit for TIME, TSTART, and TSTOP')
        })

        # TODO : The name of FIRST_FFI (and LAST_FFI) is too long to be a header kwd value.
        # Find a way to include these in the headers without breaking astropy (maybe abbreviate?).
        # primary_header['FIRST_FFI'] = (self.first_ffi, 'the FFI used for the primary header 
        # keyword values, except TSTOP')
        # primary_header['LAST_FFI'] = (self.last_ffi, 'the FFI used for the TSTOP keyword value')

        if self._product == 'TICA':
            # Adding some missing keywords for TICA cutouts
            primary_header.update({
                'EXTVER': ('1', 'extension version number (not format version)'),
                'SIMDATA': (False, 'file is based on simulated data'),
                'NEXTEND': ('2', 'number of standard extensions'),
                'TSTART': (primary_header['STARTTJD'], 'observation start time in TJD of first FFI'),
                'TSTOP': (primary_header['ENDTJD'], 'observation stop time in TJD of last FFI'),
                'CAMERA': (primary_header['CAMNUM'], 'Camera number'),
                'CCD': (primary_header['CCDNUM'], 'CCD chip number'),
                'ASTATE': (None, 'archive state F indicates single orbit processing'),
                'CRMITEN': (primary_header['CRM'], 'spacecraft cosmic ray mitigation enabled'),
                'CRBLKSZ': (None, '[exposures] s/c cosmic ray mitigation block siz'),
                'FFIINDEX': (primary_header['CADENCE'], 'number of FFI cadence interval of first FFI'),
                'DATA_REL': (None, 'data release version number'),
                'FILEVER': (None, 'file format version'),
                'RADESYS': (None, 'reference frame of celestial coordinates'),
                'SCCONFIG': (None, 'spacecraft configuration ID'),
                'TIMVERSN': (None, 'OGIP memo number for file format')
            })

            date_obs = Time(primary_header['TSTART'] + primary_header['BJDREFI'], format='jd').iso
            date_end = Time(primary_header['TSTOP'] + primary_header['BJDREFI'], format='jd').iso
            primary_header.update({
                'DATE-OBS': (date_obs, 'TSTART as Julian Date of first FFI'),
                'DATE-END': (date_end, 'TSTOP as Julian Date of last FFI'),
            })

        # Remove unnecessary keywords
        # Bulk removal with wildcards. Most of these should only live in EXT 1 header.
        delete_kwds_wildcards = ['SC_*', 'RMS*', 'A_*', 'AP_*', 'B_*', 'BP*', 'GAIN*', 'TESS_*', 'CD*',
                                 'CT*', 'CRPIX*', 'CRVAL*', 'MJD*']
        # Removal of specific kwds not relevant for cutouts. Most likely these describe a single FFI, and not
        # the whole cube, which is misleading because we are working with entire stacks of FFIs. Other keywords 
        # are analogs to ones that have already been added to the primary header in the lines above.
        delete_kwds = ['COMMENT', 'FILTER', 'TIME', 'EXPTIME', 'ACS_MODE', 'DEC_TARG', 'FLXWIN', 'RA_TARG',
                       'CCDNUM', 'CAMNUM', 'WCSGDF', 'UNITS', 'CADENCE', 'SCIPIXS', 'INT_TIME', 'PIX_CAT',
                       'REQUANT', 'DIFF_HUF', 'PRIM_HUF', 'QUAL_BIT', 'SPM', 'STARTTJD', 'ENDTJD', 'CRM',
                       'TJD_ZERO', 'CRM_N', 'ORBIT_ID', 'MIDTJD']
        
        for kwd in delete_kwds_wildcards + delete_kwds:
            primary_header.pop(kwd, None)

        # Compute and update TELAPSE keyword
        telapse = primary_header.get('TSTOP', 0) - primary_header.get('TSTART', 0)
        primary_header['TELAPSE '] = (telapse, '[d] TSTOP - TSTART')

        # Update DATE comment to be more explicit
        primary_header['DATE'] = (primary_header['DATE'], 'FFI cube creation date')

        # Specifying that some of these headers keyword values are inherited from the first FFI
        if self._product == 'SPOC':
            primary_header.update({
                'TSTART': (primary_header['TSTART'], 'observation start time in TJD of first FFI'),
                'TSTOP': (primary_header['TSTOP'], 'observation stop time in TJD of last FFI'),
                'DATE-OBS': (primary_header['DATE-OBS'], 'TSTART as UTC calendar date of first FFI'),
                'DATE-END': (primary_header['DATE-END'], 'TSTOP as UTC calendar date of last FFI'),
                'FFIINDEX': (primary_header['FFIINDEX'], 'number of FFI cadence interval of first FFI')
            })

        # Add missing target-specific metadata
        primary_header.update({
            'OBJECT': ('', 'string version of target ID'),
            'TCID': (0, 'unique TESS target identifier'),
            'PXTABLE': (0, 'pixel table ID'),
            'PMRA': (0.0, '[mas/yr] RA proper motion'),
            'PMDEC': (0.0, '[mas/yr] Dec proper motion'),
            'PMTOTAL': (0.0, '[mas/yr] Total proper motion'),
            'TESSMAG': (0.0, '[mag] TESS magnitude'),
            'TEFF': (0.0, '[K] Effective temperature'),
            'LOGG': (0.0, '[cm/s2] log10 surface gravity'),
            'MH': (0.0, '[log10([M/H])] Metallicity'),
            'RADIUS': (0.0, '[solar radii] Stellar radius'),
            'TICVER': (0, 'TIC version'),
            'TICID': (None, 'unique TESS target identifier')
        })

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
            # We'll skip these TICA-specific image keywords that are single-FFI specific
            # or just not helpful
            if key not in {'TIME', 'EXPTIME', 'FILTER'}:
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

    def _build_tpf(self, cube_fits: fits.HDUList, cutout: 'CubeCutoutInstance') -> fits.HDUList:
        """
        Build the cutout target pixel file (TPF) and format it to match TESS pipeline TPFs.

        Parameters
        ---------
        cube_fits : `~astropy.io.fits.hdu.hdulist.HDUList`
            The cube hdu list.
        cutout : `CubeCutout.CubeCutoutInstance`
            The cutout object.

        Returns
        -------
        cutout_hdu_list :  `~astropy.io.fits.HDUList`
            Target pixel file HDUList
        """
        # Copy the primary HDU and update its header
        primary_hdu = cube_fits[0].copy()
        self._update_primary_header(primary_hdu.header)

        cols = []  # list to store FITS table columns
        img_cube = cutout.data
        empty_arr = np.zeros_like(img_cube)  # precompute empty array for missing data
        empty_single = empty_arr[:, 0, 0]  # extract single-value empty array for time & position

        # Define FITS column format and dimension
        pixel_format = f'{img_cube[0].size}E'
        pixel_dim = f'{img_cube[0].shape[::-1]}'

        # Definte time-related keywords based on product type
        start, stop = ("TSTART", "TSTOP") if self._product == "SPOC" else ("STARTTJD", "ENDTJD")
        cols.append(fits.Column(name='TIME', format='D', unit='BJD - 2457000, days', disp='D14.7',
                                array=(cube_fits[2].columns[start].array + cube_fits[2].columns[stop].array) / 2))

        if self._product == 'SPOC':
            cols.append(fits.Column(name='TIMECORR', format='E', unit='d', disp='E14.7',
                                    array=cube_fits[2].columns['BARYCORR'].array))

        # Define cadence number (zero-filled for SPOC)
        cadence_array = empty_single if self._product == 'SPOC' else cube_fits[2].columns['CADENCE'].array
        cols.append(fits.Column(name='CADENCENO', format='J', disp='I10', array=cadence_array))

        # Define flux-related columns
        pixel_unit = 'e-/s' if self._product == 'SPOC' else 'e-'
        flux_err_array = cutout.uncertainty if self._product == 'SPOC' else empty_arr
        cols.extend([
            fits.Column(name='RAW_CNTS', format=pixel_format.replace('E', 'J'), unit='count',
                        dim=pixel_dim, disp='I8', array=empty_arr - 1, null=-1),
            fits.Column(name='FLUX', format=pixel_format, dim=pixel_dim, unit=pixel_unit,
                        disp='E14.7', array=img_cube),
            fits.Column(name='FLUX_ERR', format=pixel_format, dim=pixel_dim, unit=pixel_unit,
                        disp='E14.7', array=flux_err_array)
        ])
   
        # Add background columns (identical zero arrays)
        for col_name in ['FLUX_BKG', 'FLUX_BKG_ERR']:
            cols.append(fits.Column(name=col_name, format=pixel_format, dim=pixel_dim,
                                    unit=pixel_unit, disp='E14.7', array=empty_arr))

        # Add the quality flags
        data_quality = 'DQUALITY' if self._product == 'SPOC' else 'QUAL_BIT'
        cols.append(fits.Column(name='QUALITY', format='J', disp='B16.16',
                                array=cube_fits[2].columns[data_quality].array))

        # Add position correction info (zero-filled)
        for col_name in ['POS_CORR1', 'POS_CORR2']:
            cols.append(fits.Column(name=col_name, format='E', unit='pixel', disp='E14.7', array=empty_single))

        # Add the FFI_FILE column (not in the pipeline tpfs)
        cols.append(fits.Column(name='FFI_FILE', format='38A', unit='pixel',
                                array=cube_fits[2].columns['FFI_FILE'].array))
        
        # Create the table HDU
        table_hdu = fits.BinTableHDU.from_columns(cols)
        table_hdu.header['EXTNAME'] = 'PIXELS'
        table_hdu.header['INHERIT'] = True
    
        # Add the WCS keywords to the columns and remove from the header
        self._add_column_wcs(table_hdu.header, self._get_cutout_wcs_dict(cutout.wcs, cutout.cutout_lims))

        # Add the extra image keywords
        self._add_img_kwds(table_hdu.header)

        # Build the aperture HDU
        aperture_hdu = fits.ImageHDU(data=cutout.aperture)
        aperture_hdu.header['EXTNAME'] = 'APERTURE'
        # Copy WCS headers
        for kwd, val, cmt in cutout.wcs.to_header().cards:
            aperture_hdu.header.set(kwd, val, cmt)
        # Add extra aperture keywords (TESS specific)
        aperture_hdu.header.set('NPIXMISS', None, 'Number of op. aperture pixels not collected')
        aperture_hdu.header.set('NPIXSAP', None, 'Number of pixels in optimal aperture')
        # Add goodness-of-fit keywords
        for key, val in cutout.wcs_fit.items():
            aperture_hdu.header[key] = tuple(val)
        aperture_hdu.header['INHERIT'] = True
    
        # Assemble the HDUList
        cutout_hdu_list = fits.HDUList([primary_hdu, table_hdu, aperture_hdu])
        
        # Apply header inheritance
        self._apply_header_inherit(cutout_hdu_list)

        return cutout_hdu_list
    
    def _cutout_file(self, file: Union[str, Path, S3Path]):
        """
        Make a cutout from a single cube file.

        Parameters
        ----------
        file : str, Path, or S3Path
            The path to the cube file.
        """
        # Read in file data
        cube = self._load_file_data(file)

        # Parse table info
        cube_wcs = self._parse_table_info(cube[2].data)

        # Get cutouts
        has_uncert = self._product == 'SPOC'
        try:
            cutout = self.CubeCutoutInstance(cube, cube_wcs, has_uncert, self)
        except InvalidQueryError:
            warnings.warn(f'Cutout footprint does not overlap with data in {file}, skipping...', DataWarning)
            cube.close()
            return

        # Build TPF
        tpf = self._build_tpf(cube, cutout)
        cube.close()

        # Log coordinates
        log.debug('Cutout center coordinate: %s, %s', self._coordinates.ra.deg, self._coordinates.dec.deg)

        # Get cutout WCS info
        max_dist, sigma = cutout.wcs_fit['WCS_MSEP'][0], cutout.wcs_fit['WCS_SIG'][0]
        log.debug('Maximum distance between approximate and true location: %s', max_dist)
        log.debug('Error in approximate WCS (sigma): %.4f', sigma)

        # Store cutouts with filename
        self.cutouts_by_file[file] = cutout
        self.tpf_cutouts_by_file[file] = tpf
    
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

    def write_as_tpf(self, output_dir: Union[str, Path] = '.', output_file: str = None):
        """
        Write the cutouts to disk as target pixel files.

        Parameters
        ----------
        output_dir : str | Path
            The output directory where the cutout files will be saved.
        output_file : str
            The output filename. If not provided or more than one cutout is created, the filename 
            will be generated based on the input filename. This parameter is included so that
            ``CubeCutout`` is backwards compatible with ``CutoutFactory.cube_cut``.

        Returns
        -------
        cutouts : list
            List of file paths to cutout target pixel files.
        """
        # Track write time
        write_time = monotonic()

        # Ensure that output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        cutout_paths = []
        for file, cutout in self.cutouts_by_file.items():
            # Determine file name
            if not output_file or len(self._input_files) > 1:
                cutout_lims = cutout.cutout_lims
                width = cutout_lims[0, 1] - cutout_lims[0, 0]
                height = cutout_lims[1, 1] - cutout_lims[1, 0]
                filename = '{}_{:7f}_{:7f}_{}x{}_astrocut.fits'.format(
                    Path(file).stem.rstrip('-cube'),
                    self._coordinates.ra.value,
                    self._coordinates.dec.value,
                    width,
                    height)
            else:
                filename = output_file

            # Write the cutout TPF
            cutout_path = Path(output_dir) / filename

            # Get the corresponding cutout TPF
            tpf = self.tpf_cutouts_by_file[file]

            with warnings.catch_warnings():
                # Suppress FITS verification warnings
                warnings.simplefilter('ignore', fits.verify.VerifyWarning)
                tpf.writeto(cutout_path, overwrite=True, checksum=True)
            
            # Log file path
            log.debug('Target pixel file path: %s', cutout_path)
            # Append file path or memory object
            cutout_paths.append(cutout_path.as_posix())

        log.debug('Write time: %.2f sec', (monotonic() - write_time))
        return cutout_paths
    
    class CubeCutoutInstance:
        """
        Represents an individual cutout with its own data, uncertainty, and aperture arrays.

        Args
        ----
        cube : `~astropy.io.fits.HDUList`
            The input cube.
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
        """

        def __init__(self, cube: fits.HDUList, cube_wcs: WCS, has_uncert: bool, parent: 'CubeCutout'):
            self._parent = parent
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

        def _get_full_cutout_wcs(self, cube_wcs: WCS, cube_table_header: fits.Header) -> WCS:
            """
            Adjust the full FFI WCS for the cutout WCS. Modifies CRPIX values and adds physical WCS keywords.

            Parameters
            ----------
            cube_wcs : `~astropy.wcs.WCS`
                The WCS for the input cube.
            cube_table_header :  `~astropy.io.fits.Header`
                The FFI cube header for the data table extension. This allows the cutout WCS information
                to more closely match the mission TPF format.

            Returns
            --------
            response :  `~astropy.wcs.WCS`
                The cutout WCS object including SIP distortions.
            """
            wcs_header = cube_wcs.to_header(relax=True)

            # Using table comments if available
            for kwd in wcs_header:
                if cube_table_header.get(kwd, None):
                    wcs_header.comments[kwd] = cube_table_header[kwd]

            # Adjusting the CRPIX values based on cutout limits
            wcs_header['CRPIX1'] -= self.cutout_lims[0, 0]
            wcs_header['CRPIX2'] -= self.cutout_lims[1, 0]

            # Add physical WCS keywords
            physical_wcs_keys = {
                'WCSNAMEP': ('PHYSICAL', 'name of world coordinate system alternate P'),
                'WCSAXESP': (2, 'mumber of WCS physical axes'),
                'CTYPE1P': ('RAWX', 'physical WCS axis 1 type (CCD column)'),
                'CUNIT1P': ('PIXEL', 'physical WCS axis 1 unit'),
                'CRPIX1P': (1, 'reference CCD column'),
                'CRVAL1P': (self.cutout_lims[0, 0] + 1, 'value at reference CCD column'),
                'CDELT1P': (1.0, 'physical WCS axis 1 step'),
                'CTYPE2P': ('RAWY', 'physical WCS axis 2 type (CCD row)'),
                'CUNIT2P': ('PIXEL', 'physical WCS axis 2 unit'),
                'CRPIX2P': (1, 'reference CCD row'),
                'CRVAL2P': (self.cutout_lims[1, 0] + 1, 'value at reference CCD row'),
                'CDELT2P': (1.0, 'physical WCS axis 2 step'),
            }

            for key, (value, comment) in physical_wcs_keys.items():
                wcs_header[key] = (value, comment)

            return WCS(wcs_header)

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
