# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements the cutout functionality."""

import os
import warnings

from time import time
from itertools import product

import astropy.units as u
import numpy as np

from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from . import __version__ 
from .exceptions import InputWarning, InvalidQueryError

# Note: Use the astropy function if available, TODO: fix > 4.3 astropy fitting
import astropy
if astropy.utils.minversion(astropy, "4.0.2") and (float(astropy.__version__[:3]) < 4.3):
    from astropy.wcs.utils import fit_wcs_from_points
else:
    from .utils.wcs_fitting import fit_wcs_from_points


class CutoutFactory():
    """
    Class for creating image cutouts.

    This class encompasses all of the cutout functionality.  
    In the current version this means creating cutout target pixel files from both 
    SPOC (Science Processing Operations Center) and TICA (Tess Image CAlibration) 
    full frame image cubes.

    Future versions will include more generalized cutout functionality.
    """

    def __init__(self):
        """
        Initialization function.
        """

        self.cube_wcs = None  # WCS information from the image cube
        self.cutout_wcs = None  # WCS information (linear) for the cutout
        self.cutout_wcs_fit = {'WCS_MSEP': [None, "[deg] Max offset between cutout WCS and FFI WCS"],
                               'WCS_SIG': [None, "[deg] Error measurement of cutout WCS fit"]}
        
        self.cutout_lims = np.zeros((2, 2), dtype=int)  # Cutout pixel limits, [[ymin,ymax],[xmin,xmax]]
        self.center_coord = None  # Central skycoord
        
        # Extra keywords from the FFI image headers in SPOC.
        # These are applied to both SPOC and TICA cutouts for consistency.
        self.img_kwds = {"BACKAPP": [None, "background is subtracted"],
                         "CDPP0_5": [None, "RMS CDPP on 0.5-hr time scales"],
                         "CDPP1_0": [None, "RMS CDPP on 1.0-hr time scales"],
                         "CDPP2_0": [None, "RMS CDPP on 2.0-hr time scales"],
                         "CROWDSAP": [None, "Ratio of target flux to total flux in op. ap."],
                         "DEADAPP": [None, "deadtime applied"], 
                         "DEADC": [None, "deadtime correction"],
                         "EXPOSURE": [None, "[d] time on source"],
                         "FLFRCSAP": [None, "Frac. of target flux w/in the op. aperture"],
                         "FRAMETIM": [None, "[s] frame time [INT_TIME + READTIME]"],
                         "FXDOFF": [None, "compression fixed offset"],
                         "GAINA": [None, "[electrons/count] CCD output A gain"],
                         "GAINB": [None, "[electrons/count] CCD output B gain"],
                         "GAINC": [None, "[electrons/count] CCD output C gain"],
                         "GAIND": [None, "[electrons/count] CCD output D gain"],
                         "INT_TIME": [None, "[s] photon accumulation time per frame"],
                         "LIVETIME": [None, "[d] TELAPSE multiplied by DEADC"],
                         "MEANBLCA": [None, "[count] FSW mean black level CCD output A"],
                         "MEANBLCB": [None, "[count] FSW mean black level CCD output B"],
                         "MEANBLCC": [None, "[count] FSW mean black level CCD output C"],
                         "MEANBLCD": [None, "[count] FSW mean black level CCD output D"],
                         "NREADOUT": [None, "number of read per cadence"],
                         "NUM_FRM": [None, "number of frames per time stamp"],
                         "READNOIA": [None, "[electrons] read noise CCD output A"],
                         "READNOIB": [None, "[electrons] read noise CCD output B"],
                         "READNOIC": [None, "[electrons] read noise CCD output C"],
                         "READNOID": [None, "[electrons] read noise CCD output D"],
                         "READTIME": [None, "[s] readout time per frame"],
                         "TIERRELA": [None, "[d] relative time error"],
                         "TIMEDEL": [None, "[d] time resolution of data"],
                         "TIMEPIXR": [None, "bin time beginning=0 middle=0.5 end=1"],
                         "TMOFST11": [None, "(s) readout delay for camera 1 and ccd 1"],
                         "VIGNAPP": [None, "vignetting or collimator correction applied"]}

        
    def _parse_table_info(self, product, table_data, verbose=False):
        """
        Takes the header and the middle entry from the cube table (EXT 2) of image header data,
        builds a WCS object that encapsulates the given WCS information,
        and collects into a dictionary the other keywords we care about.  

        The WCS is stored in ``self.cube_wcs``, and the extra keywords in ``self.img_kwds``

        Parameters
        ----------
        product : str
            The product type to make the cutouts from.
            Can either be 'SPOC' or 'TICA'.
        table_data : `~astropy.io.fits.fitsrec.FITS_rec`
            The cube image header data table.
        """

        # Populating `table_row` with the primary header keywords 
        # of the middle FFI
        data_ind = len(table_data)//2
        table_row = None

        while table_row is None:
            table_row = table_data[data_ind]

            # Making sure we have a row with wcs info.
            wcsaxes_keyword = 'WCSAXES' if product == 'SPOC' else 'WCAX3' 
            if table_row[wcsaxes_keyword] != 2:
                table_row = None
                data_ind += 1
                if data_ind == len(table_data):
                    raise wcs.NoWcsKeywordsFoundError("No FFI rows contain valid WCS keywords.")

        if verbose:
            print("Using WCS from row {} out of {}".format(data_ind, len(table_data)))

        # Turning the table row into a new header object
        wcs_header = fits.header.Header()
        for col in table_data.columns:
            
            wcs_val = table_row[col.name]
            if (not isinstance(wcs_val, str)) and (np.isnan(wcs_val)):
                continue  # Just skip nans

            wcs_header[col.name] = wcs_val
            
        # Setting the cube wcs
        self.cube_wcs = wcs.WCS(wcs_header, relax=True)

        # Filling the img_kwds dictionary while we are here
        for kwd in self.img_kwds:
            self.img_kwds[kwd][0] = wcs_header.get(kwd)
        # Adding the info about which FFI we got the WCS info from
        self.img_kwds["WCS_FFI"] = [table_data[data_ind]["FFI_FILE"],
                                    "FFI used for cutout WCS"]

            
    def _get_cutout_limits(self, cutout_size):
        """
        Takes the center coordinates, cutout size, and the wcs from
        which the cutout is being taken and returns the x and y pixel limits
        for the cutout.

        Parameters
        ----------
        cutout_size : array
            [ny,nx] in with ints (pixels) or astropy quantities

        Returns
        -------
        response : `numpy.array`
            The cutout pixel limits in an array of the form [[ymin,ymax],[xmin,xmax]]
        """
        
        # Note: This is returning the center pixel in 1-up
        try:
            center_pixel = self.center_coord.to_pixel(self.cube_wcs, 1)
        except wcs.NoConvergence:  # If wcs can't converge, center coordinate is far from the footprint
            raise InvalidQueryError("Cutout location is not in cube footprint!")

        lims = np.zeros((2, 2), dtype=int)

        for axis, size in enumerate(cutout_size):

            if not isinstance(size, u.Quantity):  # assume pixels
                dim = size / 2
            elif size.unit == u.pixel:  # also pixels
                dim = size.value / 2
            elif size.unit.physical_type == 'angle':
                pixel_scale = u.Quantity(wcs.utils.proj_plane_pixel_scales(self.cube_wcs)[axis],
                                         self.cube_wcs.wcs.cunit[axis])
                dim = (size / pixel_scale).decompose() / 2

            lims[axis, 0] = int(np.round(center_pixel[axis] - 1 - dim))
            lims[axis, 1] = int(np.round(center_pixel[axis] - 1 + dim))

            # The case where the requested area is so small it rounds to zero
            if lims[axis, 0] == lims[axis, 1]:
                lims[axis, 0] = int(np.floor(center_pixel[axis] - 1))
                lims[axis, 1] = int(np.ceil(center_pixel[axis] - 1))

        # Checking at least some of the cutout is on the cube
        if ((lims[0, 0] <= 0) and (lims[0, 1] <= 0)) or ((lims[1, 0] <= 0) and (lims[1, 1] <= 0)):
            raise InvalidQueryError("Cutout location is not in cube footprint!")

        self.cutout_lims = lims


    def _get_full_cutout_wcs(self, cube_table_header):
        """
        Starting with the full FFI WCS and adjusting it for the cutout WCS.
        Adjusts CRPIX values and adds physical WCS keywords.

        Parameters
        ----------
        cube_table_header :  `~astropy.io.fits.Header`
           The FFI cube header for the data table extension. This allows the cutout WCS information
           to more closely match the mission TPF format.

        Resturns
        --------
        response :  `~astropy.wcs.WCS`
            The cutout WCS object including SIP distortions.
        """
        
        wcs_header = self.cube_wcs.to_header(relax=True)

        # Using table comment rather than the default ones if available
        for kwd in wcs_header:
            if cube_table_header.get(kwd, None):
                wcs_header.comments[kwd] = cube_table_header[kwd]

        # Adjusting the CRPIX values
        wcs_header["CRPIX1"] -= self.cutout_lims[0, 0]
        wcs_header["CRPIX2"] -= self.cutout_lims[1, 0]

        # Adding the physical wcs keywords
        wcs_header.set("WCSNAMEP", "PHYSICAL", "name of world coordinate system alternate P")
        wcs_header.set("WCSAXESP", 2, "number of WCS physical axes")
    
        wcs_header.set("CTYPE1P", "RAWX", "physical WCS axis 1 type CCD col")
        wcs_header.set("CUNIT1P", "PIXEL", "physical WCS axis 1 unit")
        wcs_header.set("CRPIX1P", 1, "reference CCD column")
        wcs_header.set("CRVAL1P", self.cutout_lims[0, 0] + 1, "value at reference CCD column")
        wcs_header.set("CDELT1P", 1.0, "physical WCS axis 1 step")
                
        wcs_header.set("CTYPE2P", "RAWY", "physical WCS axis 2 type CCD col")
        wcs_header.set("CUNIT2P", "PIXEL", "physical WCS axis 2 unit")
        wcs_header.set("CRPIX2P", 1, "reference CCD row")
        wcs_header.set("CRVAL2P", self.cutout_lims[1, 0] + 1, "value at reference CCD row")
        wcs_header.set("CDELT2P", 1.0, "physical WCS axis 2 step")

        return wcs.WCS(wcs_header)

    
    def _fit_cutout_wcs(self, cutout_wcs, cutout_shape):
        """
        Given a full (including SIP coefficients) wcs for the cutout, 
        calculate the best fit linear wcs, and a measure of the goodness-of-fit.
        
        The new WCS is stored in ``self.cutout_wcs``.
        Goodness-of-fit measures are returned and stored in ``self.cutout_wcs_fit``.

        Parameters
        ----------
        cutout_wcs :  `~astropy.wcs.WCS`
            The full (including SIP coefficients) cutout WCS object 
        cutout_shape : tuple
            The shape of the cutout in the form (width, height).

        Returns
        -------
        response : tuple
            Goodness-of-fit statistics. (max dist, sigma)
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
        y, x = cutout_shape
        i = 1
        while (x/i)*(y/i) > 100:
            i += 1
            
        xvals = list(reversed(range(x//2, -1, -i)))[:-1] + list(range(x//2, x, i))
        if xvals[-1] != x-1:
            xvals += [x-1]
        if xvals[0] != 0:
            xvals = [0] + xvals
        
        yvals = list(reversed(range(y//2, -1, -i)))[:-1] + list(range(y//2, y, i))
        if yvals[-1] != y-1:
            yvals += [y-1]
        if yvals[0] != 0:
            yvals = [0] + yvals
        
        pix_inds = np.array(list(product(xvals, yvals)))
        world_pix = SkyCoord(cutout_wcs.all_pix2world(pix_inds, 0), unit='deg')

        # Getting the fit WCS
        linear_wcs = fit_wcs_from_points([pix_inds[:, 0], pix_inds[:, 1]], world_pix, proj_point='center')

        self.cutout_wcs = linear_wcs

        # Checking the fit (we want to use all of the pixels for this)
        pix_inds = np.array(list(product(list(range(cutout_shape[1])), list(range(cutout_shape[0])))))
        world_pix = SkyCoord(cutout_wcs.all_pix2world(pix_inds, 0), unit='deg')
        world_pix_new = SkyCoord(linear_wcs.all_pix2world(pix_inds, 0), unit='deg')

        dists = world_pix.separation(world_pix_new).to('deg')
        sigma = np.sqrt(sum(dists.value**2))

        self.cutout_wcs_fit['WCS_MSEP'][0] = dists.max().value
        self.cutout_wcs_fit['WCS_SIG'][0] = sigma

        return (dists.max(), sigma)
    
  
    def _get_cutout_wcs_dict(self):
        """
        Transform the cutout WCS object into the cutout column WCS keywords.
        Adds the physical keywords for transformation back from cutout to location on FFI.
        This is a very TESS specific function.
        
        Returns
        -------
        response: dict
            Cutout wcs column header keywords as dictionary of 
            ``{<kwd format string>: [value, desc]} pairs.``
        """
        wcs_header = self.cutout_wcs.to_header()

        cutout_wcs_dict = dict()

        ## Cutout array keywords ##

        cutout_wcs_dict["WCAX{}"] = [wcs_header['WCSAXES'], "number of WCS axes"]
        # TODO: check for 2? this must be two

        cutout_wcs_dict["1CTYP{}"] = [wcs_header["CTYPE1"], "right ascension coordinate type"]
        cutout_wcs_dict["2CTYP{}"] = [wcs_header["CTYPE2"], "declination coordinate type"]
        
        cutout_wcs_dict["1CRPX{}"] = [wcs_header["CRPIX1"], "[pixel] reference pixel along image axis 1"]
        cutout_wcs_dict["2CRPX{}"] = [wcs_header["CRPIX2"], "[pixel] reference pixel along image axis 2"]
    
        cutout_wcs_dict["1CRVL{}"] = [wcs_header["CRVAL1"], "[deg] right ascension at reference pixel"]
        cutout_wcs_dict["2CRVL{}"] = [wcs_header["CRVAL2"], "[deg] declination at reference pixel"]

        cutout_wcs_dict["1CUNI{}"] = [wcs_header["CUNIT1"], "physical unit in column dimension"]
        cutout_wcs_dict["2CUNI{}"] = [wcs_header["CUNIT2"], "physical unit in row dimension"]

        cutout_wcs_dict["1CDLT{}"] = [wcs_header["CDELT1"], "[deg] pixel scale in RA dimension"]
        cutout_wcs_dict["2CDLT{}"] = [wcs_header["CDELT1"], "[deg] pixel scale in DEC dimension"]

        cutout_wcs_dict["11PC{}"] = [wcs_header["PC1_1"], "Coordinate transformation matrix element"]
        cutout_wcs_dict["12PC{}"] = [wcs_header["PC1_2"], "Coordinate transformation matrix element"]
        cutout_wcs_dict["21PC{}"] = [wcs_header["PC2_1"], "Coordinate transformation matrix element"]
        cutout_wcs_dict["22PC{}"] = [wcs_header["PC2_2"], "Coordinate transformation matrix element"]

        
        ## Physical keywords ##
        # TODO: Make sure these are correct
        cutout_wcs_dict["WCSN{}P"] = ["PHYSICAL", "table column WCS name"]
        cutout_wcs_dict["WCAX{}P"] = [2, "table column physical WCS dimensions"]
    
        cutout_wcs_dict["1CTY{}P"] = ["RAWX", "table column physical WCS axis 1 type, CCD col"]
        cutout_wcs_dict["2CTY{}P"] = ["RAWY", "table column physical WCS axis 2 type, CCD row"]
    
        cutout_wcs_dict["1CUN{}P"] = ["PIXEL", "table column physical WCS axis 1 unit"]
        cutout_wcs_dict["2CUN{}P"] = ["PIXEL", "table column physical WCS axis 2 unit"]
    
        cutout_wcs_dict["1CRV{}P"] = [self.cutout_lims[0, 0] + 1,
                                      "table column physical WCS ax 1 ref value"]
        cutout_wcs_dict["2CRV{}P"] = [self.cutout_lims[1, 0] + 1,
                                      "table column physical WCS ax 2 ref value"]

        # TODO: can we calculate these? or are they fixed?
        cutout_wcs_dict["1CDL{}P"] = [1.0, "table column physical WCS a1 step"]    
        cutout_wcs_dict["2CDL{}P"] = [1.0, "table column physical WCS a2 step"]
    
        cutout_wcs_dict["1CRP{}P"] = [1, "table column physical WCS a1 reference"]
        cutout_wcs_dict["2CRP{}P"] = [1, "table column physical WCS a2 reference"]

        return cutout_wcs_dict


    def _get_cutout(self, transposed_cube, verbose=True):
        """
        Making a cutout from an image/uncertainty cube that has been transposed 
        to have time on the longest axis.
        
        Parameters
        ----------
        transposed_cube : `numpy.array`
            Transposed image/uncertainty array.
        verbose :  bool
            Optional. If true intermediate information is printed. 

        Returns
        -------
        response :  `numpy.array`, `numpy.array`, `numpy.array`
            The untransposed image cutout array,
            the untransposeduncertainty cutout array,
            and the aperture array (an array the size of a single cutout 
            that is 1 where there is image data and 0 where there isn't)
        """

        # These limits are not guarenteed to be within the image footprint
        xmin, xmax = self.cutout_lims[1]
        ymin, ymax = self.cutout_lims[0]

        # Get the image array limits
        xmax_cube, ymax_cube, _, _ = transposed_cube.shape

        # Adjust limits and figuring out the padding
        padding = np.zeros((3, 2), dtype=int)
        if xmin < 0:
            padding[1, 0] = -xmin
            xmin = 0
        if ymin < 0:
            padding[2, 0] = -ymin
            ymin = 0
        if xmax > xmax_cube:
            padding[1, 1] = xmax - xmax_cube
            xmax = xmax_cube
        if ymax > ymax_cube:
            padding[2, 1] = ymax - ymax_cube
            ymax = ymax_cube       
        
        # Doing the cutout
        cutout = transposed_cube[xmin:xmax, ymin:ymax, :, :]
    
        img_cutout = cutout[:, :, :, 0].transpose((2, 0, 1))
        uncert_cutout = cutout[:, :, :, 1].transpose((2, 0, 1))
    
        # Making the aperture array
        aperture = np.ones((xmax-xmin, ymax-ymin), dtype=np.int32)

        # Adding padding to the cutouts so that it's the expected size
        if padding.any():  # only do if we need to pad
            img_cutout = np.pad(img_cutout, padding, 'constant', constant_values=np.nan)
            uncert_cutout = np.pad(uncert_cutout, padding, 'constant', constant_values=np.nan)
            aperture = np.pad(aperture, padding[1:], 'constant', constant_values=0)

        if verbose:
            print("Image cutout cube shape: {}".format(img_cutout.shape))
            print("Uncertainty cutout cube shape: {}".format(uncert_cutout.shape))
    
        return img_cutout, uncert_cutout, aperture


    def _update_primary_header(self, product, primary_header):
        """
        Updates the primary header for the cutout target pixel file by filling in 
        the object ra and dec with the central cutout coordinates and filling in
        the rest of the TESS target pixel file keywords wil 0/empty strings
        as we do not have access to this information.
        This is a TESS-specific function.

        Parameters
        ----------
        product : str
            The product type to make the cutouts from.
            Can either be 'SPOC' or 'TICA'.
        primary_header : `~astropy.io.fits.Header`
            The primary header from the cube file that will be modified in place for the cutout.
        """

        # Adding cutout specific headers
        primary_header['CREATOR'] = ('astrocut', 'software used to produce this file')
        primary_header['PROCVER'] = (__version__, 'software version')
        primary_header['FFI_TYPE'] = (product, 'the FFI type used to make the cutouts')
        # TODO : The name of FIRST_FFI (and LAST_FFI) is too long to be a header kwd value.
        # Find a way to include these in the headers without breaking astropy (maybe abbreviate?).
        # primary_header['FIRST_FFI'] = (self.first_ffi, 'the FFI used for the primary header 
        # keyword values, except TSTOP')
        # primary_header['LAST_FFI'] = (self.last_ffi, 'the FFI used for the TSTOP keyword value')

        primary_header['RA_OBJ'] = (self.center_coord.ra.deg, '[deg] right ascension')
        primary_header['DEC_OBJ'] = (self.center_coord.dec.deg, '[deg] declination')

        timeref = 'SOLARSYSTEM' if product == 'SPOC' else None
        tassign = 'SPACECRAFT' if product == 'SPOC' else None
        primary_header['TIMEREF'] = (timeref, 'barycentric correction applied to times')        
        primary_header['TASSIGN'] = (tassign, 'where time is assigned')
        primary_header['TIMESYS'] = ('TDB', 'time system is Barycentric Dynamical Time (TDB)')

        primary_header['BJDREFI'] = (2457000, 'integer part of BTJD reference date')           
        primary_header['BJDREFF'] = (0.00000000, 'fraction of the day in BTJD reference date')    
        primary_header['TIMEUNIT'] = ('d', 'time unit for TIME, TSTART and TSTOP')

        delete_kwds_wildcards = ['SC_*', 'RMS*', 'A_*', 'AP_*', 'B_*', 'BP*', 'GAIN*', 'TESS_*', 'CD*',
                                 'CT*', 'CRPIX*', 'CRVAL*', 'MJD*']
        delete_kwds = ['COMMENT', 'FILTER', 'TIME', 'EXPTIME', 'ACS_MODE', 'DEC_TARG', 'FLXWIN', 'RA_TARG',
                       'CCDNUM', 'CAMNUM', 'WCSGDF', 'UNITS', 'CADENCE', 'SCIPIXS', 'INT_TIME', 'PIX_CAT',
                       'REQUANT', 'DIFF_HUF', 'PRIM_HUF', 'QUAL_BIT', 'SPM', 'STARTTJD', 'ENDTJD', 'CRM',
                       'TJD_ZERO', 'CRM_N', 'ORBIT_ID', 'MIDTJD']

        if product == 'TICA':

            # Adding some missing kwds not in TICA (but in Ames-produced SPOC ffis)
            primary_header['EXTVER'] = ('1', 'extension version number (not format version)')
            primary_header['SIMDATA'] = (False, 'file is based on simulated data')
            primary_header['NEXTEND'] = ('2', 'number of standard extensions')
            primary_header['TSTART'] = (primary_header['STARTTJD'], 'observation start time in TJD of first FFI')
            primary_header['TSTOP'] = (primary_header['ENDTJD'], 'observation stop time in TJD of last FFI')
            primary_header['CAMERA'] = (primary_header['CAMNUM'], 'Camera number')
            primary_header['CCD'] = (primary_header['CCDNUM'], 'CCD chip number')
            primary_header['ASTATE'] = (None, 'archive state F indicates single orbit processing')
            primary_header['CRMITEN'] = (primary_header['CRM'], 'spacecraft cosmic ray mitigation enabled')
            primary_header['CRBLKSZ'] = (None, '[exposures] s/c cosmic ray mitigation block siz')
            primary_header['FFIINDEX'] = (primary_header['CADENCE'], 'number of FFI cadence interval of first FFI')
            primary_header['DATA_REL'] = (None, 'data release version number')

            date_obs = Time(primary_header['TSTART']+primary_header['BJDREFI'], format='jd').iso
            date_end = Time(primary_header['TSTOP']+primary_header['BJDREFI'], format='jd').iso
            primary_header['DATE-OBS'] = (date_obs, 'TSTART as UTC calendar date of first FFI')
            primary_header['DATE-END'] = (date_end, 'TSTOP as UTC calendar date of last FFI')

            primary_header['FILEVER'] = (None, 'file format version')
            primary_header['RADESYS'] = (None, 'reference frame of celestial coordinates')
            primary_header['SCCONFIG'] = (None, 'spacecraft configuration ID')
            primary_header['TIMVERSN'] = (None, 'OGIP memo number for file format')

            # Bulk removal with wildcards. Most of these should only live in EXT 1 header.
            for kwd in delete_kwds_wildcards:
                try:
                    del primary_header[kwd]
                except KeyError:
                    continue

            # Removal of specific kwds not relevant for cutouts.
            # Most likely these describe a single FFI, and not
            # the whole cube, which is misleading because we are 
            # working with entire stacks of FFIs. Other keywords 
            # are analogs to ones that have already been added 
            # to the primary header in the lines above.
            for kwd in delete_kwds:
                try:
                    del primary_header[kwd]
                except KeyError:
                    continue

        telapse = primary_header.get("TSTOP", 0) - primary_header.get("TSTART", 0)
        primary_header['TELAPSE '] = (telapse, '[d] TSTOP - TSTART')

        # Updating card comment to be more explicit
        primary_header['DATE'] = (primary_header['DATE'], 'FFI cube creation date')

        # Specifying that some of these headers keyword values are inherited from the first FFI
        if product == 'SPOC':
            primary_header['TSTART'] = (primary_header['TSTART'], 'observation start time in TJD of first FFI')
            primary_header['TSTOP'] = (primary_header['TSTOP'], 'observation stop time in TJD of last FFI')
            primary_header['DATE-OBS'] = (primary_header['DATE-OBS'], 'TSTART as UTC calendar date of first FFI')
            primary_header['DATE-END'] = (primary_header['DATE-END'], 'TSTOP as UTC calendar date of last FFI')
            primary_header['FFIINDEX'] = (primary_header['FFIINDEX'], 'number of FFI cadence interval of first FFI')

        # These are all the things in the TESS pipeline tpfs about the object that we can't fill
        primary_header['OBJECT'] = ("", 'string version of target id ')
        primary_header['TCID'] = (0, 'unique tess target identifier')
        primary_header['PXTABLE'] = (0, 'pixel table id') 
        primary_header['PMRA'] = (0.0, '[mas/yr] RA proper motion') 
        primary_header['PMDEC'] = (0.0, '[mas/yr] Dec proper motion') 
        primary_header['PMTOTAL'] = (0.0, '[mas/yr] total proper motion') 
        primary_header['TESSMAG'] = (0.0, '[mag] TESS magnitude') 
        primary_header['TEFF'] = (0.0, '[K] Effective temperature') 
        primary_header['LOGG'] = (0.0, '[cm/s2] log10 surface gravity') 
        primary_header['MH'] = (0.0, '[log10([M/H])] metallicity') 
        primary_header['RADIUS'] = (0.0, '[solar radii] stellar radius')
        primary_header['TICVER'] = (0, 'TICVER')
        primary_header['TICID'] = (None, 'unique tess target identifier')

    
    def _add_column_wcs(self, table_header, wcs_dict):
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

        wcs_col = False  # says if column is one that requires wcs info
    
        for kwd in table_header:

            # Adding header descriptions for the table keywords
            if "TTYPE" in kwd:
                table_header.comments[kwd] = "column name"
            elif "TFORM" in kwd:
                table_header.comments[kwd] = "column format"
            elif "TUNIT" in kwd:
                table_header.comments[kwd] = "unit"
            elif "TDISP" in kwd:
                table_header.comments[kwd] = "display format"
            elif "TDIM" in kwd:
                table_header.comments[kwd] = "multi-dimensional array spec"
                wcs_col = True  # if column holds 2D array need to add wcs info
            elif "TNULL" in kwd:
                table_header.comments[kwd] = "null value"

            # Adding wcs info if necessary
            if (kwd[:-1] == "TTYPE") and wcs_col:
                wcs_col = False  # reset
                for wcs_key, (val, com) in wcs_dict.items():
                    table_header.insert(kwd, (wcs_key.format(int(kwd[-1])-1), val, com))

        
    def _add_img_kwds(self, table_header):
        """
        Adding extra keywords to the table header.

        Parameters
        ----------
        table_header : `~astropy.io.fits.Header`
            The table header to add keywords to.  It will be modified in place.
        """

        for key in self.img_kwds:
            # We'll skip these TICA-specific image keywords that are single-FFI specific
            # or just not helpful
            if (key == 'TIME') or (key == 'EXPTIME') or (key == 'FILTER'):
                continue
            table_header[key] = tuple(self.img_kwds[key])

            
    def _apply_header_inherit(self, hdu_list):
        """
        The INHERIT keyword indicated that keywords from the primary header should be duplicated in 
        the headers of all subsequent extensions.  This function performs this addition in place to 
        the given hdu list.
        
        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            The hdu list to aaply the INHERIT keyword to.
        """
    
        primary_header = hdu_list[0].header

        reserved_kwds = ["COMMENT", "SIMPLE", "BITPIX", "EXTEND", "NEXTEND"]

        for hdu in hdu_list[1:]:
            if hdu.header.get("INHERIT", False):
                for kwd in primary_header:
                    if (kwd not in hdu.header) and (kwd not in reserved_kwds):
                        hdu.header[kwd] = (primary_header[kwd], primary_header.comments[kwd])
            

    def _build_tpf(self, product, cube_fits, img_cube, uncert_cube, cutout_wcs_dict, aperture):
        """
        Building the cutout target pixel file (TPF) and formatting it to match TESS pipeline TPFs.

        Paramters
        ---------
        product : str
            The product type to make the cutouts from.
            Can either be 'SPOC' or 'TICA'.
        cube_fits : `~astropy.io.fits.hdu.hdulist.HDUList`
            The cube hdu list.
        img_cube : `numpy.array`
            The untransposed image cutout array
        uncert_cube : `numpy.array`
            The untransposed uncertainty cutout array
        cutout_wcs_dict : dict
            Dictionary of wcs keyword/value pairs to be added to each array 
            column in the cutout table header.
        aperture : `numpy.array`
            The aperture array (an array the size of a single cutout 
            that is 1 where there is image data and 0 where there isn't)        
        verbose : bool
            Optional. If true intermediate information is printed. 

        Returns
        -------
        response :  `~astropy.io.fits.HDUList`
            Target pixel file HDU list
        """
        
        # The primary hdu is just the main header, which is the same
        # as the one on the cube file
        primary_hdu = cube_fits[0]
        self._update_primary_header(product, primary_hdu.header)

        cols = list()

        # Adding the cutouts
        tform = str(img_cube[0].size) + "E"
        dims = str(img_cube[0].shape[::-1])
        empty_arr = np.zeros(img_cube.shape)
        # Adding the Time relates columns
        start = 'TSTART' if product == 'SPOC' else 'STARTTJD'
        stop = 'TSTOP' if product == 'SPOC' else 'ENDTJD'
        cols.append(fits.Column(name='TIME', format='D', unit='BJD - 2457000, days', disp='D14.7',
                                array=(cube_fits[2].columns[start].array + cube_fits[2].columns[stop].array)/2))

        if product == 'SPOC':
            cols.append(fits.Column(name='TIMECORR', format='E', unit='d', disp='E14.7',
                                    array=cube_fits[2].columns['BARYCORR'].array))

        # Adding CADENCENO as zeros for SPOC b/c we don't have this info
        cadence_array = empty_arr[:, 0, 0] if product == 'SPOC' else cube_fits[2].columns['CADENCE'].array
        cols.append(fits.Column(name='CADENCENO', format='J', disp='I10', array=cadence_array))

        # Adding counts (-1 b/c we don't have data)
        cols.append(fits.Column(name='RAW_CNTS', format=tform.replace('E', 'J'), unit='count', dim=dims, disp='I8',
                                array=empty_arr-1, null=-1))

        # Adding flux and flux_err (data we actually have!)
        pixel_unit = 'e-/s' if product == 'SPOC' else 'e-'
        cols.append(fits.Column(name='FLUX', format=tform, dim=dims, unit=pixel_unit, disp='E14.7', array=img_cube))
        cols.append(fits.Column(name='FLUX_ERR', format=tform, dim=dims, unit=pixel_unit, disp='E14.7',
                                array=uncert_cube)) 
   
        # Adding the background info (zeros b.c we don't have this info)
        cols.append(fits.Column(name='FLUX_BKG', format=tform, dim=dims, unit=pixel_unit, disp='E14.7',
                                array=empty_arr))
        cols.append(fits.Column(name='FLUX_BKG_ERR', format=tform, dim=dims,
                                unit=pixel_unit, disp='E14.7', array=empty_arr))

        # Adding the quality flags
        data_quality = 'DQUALITY' if product == 'SPOC' else 'QUAL_BIT'
        cols.append(fits.Column(name='QUALITY', format='J', disp='B16.16',
                                array=cube_fits[2].columns[data_quality].array))

        # Adding the position correction info (zeros b.c we don't have this info)
        cols.append(fits.Column(name='POS_CORR1', format='E', unit='pixel', disp='E14.7', array=empty_arr[:, 0, 0]))
        cols.append(fits.Column(name='POS_CORR2', format='E', unit='pixel', disp='E14.7', array=empty_arr[:, 0, 0]))

        # Adding the FFI_FILE column (not in the pipeline tpfs)
        cols.append(fits.Column(name='FFI_FILE', format='38A', unit='pixel',
                                array=cube_fits[2].columns['FFI_FILE'].array))
        
        # making the table HDU
        table_hdu = fits.BinTableHDU.from_columns(cols)
        table_hdu.header['EXTNAME'] = 'PIXELS'
        table_hdu.header['INHERIT'] = True
    
        # Adding the wcs keywords to the columns and removing from the header
        self._add_column_wcs(table_hdu.header, cutout_wcs_dict)

        # Adding the extra image keywords
        self._add_img_kwds(table_hdu.header)

        # Building the aperture HDU
        aperture_hdu = fits.ImageHDU(data=aperture)
        aperture_hdu.header['EXTNAME'] = 'APERTURE'
        for kwd, val, cmt in self.cutout_wcs.to_header().cards: 
            aperture_hdu.header.set(kwd, val, cmt)

        # Adding extra aperture keywords (TESS specific)
        aperture_hdu.header.set("NPIXMISS", None, "Number of op. aperture pixels not collected")
        aperture_hdu.header.set("NPIXSAP", None, "Number of pixels in optimal aperture")

        # Adding goodness-of-fit keywords
        for key in self.cutout_wcs_fit:
            aperture_hdu.header[key] = tuple(self.cutout_wcs_fit[key])
        
        aperture_hdu.header['INHERIT'] = True
    
        cutout_hdu_list = fits.HDUList([primary_hdu, table_hdu, aperture_hdu])
        
        self._apply_header_inherit(cutout_hdu_list)

        return cutout_hdu_list


    def cube_cut(self, cube_file, coordinates, cutout_size,
                 product='SPOC', target_pixel_file=None, output_path=".", verbose=False):
        """
        Takes a cube file (as created by `~astrocut.CubeFactory`), and makes a cutout target pixel 
        file of the given size around the given coordinates. The target pixel file is formatted like
        a TESS pipeline target pixel file.

        Parameters
        ----------
        cube_file : str
            The cube file containing all the images to be cutout.  
            Must be in the format returned by ~astrocut.make_cube.
        coordinates : str or `astropy.coordinates.SkyCoord` object
            The position around which to cutout. 
            It may be specified as a string ("ra dec" in degrees) 
            or as the appropriate `~astropy.coordinates.SkyCoord` object.
        cutout_size : int, array-like, `~astropy.units.Quantity`
            The size of the cutout array. If ``cutout_size``
            is a scalar number or a scalar `~astropy.units.Quantity`,
            then a square cutout of ``cutout_size`` will be created.  If
            ``cutout_size`` has two elements, they should be in ``(ny, nx)``
            order.  Scalar numbers in ``cutout_size`` are assumed to be in
            units of pixels. `~astropy.units.Quantity` objects must be in pixel or
            angular units.
        product : str
            The product type to make the cutouts from.
            Can either be 'SPOC' or 'TICA' (default is 'SPOC').
        target_pixel_file : str
            Optional. The name for the output target pixel file. 
            If no name is supplied, the file will be named: 
            ``<cube_file_base>_<ra>_<dec>_<cutout_size>_astrocut.fits``
        output_path : str
            Optional. The path where the output file is saved. 
            The current directory is default.
        verbose : bool
            Optional. If true intermediate information is printed. 

        Returns
        -------
        response: string or None
            If successfull, returns the path to the target pixel file, 
            if unsuccessful returns None.
        """

        if verbose:
            start_time = time()

        warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)
        with fits.open(cube_file, mode='denywrite', memmap=True) as cube:

            # Get the info we need from the data table
            self._parse_table_info(product, cube[2].data, verbose)

            if isinstance(coordinates, SkyCoord):
                self.center_coord = coordinates
            else:
                self.center_coord = SkyCoord(coordinates, unit='deg')

            if verbose:
                print("Cutout center coordinate: {},{}".format(self.center_coord.ra.deg,
                                                               self.center_coord.dec.deg))

            # Making size into an array [ny, nx]
            if np.isscalar(cutout_size):
                cutout_size = np.repeat(cutout_size, 2)

            if isinstance(cutout_size, u.Quantity):
                cutout_size = np.atleast_1d(cutout_size)
                if len(cutout_size) == 1:
                    cutout_size = np.repeat(cutout_size, 2)

            if len(cutout_size) > 2:
                warnings.warn("Too many dimensions in cutout size, only the first two will be used.",
                              InputWarning)
                cutout_size = cutout_size[:2]
                
            # Get cutout limits
            self._get_cutout_limits(cutout_size)

            if verbose:
                print("xmin,xmax: {}".format(self.cutout_lims[1]))
                print("ymin,ymax: {}".format(self.cutout_lims[0]))

            # Make the cutout
            img_cutout, uncert_cutout, aperture = self._get_cutout(cube[1].data, verbose=verbose)

            # Get cutout wcs info
            cutout_wcs_full = self._get_full_cutout_wcs(cube[2].header)
            max_dist, sigma = self._fit_cutout_wcs(cutout_wcs_full, img_cutout.shape[1:])
            if verbose:
                print("Maximum distance between approximate and true location: {}".format(max_dist))
                print("Error in approximate WCS (sigma): {}".format(sigma))
                
            cutout_wcs_dict = self._get_cutout_wcs_dict()
    
            # Build the TPF
            tpf_object = self._build_tpf(product, cube, img_cutout, uncert_cutout, cutout_wcs_dict, aperture)

            if verbose:
                write_time = time()

            if not target_pixel_file:
                _, flename = os.path.split(cube_file)

                width = self.cutout_lims[0, 1]-self.cutout_lims[0, 0]
                height = self.cutout_lims[1, 1]-self.cutout_lims[1, 0]
                target_pixel_file = "{}_{:7f}_{:7f}_{}x{}_astrocut.fits".format(flename.rstrip('.fits').rstrip("-cube"),
                                                                                self.center_coord.ra.value,
                                                                                self.center_coord.dec.value,
                                                                                width,
                                                                                height)
            target_pixel_file = os.path.join(output_path, target_pixel_file)
            
        
            if verbose:
                print("Target pixel file: {}".format(target_pixel_file))

            # Make sure the output directory exists
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        
            # Write the TPF
            tpf_object.writeto(target_pixel_file, overwrite=True, checksum=True)

        if verbose:
            print("Write time: {:.2} sec".format(time()-write_time))
            print("Total time: {:.2} sec".format(time()-start_time))

        return target_pixel_file
