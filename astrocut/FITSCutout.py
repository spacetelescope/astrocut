from datetime import date
import pathlib
from typing import Union
import warnings

from astropy import log as astropy_log
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from s3path import S3Path

from .exceptions import DataWarning, InvalidQueryError
from .Cutout import Cutout
from . import __version__

class FITSCutout(Cutout):

    # Makes a cutout of a SINGLE fits file and returns an HDUlist

    def __init__(self, input_file: Union[str, pathlib.Path, S3Path], coordinates, cutout_size: int = 25,
                 fill_value: Union[int, float] = np.nan, write_file: bool = True,
                 output_file: str = 'cutout.fits', extension=None, correct_wcs: bool = False, verbose: bool = True):
        super().__init__(input_file, coordinates, cutout_size, fill_value, write_file, output_file, verbose)
        
        # If a single extension is given, make it a list
        if isinstance(extension, int):
            extension = [extension]
        self.extension = extension

        self.correct_wcs = correct_wcs
        self.num_empty = 0


    def _parse_extensions(self):
        """
        Given a list of image extensions available in the file with infile_name, cross-match with
        user input extensions to figure out which extensions to use for cutout.

        Parameters
        ----------
        infile_exts : array
        infile_name : str
        user_exts : int, list of ints, None, or 'all'
            Optional, default None. Default is to cutout the first extension that has image data.
            The user can also supply one or more extensions to cutout from (integers), or "all".

        Returns
        -------
        response : array
            List of extensions to be cutout.
        """
    
        if len(self.infile_exts) == 0:
            warnings.warn(f"No image extensions with data found in {self.input_file}, skipping...",
                          DataWarning)
            return []
                
        if self.extension is None:
            cutout_exts = self.infile_exts[:1]  # Take the first image extension
        elif self.extension == 'all':
            cutout_exts = self.infile_exts  # Take all the extensions
        else:  # User input extentions
            cutout_exts = [x for x in self.infile_exts if x in self.extension]
            if len(cutout_exts) < len(self.extension):
                warnings.warn((f"Not all requested extensions in {self.input_file} are image extensions or have data, "
                               f"extension(s) {','.join([x for x in self.extension if x not in cutout_exts])} will be skipped."),
                               DataWarning)

        return cutout_exts


    def _load_data(self):
        fsspec_kwargs = {"anon": True} if 's3://' in self.input_file else None

        self.hdulist = fits.open(self.input_file, mode='denywrite', memmap=True, fsspec_kwargs=fsspec_kwargs)

        # Sorting out which extension(s) to cutout
        self.infile_exts = np.where([x.is_image and (x.data is not None) for x in self.hdulist])[0]
        self.cutout_inds = self._parse_extensions()


    def _hducut(self, ind):
        img_hdu = self.hdulist[ind]
        hdu_header = fits.Header(img_hdu.header, copy=True)

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

        img_cutout = Cutout2D(img_hdu.data,
                              position=self.coordinates,
                              wcs=img_wcs,
                              size=self.cutout_size,
                              mode='partial')
        cutout_wcs = img_cutout.wcs

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
        cutout_data = img_cutout.data
        if (cutout_data == 0).all() or (np.isnan(cutout_data)).all():
            hdu_header["EMPTY"] = (True, "Indicates no data in cutout image.")
            self.num_empty += 1

        # Adding a few more keywords
        hdu_header["ORIG_EXT"] = (ind, "Extension in original file.")
        if not hdu_header.get("ORIG_FLE") and self.hdulist[0].header.get("FILENAME"):
            hdu_header["ORIG_FLE"] = self.hdulist[0].header.get("FILENAME")

        self.hdulist.close()

        return fits.ImageHDU(header=hdu_header, data=img_cutout.data)
    

    def _build_astrocut_primaryhdu(self, **keywords):
        """
        TODO: Document
        """

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header.extend([("ORIGIN", 'STScI/MAST', "institution responsible for creating this file"),
                                ("DATE", str(date.today()), "file creation date"),
                                ('PROCVER', __version__, 'software version')])
        for kwd in keywords:
            primary_hdu.header[kwd] = keywords[kwd]

        return primary_hdu
    

    def _construct_fits_from_hdus(self):
        """
        Make one or more cutout hdus to a single fits object, optionally save the file to disk.

        Parameters
        ----------
        cutout_hdus : list or `~astropy.io.fits.hdu.image.ImageHDU`
            The `~astropy.io.fits.hdu.image.ImageHDU` object(s) to be written to the fits file.
        output_path : str
            The full path to the output fits file.
        center_coord : `~astropy.coordinates.sky_coordinate.SkyCoord`
            The center coordinate of the image cutouts.  TODO: make more general?
        """

        if isinstance(self.cutout_hdus, fits.hdu.image.ImageHDU):
            self.cutout_hdus = [self.cutout_hdus]
        
        # Setting up the Primary HDU
        keywords = dict()
        if self.coordinates:
            keywords = {"RA_OBJ": (self.coordinates.ra.deg, '[deg] right ascension'),
                        "DEC_OBJ": (self.coordinates.dec.deg, '[deg] declination')}
        primary_hdu = self._build_astrocut_primaryhdu(**keywords)

        self.cutout_hdulist = fits.HDUList([primary_hdu] + self.cutout_hdus)
    

    def _write_fits(self):
        # Writing out the hdu often causes a warning as the ORIG_FLE card description is truncated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            self.cutout_hdulist.writeto(self.output_file, overwrite=True, checksum=True)


    def make_cutout(self):
        # load data
        self._load_data()

        # create HDU cutouts
        self.cutout_hdus = []
        for ind in self.cutout_inds:
            try:
                cutout = self._hducut(ind)
                self.cutout_hdus.append(cutout)
            except OSError as err:
                warnings.warn((f"Error {err} encountered when performing cutout on {self.input_file}, "
                               f"extension {ind}, skipping..."),
                               DataWarning)
                self.num_empty += 1

        # If no cutouts contain data, raise exception
        if self.num_empty == len(self.cutout_inds):
            raise InvalidQueryError("Cutout contains no data! (Check image footprint.)")
        
        self._construct_fits_from_hdus()
        
        if self.write_file:
            self._write_fits()

        return self.cutout_hdulist
    