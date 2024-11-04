from datetime import date
from pathlib import Path
from typing import List, Union
import warnings

from astropy import log as astropy_log
from astropy.nddata import Cutout2D, NoOverlapError
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from s3path import S3Path

from .exceptions import DataWarning, InvalidQueryError
from .Cutout import Cutout
from . import __version__, log

class FITSCutout(Cutout):


    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates, cutout_size: int = 25,
                 fill_value: Union[int, float] = np.nan, extension=None, single_outfile: bool = True, 
                 cutout_prefix: str = 'cutout', output_dir: str = '.', memory_only: bool = False, verbose: bool = True):
        super().__init__(input_files, coordinates, cutout_size, fill_value, memory_only, output_dir, verbose)
        
        # If a single extension is given, make it a list
        if isinstance(extension, int):
            extension = [extension]
        self.extension = extension
        self.single_outfile = single_outfile
        self.cutout_prefix = cutout_prefix
        self.num_empty = 0


    def _parse_extensions(self, input_file, infile_exts):
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
    
        if len(infile_exts) == 0:
            warnings.warn(f"No image extensions with data found in {input_file}, skipping...",
                          DataWarning)
            return []
                
        if self.extension is None:
            cutout_exts = infile_exts[:1]  # Take the first image extension
        elif self.extension == 'all':
            cutout_exts = infile_exts  # Take all the extensions
        else:  # User input extentions
            cutout_exts = [x for x in infile_exts if x in self.extension]
            if len(cutout_exts) < len(self.extension):
                warnings.warn((f"Not all requested extensions in {input_file} are image extensions or have data, "
                               f"extension(s) {','.join([x for x in self.extension if x not in cutout_exts])} will be skipped."),
                               DataWarning)

        return cutout_exts


    def _load_data(self, input_file):
        fsspec_kwargs = {"anon": True} if 's3://' in input_file else None

        hdulist = fits.open(input_file, mode='denywrite', memmap=True, fsspec_kwargs=fsspec_kwargs)

        # Sorting out which extension(s) to cutout
        infile_exts = np.where([x.is_image and (x.data is not None) for x in hdulist])[0]
        cutout_inds = self._parse_extensions(input_file, infile_exts)

        return (hdulist, cutout_inds)


    def _hducut(self, img_hdu):
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

        log.debug("Original image shape: %s", img_hdu.data.shape)
        img_cutout = Cutout2D(img_hdu.data,
                              position=self.coordinates,
                              wcs=img_wcs,
                              size=(self.cutout_size[1], self.cutout_size[0]),
                              mode='partial',
                              fill_value=self.fill_value)
        log.debug("Image cutout shape: %s", img_cutout.shape)
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

        return fits.ImageHDU(header=hdu_header, data=img_cutout.data)
    

    def _construct_fits_from_hdus(self, cutout_hdus):
        """
        Make one or more cutout hdus to a single fits object.

        Parameters
        ----------
        cutout_hdus : list or `~astropy.io.fits.hdu.image.ImageHDU`
            The `~astropy.io.fits.hdu.image.ImageHDU` object(s) to be written to the fits file.
        output_path : str
            The full path to the output fits file.
        center_coord : `~astropy.coordinates.sky_coordinate.SkyCoord`
            The center coordinate of the image cutouts.  TODO: make more general?
        """
        if isinstance(cutout_hdus, fits.hdu.image.ImageHDU):
            cutout_hdus = [cutout_hdus]
        
        # Setting up the Primary HDU
        keywords = dict()
        if self.coordinates:
            keywords = {"RA_OBJ": (self.coordinates.ra.deg, '[deg] right ascension'),
                        "DEC_OBJ": (self.coordinates.dec.deg, '[deg] declination')}

        # Build the primary HDU with keywords
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header.extend([("ORIGIN", 'STScI/MAST', "institution responsible for creating this file"),
                                   ("DATE", str(date.today()), "file creation date"),
                                   ('PROCVER', __version__, 'software version')])
        for kwd in keywords:
            primary_hdu.header[kwd] = keywords[kwd]

        return fits.HDUList([primary_hdu] + cutout_hdus)
    

    def _write_cutout(self, cutout_fits, cutout_path):
        # Writing out the hdu often causes a warning as the ORIG_FLE card description is truncated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            cutout_fits.writeto(cutout_path, overwrite=True, checksum=True)


    def cutout(self):
        cutout_hdu_dict = {}
        num_cutouts = 0
        for file in self.input_files:
            # Load data
            hdulist, cutout_inds = self._load_data(file)

            # create HDU cutouts
            num_cutouts += len(cutout_inds)
            cutout_hdus = []
            for ind in cutout_inds:
                try:
                    cutout = self._hducut(hdulist[ind])

                    # Adding a few more keywords
                    cutout.header["ORIG_EXT"] = (ind, "Extension in original file.")
                    if not cutout.header.get("ORIG_FLE") and hdulist[0].header.get("FILENAME"):
                        cutout.header["ORIG_FLE"] = hdulist[0].header.get("FILENAME")

                    cutout_hdus.append(cutout)
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
        if self.num_empty == num_cutouts:
            raise InvalidQueryError("Cutout contains no data! (Check image footprint.)")
        
        # Make sure that output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.single_outfile:
            log.debug('Returning cutout as a single FITS file.')
            cutout_hdus = [x for file in cutout_hdu_dict for x in cutout_hdu_dict[file]]
            cutout_fits = self._construct_fits_from_hdus(cutout_hdus)

            if self.memory_only:
                return [cutout_fits]
            else:
                cutout_path = "{}_{:7f}_{:7f}_{}-x-{}_astrocut.fits".format(self.cutout_prefix,
                                                                            self.coordinates.ra.value,
                                                                            self.coordinates.dec.value,
                                                                            str(self.cutout_size[0]).replace(' ', ''), 
                                                                            str(self.cutout_size[1]).replace(' ', ''))
                cutout_path = Path(self.output_dir, cutout_path)
                self._write_cutout(cutout_fits, cutout_path)
                log.debug("Cutout fits file: %s", cutout_path)
                return cutout_path.as_posix()
        else:  # One output file per input file
            log.debug('Returning cutouts as individual FITS files.')

            all_fits = []
            for file, cutout_list in cutout_hdu_dict.items():
                if np.array([x.header.get("EMPTY") for x in cutout_list]).all():
                    warnings.warn(f"Cutout of {file} contains no data and will not be written.",
                                DataWarning)
                    continue

                cutout_fits = self._construct_fits_from_hdus(cutout_list)
                if self.memory_only:
                    all_fits.append(cutout_fits)
                else:
                    filename = "{}_{:7f}_{:7f}_{}-x-{}_astrocut.fits".format(Path(file).name.rstrip('.fits'),
                                                                             self.coordinates.ra.value,
                                                                             self.coordinates.dec.value,
                                                                             str(self.cutout_size[0]).replace(' ', ''), 
                                                                             str(self.cutout_size[1]).replace(' ', ''))
                    cutout_path = Path(self.output_dir, filename)
                    self._write_cutout(cutout_fits, cutout_path)
                    all_fits.append(cutout_path.as_posix())

            return all_fits
    