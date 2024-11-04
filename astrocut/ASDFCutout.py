import copy
import pathlib
from typing import Tuple, Union

import asdf
import gwcs
import numpy as np
import requests
import s3fs
from astropy.io.fits import writeto
from astropy.modeling import models
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.units import Quantity
from astropy.wcs import WCS
from s3path import S3Path

from . import log
from .Cutout import Cutout

class ASDFCutout(Cutout):


    def __init__(self, input_files: Union[str, pathlib.Path, S3Path], coordinates, cutout_size: int = 25,
                 fill_value: Union[int, float] = np.nan, memory_only: bool = False, output_dir: str ='.', 
                 output_file: str = 'asdf_cutout.fits', key: str = None, secret: str = None, token: str = None,
                 verbose: bool = True):
        super().__init__(input_files, coordinates, cutout_size, fill_value, memory_only, output_dir, verbose)
        self.output_file = output_file
        self.key = key
        self.secret = secret
        self.token = token


    def _get_cloud_http(self, input_file) -> str:
        """ 
        Get the HTTP URI of a cloud resource from an S3 URI.

        Parameters
        ----------
        s3_uri : string | S3Path
            the S3 URI of the cloud resource
        key : string
            Default None. Access key ID for S3 file system.
        secret : string
            Default None. Secret access key for S3 file system.
        token : string
            Default None. Security token for S3 file system.
        verbose : bool
            Default False. If true intermediate information is printed.
        """
        # check if public or private by sending an HTTP request
        s3_path = S3Path.from_uri(input_file) if isinstance(input_file, str) else input_file
        url = f'https://{s3_path.bucket}.s3.amazonaws.com/{s3_path.key}'
        resp = requests.head(url, timeout=10)
        is_anon = False if resp.status_code == 403 else True
        if not is_anon:
            log.debug('Attempting to access private S3 bucket: %s', s3_path.bucket)

        # create file system and get URL of file
        fs = s3fs.S3FileSystem(anon=is_anon, key=self.key, secret=self.secret, token=self.token)
        with fs.open(input_file, 'rb') as f:
            return f.url()


    def _read_from_asdf(self):
        # Override in RomanCutout.py
        pass


    def _load_data(self, input_file):
        # if file comes from AWS cloud bucket, get HTTP URL to open with asdf
        if (isinstance(input_file, str) and input_file.startswith('s3://')) or isinstance(input_file, S3Path):
            input_file = self._get_cloud_http(input_file)

        # Get data and GWCS object from ASDF input file
        data, gwcs = self._read_from_asdf(input_file)

        # convert the gwcs object into a wcs
        # Update WCS header with some keywords that it's missing.
        # Otherwise, it won't work with astropy.wcs tools (TODO: Figure out why. What are these keywords for?)
        header = gwcs.to_fits_sip()
        for k in ['cpdis1', 'cpdis2', 'det2im1', 'det2im2', 'sip']:
            if k not in header:
                header[k] = 'na'

        # New WCS object with updated header
        wcs = WCS(header)

        return (data, wcs, gwcs)
        

    def _slice_gwcs(self, cutout, gwcs) -> gwcs.wcs.WCS:
        """ 
        Slice the original gwcs object.

        "Slices" the original gwcs object down to the cutout shape.  This is a hack
        until proper gwcs slicing is in place a la fits WCS slicing.  The ``slices``
        keyword input is a tuple with the x, y cutout boundaries in the original image
        array, e.g. ``cutout.slices_original``.  Astropy Cutout2D slices are in the form
        ((ymin, ymax, None), (xmin, xmax, None))

        Parameters
        ----------
        gwcsobj : gwcs.wcs.WCS
            the original gwcs from the input image
        slices : Tuple[slice, slice]
            the cutout x, y slices as ((ymin, ymax), (xmin, xmax))

        Returns
        -------
        gwcs.wcs.WCS
            The sliced gwcs object
        """
        tmp = copy.deepcopy(gwcs)

        # get the cutout array bounds and create a new shift transform to the cutout
        # add the new transform to the gwcs
        slices = cutout.slices_original
        xmin, xmax = slices[1].start, slices[1].stop
        ymin, ymax = slices[0].start, slices[0].stop
        shape = (ymax - ymin, xmax - xmin)
        offsets = models.Shift(xmin, name='cutout_offset1') & models.Shift(ymin, name='cutout_offset2')
        tmp.insert_transform('detector', offsets, after=True)

        # modify the gwcs bounding box to the cutout shape
        tmp.bounding_box = ((0, shape[0] - 1), (0, shape[1] - 1))
        tmp.pixel_shape = shape[::-1]
        tmp.array_shape = shape
        return tmp
    

    def _write_fits(self, cutout):
        # check if the data is a quantity and get the array data
        if isinstance(cutout.data, Quantity):
            data = cutout.data.value
        else:
            data = cutout.data

        writeto(self.output_file, data=data, header=cutout.wcs.to_header(relax=True), overwrite=True)


    def _write_asdf(self, cutout, gwcs):
        # slice the origial gwcs to the cutout
        sliced_gwcs = self._slice_gwcs(cutout, gwcs)

        # create the asdf tree
        tree = {'roman': {'meta': {'wcs': sliced_gwcs}, 'data': cutout.data}}
        af = asdf.AsdfFile(tree)

        # Write the data to a new file
        af.write_to(self.output_file)


    def _write_cutout(self, cutout, gwcs):
        # check the output file type
        out = pathlib.Path(self.output_file)
        write_as = out.suffix or '.fits'
        self.output_file = self.output_file if out.suffix else str(out) + write_as

        if write_as == '.fits':
            self._write_fits(cutout)
        elif write_as == '.asdf':
            self._write_asdf(cutout, gwcs)


    def cutout(self):

        cutouts = []
        for file in self.input_files:

            # load data
            data, wcs, gwcs = self._load_data(file)

            # create the cutout
            try:
                cutout = Cutout2D(data,
                                  position=self.coordinates,
                                  wcs=wcs,
                                  size=self.cutout_size,
                                  mode='partial',
                                  fill_value=self.fill_value)
            except NoOverlapError as e:
                raise RuntimeError('Could not create 2d cutout.  The requested cutout does not overlap with the '
                                'original image.') from e
        
            if not self.memory_only:
                self._write_cutout(cutout, gwcs)
            cutouts.append(cutout)
        
        return cutouts if len(cutouts) > 1 else cutouts[0]