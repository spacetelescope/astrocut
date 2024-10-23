from abc import abstractmethod, ABC
import pathlib
from typing import Union

import numpy as np
from s3path import S3Path
from astropy.coordinates import SkyCoord

from astrocut.utils.utils import _handle_verbose, parse_size_input


class Cutout(ABC):

    def __init__(self, input_file: Union[str, pathlib.Path, S3Path], coordinates, cutout_size: int = 25,
                 fill_value: Union[int, float] = np.nan, write_file: bool = True,
                 output_file: str = 'cutout.fits', verbose: bool = True):
        
        # Log messages according to verbosity
        _handle_verbose(verbose)

        # Get coordinates as a SkyCoord object
        if not isinstance(coordinates, SkyCoord):
            coordinates = SkyCoord(coordinates, unit='deg')
        self.coordinates = coordinates

        # Turning the cutout size into a 2 member array
        cutout_size = parse_size_input(cutout_size)
        self.cutout_size = cutout_size

        self.input_file = input_file
        self.fill_value = fill_value
        self.write_file = write_file
        self.output_file = output_file
        self.data = None
        self.wcs = None


    @abstractmethod
    def _load_data(self):
        # Override
        pass


    @abstractmethod
    def make_cutout(self):
        # Override
        pass


    @abstractmethod
    def _write_fits(self):
        # Override
        pass


    def _write_asdf(self):
        # Override
        pass


    def _write_cutout(self):
        # check the output file type
        out = pathlib.Path(self.output_file)
        write_as = out.suffix or '.fits'
        self.output_file = self.output_file if out.suffix else str(out) + write_as

        if write_as == '.fits':
            self._write_fits()
        elif write_as == '.asdf':
            self._write_asdf()