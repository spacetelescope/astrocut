from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Union

import numpy as np
from s3path import S3Path
from astropy.coordinates import SkyCoord

from . import log
from .utils.utils import _handle_verbose, parse_size_input


class Cutout(ABC):

    def __init__(self, input_files: List[Union[str, Path, S3Path]], coordinates, cutout_size: int = 25,
                 fill_value: Union[int, float] = np.nan, memory_only: bool = False,
                 output_dir: str = '.', verbose: bool = True):
        
        # Log messages according to verbosity
        _handle_verbose(verbose)

        # Making sure we have an array of images
        if isinstance(input_files, str) or isinstance(input_files, Path):
            input_files = [input_files]

        # Get coordinates as a SkyCoord object
        if coordinates and not isinstance(coordinates, SkyCoord):
            coordinates = SkyCoord(coordinates, unit='deg')
        self.coordinates = coordinates
        log.debug('Coordinates: %s', self.coordinates)

        # Turning the cutout size into a 2 member array
        self.cutout_size = parse_size_input(cutout_size)
        log.debug('Cutout size: %s', self.cutout_size)

        self.input_files = input_files
        self.fill_value = fill_value
        self.memory_only = memory_only
        self.output_dir = output_dir
        self.verbose = verbose


    @abstractmethod
    def _load_data(self):
        # Override
        pass


    @abstractmethod
    def cutout(self):
        # Override
        pass


    @abstractmethod
    def _write_cutout(self):
        # Override
        pass