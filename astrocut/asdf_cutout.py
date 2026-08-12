import sys
import warnings
from contextlib import nullcontext
from copy import deepcopy
from datetime import date
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import asdf
import gwcs
import numpy as np
from asdf.tags.core.ndarray import NDArrayType
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import models
from astropy.nddata.utils import Cutout2D, NoOverlapError
from astropy.table import Table
from astropy.units import Quantity
from astropy.utils.decorators import deprecated_renamed_argument
from astropy.wcs import WCS
from packaging.version import Version
from PIL.Image import Image
from s3path import S3Path

from . import __version__, log
from .exceptions import DataWarning, InputWarning, InvalidInputError, InvalidQueryError, ModuleWarning
from .image_cutout import ImageCutout


class ASDFCutout(ImageCutout):
    """
    Class for creating cutouts from ASDF files.

    Parameters
    ----------
    input_files : list
        List of input image files.
    coordinates : str | `~astropy.coordinates.SkyCoord` | list
        Coordinates of the center of the cutout. You can pass a single coordinate or a list of coordinates.
    cutout_size : int | array | list | tuple | `~astropy.units.Quantity`
        Size of the cutout array.
    fill_value : int | float
        Value to fill the cutout with if the cutout is outside the image. Default is np.nan. If the input data array
        has an integer data type, the fill value will be converted to an integer (e.g., a fill value of 1.0 will be
        converted to 1). If the conversion fails, it will default to 0.
    key : str
        Optional, default None. Access key ID for S3 file system.
    secret : str
        Optional, default None. Secret access key for S3 file system.
    token : str
        Optional, default None. Security token for S3 file system.
    lite : bool
        Optional, default True. If True, the cutout will be created in "lite" mode,
        which means that it will only contain the data and an updated world coordinate system.
        If False, cutouts will be made from all arrays in the input file (e.g., data, error,
        uncertainty, variance, etc.) where the last two dimensions match the shape of the science data array.
        It also preserves all of the metadata from the input file.
    asdf_kwargs : dict, optional
        Keyword arguments passed to `asdf.open` when reading input files. By default,
        `memmap=True` is applied unless explicitly overridden.
    verbose : bool
        If True, log messages are printed to the console.

    Attributes
    ----------
    cutouts : list
        The cutouts as a list of `astropy.nddata.Cutout2D` objects.
    cutouts_by_file : dict
        The cutouts as `astropy.nddata.Cutout2D` objects stored by input filename.
    fits_cutouts : list
        The cutouts as a list `astropy.io.fits.HDUList` objects.
    asdf_cutouts : list
        The cutouts as a list of `asdf.AsdfFile` objects.
    image_cutouts : list
        List of `~PIL.Image.Image` objects representing the cutouts.

    Methods
    -------
    cutout()
        Generate cutouts from a list of input images.
    write_as_fits(output_dir)
        Write the cutouts to disk or memory in FITS format.
    write_as_asdf(output_dir)
        Write the cutouts to disk or memory in ASDF format.
    """

    def __init__(
        self,
        input_files: List[Union[str, Path, S3Path]],
        coordinates: Union[SkyCoord, str, List[Union[SkyCoord, str]]],
        cutout_size: Union[int, np.ndarray, Quantity, List[int], Tuple[int]] = 25,
        fill_value: Union[int, float] = np.nan,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        token: Optional[str] = None,
        lite: Optional[bool] = True,
        asdf_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        super().__init__(input_files, coordinates, cutout_size, fill_value, verbose=verbose)

        # Must be using Python 3.11 or higher to support stdatamodels and ASDF-in-FITS embedding
        self._py311_or_higher = sys.version_info >= (3, 11)
        self._asdf_in_fits = None  # Will be set to the asdf_in_fits module if available

        # Assign AWS credential attributes
        self._key = key
        self._secret = secret
        self._token = token
        self._mission_kwd = "roman"

        # Store cutouts as a table
        self._cutouts = None  # Store Cutout2D objects
        self._asdf_cutouts = None  # Store ASDF objects
        self._fits_cutouts = None  # Store FITS objects
        self._asdf_trees = {}  # Store ASDF trees for each cutout
        self._lite = lite  # Flag for lite mode
        if asdf_kwargs is not None and not isinstance(asdf_kwargs, dict):
            raise InvalidInputError("asdf_kwargs must be a dictionary.")
        self._asdf_open_kwargs = dict(asdf_kwargs or {})
        self._asdf_open_kwargs.setdefault("memmap", True)
        self._primary_header_template = None  # Optional template for primary header keywords
        self._fill_value_cache = {}  # Cache for converted fill values based on input data types
        self._gwcs_to_fits_cache = {}  # Cache for converted GWCS to FITS WCS objects

        # Make cutouts
        self.cutout()

    @property
    def cutouts(self) -> Table:
        """
        Return the cutouts as an `astropy.table.Table` with columns for input file, coordinate, and the corresponding
        `astropy.nddata.Cutout2D` object.
        """
        if self._cutouts is not None:
            return self._cutouts

        self._cutouts = self.get_cutouts()
        return self._cutouts

    @property
    def asdf_cutouts(self) -> Table:
        """
        Return the cutouts as a list of `asdf.AsdfFile` objects.
        """
        if self._asdf_cutouts is not None:
            return self._asdf_cutouts

        self._asdf_cutouts = self.get_asdf_cutouts()
        return self._asdf_cutouts

    @property
    def fits_cutouts(self) -> Table:
        """
        Return the cutouts as a list `astropy.io.fits.HDUList` objects.
        """
        if self._fits_cutouts is not None:
            return self._fits_cutouts

        self._fits_cutouts = self.get_fits_cutouts()
        return self._fits_cutouts

    def get_cutouts(
        self,
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> Table:
        """
        Get the cutouts as an `astropy.table.Table` with columns for input file, coordinate, and the corresponding
        `astropy.nddata.Cutout2D` object.

        Parameters
        ----------
        input_files : list
            Optional. List of input image files to include in the output. If not specified, all input files will be
            included.
        coordinates : list
            Optional. List of coordinates to include in the output. If not specified, all coordinates will be included.

        Returns
        -------
        cutouts : `astropy.table.Table`
            Table with columns for input file, coordinate, and the corresponding `astropy.nddata.Cutout2D` object.
        """
        if self._cutouts is not None:
            # Filter existing cutouts by input_files and coordinates if provided
            return self._return_filtered_table(self._cutouts, input_files=input_files, coordinates=coordinates)

        return self._build_cutout_table(
            self.iter_cutouts(input_files=input_files, coordinates=coordinates),
        )

    def _return_filtered_table(
        self,
        table: Table,
        input_files: Optional[List[Union[str, Path, S3Path]]],
        coordinates: Optional[List[Union[SkyCoord, str]]],
    ) -> Table:
        """
        Return a filtered version of the input table based on the specified input files and coordinates.

        Parameters
        ----------
        table : `astropy.table.Table`
            The input table to filter.
        input_files : list, optional
            List of input image files to include in the output. If not specified, all input files will be included.
        coordinates : list, optional
            List of coordinates to include in the output. If not specified, all coordinates will be included.

        Returns
        -------
        filtered_table : `astropy.table.Table`
            The filtered table containing only the specified input files and coordinates.
        """
        files_to_include, coords_to_include = self._resolve_selection(input_files, coordinates)
        mask = np.ones(len(table), dtype=bool)

        if input_files is not None:
            mask &= np.isin(table["file"], files_to_include)

        if coordinates is not None:
            coord_keys = [coord.to_string(precision=8) for coord in table["coordinate"]]
            mask &= np.isin(coord_keys, coords_to_include)

        return table[mask]

    def iter_cutouts(
        self,
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> Iterator[Tuple[str, SkyCoord, Cutout2D]]:
        """
        Yield base `~astropy.nddata.Cutout2D` cutouts lazily for selected file/coordinate pairs.

        Parameters
        ----------
        input_files : list
            Optional. List of input image files to include in the output. If not specified, all input files will be
            included.
        coordinates : list
            Optional. List of coordinates to include in the output. If not specified, all coordinates will be included.

        Yields
        ------
        tuple
            Tuples of (input file, coordinate, `astropy.nddata.Cutout2D`).
        """
        for file, coord in self.iter_file_coord_pairs(input_files=input_files, coordinates=coordinates):
            yield file, SkyCoord(coord, unit="deg"), self.cutouts_by_file[file][coord]

    def get_asdf_cutouts(
        self,
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> Table:
        """
        Get the cutouts as an `astropy.table.Table` with columns for input file, coordinate,
        and the corresponding `asdf.AsdfFile` object.

        Parameters
        ----------
        input_files : list
            Optional. List of input image files to include in the output. If not specified, all input files will be
            included.
        coordinates : list
            Optional. List of coordinates to include in the output. If not specified, all coordinates will be included.

        Returns
        -------
        asdf_cutouts : `astropy.table.Table`
            Table with columns for input file, coordinate, and the corresponding `asdf.AsdfFile` object representing
            the cutout.
        """
        if self._asdf_cutouts is not None:
            return self._return_filtered_table(self._asdf_cutouts, input_files=input_files, coordinates=coordinates)

        return self._build_cutout_table(
            self.iter_asdf_cutouts(input_files=input_files, coordinates=coordinates),
        )

    def iter_asdf_cutouts(
        self,
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> Iterator[Tuple[str, SkyCoord, asdf.AsdfFile]]:
        """
        Yield ASDF cutouts lazily for the selected file/coordinate pairs.

        Parameters
        ----------
        input_files : list
            Optional. List of input image files to include in the output. If not specified, all input files will be
            included.
        coordinates : list
            Optional. List of coordinates to include in the output. If not specified, all coordinates will be included.

        Yields
        ------
        tuple
            Tuples of (input file, coordinate, `asdf.AsdfFile`).
        """
        for row in self._return_filtered_table(self.cutouts, input_files=input_files, coordinates=coordinates):
            file = row["file"]
            coord_obj = row["coordinate"]
            cutout = row["cutout"]
            coord_key = coord_obj.to_string(precision=8)
            tree = self._asdf_trees[file][coord_key]

            af = asdf.AsdfFile(tree)
            ra, dec = coord_key.split()
            af.add_history_entry(
                f"Cutout of size {cutout.shape} at sky coordinates ({ra}, {dec})",
                software={
                    "name": "astrocut",
                    "author": "Space Telescope Science Institute",
                    "version": __version__,
                    "homepage": "https://astrocut.readthedocs.io/en/latest/",
                },
            )
            yield file, coord_obj, af

    def get_fits_cutouts(
        self,
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> Table:
        """
        Get the cutouts as an `astropy.table.Table` with columns for input file, coordinate, and the
        corresponding `astropy.io.fits.HDUList` object.

        Parameters
        ----------
        input_files : list
            Optional. List of input image files to include in the output. If not specified, all input files will be
            included.
        coordinates : list
            Optional. List of coordinates to include in the output. If not specified, all coordinates will be included.

        Returns
        -------
        fits_cutouts : `astropy.table.Table`
            Table with columns for input file, coordinate, and the corresponding `astropy.io.fits.HDUList` object
            representing the cutout.
        """
        if self._fits_cutouts is not None:
            return self._return_filtered_table(self._fits_cutouts, input_files=input_files, coordinates=coordinates)

        return self._build_cutout_table(
            self.iter_fits_cutouts(input_files=input_files, coordinates=coordinates),
            object_cutout=True,
        )

    def iter_fits_cutouts(
        self,
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> Iterator[Tuple[str, SkyCoord, fits.HDUList]]:
        """
        Yield FITS cutouts lazily for the selected file/coordinate pairs.

        Parameters
        ----------
        input_files : list
            Optional. List of input image files to include in the output. If not specified, all input files will be
            included.
        coordinates : list
            Optional. List of coordinates to include in the output. If not specified, all coordinates will be included.

        Yields
        ------
        tuple
            Tuples of (input file, coordinate, `astropy.io.fits.HDUList`).
        """
        self._check_asdf_in_fits_support()

        today_str = str(date.today())
        for row in self._return_filtered_table(self.cutouts, input_files=input_files, coordinates=coordinates):
            file = row["file"]
            coord_obj = row["coordinate"]
            cutout = row["cutout"]
            coord_key = coord_obj.to_string(precision=8)

            source_tree = self._asdf_trees[file][coord_key]
            # Build a metadata-only tree for FITS embedding without mutating cached ASDF trees.
            tree = {self._mission_kwd: {"meta": source_tree[self._mission_kwd]["meta"]}}

            ra, dec = coord_key.split()
            if self._primary_header_template is None:
                self._primary_header_template = fits.Header(
                    [
                        ("ORIGIN", "STScI/MAST"),
                        ("PROCVER", __version__),
                    ]
                )
            primary_header = self._primary_header_template.copy()
            primary_header["RA_OBJ"] = float(ra)
            primary_header["DEC_OBJ"] = float(dec)
            primary_header["DATE"] = today_str
            primary_hdu = fits.PrimaryHDU(header=primary_header)

            image_hdu = fits.ImageHDU(data=cutout.data, header=cutout.wcs.to_header(relax=True))
            image_hdu.header["ORIG_FLE"] = file
            image_hdu.header["EXTNAME"] = "CUTOUT"
            hdul = fits.HDUList([primary_hdu, image_hdu])

            if self._asdf_in_fits is not None:
                hdul = self._asdf_in_fits.to_hdulist(tree, hdul)

            yield file, coord_obj, hdul

    def _check_asdf_in_fits_support(self):
        """
        Check if the `stdatamodels` package is available and meets the version requirement for ASDF-in-FITS embedding.
        """
        if self._asdf_in_fits is not None:
            return

        # Try to import stdatamodels for ASDF-in-FITS embedding
        if self._py311_or_higher:
            try:
                # Check version of stdatamodels
                from stdatamodels import __version__ as stdata_version
                from stdatamodels import asdf_in_fits

                if Version(stdata_version) < Version("4.1.0"):
                    warnings.warn(
                        "The `stdatamodels` package is not available in the correct version (>=4.1.0); "
                        "ASDF-in-FITS embedding will be skipped for these cutouts. Install the optional "
                        'dependency with: pip install "astrocut[all]" or pip install stdatamodels>=4.1.0',
                        ModuleWarning,
                    )
                else:
                    self._asdf_in_fits = asdf_in_fits
            except ImportError:
                warnings.warn(
                    "The `stdatamodels` package cannot be imported; ASDF-in-FITS embedding will be "
                    "skipped for these cutouts. Install the optional dependency with: "
                    'pip install "astrocut[all]" or pip install stdatamodels>=4.1.0',
                    ModuleWarning,
                )
        else:
            warnings.warn(
                "ASDF-in-FITS embedding requires Python 3.11 or higher. Skipping embedding for these cutouts.",
                ModuleWarning,
            )

    def get_image_cutouts(
        self,
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
        stretch: Optional[str] = "asinh",
        minmax_percent: Optional[List[int]] = None,
        minmax_value: Optional[List[int]] = None,
        invert: Optional[bool] = False,
        colorize: Optional[bool] = False,
        flip_orientation: Optional[bool] = True,
    ) -> Table:
        """
        Get the cutouts as an `astropy.table.Table` with columns for input file, coordinate, and the corresponding
        `~PIL.Image` object representing the image cutout.

        Parameters
        ----------
        input_files : list
            Optional. List of input image files to include in the output. If not specified, all input files will be
            included.
        coordinates : list
            Optional. List of coordinates to include in the output. If not specified, all coordinates will be included.
        stretch : str
            Optional, default 'asinh'. The stretch to apply to the image array.
            Valid values are: asinh, sinh, sqrt, log, linear
        minmax_percent : array
            Optional. Interval based on a keeping a specified fraction of pixels (can be asymmetric)
            when scaling the image. The format is [lower percentile, upper percentile], where pixel
            values below the lower percentile and above the upper percentile are clipped.
            Only one of minmax_percent and minmax_value should be specified.
        minmax_value : array
            Optional. Interval based on user-specified pixel values when scaling the image.
            The format is [min value, max value], where pixel values below the min value and above
            the max value are clipped.
            Only one of minmax_percent and minmax_value should be specified.
        invert : bool
            Optional, default False.  If True the image is inverted (light pixels become dark and vice versa).
        colorize : bool
            Optional, default False. If True, the first three cutouts will be combined into a single RGB image.
        flip_orientation : bool
            Optional, default True. If True, the cutout images are flipped vertically to match the orientation
            of the input images.

        Returns
        -------
        image_cutouts : `astropy.table.Table`
            Table with columns for input file(s), coordinate, and the corresponding `~PIL.Image` object representing
            the image cutout.
        """
        image_iter = self.iter_image_cutouts(
            input_files=input_files,
            coordinates=coordinates,
            stretch=stretch,
            minmax_percent=minmax_percent,
            minmax_value=minmax_value,
            invert=invert,
            colorize=colorize,
            flip_orientation=flip_orientation,
        )
        return self._build_cutout_table(image_iter, object_cutout=True)

    def iter_image_cutouts(
        self,
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
        stretch: Optional[str] = "asinh",
        minmax_percent: Optional[List[int]] = None,
        minmax_value: Optional[List[int]] = None,
        invert: Optional[bool] = False,
        colorize: Optional[bool] = False,
        flip_orientation: Optional[bool] = True,
    ) -> Iterator[Tuple[str, SkyCoord, Image]]:
        """
        Yield image cutouts lazily for the selected file/coordinate pairs.

        Parameters
        ----------
        input_files : list
            Optional. List of input image files to include in the output. If not specified, all input files will be
            included.
        coordinates : list
            Optional. List of coordinates to include in the output. If not specified, all coordinates will be included.
        stretch : str
            Optional, default 'asinh'. The stretch to apply to the image array.
            Valid values are: asinh, sinh, sqrt, log, linear
        minmax_percent : array
            Optional. Interval based on a keeping a specified fraction of pixels (can be asymmetric)
            when scaling the image. The format is [lower percentile, upper percentile], where pixel
            values below the lower percentile and above the upper percentile are clipped.
            Only one of minmax_percent and minmax_value should be specified.
        minmax_value : array
            Optional. Interval based on user-specified pixel values when scaling the image.
            The format is [min value, max value], where pixel values below the min value and above
            the max value are clipped.
            Only one of minmax_percent and minmax_value should be specified.
        invert : bool
            Optional, default False. If True the image is inverted (light pixels become dark and vice versa).
        colorize : bool
            Optional, default False. If True, the first three cutouts will be combined into a single RGB image.
        flip_orientation : bool
            Optional, default True. If True, the cutout images are flipped vertically to match the orientation
            of the input images.

        Yields
        ------
        tuple
            Tuples of (input file(s), coordinate, `~PIL.Image`).
        """
        yield from self._iter_image_cutout_rows(
            input_files=input_files,
            coordinates=coordinates,
            stretch=stretch,
            minmax_percent=minmax_percent,
            minmax_value=minmax_value,
            invert=invert,
            colorize=colorize,
            flip_orientation=flip_orientation,
        )

    def _iter_selected_cutouts(
        self,
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> Iterator[Tuple[str, SkyCoord, Cutout2D]]:
        """
        Yield selected ASDF cutouts as (file, coordinate, cutout).
        """
        for file, coord in self.iter_file_coord_pairs(input_files=input_files, coordinates=coordinates):
            yield file, SkyCoord(coord, unit="deg"), self.cutouts_by_file[file][coord]

    def _warn_too_many_color_cutouts(self, coord: SkyCoord):
        """
        ASDF-specific warning when more than three cutouts exist for one coordinate.
        """
        warnings.warn(
            f"More than 3 cutouts found for coordinate {coord.to_string(precision=8)}. "
            "Only the first three will be used for the color cutout.",
            InputWarning,
        )

    def _handle_insufficient_color_cutouts(self, coord: SkyCoord) -> bool:
        """
        ASDF-specific policy for insufficient RGB inputs: warn and skip this coordinate.
        """
        warnings.warn(
            f"Color cutouts require 3 input images (RGB) for coordinate {coord}. "
            "If you supplied 3 images one of the cutouts may have been empty.",
            InputWarning,
        )
        return False

    def iter_file_coord_pairs(
        self,
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> Iterator[Tuple[str, str]]:
        """
        Yield tuples for each valid (input_file, coordinate) pair.

        Parameters
        ----------
        input_files : list
            Optional. List of input image files to include in the output. If not specified, all input files will be
            included.
        coordinates : list
            Optional. List of coordinates to include in the output. If not specified, all coordinates will be included.

        Returns
        -------
        iterator
            Iterator of (input_file, coordinate) for valid pairs.
        """
        files_to_include, coords_to_include = self._resolve_selection(input_files, coordinates)
        for file in files_to_include:
            for coord in coords_to_include:
                if coord not in self.cutouts_by_file[file]:
                    continue  # Skip coordinates that are not associated with this file
                yield file, coord

    def _resolve_selection(
        self,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> Tuple[List[Union[str, Path, S3Path]], List[SkyCoord]]:
        """
        Resolve the input files and coordinates to include in the cutout results.

        Parameters
        ----------
        input_files : list, optional
            List of input files to include. If None, all files in the cutout results are included.
        coordinates : list, optional
            List of coordinates to include. If None, all coordinates in the cutout results are included.

        Returns
        -------
        files_to_include : list
            List of input files to include in the cutout results.
        coords_to_include : list
            List of string coordinates to include in the cutout results.
        """
        # Determine which files to include
        if input_files is None:
            files_to_include = list(self.cutouts_by_file.keys())
        else:
            files_to_include = input_files if isinstance(input_files, (list, tuple)) else [input_files]
            files_to_include = [str(file) for file in files_to_include]

            for file in files_to_include:
                if file not in self.cutouts_by_file:
                    raise InvalidInputError(f"Input file {file} is not in the cutout results.")

        # Determine which coordinates to include
        all_coords = []
        for file in files_to_include:
            all_coords.extend(self.cutouts_by_file[file].keys())
        # Remove duplicates while preserving order
        all_coords = list(dict.fromkeys(all_coords))

        if coordinates is None:
            coords_to_include = all_coords
        else:
            coords_to_include = self._normalize_coordinates_input(coordinates)
            for i, coord in enumerate(coords_to_include):
                if isinstance(coord, SkyCoord):
                    coords_to_include[i] = coord.to_string(precision=8)
                elif isinstance(coord, str):
                    try:
                        coords_to_include[i] = SkyCoord(coord, unit="deg").to_string(precision=8)
                    except Exception as e:
                        raise InvalidInputError(f"Invalid coordinate string: {coord}. Error: {e}")
                else:
                    raise InvalidInputError(f"Coordinate {coord} is not a valid SkyCoord or string.")

            for coord in coords_to_include:
                if coord not in all_coords:
                    raise InvalidInputError(f"Input coordinate {coord} is not in the cutout results.")

        return files_to_include, coords_to_include

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
            raise InvalidQueryError("Cutout contains no data! (Check image footprint.)")

        # Log total time elapsed
        log.debug("Total time: %.2f sec", monotonic() - start_time)

        return self.cutouts

    def _get_cloud_file(self, input_file: Union[str, S3Path]):
        """
        Open a cloud-hosted file using fsspec.

        Parameters
        ----------
        input_file : str | S3Path
            The input file S3 URI.

        Returns
        -------
        file-like object
            An open binary file handle for the cloud resource.
        """
        # Import fsspec here to avoid adding it as a dependency for users who don't need cloud support
        import fsspec

        fsspec_kwargs = {}
        if self._key is None and self._secret is None and self._token is None:
            fsspec_kwargs["anon"] = True
        else:
            if self._key is not None:
                fsspec_kwargs["key"] = self._key
            if self._secret is not None:
                fsspec_kwargs["secret"] = self._secret
            if self._token is not None:
                fsspec_kwargs["token"] = self._token

        return fsspec.open(input_file, mode="rb", **fsspec_kwargs)

    def _convert_gwcs_to_fits_wcs(self, gwcsobj: gwcs.wcs.WCS) -> WCS:
        """
        Convert a GWCS object to an approximated FITS WCS object.

        Parameters
        ----------
        gwcsobj : gwcs.wcs.WCS
            The GWCS object to convert.

        Returns
        -------
        wcs_updated : `~astropy.wcs.WCS`
            The approximated FITS WCS object.
        """
        if gwcsobj in self._gwcs_to_fits_cache:
            return self._gwcs_to_fits_cache[gwcsobj]

        # Convert the gwcs object to an astropy FITS WCS header
        header = gwcsobj.to_fits_sip()

        # Update WCS header with some keywords that it's missing.
        # Otherwise, it won't work with astropy.wcs tools (TODO: Figure out why. What are these keywords for?)
        for k in ["cpdis1", "cpdis2", "det2im1", "det2im2", "sip"]:
            if k not in header:
                header[k] = "na"

        # New WCS object with updated header
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs_updated = WCS(header)

        self._gwcs_to_fits_cache[gwcsobj] = wcs_updated
        return wcs_updated

    def _get_fill_value(self, dtype: np.dtype) -> Union[int, float]:
        """
        Get the appropriate fill value for a given data type, converting if necessary.

        Parameters
        ----------
        dtype : np.dtype
            The data type of the input array.

        Returns
        -------
        fill_value : int | float
            The fill value converted to the appropriate type if necessary.
        """
        if dtype in self._fill_value_cache:
            return self._fill_value_cache[dtype]

        fill_value = self._fill_value
        if np.issubdtype(dtype, np.integer) and not isinstance(fill_value, int):
            log.debug("Input data array has integer data type, converting fill_value to integer.")
            try:
                fill_value = int(self._fill_value)
            except ValueError:
                fill_value = 0  # Default to 0 if conversion fails

        self._fill_value_cache[dtype] = fill_value
        return fill_value

    def _make_cutout(self, array: np.ndarray, position: tuple, wcs: WCS) -> Cutout2D:
        """
        Helper to generate a Cutout2D and return plain ndarray data.

        Parameters
        ----------
        array : np.ndarray
            The input data array.
        position : tuple
            The (x, y) position of the cutout center.
        wcs : WCS
            The WCS object associated with the input array.

        Returns
        -------
        cutout : Cutout2D
            The generated cutout.
        """
        # If the array has an integer data type, fill_value must be an integer
        fill_value = self._get_fill_value(array.dtype)

        cutout = Cutout2D(
            array,
            position=position,
            wcs=wcs,
            size=(self._cutout_size[1], self._cutout_size[0]),
            mode="partial",
            fill_value=fill_value,
            # Keep cutouts detached from source arrays so downstream serialization
            # does not preserve references to full-size parent data.
            copy=True,
        )

        # Strip units if present
        if isinstance(cutout.data, Quantity):
            cutout.data = cutout.data.value

        return cutout

    def _apply_cutout_slices(self, array: np.ndarray, data_cutout: Cutout2D) -> np.ndarray:
        """
        Apply an existing Cutout2D footprint to another aligned array.

        Parameters
        ----------
        array : np.ndarray
            The input array to apply the cutout slices to.
        data_cutout : Cutout2D
            The Cutout2D object containing the original cutout slices.

        Returns
        -------
        result : np.ndarray
            The cutout array with the same shape as the input array, where the cutout region is filled
            with data from the input array and the rest is filled with the fill value.
        """
        orig_slices = data_cutout.slices_original
        cutout_slices = data_cutout.slices_cutout
        out_shape = data_cutout.data.shape
        fill_value = self._get_fill_value(array.dtype)

        # Build a result array for the cutout filled with the fill value
        result = np.empty(array.shape[:-2] + out_shape, dtype=array.dtype)

        # Insert original data into the cutout region of the result array
        result[..., cutout_slices[0], cutout_slices[1]] = array[..., orig_slices[0], orig_slices[1]]

        # Fill the rest of the result array with the fill value
        result[..., : cutout_slices[0].start, :] = fill_value
        result[..., cutout_slices[0].stop :, :] = fill_value
        result[..., :, : cutout_slices[1].start] = fill_value
        result[..., :, cutout_slices[1].stop :] = fill_value

        return result

    def _get_cutout_data(self, mission_tree: dict, wcs: WCS, pixel_coords: Tuple[int, int]) -> Cutout2D:
        """
        Get the cutout data from the input image.

        Parameters
        ----------
        mission_tree : dict
            The mission-specific tree of the input file.
        wcs : `~astropy.wcs.WCS`
            The approximated WCS of the input image.
        pixel_coords : tuple
            The pixel coordinates closest to the center of the cutout.

        Returns
        -------
        img_cutout : `~astropy.nddata.Cutout2D`
            The cutout object.
        """
        # Shape of data array
        mission_data = mission_tree["data"]
        data_shape = mission_data.shape

        # Make data cutout
        data_cutout = self._make_cutout(mission_data, pixel_coords, wcs)

        # If full cutout, apply the same cutout slices to other arrays in the mission tree that
        # are aligned with the data array, i.e. have the same shape in the last two dimensions
        if not self._lite:
            for key, obj in mission_tree.items():
                if not isinstance(obj, (np.ndarray, NDArrayType)):
                    continue  # Skip non-array objects

                shape = obj.shape
                if shape[-2:] != data_shape[-2:]:
                    continue  # Skip arrays not aligned with science data

                log.debug("Original %s shape: %s", key, shape)

                arr_cutout = self._apply_cutout_slices(obj, data_cutout)
                mission_tree[key] = arr_cutout

                log.debug("%s cutout shape: %s", key, arr_cutout.shape)

        return data_cutout

    def _slice_gwcs(self, cutout: Cutout2D, gwcs: gwcs.wcs.WCS) -> gwcs.wcs.WCS:
        """
        Slice the original gwcs object.

        "Slices" the original gwcs object down to the cutout shape.  This is a hack
        until proper gwcs slicing is in place a la fits WCS slicing.  The ``slices``
        keyword input is a tuple with the x, y cutout boundaries in the original image
        array, e.g. ``cutout.slices_original``.  Astropy Cutout2D slices are in the form
        ((ymin, ymax, None), (xmin, xmax, None))

        Parameters
        ----------
        cutout : astropy.nddata.Cutout2D
            The cutout object.
        gwcs : gwcs.wcs.WCS
            The original GWCS from the input image.

        Returns
        -------
        gwcs.wcs.WCS
            The sliced GWCS object.
        """
        # Create copy of original gwcs object
        tmp = deepcopy(gwcs)

        # Get the cutout array bounds and create a new shift transform to the cutout
        # Add the new transform to the gwcs
        slices = cutout.slices_original
        xmin, xmax = slices[1].start, slices[1].stop
        ymin, ymax = slices[0].start, slices[0].stop
        shape = (xmax - xmin, ymax - ymin)
        offsets = models.Shift(xmin, name="cutout_offset1") & models.Shift(ymin, name="cutout_offset2")
        tmp.insert_transform("detector", offsets, after=True)

        # Modify the gwcs bounding box to the cutout shape
        tmp.bounding_box = ((0, shape[0] - 1), (0, shape[1] - 1))
        tmp.pixel_shape = shape
        tmp.array_shape = shape[::-1]
        return tmp

    def _cutout_file(self, file: Union[str, Path, S3Path]):
        """
        Create a cutout from a single input file.

        Parameters
        ----------
        file : str | Path | S3Path
            The input file to create a cutout from.
        """
        input_file = str(file)
        cloud_file = None
        # If file comes from AWS cloud bucket, open it with fsspec and pass the file handle to ASDF.
        if (isinstance(file, str) and file.startswith("s3://")) or isinstance(file, S3Path):
            cloud_file = self._get_cloud_file(file)

        if cloud_file is not None:
            asdf_file = cloud_file
        else:
            asdf_file = nullcontext(file)

        with asdf_file as file_handle:
            with asdf.open(file_handle, **self._asdf_open_kwargs) as af:
                # Load the data from the input file
                tree = af.tree
                mission_tree = tree[self._mission_kwd] if self._mission_kwd in tree else None
                if mission_tree is None:
                    warnings.warn(
                        f"File {input_file} does not contain the expected mission keyword '{self._mission_kwd}'. "
                        "Skipping...",
                        DataWarning,
                    )
                    return

                # Skip if the file does not contain a GWCS object
                gwcs = mission_tree["meta"].get("wcs", None)
                if gwcs is None:
                    warnings.warn(f"File {input_file} does not contain a GWCS object. Skipping...", DataWarning)
                    return

                new_mission_tree = {"meta": mission_tree.get("meta", {})}

                data_shape = mission_tree["data"].shape

                for key, value in mission_tree.items():
                    if isinstance(value, (np.ndarray, NDArrayType)):
                        if value.shape[-2:] == data_shape[-2:]:
                            new_mission_tree[key] = value

                # For each requested coordinate, attempt to make a cutout if it overlaps
                file_cutouts = {}  # dictionary to hold cutouts for this file, keyed by coordinate
                wcs = self._convert_gwcs_to_fits_wcs(gwcs)
                for coord in self._coordinates:
                    pixel_coords = get_center_pixel(gwcs, coord.ra.value, coord.dec.value)

                    if any(np.isnan(pixel_coords)):
                        warnings.warn(
                            f"Coordinate {coord} is outside the footprint of file {input_file}. "
                            "Skipping this coordinate for this file.",
                            DataWarning,
                        )
                        continue

                    # Make a per-coordinate copy because _get_cutout_data may modify the mission_tree
                    # in place if not in lite mode
                    coord_mission_tree = dict(new_mission_tree)

                    try:
                        data_cutout = self._get_cutout_data(coord_mission_tree, wcs, pixel_coords)
                    except NoOverlapError:
                        warnings.warn(
                            f"Cutout of {input_file} at {coord} does not overlap the image. "
                            "Skipping this coordinate for this file.",
                            DataWarning,
                        )
                        # Skip coordinates that do not overlap this file
                        continue

                    data = data_cutout.data
                    if not np.any(np.nan_to_num(data, nan=0.0)):
                        # No useful data for this coordinate in this file
                        warnings.warn(f"Cutout of {input_file} at {coord} contains no data, skipping...", DataWarning)
                        continue

                    # Store the Cutout2D object and associated metadata
                    coord_key = coord.to_string(precision=8)
                    file_cutouts[coord_key] = data_cutout
                    self.cutouts_by_file.setdefault(input_file, {})[coord_key] = data_cutout

                    sliced_gwcs = self._slice_gwcs(data_cutout, gwcs)

                    if self._lite:
                        lite_tree = {
                            self._mission_kwd: {
                                "meta": {"wcs": sliced_gwcs, "orig_file": input_file, "coordinate": coord_key},
                                "data": data,
                            }
                        }
                        self._asdf_trees.setdefault(input_file, {})[coord_key] = lite_tree
                    else:
                        coord_mission_tree["meta"]["wcs"] = sliced_gwcs
                        coord_mission_tree["meta"]["orig_file"] = input_file
                        coord_mission_tree["meta"]["coordinate"] = coord_key
                        self._asdf_trees.setdefault(input_file, {})[coord_key] = {self._mission_kwd: coord_mission_tree}

    def _make_cutout_filename(self, file: str, output_format: str, coord: str) -> str:
        """
        Generate a standardized filename for the cutout.

        Overrides the superclass method to include the '_lite' tag if applicable and the output format.

        Parameters
        ----------
        file : str
            The input file name.
        output_format : str
            The output format to write the cutout to. Options are '.fits' and '.asdf'.
        coord : str
            The coordinate string associated with the cutout.

        Returns
        -------
        filename : str
            The generated filename for the cutout.
        """
        ra, dec = coord.split()

        return "{}_{:.7f}_{:.7f}_{}-x-{}{}_astrocut{}".format(
            Path(file).stem,
            float(ra),
            float(dec),
            str(self._cutout_size[0]).replace(" ", ""),
            str(self._cutout_size[1]).replace(" ", ""),
            "_lite" if self._lite else "",
            output_format,
        )

    def write_as_fits(
        self,
        output_dir: Union[str, Path] = ".",
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> List[str]:
        """
        Write the cutouts to disk or memory in FITS format.

        Parameters
        ----------
        output_dir : str | Path
            The output directory to write the cutouts to. Defaults to the current directory.

        Returns
        -------
        list
            A list of paths to the cutout FITS files.
        """
        return self._write_as_format(
            output_format=".fits", output_dir=output_dir, input_files=input_files, coordinates=coordinates
        )

    def write_as_asdf(
        self,
        output_dir: Union[str, Path] = ".",
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> List[str]:
        """
        Write the cutouts to disk or memory in ASDF format.

        Parameters
        ----------
        output_dir : str | Path
            The output directory to write the cutouts to. Defaults to the current directory.
        validate_output : bool
            Whether to validate the output ASDF file. Defaults to True. Setting to False can
            speed up writing for large numbers of cutouts, but should only be used if you
            trust the output is valid.

        Returns
        -------
        list
            A list of paths to the cutout ASDF files.
        """
        return self._write_as_format(
            output_format=".asdf", output_dir=output_dir, input_files=input_files, coordinates=coordinates
        )

    def _write_as_format(
        self,
        output_format: str,
        output_dir: Union[str, Path] = ".",
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> List[str]:
        """
        Write the cutout to disk in the specified output format.

        Parameters
        ----------
        output_format : str
            The output format to write the cutout to. Options are '.fits' and '.asdf'.
        output_dir : str | Path
            The output directory to write the cutouts to

        Returns
        -------
        cutout_paths : list
            The path(s) to the cutout file(s) or the cutout memory objects.
        """
        if output_format == ".asdf":
            iterator = self.iter_asdf_cutouts(input_files=input_files, coordinates=coordinates)
        elif output_format == ".fits":
            iterator = self.iter_fits_cutouts(input_files=input_files, coordinates=coordinates)
        else:
            raise InvalidInputError(
                f'Output format {output_format} is not recognized. Valid options are ".asdf" and ".fits".'
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cutout_paths = []  # List to store paths to cutout files

        for file, coord_obj, cutout_obj in iterator:
            coord = coord_obj.to_string(precision=8)
            filename = self._make_cutout_filename(file, output_format, coord=coord)
            cutout_path = Path(output_dir, filename)

            if output_format == ".fits":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cutout_obj.writeto(cutout_path, overwrite=True, checksum=True)
            elif output_format == ".asdf":
                cutout_obj.write_to(cutout_path)

            cutout_paths.append(cutout_path.as_posix())

        log.debug("Cutout filepaths: %s", cutout_paths)
        return cutout_paths

    def write_as_img(
        self,
        *,
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
        stretch: Optional[str] = "asinh",
        minmax_percent: Optional[List[int]] = None,
        minmax_value: Optional[List[int]] = None,
        invert: Optional[bool] = False,
        colorize: Optional[bool] = False,
        output_format: str = ".jpg",
        output_dir: Union[str, Path] = ".",
        cutout_prefix: str = "cutout",
        flip_orientation: Optional[bool] = True,
    ) -> Union[str, List[str]]:
        """
        Write the cutout to memory or to a file in an image format. If colorize is set, the first 3 cutouts
        will be combined into a single RGB image. Otherwise, each cutout will be written to a separate file.

        Parameters
        ----------
        input_files : list
            Optional. List of input image files to include in the output. If not specified, all input files will be
            included.
        coordinates : list
            Optional. List of coordinates to include in the output. If not specified, all coordinates will be included.
        stretch : str
            Optional, default 'asinh'. The stretch to apply to the image array.
            Valid values are: asinh, sinh, sqrt, log, linear
        minmax_percent : array
            Optional. Interval based on a keeping a specified fraction of pixels (can be asymmetric)
            when scaling the image. The format is [lower percentile, upper percentile], where pixel
            values below the lower percentile and above the upper percentile are clipped.
            Only one of minmax_percent and minmax_value shoul be specified.
        minmax_value : array
            Optional. Interval based on user-specified pixel values when scaling the image.
            The format is [min value, max value], where pixel values below the min value and above
            the max value are clipped.
            Only one of minmax_percent and minmax_value should be specified.
        invert : bool
            Optional, default False.  If True the image is inverted (light pixels become dark and vice versa).
        colorize : bool
            Optional, default False. If True, the first three cutouts will be combined into a single RGB image.
        flip_orientation : bool
            Optional, default True. If True, the cutout images are flipped vertically to match the
            orientation of the input images.
        output_format : str
            Optional, default '.jpg'. The output format for the cutout image(s).
        output_dir : str | `~pathlib.Path`
            Optional, default '.'. The directory to write the cutout image(s) to.
        cutout_prefix : str
            Optional, default 'cutout'. The prefix to add to the cutout image file name.

        Returns
        -------
        cutout_path : List[Path]
            Path(s) to the written cutout files.

        Raises
        ------
        InvalidInputError
            If less than three inputs were provided for a colorized cutout.
        """
        # Parse the output format
        output_format = self._parse_output_format(output_format)

        image_cutouts = self.iter_image_cutouts(
            input_files=input_files,
            coordinates=coordinates,
            stretch=stretch,
            minmax_percent=minmax_percent,
            minmax_value=minmax_value,
            invert=invert,
            colorize=colorize,
            flip_orientation=flip_orientation,
        )

        # Create the output directory if it does not exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cutout_paths = []  # List to store paths to cutout files

        # Set up output files and write them
        if colorize:  # Combine first three cutouts into a single RGB image
            for _, coord, img in image_cutouts:
                # Write the colorized cutout to disk
                filename = "{}_{:.7f}_{:.7f}_{}-x-{}_astrocut{}".format(
                    cutout_prefix,
                    coord.ra.deg,
                    coord.dec.deg,
                    str(self._cutout_size[0]).replace(" ", ""),
                    str(self._cutout_size[1]).replace(" ", ""),
                    output_format,
                )

                # Attempt to write image to file
                cutout_path = Path(output_dir, filename).as_posix()
                success = self._save_img_to_file(img, cutout_path)
                if success:
                    cutout_paths.append(cutout_path)

        else:  # Write each cutout to a separate image file
            for file, coord, img in image_cutouts:
                filename = "{}_{:.7f}_{:.7f}_{}-x-{}_astrocut{}".format(
                    Path(file).stem,
                    coord.ra.deg,
                    coord.dec.deg,
                    str(self._cutout_size[0]).replace(" ", ""),
                    str(self._cutout_size[1]).replace(" ", ""),
                    output_format,
                )

                # Attempt to write image to file
                cutout_path = Path(output_dir, filename).as_posix()
                success = self._save_img_to_file(img, cutout_path)
                if success:
                    cutout_paths.append(cutout_path)

        log.debug("Cutout filepaths: {}".format(cutout_paths))
        return cutout_paths

    def write_as_zip(
        self,
        output_dir: Union[str, Path] = ".",
        filename: Union[str, Path, None] = None,
        *,
        output_format: str = ".asdf",
        input_files: Optional[List[Union[str, Path, S3Path]]] = None,
        coordinates: Optional[List[Union[SkyCoord, str]]] = None,
    ) -> str:
        """
        Package the ASDF or FITS cutouts into a zip archive without writing intermediates.

        Parameters
        ----------
        output_dir : str | Path, optional
            Directory where the zip will be created. Default '.'.
        filename : str | Path | None, optional
            Name (or path) of the output zip file. If not provided, defaults to
            'astrocut_{ra}_{dec}_{size}.zip'. If provided without a '.zip' suffix,
            the suffix is added automatically.
        output_format : str, optional
            Either '.asdf' (default) or '.fits'. Determines which in-memory representation is zipped.

        Returns
        -------
        str
            Path to the created zip file.
        """
        fmt = output_format.lower().strip()
        fmt = "." + fmt if not fmt.startswith(".") else fmt
        if fmt not in (".asdf", ".fits"):
            raise InvalidInputError("File format must be either '.asdf' or '.fits'")

        if filename is None:
            filename = f"astrocut_{fmt[1:]}_cutouts.zip"

        if fmt == ".asdf":

            def iterator_factory():
                return self.iter_asdf_cutouts(input_files=input_files, coordinates=coordinates)
        else:

            def iterator_factory():
                return self.iter_fits_cutouts(input_files=input_files, coordinates=coordinates)

        def build_entries():
            for file, coord_obj, cutout_obj in iterator_factory():
                coord = coord_obj.to_string(precision=8)
                arcname = self._make_cutout_filename(file, fmt, coord=coord)
                yield arcname, cutout_obj

        return self._write_cutouts_to_zip(output_dir=output_dir, filename=filename, build_entries=build_entries)


@deprecated_renamed_argument(
    "output_file",
    None,
    "1.0.0",
    warning_type=DeprecationWarning,
    message="`output_file` is non-operational and will be removed in a future version.",
)
def asdf_cut(
    input_files: List[Union[str, Path, S3Path]],
    ra: float,
    dec: float,
    cutout_size: int = 25,
    output_file: Union[str, Path] = "example_roman_cutout.fits",
    write_file: bool = True,
    fill_value: Union[int, float] = np.nan,
    output_dir: Union[str, Path] = ".",
    output_format: str = ".asdf",
    key: str = None,
    secret: str = None,
    token: str = None,
    lite: bool = True,
    verbose: bool = False,
) -> Cutout2D:
    """
    Takes one of more ASDF input files (`input_files`) and generates a cutout of designated size `cutout_size`
    around the given coordinates (`coordinates`). The cutout is written to a file or returned as an object.

    This function is maintained for backwards compatibility. For maximum flexibility, we recommend using the
    ``ASDFCutout`` class directly.

    Parameters
    ----------
    input_file : str | Path | S3Path
        The input ASDF file.
    ra : float
        The right ascension of the central cutout.
    dec : float
        The declination of the central cutout.
    cutout_size : int
        Optional, default 25. The image cutout pixel size.
        Note: Odd values for `cutout_size` generally result in a cutout that is more accurately
        centered on the target coordinates compared to even values, due to the symmetry of the
        pixel grid.
    output_file : str | Path
        Optional, default "example_roman_cutout.fits". The name of the output cutout file.
        This parameter is deprecated and will be removed in a future version.
    write_file : bool
        Optional, default True. Flag to write the cutout to a file or not.
    fill_value: int | float
        Optional, default `np.nan`. The fill value for pixels outside the original image.
    output_dir : str | Path
        Optional, default ".". The directory to write the cutout file(s) to.
    output_format : str
        Optional, default ".asdf". The format of the output cutout file. If `write_file` is False,
        then cutouts will be returned as `asdf.AsdfFile` objects if `output_format` is ".asdf" or
        as `astropy.io.fits.HDUList` objects if `output_format` is ".fits".
    key : string
        Default None. Access key ID for S3 file system. Only applicable if `input_file` is a
        cloud resource.
    secret : string
        Default None. Secret access key for S3 file system. Only applicable if `input_file` is a
        cloud resource.
    token : string
        Default None. Security token for S3 file system. Only applicable if `input_file` is a
        cloud resource.
    lite : bool
        Optional, default True. If True, the cutout will be created in "lite" mode,
        which means that it will only contain the data and an updated world coordinate system.
        If False, cutouts will be made from all arrays in the input file (e.g., data, error,
        uncertainty, variance, etc.) where the last two dimensions match the shape of the science data array.
        It also preserves all of the metadata from the input file.
    verbose : bool
        Default False. If True, intermediate information is printed.

    Returns
    -------
    response : str | list
        A list of cutout file paths if `write_file` is True, otherwise a list of cutout objects.
    """
    asdf_cutout = ASDFCutout(
        input_files,
        f"{ra} {dec}",
        cutout_size,
        fill_value,
        key=key,
        secret=secret,
        token=token,
        lite=lite,
        verbose=verbose,
    )

    if not write_file:  # Returns as Cutout2D objects
        return asdf_cutout.cutouts

    # Get output format in standard form
    output_format = f".{output_format}" if not output_format.startswith(".") else output_format
    output_format = output_format.lower()

    if output_format == ".asdf":
        return asdf_cutout.write_as_asdf(output_dir)
    elif output_format == ".fits":
        return asdf_cutout.write_as_fits(output_dir)
    else:
        # Error if output format not recognized
        raise InvalidInputError(
            f'Output format {output_format} is not recognized. Valid options are ".asdf" and ".fits".'
        )


def get_center_pixel(gwcsobj: gwcs.wcs.WCS, ra: float, dec: float) -> Tuple[Tuple[int, int], WCS]:
    """
    Get the closest pixel location on an input image for a given set of coordinates.

    Parameters
    ----------
    gwcsobj : gwcs.wcs.WCS
        The GWCS object.
    ra : float
        The right ascension of the input coordinates.
    dec : float
        The declination of the input coordinates.

    Returns
    -------
    pixel_coords : tuple
        The (row, col) pixel coordinates of the input coordinates.
    """
    # Map the coordinates to a pixel's location on the 2d image
    row, col = gwcsobj.invert(np.atleast_1d(ra), np.atleast_1d(dec), with_bounding_box=False)
    row_pix = float(row.value[0]) if isinstance(row, Quantity) else float(row[0])
    col_pix = float(col.value[0]) if isinstance(col, Quantity) else float(col[0])
    pixel_coords = (row_pix, col_pix)

    return pixel_coords
