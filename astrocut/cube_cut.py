# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements the cutout functionality."""

from typing import Literal, Union

from .CubeCutout import CubeCutout


class CutoutFactory():
    """
    Class for creating image cutouts.

    This class encompasses all of the cutout functionality.  
    In the current version this means creating cutout target pixel files from both 
    SPOC (Science Processing Operations Center) and TICA (Tess Image CAlibration) 
    full frame image cubes.

    Future versions will include more generalized cutout functionality.
    """

    def cube_cut(
        self,
        cube_file,
        coordinates,
        cutout_size,
        product="SPOC",
        target_pixel_file=None,
        output_path=".",
        memory_only=False,
        threads: Union[int, Literal["auto"]] = 1,
        verbose=False,
    ):
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
        memory_only : bool
            Optional. If true, the cutout is made in memory only and not saved to disk.
            Default is False.
        threads : int, "auto", default=1
            Number of threads to use when making remote (e.g. s3) cutouts, will not use threads for local access
            <=1 disables the threadpool, >1 sets threadpool to the specified number of threads,
            "auto" uses `concurrent.futures.ThreadPoolExecutor`'s default: cpu_count + 4, limit to max of 32
        verbose : bool
            Optional. If true intermediate information is printed.

        Returns
        -------
        response: string or None
            If successful, returns the path to the target pixel file,
            if unsuccessful returns None.
        """
        cube_cutout = CubeCutout(input_files=cube_file,
                                 coordinates=coordinates,
                                 cutout_size=cutout_size,
                                 product=product,
                                 threads=threads,
                                 verbose=verbose)
        
        # Assign these attributes to be backwards compatible
        cutout_obj = cube_cutout.cutouts_by_file[cube_file]
        self.cube_wcs = cutout_obj.cube_wcs
        self.center_coord = cube_cutout._coordinates
        self.cutout_lims = cutout_obj.cutout_lims
        self.cutout_wcs = cutout_obj.wcs
        
        if memory_only:
            return cube_cutout.tpf_cutouts[0]
        
        return cube_cutout.write_as_tpf(output_dir=output_path, 
                                        output_file=target_pixel_file)[0]
