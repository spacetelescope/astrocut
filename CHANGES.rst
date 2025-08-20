1.1.0 (Unreleased)
-------------------

- Bugfix for transposed GWCS bounding box for ASDF cutouts. [#160]
- Bugfix to correct ``array_shape`` and ``pixel_shape`` for GWCS objects. [#160]
- By default, ``ASDFCutout`` makes cutouts of all arrays in the input file (e.g., data, error, uncertainty, variance, etc.)
  where the last two dimensions match the shape of the science data array. [#158]
- By default, ASDF cutouts now preserve all metadata from the input file. [#158]
- Add ``lite`` parameter to ``ASDFCutout`` to create minimal cutouts with only the science data and updated world coordinate system. [#158]
- Add history entry to ASDF cutouts specifying the cutout shape and center coordinates. [#158]
- Remove TICA (TESS Image Calibration) as an option for the ``product`` parameter in ``TessFootprintCutout``. [#161]
- Deprecate the ``TicaCubeFactory`` class. [#161]
- Deprecate the ``product`` parameter in the ``TessCubeCutout`` class, the ``TessFootprintCutout`` class, the ``cube_cut`` function,
  the ``CutoutFactory.cube_cut`` function, and the ``cube_cutout_from_footprint`` function. [#161]


1.0.1 (2025-05-12)
-------------------

- Bugfix so ``ASDFCutout.get_center_pixel`` preserves the GWCS bounding box. [#154]
- Bugfix in ``ASDFCutout`` to use deep copies of data and GWCS to avoid links to original ASDF input.

1.0.0 (2025-04-28)
-------------------

- Introduce generalized cutout architecture with ``Cutout``, ``ImageCutout``, and ``FITSCutout`` classes. [#136]
- Deprecate ``correct_wcs`` parameter in ``fits_cut`` as non-operational. [#136]
- Add ``ASDFCutout`` class as a specialized cutout class for ASDF files. [#137]
- Allow ``ASDFCutout`` and ``asdf_cut`` to accept multiple input files. [#137]
- Deprecated ``output_file`` parameter in ``asdf_cut`` in favor of making outputs from a batch of input files.. [#137]
- Return ASDF cutouts in memory as ``astropy.nddata.Cutout2D`` objects, ``asdf.AsdfFile`` objects, or ``astropy.io.fits.HDUList`` objects. [#137]
- Enable output of ASDF cutouts in image formats. [#137]
- Refactor ``TicaCubeFactory`` to inherit from ``CubeFactory``. [#143]
- Optimize ``CubeFactory._update_info_table`` to open FITS files only once. [#143]
- Add ``TessCubeCutout`` class as a concrete implementation of abstract ``CubeCutout`` with TESS-specific logic. [#146]
- Introduce ``TessCubeCutout.CubeCutoutInstance`` inner class for per-cutout attributes. [#146]
- Enable in-memory output for ``TessCubeCutout`` instances. [#146]
- Add ``TessFootprintCutout`` class as a concrete implementation of abstract ``FootprintCutout`` with TESS-specific logic. [#149]
- Enable in-memory output for ``TessFootprintCutout`` instances. [#149]
- Bugfix so ASDF cutouts store a copy of the cutout data rather than a view into the original data. [#153]


0.12.0 (2025-01-21)
--------------------

- Implement and document ``cube_cut_from_footprint`` function to generate cutouts from TESS image cube files hosted on the S3 cloud. [#127]
- Bugfix to properly catch input TICA product files in ``CubeFactory``. [#129]
- Add a logging framework. [#131]
- Improve performance of FITS image cutouts by using the ``section`` attribute of ``ImageHDU`` objects to access data more efficiently. [#132]
- Bugfix when writing multiple output files to memory in ``fits_cut``. [#132]


0.11.1 (2024-07-31)
--------------------

- ``asdf_cut`` function now accepts `pathlib.Path` and `s3path.S3Path` objects as an input file. [#119]
- Bugfix for accessing private resources on the cloud in the ``asdf_cut`` function. [#121]
- Add ``key``, ``secret``, and ``token`` parameters to ``asdf_cut`` for accessing private S3 buckets. [#124]


0.11.0 (2024-05-28)
--------------------

- Add functionality for creating cutouts from the ASDF file format [#105]
- Update ASDF cutout function to support an `astropy.Quantity` object as input data [#114]
- Return an `astropy.nddata.Cutout2D` object from ASDF cutout function [#114]
- Preserve original cutout shape when requesting an ASDF cutout that is partially outside of image bounds [#115]
- Output ASDF cutout as either a FITS file or an ASDF file [#116]
- Support S3 URI strings as input to ASDF cutout function [#117]
- Drop support for Python 3.8 [#112]


0.10.0 (2023-10-23)
--------------------

- Improve file checking prior to cutout creation to avoid errors [#52]
- Fix broken tests from GitHub Actions CI run [#56]
- Fix error resulting from forward slash in target name [#55]
- MNT: Update codecov-action version to v2 [#53]
- Make cubes out of TICA FFIs [#59]
- Make cutouts out of TICA cubes [#60]
- Fix bug for not catching duplicate ffis [#69]
- Add max_memory arg to update_cube [#71]
- Hotfix for cube_cut checking for valid WCS info [#70]
- Add remote cutout functionality (requires astropy 5.2 or above) [#76]
- Error handling for CubeFactory and TicaCubeFactory [#85]
- Cutout in threadpool [#84]
- Document multithreading enhancement [#86]
- Remove error array dimension from TicaCubeFactory [#87]
- Adapt CutoutFactory to account for error-less TICA Cubes [#88]
- Update .readthedocs.yml with Python 3.11 [#89]
- Update cube and cutout unit tests [#90]
- Update docs to reflect changes in TICA cube format [#93]
- Cloud functionality for astrocut.fits_cut() [#95]
- Use GitHub Actions for publishing new releases to PyPI [#97]
- Update deprecated license_file kwd [#103]


0.9 (2021-08-10)
----------------

- Add cutout combine functionality [#45]


0.8 (2021-07-02)
----------------

- Add moving target cutout functionality [#40]
  

0.7 (2020-08-19)
----------------

- Add iterative cubing and user selected max memory [#35]


0.6 (2020-05-20)
----------------
- Update wcs fitting to match Astropy (and use Astropy when available) [#29]
- Limit the number of pixels used for WCS fitting to 100 [#30]
- Deprecate drop_after and handle inconsistant wcs keywords automatically [#31]
- Change the memmap access mode from ACCESS_COPY to ACCESS_READ to lower memory usage. [#33]


0.5 (2020-01-13)
----------------
- Adding fits_cut function [#17]
- Doc update (explain time column) [#19]
- Adding img_cut and normalize_img [#21]
- Improve cutout filenames, change minmax_cut to minmax_value [#24]
- Add error handling when reading data raises an exception [#28]

0.4 (2019-06-21)
----------------

- Adding more unit tests and coveralls setup [#11]
- Adding workaround for FFIs with bad WCS info [#12]
- Adding linear WCS approximation for cutouts [#14]


0.3 (2019-05-03)
----------------

- Formatting update. [#5]
- Making the sperture extension use integers. [#6]
- Setting the creator keyword to astrocute. [#7]
- Adding automated testing. [#8]
- Uniform formatting on target pixel file names. [#10]

0.2 (2018-12-05)
----------------

- Improved WCS handling
- Additional TESS keywords
- A handlful of bugfixes


0.1 (2018-10-26)
----------------

- Initial release.  Includes features!
