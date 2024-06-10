0.12.0 (Unreleased)
--------------------

- asdf_cut() function now accepts pathlib.Path and s3path.S3Path objects as an input file [#119]

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
