0.8 (unreleased)
----------------

- No changes yet
  

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
