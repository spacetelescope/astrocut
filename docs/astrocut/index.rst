
**********************
Astrocut Documentation
**********************

 
Introduction
============

Astrocut provides tools for making cutouts from sets of astronomical images with shared footprints. It is under active development. 

Three main areas of functionality are included:

- Solving the specific problem of creating image cutouts from Sectors of Transiting Exoplanet Survey Satellite (TESS) full-frame images, and related High-Level Science Product images (TICA).
- General FITS file cutouts including from single images and sets of images with shared WCS/pixel scale.
- Cutout post-processing functionality, including centering cutouts along a path (for moving targets) and combining cutouts.



FITS File Image Cutouts
=======================

These functions provide general purpose astronomical cutout functionality for FITS files.
There are two main cutout functions, `~astrocut.fits_cut` for creating cutout FITS files,
and `~astrocut.img_cut` for creating cutout JPG or PNG files. An image normalization
(`~astrocut.normalize_img`) function is also available.

Creating FITS Cutouts
---------------------

The function `~astrocut.fits_cut` takes one or more FITS files and performs the same cutout
on each, returning the result either in a single FITS file or as one FITS file per cutout.
It is important to remember that while the expectation is that all input images are aligned
and have the same pixel scale, no checking is done by Astrocut.

The cutout FITS file format is described `here <file_formats.html#fits-cutout-files>`__.

.. code-block:: python

                >>> from astrocut import fits_cut
                >>> from astropy.io import fits
                >>> from astropy.coordinates import SkyCoord
                
                >>> input_files = ["https://archive.stsci.edu/pub/hlsp/candels/cosmos/cos-tot/v1.0/hlsp_candels_hst_acs_cos-tot-sect23_f606w_v1.0_drz.fits",
                ...                "https://archive.stsci.edu/pub/hlsp/candels/cosmos/cos-tot/v1.0/hlsp_candels_hst_acs_cos-tot-sect23_f814w_v1.0_drz.fits"]

                >>> center_coord = SkyCoord("150.0945 2.38681", unit='deg')
                >>> cutout_size = [200,300]
                
                >>> cutout_file = fits_cut(input_files, center_coord, cutout_size, single_outfile=True)  #doctest: +SKIP
                >>> print(cutout_file)    #doctest: +SKIP
                ./cutout_150.094500_2.386810_200-x-300_astrocut.fits

                >>> cutout_hdulist = fits.open(cutout_file)  #doctest: +SKIP
                >>> cutout_hdulist.info() #doctest: +SKIP
                Filename: ./cutout_150.094500_2.386810_200-x-300_astrocut.fits
                No.    Name      Ver    Type      Cards   Dimensions   Format
                  0  PRIMARY       1 PrimaryHDU      11   ()      
                  1  CUTOUT        1 ImageHDU        44   (200, 300)   float32   
                  2  CUTOUT        1 ImageHDU        44   (200, 300)   float32

                  
The cutout(s) can also be returned in memory as `~astropy.io.fits.HDUList` object(s).

.. code-block:: python

                >>> from astrocut import fits_cut
                >>> from astropy.io import fits
                >>> from astropy.coordinates import SkyCoord
                
                >>> input_files = ["https://archive.stsci.edu/pub/hlsp/candels/cosmos/cos-tot/v1.0/hlsp_candels_hst_acs_cos-tot-sect23_f606w_v1.0_drz.fits",
                ...                "https://archive.stsci.edu/pub/hlsp/candels/cosmos/cos-tot/v1.0/hlsp_candels_hst_acs_cos-tot-sect23_f814w_v1.0_drz.fits"]

                >>> center_coord = SkyCoord("150.0945 2.38681", unit='deg')
                >>> cutout_size = [200,300]
                
                >>> cutout_list = fits_cut(input_files, center_coord, cutout_size,
                ...                        single_outfile=False, memory_only=True)  #doctest: +SKIP
                >>> cutout_list[0].info() #doctest: +SKIP
                Filename: ./cutout_150.094500_2.386810_200-x-300_astrocut.fits
                No.    Name      Ver    Type      Cards   Dimensions   Format
                  0  PRIMARY       1 PrimaryHDU      11   ()      
                  1  CUTOUT        1 ImageHDU        44   (200, 300)   float32   
                  2  CUTOUT        1 ImageHDU        44   (200, 300)   float32

                  
`~astrocut.fits_cut` also has the ability to take FITS files from the cloud using S3 URIs. 

.. code-block:: python

                >>> from astrocut import fits_cut
                >>> from astropy.io import fits
                >>> from astropy.coordinates import SkyCoord

                >>> s3_uri = "s3://stpubdata/hst/public/j8pu/j8pu0y010/j8pu0y010_drz.fits"

                >>> center_coord = SkyCoord("150.42838	2.421421", unit='deg')
                >>> cutout_size = [100,100]
                
                >>> cutout_file = fits_cut(input_files, center_coord, cutout_size,
                ...                        single_outfile=True, memory_only=True)  #doctest: +SKIP
                >>> cutout_hdulist = fits.open(cutout_file)  #doctest: +SKIP
                >>> cutout_hdulist.info() #doctest: +SKIP
                Filename: ./cutout_150.428380_2.421421_100-x-100_astrocut.fits
                No.    Name      Ver    Type      Cards   Dimensions   Format
                  0  PRIMARY       1 PrimaryHDU      11   ()      
                  1  CUTOUT        1 ImageHDU        97   (100, 100)   float32
                

Creating Image Cutouts
----------------------
                  
The function `~astrocut.img_cut` takes one or more FITS files and performs the same cutout
on each, returning a single JPG or PNG file for each cutout.
It is important to remember that while the expectation is that all input images are
aligned and have the same pixel scale, no checking is done by Astrocut.

.. code-block:: python

                >>> from astrocut import img_cut
                >>> from astropy.coordinates import SkyCoord
                >>> from PIL import Image
                
                >>> input_files = ["https://archive.stsci.edu/pub/hlsp/candels/cosmos/cos-tot/v1.0/hlsp_candels_hst_acs_cos-tot-sect23_f606w_v1.0_drz.fits",
                ...                "https://archive.stsci.edu/pub/hlsp/candels/cosmos/cos-tot/v1.0/hlsp_candels_hst_acs_cos-tot-sect23_f814w_v1.0_drz.fits"]

                >>> center_coord = SkyCoord("150.0945 2.38681", unit='deg')
                >>> cutout_size = [200,300]
                
                >>> png_files = img_cut(input_files, center_coord, cutout_size, img_format='png')    #doctest: +SKIP
                >>> print(png_files[0])    #doctest: +SKIP
                ./hlsp_candels_hst_acs_cos-tot-sect23_f606w_v1.0_drz_150.094500_2.386810_200-x-300_astrocut.png

                >>> Image.open(png_files[1]) #doctest: +SKIP
                
.. image:: imgs/png_ex_cutout.png

Color images can also be produced using `~astrocut.img_cut`, given three input files, which will be
treated as the R, G, and B channels, respectively.

.. code-block:: python

                >>> from astrocut import img_cut
                >>> from astropy.coordinates import SkyCoord
                >>> from PIL import Image
                
                >>> input_files = ["https://archive.stsci.edu/pub/hlsp/goods/v2/h_nz_sect14_v2.0_drz_img.fits",
                ...                "https://archive.stsci.edu/pub/hlsp/goods/v2/h_ni_sect14_v2.0_drz_img.fits",
                ...                "https://archive.stsci.edu/pub/hlsp/goods/v2/h_nv_sect14_v2.0_drz_img.fits"]
                
                >>> center_coord = SkyCoord("189.51522 62.2865221", unit='deg')
                >>> cutout_size = [200,300]
                
                >>> color_image = img_cut(input_files, center_coord, cutout_size, colorize=True)   #doctest: +SKIP
                >>> print(color_image)    #doctest: +SKIP
                ./cutout_189.515220_62.286522_200-x-300_astrocut.jpg
                
                >>> Image.open(color_image) #doctest: +SKIP
                
.. image:: imgs/color_ex_cutout.png         


      
TESS Full-Frame Image Cutouts
=============================

There are two parts of the package involved in creating cutouts from TESS full-frame images (FFIs).
First, the `~astrocut.CubeFactory` (if working with SPOC products, or `~astrocut.TicaCubeFactory` if working
with TICA FFIs) class allows you to create a large image cube from a list of FFI files.
This is what allows the cutout operation to be performed efficiently.
Next, the `~astrocut.CutoutFactory` class performs the actual cutout and builds
a target pixel file (TPF) that is similar to the TESS Mission-produced TPFs.

The basic procedure is to first create an image cube from individual FFI files
(this only needs to be completed once per set of FFIs), and to then make individual cutout TPFs from this
large cube file for targets of interest. Note, you can only make cubes from a set of FFIs
with the same product type (i.e., only SPOC *or* only TICA products) that were observed in 
the same Sector, camera, and CCD.
If you are creating a small number of cutouts, the TESSCut web service
may suit your needs: `mast.stsci.edu/tesscut <https://mast.stsci.edu/tesscut/>`_
 
Making Image Cubes
------------------

.. important::
   **Time-Memory Trade-off**

   Making an image cube is a simple operation, but comes with an important
   time-memory trade-off.

   The ``max_memory`` argument determines the maximum memory in GB that will be used
   for the image data cube while it is being built. This is the amount of memory required 
   *only* for the data cube, so is somewhat smaller than the total amount of memory needed
   for the program to run. You should never set it to your system's total memory.

   Because of this, cube files do not need to allocate their total size in
   memory all at once. Instead, a smaller memory allocation can be used while
   the cube file is constructed; however, this will significantly increase the
   execution time as bytes are swapped into and out of the memory allocation 
   being used. The default value of 50 GB was chosen because it fits all of the
   TESS FFIs from a single Prime Mission Sector (Sectors 1-26); however, in the
   current TESS Extended Mission 2, where 6 times more FFIs are observed per Sector
   (compared to the number of FFIs observed per Sector in the Prime Mission), 50 GB
   is not enough space to hold all of the FFIs in memory, and the cubes will be 
   written in multiple blocks. With the default settings, on a system with 64 GB of
   memory, it takes about 3 hours to build a single cube file. On a system with less
   memory or where ``max_memory`` is set to a value less than 50 GB, more passes 
   through the list of files are required, and the time to create a cube can increase
   significantly.
   

Assuming that you have set of calibrated TESS (or TICA) FFI files stored locally, you can
create a cube using the `~astrocut.CubeFactory.make_cube` method (or 
`~astrocut.TicaCubeFactory.make_cube` for TICA products). By default, both `~astrocut.CubeFactory.make_cube` 
and `~astrocut.TicaCubeFactory.make_cube` run in verbose mode and prints out progress; setting `verbose` to false will silence
all output.

The output image cube file format is described `here <file_formats.html#cube-files>`__.

.. code-block:: python

                >>> from astrocut import CubeFactory
                >>> from glob import glob
                >>> from astropy.io import fits
                
                >>> my_cuber = CubeFactory()
                >>> input_files = glob("data/*ffic.fits") 
                
                >>> cube_file = my_cuber.make_cube(input_files) #doctest: +SKIP
                Completed file 0
                Completed file 1
                Completed file 2
                .
                .
                .
                Completed file 142
                Completed file 143
                Total time elapsed: 46.42 sec
                File write time: 8.82 sec

                >>> print(cube_file) #doctest: +SKIP
                img-cube.fits

                >>> cube_hdu = fits.open(cube_file) #doctest: +SKIP
                >>> cube_hdu.info()  #doctest: +SKIP
                Filename: img-cube.fits
                No.    Name      Ver    Type      Cards   Dimensions   Format
                0  PRIMARY       1 PrimaryHDU      28   ()      
                1                1 ImageHDU         9   (2, 144, 2136, 2078)   float32   
                2                1 BinTableHDU    302   144R x 147C   [24A, J, J, J, J, J, J, D, 24A, J, 24A, 24A, J, J, D, 24A, 24A, 24A, J, D, 24A, D, D, D, D, 24A, 24A, D, D, D, D, D, 24A, D, D, D, D, J, D, D, D, D, D, D, D, D, D, D, D, D, J, J, D, J, J, J, J, J, J, J, J, J, J, D, J, J, J, J, J, J, D, J, J, J, J, J, J, D, J, J, J, J, J, J, D, J, J, J, J, J, J, J, J, 24A, D, J, 24A, 24A, D, D, D, D, D, D, D, D, J, J, D, D, D, D, D, D, J, J, D, D, D, D, D, D, D, D, D, D, D, D, 24A, J, 24A, 24A, J, J, D, 24A, 24A, J, J, D, D, D, D, J, 24A, 24A, 24A]  


Making Cutout Target Pixel Files
--------------------------------

To make a cutout, you must already have an image cube from which to create the cutout.
Assuming that you have a TESS cube file stored locally, you can give the central
coordinate of your target of interest and cutout size (in either pixels or angular degrees/arcseconds `~astropy.Quantity`)
to the `~astrocut.CutoutFactory.cube_cut` function.

You can optionally specify an output TPF name; if no output name is provided, the file name will be built as:
"<cube_file_base>_<ra>_<dec>_<cutout_size>_astrocut.fits". You can optionally
also specify an output path, the directory in which the TPF will
be saved; if unspecified, this will default to the current directory.

The cutout target pixel file format is described `here <file_formats.html#target-pixel-files>`__.

.. code-block:: python

                >>> from astrocut import CutoutFactory
                >>> from astropy.io import fits

                >>> my_cutter = CutoutFactory()
                >>> cube_file = "img-cube.fits"

                >>> cutout_file = my_cutter.cube_cut(cube_file, "251.51 32.36", 5, verbose=True) #doctest: +SKIP
                Cutout center coordinate: 251.51,32.36
                xmin,xmax: [26 31]
                ymin,ymax: [149 154]
                Image cutout cube shape: (144, 5, 5)
                Uncertainty cutout cube shape: (144, 5, 5)
                Target pixel file: img_251.51_32.36_5x5_astrocut.fits
                Write time: 0.016 sec
                Total time: 0.18 sec

                >>> cutout_hdu = fits.open(cutout_file) #doctest: +SKIP
                >>> cutout_hdu.info() #doctest: +SKIP
                Filename: img_251.51_32.36_5x5_astrocut.fits
                No.    Name      Ver    Type      Cards   Dimensions   Format
                0  PRIMARY       1 PrimaryHDU      42   ()      
                1  PIXELS        1 BinTableHDU    222   144R x 12C   [D, E, J, 25J, 25E, 25E, 25E, 25E, J, E, E, 38A]   
                2  APERTURE      1 ImageHDU        45   (5, 5)   float64  


Cloud-based Cutouts
-------------------

You can also create cutout TPFs out of TESS cube files stored on MAST's AWS open data bucket.
Using cube files stored on the cloud allows you the option to implement multithreading to improve cutout generation
speed. See below for a multithreaded example, using a TESS cube file stored on AWS.

Multithreading
---------------

To use multithreading for cloud-based cutouts, set the ``threads`` argument in ``cube_cut`` to the number of threads you want to use. Alternatively, you
can set ``threads`` to ``"auto"``, which will set the number of threads based on the CPU count of your machine.
Note that ``Total Time`` results may vary from machine to machine.

.. code-block:: python

                >>> from astrocut import CutoutFactory
                >>> from astropy.coordinates import SkyCoord

                >>> my_cutter = CutoutFactory()
                >>> coord = SkyCoord(217.42893801, -62.67949189, unit="deg", frame="icrs")
                >>> cutout_size = 30
                >>> cube_file = "s3://stpubdata/tess/public/mast/tess-s0038-2-2-cube.fits"

                >>> cut_factory.cube_cut(cube_file, coordinates=coord, cutout_size=cutout_size,
                ...                      verbose=True, threads="auto") #doctest: +SKIP
                Using WCS from row 1852 out of 3705
                Cutout center coordinate: 217.42893801,-62.67949189
                xmin,xmax: [1572 1602]
                ymin,ymax: [852 882]
                Image cutout cube shape: (3705, 30, 30)
                Uncertainty cutout cube shape: (3705, 30, 30)
                Maximum distance between approximate and true location: 3.6009402965268847e-05 deg
                Error in approximate WCS (sigma): 0.0003207242331953156
                Target pixel file: ./tess-s0038-2-2_217.428938_-62.679492_30x30_astrocut.fits

                WARNING: VerifyWarning: Card is too long, comment will be truncated. [astropy.io.fits.card]

                Write time: 0.54 sec
                Total time: 4.3 sec

The same call made without multithreading enabled will result in a longer processing time, depending on the cutout size.
Note that multithreading is disabled by default.

.. code-block:: python

                >>> cut_factory.cube_cut(cube_file, coordinates=coord, cutout_size=cutout_size, 
                ...                      verbose=True) #doctest: +SKIP
                Using WCS from row 1852 out of 3705
                Cutout center coordinate: 217.42893801,-62.67949189
                xmin,xmax: [1572 1602]
                ymin,ymax: [852 882]
                Image cutout cube shape: (3705, 30, 30)
                Uncertainty cutout cube shape: (3705, 30, 30)
                Maximum distance between approximate and true location: 3.6009402965268847e-05 deg
                Error in approximate WCS (sigma): 0.0003207242331953156
                Target pixel file: ./tess-s0038-2-2_217.428938_-62.679492_30x30_astrocut.fits

                WARNING: VerifyWarning: Card is too long, comment will be truncated. [astropy.io.fits.card]

                Write time: 0.56 sec
                Total time: 7.8 sec



ASDF File Cutouts
===================

The Nancy Grace Roman Space Telescope will store its data using the Advanced Scientific Data Format (ASDF). With the `asdf_cut` function, users can create cutouts of Roman mission products.

Creating ASDF Cutouts
----------------------

The function `asdf_cut` performs a cutout of an ASDF file and returns the result as either a FITS file or an ASDF file. 
The format of the cutout is determined by the filename extension of the `output_file` parameter. In the below example, a cutout is written as a FITS file. 
The cutout FITS file format is described `here <file_formats.html#fits-cutout-files>`__.

.. code-block:: python

                >>> from astrocut import asdf_cut
                >>> from astropy.coordinates import SkyCoord
                >>> from astropy.io import fits
                
                >>> input_file = ""  # Path to local ASDF file or URI

                >>> center_coord = SkyCoord("80.15189743 29.74561219", unit='deg')
                
                >>> cutout_file = asdf_cut(input_file, center_coord.ra, center_coord.dec, cutout_size=200, 
                ...                        output_file="roman-demo.fits") #doctest: +SKIP

                >>> cutout_hdulist = fits.open(cutout_file)  #doctest: +SKIP
                >>> cutout_hdulist.info() #doctest: +SKIP
                Filename: roman-demo.fits
                No.    Name      Ver    Type      Cards   Dimensions   Format
                  0  PRIMARY       1 PrimaryHDU      25   (200, 200)   float32 

`asdf_cut` accepts S3 URIs to perform cutouts on ASDF files from the cloud.
In this example, a cutout is performed on a cloud file and written as an ASDF file. The cutout ASDF file format is described `here <file_formats.html#asdf-cutout-files>`__.

.. code-block:: python

                >>> from astrocut import asdf_cut
                >>> from astropy.coordinates import SkyCoord
                
                >>> s3_uri = "s3://..." # Complete URI

                >>> center_coord = SkyCoord("80.15189743 29.74561219", unit='deg')
                
                >>> cutout_file = asdf_cut(s3_uri, center_coord.ra, center_coord.dec, cutout_size=200, 
                ...                        output_file="roman-demo.asdf") #doctest: +SKIP

When requesting a cutout that is partially outside of image bounds, the `fill_value` parameter is used to preserve the cutout shape and fill outside pixels.



Additional Cutout Processing
============================

Path-based Cutouts
------------------

The `~astrocut.center_on_path` function allows the user to take one or more Astrocut cutout
TPF(s) and create a single cutout, centered on a moving target that crosses through
the file(s). The user can optionally pass in a target object name and FFI WCS object.

The output target pixel file format is described `here <file_formats.html#path-focused-target-pixel-files>`__.

This example starts with a path, and uses several `TESScut services <https://mast.stsci.edu/tesscut/docs/>`__
to retrieve all of the inputs for the `~astrocut.center_on_path` function. We also use the helper function
`~astrocut.path_to_footprints` that takes in a path table, cutout size, and WCS object, and returns the
cutout location/size(s) necesary to cover the entire path.

.. code-block:: python
  
                >>> import astrocut

                >>> import requests  #doctest: +SKIP

                >>> from astropy.table import Table
                >>> from astropy.coordinates import SkyCoord
                >>> from astropy.time import Time
                >>> from astropy.io import fits
                >>> from astropy import wcs

                >>> from astroquery.mast import Tesscut  #doctest: +SKIP

                >>> # The moving target path
                >>> path_table = Table({"time": Time([2458468.275827604, 2458468.900827604, 2458469.525827604,
                ...                                   2458470.150827604, 2458470.775827604], format="jd"),
                ...                     "position": SkyCoord([82.22813, 82.07676, 81.92551, 81.7746, 81.62425], 
                ...                                          [-1.5821,- 1.54791, -1.5117, -1.47359, -1.43369], unit="deg")
                ...                    })

                >>> # Getting the FFI WCS
                >>> resp = requests.get(f"https://mast.stsci.edu/tesscut/api/v0.1/ffi_wcs?sector=6&camera=1&ccd=1")  #doctest: +SKIP
                >>> ffi_wcs = wcs.WCS(resp.json()["wcs"], relax=True)  #doctest: +SKIP
                >>> print(ffi_wcs)  #doctest: +SKIP
                WCS Keywords

                Number of WCS axes: 2
                CTYPE : 'RA---TAN-SIP'  'DEC--TAN-SIP'  
                CRVAL : 86.239936828613  -0.87476283311844  
                CRPIX : 1045.0  1001.0  
                PC1_1 PC1_2  : 0.0057049915194511  7.5332427513786e-06  
                PC2_1 PC2_2  : -0.00015248404815793  0.005706631578505  
                CDELT : 1.0  1.0  
                NAXIS : 2136  2078

                >>> # Making the regular cutout (using astroquery)
                >>> size = [15,15]
                >>> footprints = astrocut.path_to_footprints(path_table["position"], size, ffi_wcs)  #doctest: +SKIP
                >>> print(footprints)  #doctest: +SKIP
                [{'coordinates': <SkyCoord (ICRS): (ra, dec) in deg
                     (81.92560877, -1.50880833)>, 'size': (37, 125)}]

                >>> manifest = Tesscut.download_cutouts(**footprints[0], sector=6)  #doctest: +SKIP
                Downloading URL https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra=81.92560876541987&dec=-1.5088083330171362&y=37&x=125&units=px&sector=6 to ./tesscut_20210707103901.zip ... [Done]
                Inflating...
                
                >>> print(manifest["Local Path"][0])  #doctest: +SKIP
                ./tess-s0006-1-1_81.925609_-1.508808_125x37_astrocut.fits

                # Centering on the moving target
                >>> mt_cutout_fle = astrocut.center_on_path(path_table, size, manifest["Local Path"], target="my_asteroid", 
                ...                                         img_wcs=ffi_wcs, verbose=False)  #doctest: +SKIP

                >>> cutout_hdu = fits.open(mt_cutout_fle)  #doctest: +SKIP
                >>> cutout_hdu.info()  #doctest: +SKIP
                Filename: ./my_asteroid_1468.9120483398438-1470.1412353515625_15-x-15_astrocut.fits
                No.    Name      Ver    Type      Cards   Dimensions   Format
                  0  PRIMARY       1 PrimaryHDU      56   ()      
                  1  PIXELS        1 BinTableHDU    152   60R x 16C   [D, E, J, 225J, 225E, 225E, 225E, 225E, J, E, E, 38A, D, D, D, D]   
                  2  APERTURE      1 ImageHDU        97   (2136, 2078)   int32  


Combining Cutouts
-----------------

The `~astrocut.CutoutsCombiner` class allows the user to take one or more Astrocut cutout
FITS files (as from  `~astrocut.fits_cut`) with a shared WCS object, and combine them into
a single cutout. This means that you should request the same cutout size in all of the images you want to combine.

The default setting combines the images with a mean combiner, such that every combined pixel is the mean of all
pixels that have data at that point. This mean combiner is made with the `~astrocut.build_default_combine_function`,
which takes the input image HDUs and allows the user to specify a null data value (default is NaN).

Users can write a custom combiner function, either by directly setting the
`~astrocut.CutoutsCombiner.combine_images` function, or by writing a custom combiner function builder
and passing it to the `~astrocut.CutoutsCombiner.build_img_combiner` function. The main reason to
write a function builder is that the `~astrocut.CutoutsCombiner.combine_images` function must work
*only* on the images being combined; any usage of header keywords, for example, must be set in that
function. See the `~astrocut.build_default_combine_function` for an example of how this works.



.. code-block:: python
  
                >>> import astrocut
                
                >>> from astropy.coordinates import SkyCoord

                >>> fle_1 = 'hst_skycell-p2381x05y09_wfc3_uvis_f275w-all-all_drc.fits'
                >>> fle_2 = 'hst_skycell-p2381x06y09_wfc3_uvis_f275w-all-all_drc.fits'

                >>> center_coord = SkyCoord("211.27128477 53.66062066", unit='deg')
                >>> size = [30,50]

                >>> cutout_1 = astrocut.fits_cut(fle_1, center_coord, size, extension='all',
                ...                     cutout_prefix="cutout_p2381x05y09", verbose=False)  #doctest: +SKIP
                >>> cutout_2 = astrocut.fits_cut(fle_2, center_coord, size, extension='all', 
                ...                     cutout_prefix="cutout_p2381x06y09", verbose=False)  #doctest: +SKIP

                >>> plt.imshow(fits.getdata(cutout_1, 1))  #doctest: +SKIP
                
.. image:: imgs/hapcut_left.png

.. code-block:: python
                
                >>> plt.imshow(fits.getdata(cutout_2, 1))  #doctest: +SKIP
                
.. image:: imgs/hapcut_right.png

.. code-block:: python

                >>> combined_cutout = astrocut.CutoutsCombiner([cutout_1, cutout_2]).combine("combined_cut.fits")  #doctest: +SKIP
                >>> plt.imshow(fits.getdata(combined_cutout, 1))  #doctest: +SKIP
                
.. image:: imgs/hapcut_combined.png        


All of the combining can be done in memory, without writing FITS files to disk as well.

.. code-block:: python
  
                >>> import astrocut
                
                >>> from astropy.coordinates import SkyCoord

                >>> fle_1 = 'hst_skycell-p2381x05y09_wfc3_uvis_f275w-all-all_drc.fits'
                >>> fle_2 = 'hst_skycell-p2381x06y09_wfc3_uvis_f275w-all-all_drc.fits'

                >>> center_coord = SkyCoord("211.27128477 53.66062066", unit='deg')
                >>> size = [30,50]

                >>> cutout_1 = astrocut.fits_cut(fle_1, center_coord, size, extension='all',
                ...                     cutout_prefix="cutout_p2381x05y09", memory_only=True)[0]  #doctest: +SKIP
                >>> cutout_2 = astrocut.fits_cut(fle_2, center_coord, size, extension='all', 
                ...                     cutout_prefix="cutout_p2381x06y09", memory_only=True)[0]  #doctest: +SKIP

                >>> plt.imshow(cutout_1[1].data)  #doctest: +SKIP
                
.. image:: imgs/hapcut_left.png

.. code-block:: python
                
                >>> plt.imshow(cutout_2[1].data)  #doctest: +SKIP
                
.. image:: imgs/hapcut_right.png

.. code-block:: python

                >>> combined_cutout = astrocut.CutoutsCombiner([cutout_1, cutout_2]).combine(memory_only=True)  #doctest: +SKIP
                >>> plt.imshow(combined_cutout[1].data)  #doctest: +SKIP
                
.. image:: imgs/hapcut_combined.png        
