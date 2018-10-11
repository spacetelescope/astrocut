
Astrocut Documentation
======================

Introduction
------------

Astrocut contains tools for creating image cutouts from sets images with
shared footprints. This package is under active development, and will
ultimately grow to encompass a range of cutout activities relevant to
images from many missions, however at this time it is focused on the
specific problem of creating image cutouts from sectors of TESS full
frame images (FFIs).

There are two parts to this package, the `~astrocut.CubeFactory` class
allows you to create a large image cube from a list of FFI files.
This is what allows the cutout operation to be performed efficiently.
The `~astrocut.CutoutFactory` class performs the actual cutout and builds
a target pixel file (TPF) that is compatible with TESS pipeline TPFs.

Getting Started
---------------

The basic workflow is to first create an image cube from individual FFI files
(this is one-time work), and then make individual cutout TPFs from this
large cube file. If you are doing a small number of cutouts, it may make
sense for you to use our tesscut web service:
`mast.stsci.edu/tesscut <https://mast.stsci.edu/tesscut/>`_
 
Making image cubes
++++++++++++++++++

Making an image cube is a simple operation, but comes with a vert important
limitation:

.. warning::
   **Memory Requirements**

   The entire cube file must be able to fit in your computer's memory!

   For a sector of TESS FFI images from a single camera/chip combination this is ~50 GB.

This operation can also take some time to run. For the 1348 FFI images of the TESS ete-6
simulated sector, it takes about 12 minutes to run on a computer with 65 GB of memory.

By default *make_cube* runs in verbose mode and prints out it's progress, however setting
verbose to false will silence all output.


.. code-block:: python

                >>> from astrocut import CubeFactory
                >>> from glob import glob
                >>> from astropy.io import fits
                >>> 
                >>> my_cuber = CubeFactory()
                >>> input_files = glob("data/*ffic.fits")
                >>> 
                >>> cube_file = my_cuber.make_cube(input_files)
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

                >>> print(cube_file)
                img-cube.fits

                >>> cube_hdu = fits.open(cube_file)
                >>> cube_hdu.info()
                Filename: img-cube.fits
                No.    Name      Ver    Type      Cards   Dimensions   Format
                0  PRIMARY       1 PrimaryHDU      28   ()      
                1                1 ImageHDU         9   (2, 144, 2136, 2078)   float32   
                2                1 BinTableHDU    302   144R x 147C   [24A, J, J, J, J, J, J, D, 24A, J, 24A, 24A, J, J, D, 24A, 24A, 24A, J, D, 24A, D, D, D, D, 24A, 24A, D, D, D, D, D, 24A, D, D, D, D, J, D, D, D, D, D, D, D, D, D, D, D, D, J, J, D, J, J, J, J, J, J, J, J, J, J, D, J, J, J, J, J, J, D, J, J, J, J, J, J, D, J, J, J, J, J, J, D, J, J, J, J, J, J, J, J, 24A, D, J, 24A, 24A, D, D, D, D, D, D, D, D, J, J, D, D, D, D, D, D, J, J, D, D, D, D, D, D, D, D, D, D, D, D, 24A, J, 24A, 24A, J, J, D, 24A, 24A, J, J, D, D, D, D, J, 24A, 24A, 24A]  


Making cutout target pixel files
++++++++++++++++++++++++++++++++

To make a cutout, you must already have an image cube to cut out from.
Assuming that that step has been completed, you simply give the central
coordinate and cutout size (in either pixels or angular `~astropy.Quanitity`)
to the *cube_cut* function.

You can either specify a target pixel file name, or it will be built as:
"<cube_file_base>_<ra>_<dec>_<cutout_size>_astrocut.fits". You can optionally
also specify a output path, the directory in which the target pixel file will
be saved, if unspecified it defaults to the current directory.

.. code-block:: python

                >>> from astrocut import CutoutFactory
                >>> from astropy.io import fits
                >>> 
                >>> my_cutter = CutoutFactory()
                >>> cube_file = "img-cube.fits"
                >>> 
                >>> cutout_file = my_cutter(cube_file, "251.51 32.36", 5, verbose=True)
                Cutout center coordinate: 251.51,32.36
                xmin,xmax: [26 31]
                ymin,ymax: [149 154]
                Image cutout cube shape: (144, 5, 5)
                Uncertainty cutout cube shape: (144, 5, 5)
                Target pixel file: img_251.51_32.36_5x5_astrocut.fits
                Write time: 0.016 sec
                Total time: 0.18 sec

                >>> cutout_hdu = fits.open(cutout_file)
                >>> cutout_hdu.info()
                Filename: img_251.51_32.36_5x5_astrocut.fits
                No.    Name      Ver    Type      Cards   Dimensions   Format
                0  PRIMARY       1 PrimaryHDU      42   ()      
                1  PIXELS        1 BinTableHDU    222   144R x 12C   [D, E, J, 25J, 25E, 25E, 25E, 25E, J, E, E, 38A]   
                2  APERTURE      1 ImageHDU        45   (5, 5)   float64  


.. automodapi:: astrocut
    :skip: test
    :skip: UnsupportedPythonError
    :no-inheritance-diagram:
