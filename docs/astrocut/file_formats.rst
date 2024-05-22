:orphan:
   
*********************
Astrocut File Formats
*********************

FITS Cutout Files
=================

FITS files output by `~astrocut.fits_cut` consist of a PrimaryHDU extension
and one or more ImageHDU extensions, each containing a single cutout.

PRIMARY PrimaryHDU (Extension 0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

========= ===================================================
Keyword   Value
========= ===================================================
SIMPLE    T (conforms to FITS standard)                     
BITPIX    8 (array data type)                               
NAXIS     0 (number of array dimensions)                    
EXTEND    Number of standard extensions                                                  
ORIGIN    STScI/MAST
DATE      File creation date                             
PROCVER   Software version                      
RA_OBJ    Center coordinate right ascension (deg)                         
DEC_OBJ   Center coordinate declination (deg)                             
CHECKSUM  HDU checksum
DATASUM   Data unit checksum
========= ===================================================

CUTOUT ImageHDU (Subsequent extension(s))
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The data in each CUTOUT extension is the cutout image. The header includes all of the
keywords from the extension that the cutout image was drawn from, with WCS keywords
updated to match the cutout image. Additionally the keyword ``ORIG_FLE`` has been added,
it contains the name of the file the cutout comes from.



ASDF Cutout Files
==================

ASDF files output by `asdf_cut` are a minimal tree structure that mirrors the format of the original Roman image file.

.. code-block:: python

    asdf_cutout = {
        "roman": {
            "meta": {
                "wcs" - the gwcs of the cutout
            },
            "data" - the cutout data
        }
    }

`wcs` is the original `gwcs` object from the input ASDF file that has been sliced into the shape of the cutout.



Cube Files
==========

See the `TESS Science Data Products Description Document <https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf#page=17>`__
for detailed information on the TESS full-frame image file format. See the `TESS Image CAlibrator Full Frame Images page <https://archive.stsci.edu/hlsp/tica>`__
for information on the format of the TICA HLSP full-frame images.


PrimaryHDU (Extension 0)
^^^^^^^^^^^^^^^^^^^^^^^^

The Primary Header of the TESS Mission, SPOC cube FITS file is the same as that from
an individual FFI with the following exceptions:

========= ===================================================
Keyword   Value
========= ===================================================
 ORIGIN   STScI/MAST
 DATE     Date the cube was created
 CAMERA   From the ImageHDU (EXT 1) of an FFI
 CCD      From the ImageHDU (EXT 1) of an FFI
 SECTOR   The TESS observing Sector, passed by the user
 DATE-OBS From the ImageHDU (EXT 1) of the Sector's first FFI
 DATE-END From the ImageHDU (EXT 1) of the Sector's last FFI
 TSTART   From the ImageHDU (EXT 1) of the Sector's first FFI
 TSTOP    From the ImageHDU (EXT 1) of the Sector's last FFI
========= ===================================================

The Primary Header of the TICA cube FITS file is the same as that from
an individual TICA FFI with the following exceptions:

========= ===================================================
Keyword   Value
========= ===================================================
 ORIGIN   STScI/MAST
 DATE     Date the cube was created
 SECTOR   The TESS observing Sector, passed by the user
 STARTTJD From the PrimaryHDU (EXT 0) of the first TICA FFI in the Sector
 MIDTJD   From the PrimaryHDU (EXT 0) of the middle TICA FFI in the Sector
 ENDTJD   From the PrimaryHDU (EXT 0) of the last TICA FFI in the Sector
 MJD-BEG  From the PrimaryHDU (EXT 0) of the first TICA FFI in the Sector
 MJD-END  From the PrimaryHDU (EXT 0) of the last TICA FFI in the Sector
========= ===================================================

ImageHDU (Extension 1)
^^^^^^^^^^^^^^^^^^^^^^

The ImageHDU extension contains the TESS (or TICA) FFI data cube.
It is 4 dimensional, with two spatial dimensions, time, data and
error flux values. Note, error flux values are only included in the 
cubes generated from SPOC products. Pixel values are 32 bit floats.
The cube dimensions are ordered in the FITS format as follows:

========= ===================================================
Keyword   Value
========= ===================================================
NAXIS     4 (number of array dimensions)                    
NAXIS1    2 (For SPOC products, data value, error value) or 1 (For TICA products, data value only)
NAXIS2    Total number of FFIs
NAXIS3    Length of first array dimension (NAXIS1 from FFIs)
NAXIS4    Length of second array dimension (NAXIS2 from FFIs)
========= ===================================================


BinTableHDU (Extension 2)
^^^^^^^^^^^^^^^^^^^^^^^^^

The BinTableHDU extension, in both the SPOC and TICA cubes, contains a table that 
holds all of the Image extension header keywords from the individual FFIs. There 
is one column for each keyword plus one additional column called "FFI_FILE" that 
contains FFI filename for each row. Each column name keyword also has an entry in the 
Image extension header, with the value being the keyword value from the FFI header.
This last column allows the FFI Image extension headers to be recreated completely if desired.


Target Pixel Files
==================

The Astrocut target pixel file (TPF) format conforms as closely as possible to the
TESS Mission TPFs. See the `TESS Science Data Products Description Document <https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf#page=23>`__
for detailed information on the TESS Mission TPF format, here it is
described how Astrocut TPFs differ from Mission pipeline TPFs.

PRIMARY PrimaryHDU (Extension 0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Primary Header of an Astrocut TPF is the same as that from
a Mission TPF with the following exceptions:

========= ====================================================
Keyword   Value
========= ====================================================
ORIGIN    STScI/MAST
CREATOR   astrocut
PROCVER   Astrocut version
SECTOR    Depends on this value having been filled in the cube

 **Mission pipeline header values Astrocut cannot populate**
--------------------------------------------------------------
OBJECT    ""
TCID      0
PXTABLE   0
PMRA      0.0
PMDEC     0.0
PMTOTAL   0.0
TESSMAG   0.0
TEFF      0.0
LOGG      0.0
MH        0.0
RADIUS    0.0
TICVER    0
TICID     None
========= ====================================================

PIXELS BinTableHDU (Extension 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Astrocut PIXELS BinTableHDU comprises the same columns as those included in
the Mission pipeline TPFs, with one addition: an extra column, ``FFI_FILE``, contains
the name of the FFI file that the row's pixels come from.

While all of the columns present in Mission pipeline TPFs are present in cutouts created
from SPOC cubes, they do not all contain data. The columns that are empty in Astrocut SPOC TPFs are:

============ ====================================================
Column       Value
============ ====================================================
CADENCENO    0 filled array in cutout shape
RAW_CNTS     -1 filled array in cutout shape
FLUX_BKG     0 filled array in cutout shape
FLUX_BKG_ERR 0 filled array in cutout shape
POS_CORR1    0
POS_CORR2    0
============ ====================================================

The ``TIME`` column is formed by taking the average of the ``TSTART`` and ``TSTOP`` values
from the corresponding FFI for each row. The ``QUALITY`` column is taken from the ``DQUALITY``
image keyword in the individual SPOC FFI files.

For cutouts created from TICA cubes, the ``TIMECORR`` column has been removed from the
PIXELS BinTableHDU. Similar to cutouts made from SPOC cubes, the other columns (aside from
the ``TIMECORR`` column) present in Mission pipeline TPFs are present in cutouts created
from TICA cubes, but do not all contain data. The columns that are empty in Astrocut TICA TPFs are:

============ ====================================================
Column       Value
============ ====================================================
RAW_CNTS     -1 filled array in cutout shape
FLUX_ERR     0 filled array in cutout shape
FLUX_BKG     0 filled array in cutout shape
FLUX_BKG_ERR 0 filled array in cutout shape
QUALITY      0
POS_CORR1    0
POS_CORR2    0
============ ====================================================

Three keywords have also been added to the PIXELS extension header to give additional information
about the cutout world coordinate system (WCS). TESS FFIs are large and therefore are described
by WCS objects that have many non-linear terms. Astrocut creates a new simpler (linear) WCS
object from the matched set of cutout pixel coordinates and sky coordinates (from the FFI WCS).
This linear WCS object will generally work very well, however at larger cutout sizes (100-200
pixels per side and above) the linear WCS fit will start to be noticeably incorrect at the edges
of the cutout. The extra keywords allow the user to determine if the linear WCS is accurate enough
for their purpose, and to retrieve the original WCS with distortion coefficients if it is needed.


+---------+----------------------------------------------------------------+
| Keyword |  Value                                                         |
+=========+================================================================+
| WCS_FFI | | The name of the FFI file used to build the original WCS      |
|         | | from which the cutout and cutout WCS were calculated.        |
+---------+----------------------------------------------------------------+
| WCS_MSEP| | The maximum separation in degrees between the cutout's       |
|         | | linear WCS and the FFI's full WCS.                           |
+---------+----------------------------------------------------------------+
| WCS_SIG | | The error in the cutout's linear WCS, calculated as          |
|         | | ``sqrt((dist(Po_ij, Pl_ij)^2)`` where ``dist(Po_ij, Pl_ij)`` |
|         | | is the angular distance in degrees between the sky position  |
|         | | of of pixel i,j in the original full WCS and the new linear  |
|         | | WCS.                                                         |
+---------+----------------------------------------------------------------+


APERTURE ImageHDU (Extension 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The APERTURE ImageHDU extension is similar to that of Mission pipeline TPFs, but contains
slightly different data. For Mission pipeline files, the aperture image gives information about
each pixel, whether it was collected and whether it was used in calculating e.g., the background flux.
Because Astrocut does not do any of the more complex calculations used in the Mission pipeline, each pixel in the
aperture image will either be 1 (pixel was collected and contains data in the cutout) or 0
(pixel is off the edge of the detector and contains no data in the cutout).


Cosmic Ray Binary Table Extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This extension is not present in Astrocut TPFs, although it is a part of the Mission pipeline TPFs.


Path Focused Target Pixel Files
===============================

When the `~astrocut.center_on_path` function is used to create cutout TPFs
where the individual image cutouts move along a path in time and space, the TPF format has to be
adjusted accordingly. It still conforms as closely as possible to the TESS Mission pipeline TPF
file format, but differs in several crucial ways. The `~astrocut.center_on_path` function works
on Astrocut TPFs, so that is the baseline file format. Only the differences
between path focused Astrocut TPFs and regular Astrocut TPFs are described here (see `Target Pixel Files`_ for
regular Astrocut TPF format).

PRIMARY PrimaryHDU (Extension 0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additional or updated keywords:

========= =======================================================
Keyword   Value
========= =======================================================
DATE      Set the the time the path focused cutout was performed
OBJECT    Moving target object name/identifier, only present if
          set by the user
========= =======================================================

Removed keywords:

========= =======================================================
Keyword   Reason
========= =======================================================
RA_OBJ    Cutout is no longer centered on a sky position
DEC_OBJ   Cutout is no longer centered on a sky position
========= =======================================================


PIXELS BinTableHDU (Extension 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additional columns:

============ ========================================================
Column       Value
============ ========================================================
TGT_X        X position of the target in the cutout array at row time
TGT_Y        Y position of the target in the cutout array at row time
TGT_RA       Right ascension (deg) of the target at row time
TGT_DEC      Declination (deg) of the target at row time
============ ========================================================

No world coordinate system (WCS) information is present, since it is no
longer common across all cutout images.


APERTURE ImageHDU (Extension 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The APERTURE extension may or may not be present in a path focussed TPF, to be present
the user must have passed an FFI WCS object into the `~astrocut.center_on_path` function.

The APERTURE ImageHDU extension of path focussed TPFs is very different from other
TESS TPFs. The aperture image, instead of being the size and shape of an individual cutout,
is the size of the full FFI image the cutouts were drawn from. All pixels used in any
individual cutout are marked with 1, while the rest of the pixels are 0, so the entire
trajectory of the cutout path is captured. Additionally the WCS information in the header
is the WCS for the original FFI, including all distortion coefficients. This can be
used in combination with the TGT_RA/DEC and TGT_X/Y columns to trace the path of the
target across the FFI footprint and calculate the WCS object for individual cutout images
if necessary.






