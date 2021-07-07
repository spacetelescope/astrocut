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
DEC_OBJ   Center coordinat declination (deg)                             
CHECKSUM  HDU checksum
DATASUM   Data unit checksum
========= ===================================================

CUTOUT ImageHDU (Subsequent extension(s))
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The data in each CUTOUT extension is the cutout image. The header includes all of the
keywords from the extension that the cutout image was drawn from, with WCS keywords
updated to match the cutout image. Additionally the keyword ``ORIG_FLE`` has been added,
it contains the name of the file the cutout comes from.



Cube Files
==========

See the `TESS Science Data Products Description Document <https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf#page=17>`__
for detailed information on the TESS full-frame image file format.


PrimaryHDU (Extension 0)
^^^^^^^^^^^^^^^^^^^^^^^^

The Primary Header of the TESS cube fits file is the same as that from
an individual FFI with the following exceptions:

========= ===================================================
Keyword   Value
========= ===================================================
 ORIGIN   STScI/MAST
 DATE     Date the cube was created
 CAMERA   From the ImageHDU (EXT 1) of an FFI
 CCD      From the ImageHDU (EXT 1) of an FFI
 Sector   The TESS observing sector, passed by the user
 DATE-OBS From the ImageHDU (EXT 1) of the sector's first FFI
 DATE-END From the ImageHDU (EXT 1) of the sector's last FFI
 TSTART   From the ImageHDU (EXT 1) of the sector's first FFI
 TSTOP    From the ImageHDU (EXT 1) of the sector's last FFI
========= ===================================================

ImageHDU (Extension 1)
^^^^^^^^^^^^^^^^^^^^^^

The ImageHDU extension contains the TESS FFI datacube.
It is 4 diminsional, the two spacial dimensions, time, and data vs.
error flux values. Pixel valules are 32 bit floats.
The cube dimensions are ordered in the FITS format as follows:

========= ===================================================
Keyword   Value
========= ===================================================
NAXIS     4 (number of array dimensions)                    
NAXIS1    2 (data value, error value)
NAXIS2    Total umber of FFis
NAXIS3    Length of first array dimension (NAXIS1 from FFIs)
NAXIS4    Length of second array dimension (NAXIS2 from FFIs)
========= ===================================================


BinTableHDU (Extension 2)
^^^^^^^^^^^^^^^^^^^^^^^^^

The BinTableHDU extension contains a table that holds all of the Image extension
header keywords from the individual FFIs. There is one column for each keyword
plus one additional column called "FFI_FILE" that contains FFI filename for each
row. Each column name keyword also has an entry in the extension header, with
the value being the keyword comment from the FFI header. This last allows the
FFI Image extension headers to be reacreated completely if desired.


Target Pixel Files
==================

The Astrocut target pixel file (TPF) format conforms as closely as possible to the
TESS mission TPFs. See the `TESS Science Data Products Description Document <https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf#page=23>`__
for detailed information on the TESS mission target pixel file format, here I
describe how Astrocut TPFs differ from mission pipeline TPFs.

PRIMARY PrimaryHDU (Extension 0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Primary Header of an Astrocut TPF is the same as that from
a mission TPF with the following exceptions:

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

The Astrocut PIXELS BinTableHDU extension comprises the same columns as that from
the mission pipeline, with one addition. The extra column is ``FFI_FILE`` and contains
the name of the FFI file that that row's pixels come from.

While all of the columns present in mission pipeline TPFs are present, they do not all
contain data. The columns that are empty in Astrocut TPFs are:

============ ====================================================
Column       Value
============ ====================================================
CADENCENO    0 filled array in cutout shape
RAW_CNTS     -1 filles array in cutout shape
FLUX_BKG     0 filled array in cutout shape
FLUX_BKG_ERR 0 filled array in cutout shape
POS_CORR1    0
POS_CORR2    0
============ ====================================================

The ``TIME`` column is formed by taking the average of the ``TSTART`` and ``TSTOP`` values
from the corresponding FFI for each row. The ``QUALITY`` column is taken from the ``DQUALITY``
image keyword in the individual FFI files.

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
| WCS_MSEP| | The maximum separation in degrees between the cutout’s       |
|         | | linear WCS and the FFI’s full WCS.                           |
+---------+----------------------------------------------------------------+
| WCS_SIG | | The error in the cutout’s linear WCS, calculated as          |
|         | | ``sqrt((dist(Po_ij, Pl_ij)^2)`` where ``dist(Po_ij, Pl_ij)`` |
|         | | is the angular distance in degrees between the sky position  |
|         | | of of pixel i,j in the original full WCS and the new linear  |
|         | | WCS.                                                         |
+---------+----------------------------------------------------------------+


APERTURE ImageHDU (Extension 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The APERTURE ImageHDU extension is similar to that of mission pipeline TPFs, but contains
slightly different data. For mission pipeline files, the aperture image gives information about
each pixel, whether it was collected and what calculations it was used in. Because astrocut does
not do any of the more complex claculations used in the mission pipeline, each pixel in the
aperture image will either be 1 (pixel was collected and contains data in the cutout) or 0
(pixel is off the edge of the detector and contains no data in the cutout).


Cosmic Ray Binary Table Extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This extension is not present in Astrocut TPFs, although it is a part of mission pipeline TPFs.


Path Focused Target Pixel Files
===============================

When the `~astrocut.center_on_path` function is used to create cutout target pixel files (TPFs)
where the individualimage cutouts move along a path in time and space, the TPF format has to be
adjusted accordingly. It still conformes as closely as possible to the TESS mission pipeline TPF
file format, but differs in several cruicial ways. The `~astrocut.center_on_path` function works
on astrocut TPFs, so that is the baseline file format. I will describe here only the differences
between path focussed astrocut TPFs and regular astrocut TPFs (see `Target Pixel Files`_ for
regular Astrocut TPF format).

PRIMARY PrimaryHDU (Extension 0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additional or updated keywords:

========= =======================================================
Keyword   Value
========= =======================================================
DATE      Set the the time the path focussed cutout was performed
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
TESS TPFs. The aperture image, instead of being the size and shape of an individeual cutout,
is the size of the full FFI image the cutouts were drawn from. All pixels used in any
individual cutout are marked with 1, while the rest of the pixels are 0, so the entire
trajectory of the cutout path is captured. Additionally the WCS information in the header
is the WCS for the original FFI, including all distortion coefficients. This can be
used in combination with the TGT_RA/DEC and TGT_X/Y columns to trace the path of the
target across the FFI footprint and calculate the WCS object for individual cutout images
if necessary.






